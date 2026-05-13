import csv
import io
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cvat_stats.parser import parse_xml, to_image_df, read_xml_date
from cvat_stats.stats import label_stats, group_stats, custom_group_stats, user_progress, sorted_labels
from cvat_stats.history import (
    load_registry, register_project, delete_project, rename_project,
    migrate_flat_to_subdir,
    load_project_log, save_project_log_entry, delete_project_log_entry, build_progress_table,
    save_snapshot, load_previous_snapshot, compute_delta, delete_snapshot, list_snapshots,
)
from cvat_stats.exporter import export_excel

st.set_page_config(page_title="CVAT Statistics", page_icon="📊", layout="wide")

_TYPE_ICON  = {"tag": "🏷️", "rectangle": "📐", "polygon": "🔷"}
_TYPE_LABEL = {"tag": "Tags", "rectangle": "Rectangles", "polygon": "Polygons"}
_MIME = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv":  "text/csv",
    ".zip":  "application/zip",
}


@st.cache_data(show_spinner=False)
def _parse_cached(file_bytes: bytes, file_name: str):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    tmp.write(file_bytes)
    tmp.close()
    tmp_path = Path(tmp.name)
    parsed   = parse_xml(tmp_path)
    img_df   = to_image_df(parsed)
    xml_date = read_xml_date(tmp_path)
    tmp_path.unlink(missing_ok=True)
    return parsed, img_df, xml_date


def _excel_bytes(summary_df, label_df, group_df, img_df, user_df, delta_df, labels) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp_path = Path(tmp.name)
    export_excel(tmp_path, summary_df, label_df, group_df, img_df, user_df, delta_df, labels)
    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return data


def _detail_csv_bytes(img_df: pd.DataFrame, labels: list) -> bytes:
    cols = ["img_name", "task_name", "assignee", "is_labeled"] + labels
    available = [c for c in cols if c in img_df.columns]
    return img_df[available].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def _worker_csv_bytes(run_date: date, project_name: str, user_df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    fields = ["date", "project", "assignee", "assigned", "labeled", "unlabeled", "pct_labeled"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for _, row in user_df.iterrows():
        labeled  = int(row.get("labeled", 0))
        assigned = int(row.get("assigned", 0))
        writer.writerow({
            "date":        run_date.isoformat(),
            "project":     project_name,
            "assignee":    row.get("assignee", ""),
            "assigned":    assigned,
            "labeled":     labeled,
            "unlabeled":   int(row.get("unlabeled", 0)),
            "pct_labeled": f"{labeled/assigned*100:.1f}%" if assigned else "0.0%",
        })
    return buf.getvalue().encode("utf-8-sig")


def _make_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _pct_badge(pct: float) -> str:
    if pct >= 80: return "🟢"
    if pct >= 50: return "🟡"
    return "🔴"


def _merge_dfs(dfs: list, all_labels: list) -> pd.DataFrame:
    if len(dfs) == 1:
        return dfs[0].copy()
    label_cols = [
        col
        for lbl in all_labels
        for col in (lbl, f"bbox_{lbl}", f"horizontal_{lbl}", f"vertical_{lbl}", f"diagonal_{lbl}")
    ]
    aligned = []
    for df in dfs:
        df = df.copy()
        for col in label_cols:
            if col not in df.columns:
                df[col] = 0
        aligned.append(df)
    return pd.concat(aligned, ignore_index=True)


def _merge_labels_types(configs: list) -> tuple:
    seen: set   = set()
    ordered     = []
    merged_types: dict = {}
    for cfg in configs:
        for lbl in cfg["parsed"].meta.labels:
            if lbl in cfg["selected_labels"] and lbl not in seen:
                ordered.append(lbl)
                seen.add(lbl)
        merged_types.update(cfg["parsed"].meta.label_types)
    return sorted_labels(ordered, merged_types), merged_types


def _combined_summary_df(
    proj_name: str, source_names: list,
    df: pd.DataFrame, labels: list, label_types: dict,
) -> pd.DataFrame:
    total   = len(df)
    labeled = int(df["is_labeled"].sum())
    rows = [
        {"metric": "project_name",           "value": proj_name,                                          "percent": ""},
        {"metric": "source_files",           "value": " + ".join(source_names),                           "percent": ""},
        {"metric": "total_images",           "value": total,                                              "percent": ""},
        {"metric": "labeled_images",         "value": labeled,                                            "percent": ""},
        {"metric": "unlabeled_images",       "value": total - labeled,                                    "percent": ""},
        {"metric": "labeled_pct",            "value": f"{labeled/total*100:.1f}%" if total else "0.0%",   "percent": ""},
        {"metric": "selected_label_classes", "value": len(labels),                                        "percent": ""},
    ]
    for lbl in sorted_labels(labels, label_types):
        img_count  = int(df[lbl].sum())          if lbl           in df.columns else 0
        bbox_total = int(df[f"bbox_{lbl}"].sum()) if f"bbox_{lbl}" in df.columns else 0
        pct = f"{img_count / total * 100:.1f}%" if total else "0.0%"
        rows.append({"metric": f"images_with_{lbl}", "value": img_count, "percent": pct})
        rows.append({"metric": f"total_bbox_{lbl}",  "value": bbox_total, "percent": ""})
    return pd.DataFrame(rows)


def _validate_same_labels(file_configs: dict) -> Optional[str]:
    if len(file_configs) <= 1:
        return None
    items = list(file_configs.values())
    ref_labels = set(items[0]["parsed"].meta.labels)
    ref_name = items[0]["parsed"].meta.project_name
    errors = []
    for cfg in items[1:]:
        other_labels = set(cfg["parsed"].meta.labels)
        other_name = cfg["parsed"].meta.project_name
        if other_labels != ref_labels:
            missing = ref_labels - other_labels
            extra = other_labels - ref_labels
            parts = []
            if missing:
                parts.append(f"missing in `{other_name}`: {', '.join(sorted(missing))}")
            if extra:
                parts.append(f"extra in `{other_name}`: {', '.join(sorted(extra))}")
            errors.append(f"**{ref_name}** vs **{other_name}** — {'; '.join(parts)}")
    return "\n\n".join(errors) if errors else None


def _unique_proj_names(all_configs: list) -> list:
    seen: dict = {}
    names = []
    for i, cfg in enumerate(all_configs):
        raw = (cfg.get("display_name") or f"project_{i}")[:30]
        if raw in seen:
            raw = f"{raw[:27]}_{i}"
        seen[raw] = True
        names.append(raw)
    return names


def _wide_summary_df(configs: list, all_labels: list, merged_types: dict, proj_names: list) -> pd.DataFrame:
    proj_dfs = [cfg["img_df"] for cfg in configs]
    totals   = [len(df) for df in proj_dfs]
    labeleds = [int(df["is_labeled"].sum()) for df in proj_dfs]
    grand_total   = sum(totals)
    grand_labeled = sum(labeleds)

    def _r(metric, per_proj, total_val, pct=""):
        r = {"metric": metric}
        for n, v in zip(proj_names, per_proj):
            r[n] = v
        r["total"] = total_val
        r["percent"] = pct
        return r

    rows = [
        _r("total_images",   totals,   grand_total),
        _r("labeled_images", labeleds, grand_labeled),
        _r("unlabeled_images",
           [t - l for t, l in zip(totals, labeleds)],
           grand_total - grand_labeled),
        _r("labeled_pct",
           [f"{l/t*100:.1f}%" if t else "0.0%" for l, t in zip(labeleds, totals)],
           f"{grand_labeled/grand_total*100:.1f}%" if grand_total else "0.0%"),
        _r("selected_label_classes", [len(all_labels)] * len(configs), len(all_labels)),
    ]
    for lbl in sorted_labels(all_labels, merged_types):
        img_counts  = [int(df[lbl].sum())           if lbl           in df.columns else 0 for df in proj_dfs]
        bbox_counts = [int(df[f"bbox_{lbl}"].sum()) if f"bbox_{lbl}" in df.columns else 0 for df in proj_dfs]
        total_img   = sum(img_counts)
        pct = f"{total_img/grand_total*100:.1f}%" if grand_total else "0.0%"
        rows.append(_r(f"images_with_{lbl}", img_counts,  total_img,          pct))
        rows.append(_r(f"total_bbox_{lbl}",  bbox_counts, sum(bbox_counts)))
    return pd.DataFrame(rows)


def _wide_label_stats_df(configs: list, all_labels: list, merged_types: dict, proj_names: list) -> pd.DataFrame:
    proj_dfs     = [cfg["img_df"] for cfg in configs]
    proj_labeled = [int(df["is_labeled"].sum()) for df in proj_dfs]
    grand_labeled = sum(proj_labeled)

    rows = []
    for lbl in sorted_labels(all_labels, merged_types):
        img_counts  = [int(df[lbl].sum())           if lbl           in df.columns else 0 for df in proj_dfs]
        bbox_counts = [int(df[f"bbox_{lbl}"].sum()) if f"bbox_{lbl}" in df.columns else 0 for df in proj_dfs]
        total_imgs  = sum(img_counts)
        total_bbox  = sum(bbox_counts)
        pct = f"{total_imgs/grand_labeled*100:.1f}%" if grand_labeled else "0.0%"
        r = {"type": merged_types.get(lbl, "tag"), "class": lbl}
        for n, ic, bc in zip(proj_names, img_counts, bbox_counts):
            r[f"{n}_imgs"] = ic
            r[f"{n}_bbox"] = bc
        r["total_imgs"]    = total_imgs
        r["total_bbox"]    = total_bbox
        r["pct_of_labeled"] = pct
        rows.append(r)
    return pd.DataFrame(rows)


def _wide_group_stats_df(configs: list, all_labels: list, merged_types: dict, proj_names: list) -> pd.DataFrame:
    proj_dfs   = [cfg["img_df"] for cfg in configs]
    proj_totals = [len(df) for df in proj_dfs]
    grand_total = sum(proj_totals)

    geo_labels = [l for l in all_labels if merged_types.get(l) in ("rectangle", "polygon")]
    orient_map = [("horizontal", "horizontal"), ("vertical", "vertical"), ("diagonal", "diagonal")]

    rows = []
    for line_kw, line_name in [("dash", "dash"), ("solid", "solid")]:
        matched = [l for l in geo_labels if line_kw in l.lower()]
        for col_key, orient_name in orient_map:
            r: dict = {"line_type": line_name, "orientation": orient_name}
            total_imgs_all = total_bbox_all = 0
            for n, df, t in zip(proj_names, proj_dfs, proj_totals):
                orient_cols = [f"{col_key}_{l}" for l in matched if f"{col_key}_{l}" in df.columns]
                if orient_cols:
                    ti = int((df[orient_cols].sum(axis=1) > 0).sum())
                    tb = int(df[orient_cols].sum().sum())
                else:
                    ti = tb = 0
                r[f"{n}_imgs"] = ti
                r[f"{n}_bbox"] = tb
                r[f"{n}_pct"]  = f"{ti/t*100:.1f}%" if t else "0.0%"
                total_imgs_all += ti
                total_bbox_all += tb
            r["total_imgs"] = total_imgs_all
            r["total_bbox"] = total_bbox_all
            r["total_pct"]  = f"{total_imgs_all/grand_total*100:.1f}%" if grand_total else "0.0%"
            rows.append(r)
    return pd.DataFrame(rows)


def _tag_filter_section(img_df: pd.DataFrame, all_labels: list, label_types: dict):
    """Live tag-filter breakdown: pick a tag → see class counts/pct within that filtered set."""
    tag_labels = [l for l in all_labels if label_types.get(l) == "tag"]

    with st.expander("🔍 Tag Filter Breakdown", expanded=False):
        if not tag_labels:
            st.caption("No tag-type labels in the current selection.")
            return

        filter_tag = st.selectbox(
            "Filter images by tag",
            options=["(no filter)"] + tag_labels,
            key="tag_filter_sel",
            help="Only images that have this tag will be counted.",
        )

        if filter_tag == "(no filter)":
            st.caption("Select a tag above to filter images and see per-class breakdown.")
            return

        if filter_tag in img_df.columns:
            filtered_df = img_df[img_df[filter_tag] > 0]
        else:
            filtered_df = img_df.iloc[0:0]

        n_filtered = len(filtered_df)
        n_total    = len(img_df)
        pct_tag    = n_filtered / n_total * 100 if n_total else 0
        st.info(
            f"**{n_filtered}** of **{n_total}** images have tag `{filter_tag}` "
            f"({pct_tag:.1f}%)",
            icon="🏷️",
        )

        show_labels = st.multiselect(
            "Classes to display (leave empty = all)",
            options=all_labels,
            default=[],
            key="tag_filter_classes",
            format_func=lambda l: f"{_TYPE_ICON.get(label_types.get(l,'tag'), '📌')} {l}",
        )
        display_labels = show_labels if show_labels else all_labels

        rows = []
        for lbl in sorted_labels(display_labels, label_types):
            n_imgs = int(filtered_df[lbl].sum()) if lbl in filtered_df.columns else 0
            pct    = f"{n_imgs / n_filtered * 100:.1f}%" if n_filtered else "0.0%"
            rows.append({
                "type":    label_types.get(lbl, "tag"),
                "class":   lbl,
                "images":  n_imgs,
                "percent": pct,
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No classes to display.")


def _group_builder_section(all_labels: list, label_types: dict) -> list:
    """UI for defining custom group stats. Returns list of group dicts."""
    _TYPE_ICON = {"tag": "🏷️", "rectangle": "📐", "polygon": "🔷"}
    groups: list = st.session_state.setdefault("custom_groups", [])

    if groups:
        for i, g in enumerate(groups):
            geo  = [l for l in g["required"] if label_types.get(l) in ("rectangle", "polygon")]
            tags = [l for l in g["required"] if label_types.get(l) == "tag"]
            parts = []
            if geo:
                parts.append(f"📐 geo ×{len(geo)} → orientation ✓")
            if tags:
                parts.append(f"🏷️ tag ×{len(tags)}")
            detail = "  ·  ".join(parts)
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{g['name']}** — `{', '.join(g['required'])}`  \n<small>{detail}</small>", unsafe_allow_html=True)
                if c2.button("❌", key=f"del_g_{i}", use_container_width=True):
                    groups.pop(i)
                    st.rerun()
    else:
        st.caption("No groups yet. Add a group below to generate statistics by label combination.")

    with st.expander("➕ Add new group", expanded=len(groups) == 0):
        new_name = st.text_input("Group name", key="new_grp_name", placeholder="e.g. Dashed Beam")

        # Show labels grouped by type for clarity
        geo_opts = [l for l in all_labels if label_types.get(l) in ("rectangle", "polygon")]
        tag_opts = [l for l in all_labels if label_types.get(l) == "tag"]
        if geo_opts:
            st.caption("📐 Bbox / Polygon labels → orientation (H/V/D) will be calculated automatically")
        if tag_opts:
            st.caption("🏷️ Tag labels → orientation will not be calculated")

        new_req = st.multiselect(
            "Must have ALL (AND logic)",
            options=all_labels,
            key="new_grp_req",
            format_func=lambda l: f"{_TYPE_ICON.get(label_types.get(l,'tag'), '📌')} {l}",
        )

        has_geo = any(label_types.get(l) in ("rectangle", "polygon") for l in new_req)
        if has_geo:
            st.info("Orientation stats (horizontal / vertical / diagonal) will be calculated for this group.", icon="↔️")
        if new_req and all(label_types.get(l) == "tag" for l in new_req):
            st.info("Tag-only group — no orientation stats.", icon="🏷️")

        if st.button("➕ Add group", key="btn_add_grp",
                     disabled=not new_name.strip() or not new_req):
            groups.append({"name": new_name.strip(), "required": list(new_req)})
            st.session_state["custom_groups"] = groups
            st.rerun()

    col_clr, _ = st.columns([1, 4])
    if col_clr.button("❌ Remove all", key="btn_clr_grps",
                      disabled=not groups, use_container_width=True):
        st.session_state["custom_groups"] = []
        st.rerun()

    return groups


def _on_project_change():
    st.session_state["_uploader_key"] = st.session_state.get("_uploader_key", 0) + 1
    st.session_state.pop("rename_proj_input", None)
    st.session_state.pop("confirm_del_proj", None)


def _sidebar() -> Optional[str]:
    if not st.session_state.get("_migrated"):
        migrate_flat_to_subdir()
        st.session_state["_migrated"] = True

    st.header("📁 Project Tracking")

    registry      = load_registry()
    project_names = sorted(v["name"] for v in registry.values())
    CREATE        = "＋ New project"
    options       = [CREATE] + project_names

    pending  = st.session_state.pop("_pending_project_sel", None)
    init_idx = options.index(pending) if (pending and pending in options) else 0

    active = st.selectbox(
        "Project", options, index=init_idx,
        key="sidebar_project_sel",
        on_change=_on_project_change,
        help="Select a project or create a new one.",
    )

    if active == CREATE:
        new_name = st.text_input("Project name", key="new_proj_name")
        desc     = st.text_input("Description (optional)", key="new_proj_desc")
        if st.button("Create project", disabled=not new_name.strip()):
            register_project(new_name.strip(), desc.strip())
            st.session_state["_pending_project_sel"] = new_name.strip()
            st.rerun()
        return None

    st.session_state["active_project"] = active

    reg_entry = next((v for v in registry.values() if v["name"] == active), {})
    desc_str  = reg_entry.get("cvat_id") or reg_entry.get("description") or "—"
    st.caption(f"Description: **{desc_str}** · created: {reg_entry.get('created','—')}")

    log     = load_project_log(active)
    entries = log.get("entries", [])
    if not entries:
        st.info("No history yet. Generate a report to start tracking.")
    else:
        st.markdown(f"**History — {len(entries)} snapshot(s)**")
        prog_df = build_progress_table(log)
        if not prog_df.empty:
            def _bold_total(row):
                if row.name == "TOTAL":
                    return ["font-weight:bold; background:#EBF3FB"] * len(row)
                return [""] * len(row)
            st.dataframe(
                prog_df.style.apply(_bold_total, axis=1),
                use_container_width=True,
            )
            st.caption("Values = new labeled/day (delta). 'total' = latest cumulative.")

        with st.expander("❌ Remove snapshot", expanded=False):
            date_opts = [e["date"] for e in reversed(entries)]
            del_date  = st.selectbox("Date to remove", date_opts, key="del_snap_date")
            if st.button("Remove", key="btn_del_snap"):
                delete_project_log_entry(active, date.fromisoformat(del_date))
                delete_snapshot(active, date.fromisoformat(del_date))
                st.rerun()

    with st.expander("⚙️ Manage project", expanded=False):
        st.markdown("**Rename**")
        new_name_val = st.text_input(
            "New name", value=active, key="rename_proj_input",
            placeholder="Enter new project name",
        )
        new_name_stripped = new_name_val.strip()
        rename_ok = bool(new_name_stripped) and new_name_stripped != active and len(new_name_stripped) <= 30
        if new_name_stripped and len(new_name_stripped) > 30:
            st.warning(f"Name too long ({len(new_name_stripped)}/30 chars). Max 30 characters.", icon="⚠️")
        if st.button("✏️ Rename", key="btn_rename_proj",
                     disabled=not rename_ok, use_container_width=True):
            if rename_project(active, new_name_stripped):
                st.session_state["_pending_project_sel"] = new_name_stripped
                st.toast(f"Renamed to '{new_name_stripped}'", icon="✏️")
                st.rerun()
            else:
                st.error("Rename failed — name may already exist.")

        st.divider()

        st.markdown("**Delete project**")
        st.caption("Removes all history and snapshots for this project.")
        confirm_del = st.checkbox("I confirm delete", key="confirm_del_proj")
        if st.button("❌ Delete", key="btn_del_proj",
                     disabled=not confirm_del, use_container_width=True, type="primary"):
            delete_project(active)
            st.session_state.pop("active_project", None)
            st.session_state.pop("sidebar_project_sel", None)
            st.toast(f"Project '{active}' deleted.", icon="❌")
            st.rerun()

    return active


def _label_filter(parsed, img_df: pd.DataFrame, file_key: str) -> list:
    labels      = parsed.meta.labels
    label_types = parsed.meta.label_types

    groups: dict = {}
    for lbl in sorted_labels(labels, label_types):
        groups.setdefault(label_types.get(lbl, "tag"), []).append(lbl)

    selected = []
    with st.expander("⚙️ Customize labels", expanded=False):
        st.caption("All labels included by default. Uncheck to exclude from the report.")
        for type_key in ["tag", "rectangle", "polygon"]:
            grp = groups.get(type_key, [])
            if not grp:
                continue
            icon  = _TYPE_ICON.get(type_key, "📌")
            title = _TYPE_LABEL.get(type_key, type_key.title())
            st.markdown(f"**{icon} {title}** — {len(grp)} class(es)")
            c_all, c_none, _ = st.columns([1, 1, 6])
            grp_key = f"grp_{type_key}_{file_key}"
            if c_all.button("All",  key=f"all_{grp_key}",  use_container_width=True):
                for lbl in grp:
                    st.session_state[f"chk_{lbl}_{file_key}"] = True
            if c_none.button("None", key=f"none_{grp_key}", use_container_width=True):
                for lbl in grp:
                    st.session_state[f"chk_{lbl}_{file_key}"] = False
            for lbl in grp:
                img_count  = int(img_df[lbl].sum())           if lbl           in img_df.columns else 0
                bbox_count = int(img_df[f"bbox_{lbl}"].sum()) if f"bbox_{lbl}" in img_df.columns else 0
                warn = " ⚠️" if img_count == 0 else ""
                if st.checkbox(
                    f"`{lbl}` — {img_count} imgs · {bbox_count} boxes{warn}",
                    value=True,
                    key=f"chk_{lbl}_{file_key}",
                ):
                    selected.append(lbl)

    n_total, n_sel = len(labels), len(selected)
    if n_sel == n_total:
        st.caption(f"All **{n_total}** label classes included.")
    elif n_sel == 0:
        st.warning("No labels selected — select at least one.", icon="⚠️")
    else:
        st.caption(f"**{n_sel} of {n_total}** label classes included.")
    return selected


with st.sidebar:
    active_project = _sidebar()


st.title("📊 CVAT Annotation Statistics")
st.caption("Upload CVAT XML exports · filter labels · generate Excel & CSV reports")

if active_project:
    st.info(f"Tracking project: **{active_project}**", icon="📁")
else:
    st.warning("Create or select a project in the sidebar before generating.", icon="👈")

st.divider()


st.subheader("1 · Upload annotation files")

if "_uploader_key" not in st.session_state:
    st.session_state["_uploader_key"] = 0

uploaded_files = st.file_uploader(
    "Drop XML files here",
    type=["xml"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['_uploader_key']}",
    label_visibility="collapsed",
)

if not uploaded_files:
    st.info("Upload one or more CVAT annotation XML files to get started.")
    st.stop()


file_configs: dict = {}

for uf in uploaded_files:
    file_bytes = uf.getvalue()
    file_key   = f"{uf.name}_{len(file_bytes)}"

    with st.spinner(f"Reading {uf.name}…"):
        parsed, img_df, xml_date = _parse_cached(file_bytes, uf.name)

    with st.container(border=True):
        hcol1, hcol2 = st.columns([3, 1])
        hcol1.markdown(f"##### 📄 {uf.name}")
        run_date = hcol2.date_input(
            "Run date", value=xml_date or date.today(), key=f"date_{file_key}",
        )
        total       = len(parsed.images)
        labeled_all = int(img_df["is_labeled"].sum())
        pct_all     = labeled_all / total * 100 if total else 0
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Images",        total)
        ic2.metric("Labeled",       f"{labeled_all} ({pct_all:.0f}%)")
        ic3.metric("Label classes", len(parsed.meta.labels))

        # Project name rename 
        default_proj = (parsed.meta.project_name or Path(uf.name).stem)[:30]
        nc1, nc2 = st.columns([4, 1])
        display_name = nc1.text_input(
            "✏️ Project name",
            value=st.session_state.get(f"proj_name_{file_key}", default_proj),
            key=f"proj_name_{file_key}",
            max_chars=30,
            help="Used as column name in Excel and CSV file names. Automatically extracted from XML.",
        )
        char_count = len(display_name.strip())
        nc2.metric("Characters", f"{char_count}/30", delta=None)

        selected_labels = _label_filter(parsed, img_df, file_key)

    file_configs[file_key] = {
        "run_date":        run_date,
        "selected_labels": selected_labels,
        "parsed":          parsed,
        "img_df":          img_df,
        "display_name":    display_name.strip() or default_proj,
        "uf_name":         uf.name,
    }


st.divider()
st.subheader("2 · Options")

with st.expander("Advanced options", expanded=False):
    all_snaps   = list_snapshots()
    snap_labels = ["Latest (default)"] + [
        f"{s['date']}  —  {s['project_name']}" for s in all_snaps
    ]
    selected_snap = st.selectbox(
        "Delta baseline — compare against", snap_labels,
        help="Snapshot to diff against. Default: most recent before run date.",
    )
    baseline_date: Optional[date] = None
    if selected_snap != "Latest (default)":
        baseline_date = date.fromisoformat(selected_snap.split("  —  ")[0].strip())

    do_revert = st.checkbox(
        "Revert (redo this day)",
        help="Delete existing snapshot for the run date before processing.",
    )


st.divider()
st.subheader("2.5 · Group Stats")

# Compute merged labels/types from current uploads for the group builder
_gb_labels: list = []
_gb_types: dict  = {}
for _cfg in file_configs.values():
    for _lbl in _cfg["selected_labels"]:
        if _lbl not in _gb_labels:
            _gb_labels.append(_lbl)
    _gb_types.update(_cfg["parsed"].meta.label_types)

custom_groups = _group_builder_section(_gb_labels, _gb_types)

_preview_df = _merge_dfs([cfg["img_df"] for cfg in file_configs.values()], _gb_labels)
_tag_filter_section(_preview_df, _gb_labels, _gb_types)

st.divider()
st.subheader("3 · Generate")

all_ready = all(len(cfg["selected_labels"]) > 0 for cfg in file_configs.values())
_, btn_col, _ = st.columns([1, 2, 1])
generate = btn_col.button(
    "▶  Generate Reports", type="primary",
    disabled=not all_ready, use_container_width=True,
)

if not generate:
    if not all_ready:
        st.error("Select at least one label per file before generating.", icon="🚫")
    st.stop()


st.divider()
st.subheader("4 · Results")

all_downloads: dict = {}
all_configs         = list(file_configs.values())

with st.spinner("Processing…"):

    if len(uploaded_files) > 1:
        for uf in uploaded_files:
            fk  = f"{uf.name}_{len(uf.getvalue())}"
            cfg = file_configs[fk]
            lbls = sorted_labels(cfg["selected_labels"], cfg["parsed"].meta.label_types)
            all_downloads[f"{cfg['display_name']}.csv"] = _detail_csv_bytes(cfg["img_df"], lbls)

    all_labels, merged_types = _merge_labels_types(all_configs)

    if len(uploaded_files) > 1:
        label_err = _validate_same_labels(file_configs)
        if label_err:
            st.error(f"Cannot merge — label classes do not match:\n\n{label_err}")
            st.stop()

    combined_df  = _merge_dfs([cfg["img_df"] for cfg in all_configs], all_labels)
    best_date    = max(cfg["run_date"] for cfg in all_configs)

    total         = len(combined_df)
    labeled_count = int(combined_df["is_labeled"].sum())
    pct           = labeled_count / total * 100 if total else 0
    total_bboxes  = sum(
        int(combined_df[f"bbox_{lbl}"].sum())
        for lbl in all_labels if f"bbox_{lbl}" in combined_df.columns
    )

    proj_key  = active_project or all_configs[0]["display_name"]
    safe_proj = proj_key.replace(" ", "_").replace("/", "-")
    src_names = [uf.name for uf in uploaded_files]

    if do_revert:
        delete_snapshot(proj_key, best_date)
        if active_project:
            delete_project_log_entry(active_project, best_date)
        st.warning(f"Reverted snapshot **{best_date}** for **{proj_key}**", icon="🔄")

    if len(uploaded_files) > 1:
        _proj_col_names = _unique_proj_names(all_configs)
        summary_df = _wide_summary_df(all_configs, all_labels, merged_types, _proj_col_names)
        label_df   = _wide_label_stats_df(all_configs, all_labels, merged_types, _proj_col_names)
        group_df   = _wide_group_stats_df(all_configs, all_labels, merged_types, _proj_col_names)
    else:
        summary_df = _combined_summary_df(proj_key, src_names, combined_df, all_labels, merged_types)
        label_df   = label_stats(combined_df, all_labels, merged_types, labeled_count)
        group_df   = group_stats(combined_df, all_labels, merged_types, total)

    # Override Group_Stats with custom groups if user defined any
    if custom_groups:
        group_df = custom_group_stats(combined_df, custom_groups, merged_types, total)

    user_df = user_progress(combined_df, all_labels, merged_types)

    prev_snap = load_previous_snapshot(proj_key, before_date=best_date, baseline_date=baseline_date)
    delta_df  = compute_delta(combined_df, prev_snap, all_labels)
    save_snapshot(proj_key, combined_df, best_date)

    if active_project:
        by_user = {
            row["assignee"]: int(row["labeled"])
            for _, row in user_df.iterrows()
            if row["assignee"] not in ("", "(unassigned)")
        }
        save_project_log_entry(
            project_name=active_project, run_date=best_date,
            total_images=total, labeled=labeled_count, by_user=by_user,
        )

    excel_data = _excel_bytes(summary_df, label_df, group_df, combined_df, user_df, delta_df, all_labels)
    all_downloads[f"summary_{best_date}_{safe_proj}.xlsx"] = excel_data

    all_downloads[f"daily_worker_{best_date}_{safe_proj}.csv"] = _worker_csv_bytes(
        best_date, proj_key, user_df,
    )

    if len(uploaded_files) == 1:
        single_labels = sorted_labels(all_configs[0]["selected_labels"], merged_types)
        all_downloads[f"{all_configs[0]['display_name']}.csv"] = _detail_csv_bytes(combined_df, single_labels)


with st.container(border=True):
    st.markdown(f"### {_pct_badge(pct)} {proj_key}")
    if len(uploaded_files) > 1:
        st.caption(f"Combined: {' + '.join(cfg['display_name'] for cfg in all_configs)}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Run date",          str(best_date))
    m2.metric("Total images",      total)
    m3.metric("Labeled",           labeled_count, f"{pct:.1f}%")
    m4.metric("BBoxes (selected)", total_bboxes)

    cap_track = f"Tracking → **{active_project}**" if active_project else "No project selected"
    cap_delta = f"Delta vs: **{prev_snap['date']}**" if prev_snap else "First run"
    st.caption(f"{cap_track}  ·  {cap_delta}")

    with st.expander("📈 Charts", expanded=True):
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Labeled vs Unlabeled**")
            st.bar_chart(
                pd.DataFrame({"Count": {"Labeled": labeled_count, "Unlabeled": total - labeled_count}}),
                color=["#70AD47"],
            )
        with ch2:
            st.markdown("**Progress per user**")
            st.bar_chart(user_df[["assignee", "labeled", "unlabeled"]].set_index("assignee"))

    with st.expander("📋 Data tables", expanded=False):
        tab_s, tab_l, tab_g, tab_u, tab_d = st.tabs(
            ["Summary", "Labels", "Groups", "Users", "Delta"]
        )
        with tab_s:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        with tab_l:
            st.dataframe(label_df.drop(columns=["type"]), use_container_width=True, hide_index=True)
        with tab_g:
            st.dataframe(group_df, use_container_width=True, hide_index=True)
        with tab_u:
            dcols = ["assignee", "assigned", "labeled", "unlabeled"] + sorted_labels(all_labels, merged_types)
            st.dataframe(user_df[dcols], use_container_width=True, hide_index=True)
        with tab_d:
            def _color_delta(val):
                if isinstance(val, (int, float)):
                    if val > 0: return "background-color:#C6EFCE"
                    if val < 0: return "background-color:#FFC7CE"
                return ""
            st.dataframe(
                delta_df.style.map(_color_delta, subset=["delta_labeled"]),
                use_container_width=True, hide_index=True,
            )

    if len(uploaded_files) > 1:
        with st.expander("📄 Per-file breakdown", expanded=False):
            for uf in uploaded_files:
                fk    = f"{uf.name}_{len(uf.getvalue())}"
                cfg   = file_configs[fk]
                df_   = cfg["img_df"]
                n_    = len(df_)
                lbl_  = int(df_["is_labeled"].sum())
                pct_  = lbl_ / n_ * 100 if n_ else 0
                n_sel = len(cfg["selected_labels"])
                st.markdown(
                    f"**📄 {uf.name}** (`{cfg['display_name']}`) — {lbl_}/{n_} labeled ({pct_:.0f}%)"
                    f" · {n_sel} labels selected"
                )


st.divider()
st.subheader("5 · Download")

with st.popover("📥 Select files to download"):
    st.markdown(f"**{len(all_downloads)} file(s) ready:**")
    selected_dl: dict = {}
    for fname, fdata in all_downloads.items():
        size_kb = len(fdata) / 1024
        if st.checkbox(f"{fname}  ({size_kb:.0f} KB)", value=True, key=f"dl_chk_{fname}"):
            selected_dl[fname] = fdata

    st.divider()
    if not selected_dl:
        st.caption("No files selected.")
    elif len(selected_dl) == 1:
        fname, fdata = next(iter(selected_dl.items()))
        mime = _MIME.get(Path(fname).suffix.lower(), "application/octet-stream")
        st.download_button(
            "⬇ Download", data=fdata, file_name=fname, mime=mime,
            key="dl_single", use_container_width=True,
        )
    else:
        zip_name = f"cvat_stats_{best_date}_{safe_proj}.zip"
        st.download_button(
            f"⬇ Download ZIP ({len(selected_dl)} files)",
            data=_make_zip(selected_dl),
            file_name=zip_name, mime="application/zip",
            key="dl_zip", use_container_width=True,
        )

st.divider()
st.success(
    f"Done — {len(uploaded_files)} file(s) · {total} images · "
    f"{labeled_count} labeled ({pct:.1f}%)",
)
