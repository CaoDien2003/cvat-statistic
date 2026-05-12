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
from cvat_stats.stats import label_stats, group_stats, user_progress, sorted_labels
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
        {"metric": "project_name",           "value": proj_name},
        {"metric": "source_files",           "value": " + ".join(source_names)},
        {"metric": "total_images",           "value": total},
        {"metric": "labeled_images",         "value": labeled},
        {"metric": "unlabeled_images",       "value": total - labeled},
        {"metric": "labeled_pct",            "value": f"{labeled/total*100:.1f}%" if total else "0.0%"},
        {"metric": "selected_label_classes", "value": len(labels)},
    ]
    for lbl in sorted_labels(labels, label_types):
        img_count  = int(df[lbl].sum())          if lbl           in df.columns else 0
        bbox_total = int(df[f"bbox_{lbl}"].sum()) if f"bbox_{lbl}" in df.columns else 0
        rows.append({"metric": f"images_with_{lbl}", "value": img_count})
        rows.append({"metric": f"total_bbox_{lbl}",  "value": bbox_total})
    return pd.DataFrame(rows)


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
        st.info("No history yet. Generate a report to start tracking.", icon="📭")
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

        with st.expander("🗑️ Remove snapshot", expanded=False):
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
        rename_ok = new_name_val.strip() and new_name_val.strip() != active
        if st.button("✏️ Rename", key="btn_rename_proj",
                     disabled=not rename_ok, use_container_width=True):
            if rename_project(active, new_name_val.strip()):
                st.session_state["_pending_project_sel"] = new_name_val.strip()
                st.toast(f"Renamed to '{new_name_val.strip()}'", icon="✏️")
                st.rerun()
            else:
                st.error("Rename failed — name may already exist.")

        st.divider()

        st.markdown("**Delete project**")
        st.caption("Removes all history and snapshots for this project.")
        confirm_del = st.checkbox("I confirm delete", key="confirm_del_proj")
        if st.button("🗑️ Delete", key="btn_del_proj",
                     disabled=not confirm_del, use_container_width=True, type="primary"):
            delete_project(active)
            st.session_state.pop("active_project", None)
            st.session_state.pop("sidebar_project_sel", None)
            st.toast(f"Project '{active}' deleted.", icon="🗑️")
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
    st.info("Upload one or more CVAT annotation XML files to get started.", icon="☝️")
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
        selected_labels = _label_filter(parsed, img_df, file_key)

    file_configs[uf.name] = {
        "run_date": run_date, "selected_labels": selected_labels,
        "parsed": parsed, "img_df": img_df,
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
            cfg  = file_configs[uf.name]
            lbls = sorted_labels(cfg["selected_labels"], cfg["parsed"].meta.label_types)
            all_downloads[f"{Path(uf.name).stem}.csv"] = _detail_csv_bytes(cfg["img_df"], lbls)

    all_labels, merged_types = _merge_labels_types(all_configs)
    combined_df  = _merge_dfs([cfg["img_df"] for cfg in all_configs], all_labels)
    best_date    = max(cfg["run_date"] for cfg in all_configs)

    total         = len(combined_df)
    labeled_count = int(combined_df["is_labeled"].sum())
    pct           = labeled_count / total * 100 if total else 0
    total_bboxes  = sum(
        int(combined_df[f"bbox_{lbl}"].sum())
        for lbl in all_labels if f"bbox_{lbl}" in combined_df.columns
    )

    proj_key  = active_project or all_configs[0]["parsed"].meta.project_name
    safe_proj = proj_key.replace(" ", "_").replace("/", "-")
    src_names = [uf.name for uf in uploaded_files]

    if do_revert:
        delete_snapshot(proj_key, best_date)
        if active_project:
            delete_project_log_entry(active_project, best_date)
        st.warning(f"Reverted snapshot **{best_date}** for **{proj_key}**", icon="🔄")

    summary_df = _combined_summary_df(proj_key, src_names, combined_df, all_labels, merged_types)
    label_df   = label_stats(combined_df, all_labels, merged_types, labeled_count)
    group_df   = group_stats(combined_df, all_labels, merged_types)
    user_df    = user_progress(combined_df, all_labels, merged_types)

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
        xml_stem = Path(uploaded_files[0].name).stem
        all_downloads[f"{xml_stem}.csv"] = _detail_csv_bytes(combined_df, single_labels)


with st.container(border=True):
    st.markdown(f"### {_pct_badge(pct)} {proj_key}")
    if len(uploaded_files) > 1:
        st.caption(f"Combined: {' + '.join(uf.name for uf in uploaded_files)}")

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
                delta_df.style.applymap(_color_delta, subset=["delta_labeled"]),
                use_container_width=True, hide_index=True,
            )

    if len(uploaded_files) > 1:
        with st.expander("📄 Per-file breakdown", expanded=False):
            for uf in uploaded_files:
                cfg   = file_configs[uf.name]
                df_   = cfg["img_df"]
                n_    = len(df_)
                lbl_  = int(df_["is_labeled"].sum())
                pct_  = lbl_ / n_ * 100 if n_ else 0
                n_sel = len(cfg["selected_labels"])
                st.markdown(
                    f"**📄 {uf.name}** — {lbl_}/{n_} labeled ({pct_:.0f}%)"
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
    icon="✅",
)
