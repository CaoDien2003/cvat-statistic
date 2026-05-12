import pandas as pd
from cvat_stats.parser import ParsedAnnotation

_TYPE_ORDER = ["tag", "polygon", "rectangle"]


def sorted_labels(labels: list, label_types: dict) -> list:
    def _key(lbl):
        t = label_types.get(lbl, "tag")
        try:
            return _TYPE_ORDER.index(t)
        except ValueError:
            return 0
    return sorted(labels, key=_key)


def summary_stats(parsed: ParsedAnnotation, img_df: pd.DataFrame, labels: list = None) -> pd.DataFrame:
    labels = labels if labels is not None else parsed.meta.labels
    total = len(parsed.images)
    labeled = int(img_df["is_labeled"].sum())
    unlabeled = total - labeled

    rows = [
        {"metric": "project_name", "value": parsed.meta.project_name},
        {"metric": "source_file", "value": str(parsed.source_file.name)},
        {"metric": "total_images", "value": total},
        {"metric": "labeled_images", "value": labeled},
        {"metric": "unlabeled_images", "value": unlabeled},
        {"metric": "labeled_pct", "value": f"{labeled / total * 100:.1f}%" if total else "0.0%"},
        {"metric": "selected_label_classes", "value": len(labels)},
    ]

    for lbl in sorted_labels(labels, parsed.meta.label_types):
        img_count = int(img_df[lbl].sum()) if lbl in img_df.columns else 0
        bbox_total = int(img_df[f"bbox_{lbl}"].sum()) if f"bbox_{lbl}" in img_df.columns else 0
        rows.append({"metric": f"images_with_{lbl}", "value": img_count})
        rows.append({"metric": f"total_bbox_{lbl}", "value": bbox_total})

    return pd.DataFrame(rows)


def label_stats(img_df: pd.DataFrame, labels: list, label_types: dict,
                labeled_count: int) -> pd.DataFrame:
    rows = []
    for lbl in sorted_labels(labels, label_types):
        img_count = int(img_df[lbl].sum()) if lbl in img_df.columns else 0
        bbox_total = int(img_df[f"bbox_{lbl}"].sum()) if f"bbox_{lbl}" in img_df.columns else 0
        pct = f"{img_count / labeled_count * 100:.1f}%" if labeled_count > 0 else "0.0%"
        rows.append({
            "type": label_types.get(lbl, "tag"),
            "class": lbl,
            "images_with_label": img_count,
            "total_bboxes": bbox_total,
            "pct_of_labeled_imgs": pct,
        })
    return pd.DataFrame(rows)


def group_stats(img_df: pd.DataFrame, labels: list, label_types: dict) -> pd.DataFrame:
    geo_labels = [l for l in labels if label_types.get(l) in ("rectangle", "polygon")]
    orient_map = [("horizontal", "horizontal"), ("vertical", "vertical"), ("diagonal", "diagonal")]

    rows = []
    for line_kw, line_name in [("dash", "dash"), ("solid", "solid")]:
        matched = [l for l in geo_labels if line_kw in l.lower()]
        for col_key, orient_name in orient_map:
            orient_cols = [f"{col_key}_{l}" for l in matched if f"{col_key}_{l}" in img_df.columns]
            if orient_cols:
                has_any = (img_df[orient_cols].sum(axis=1) > 0)
                total_imgs = int(has_any.sum())
                total_bbox = int(img_df[orient_cols].sum().sum())
            else:
                total_imgs = total_bbox = 0
            rows.append({"line_type": line_name, "orientation": orient_name,
                         "total_imgs": total_imgs, "total_bbox": total_bbox})

    return pd.DataFrame(rows)


def user_progress(img_df: pd.DataFrame, labels: list, label_types: dict) -> pd.DataFrame:
    groups = img_df.groupby("assignee")
    ordered = sorted_labels(labels, label_types)

    rows = []
    for user, grp in groups:
        row = {
            "assignee": user if user else "(unassigned)",
            "assigned": len(grp),
            "labeled": int(grp["is_labeled"].sum()),
            "unlabeled": len(grp) - int(grp["is_labeled"].sum()),
        }
        for lbl in ordered:
            row[lbl] = int(grp[lbl].sum()) if lbl in grp.columns else 0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("assignee")
