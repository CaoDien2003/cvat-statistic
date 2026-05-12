import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
import pandas as pd


@dataclass
class ProjectMeta:
    project_id: str
    project_name: str
    labels: List[str]
    label_types: Dict[str, str]
    tasks: Dict[str, dict]
    updated_date: str = ""


@dataclass
class ImageRecord:
    img_id: str
    img_name: str
    task_id: str
    task_name: str
    assignee: str
    applied_labels: Set[str] = field(default_factory=set)
    bbox_counts: Dict[str, int] = field(default_factory=dict)
    orient_by_label: Dict[str, Dict[str, int]] = field(default_factory=dict)
    horizontal_count: int = 0
    vertical_count: int = 0
    diagonal_count: int = 0

    @property
    def is_labeled(self) -> bool:
        return len(self.applied_labels) > 0


@dataclass
class ParsedAnnotation:
    source_file: Path
    meta: ProjectMeta
    images: List[ImageRecord]


def read_xml_date(xml_path: Union[str, Path]) -> Optional[date]:
    try:
        root = ET.parse(Path(xml_path)).getroot()
        raw = (
            root.findtext("meta/project/updated")
            or root.findtext("meta/task/updated")
            or ""
        )
        if raw:
            return date.fromisoformat(raw.split("T")[0].split(" ")[0])
    except Exception:
        pass
    return None


def parse_xml(xml_path: Union[str, Path]) -> ParsedAnnotation:
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta = _parse_meta(root)
    images = _parse_images(root, meta)

    return ParsedAnnotation(source_file=xml_path, meta=meta, images=images)


def _parse_meta(root: ET.Element) -> ProjectMeta:
    project = root.find("meta/project")
    task_node = root.find("meta/task")
    node = project if project is not None else task_node

    if node is None:
        raise ValueError(
            "Unsupported XML format: expected <meta/project> or <meta/task> element. "
            "Export from CVAT using 'CVAT 1.1' format."
        )

    project_id = node.findtext("id", default="")
    project_name = node.findtext("name", default="")

    labels = []
    label_types: Dict[str, str] = {}
    for lbl in node.findall("labels/label"):
        name = lbl.findtext("name", default="")
        if name:
            labels.append(name)
            label_types[name] = lbl.findtext("type", default="tag")

    tasks = {}
    if project is not None:
        for task in project.findall("tasks/task"):
            tid = task.findtext("id", default="")
            tasks[tid] = {
                "name": task.findtext("name", default=""),
                "assignee": task.findtext("assignee/username", default=""),
                "size": task.findtext("size", default="0"),
            }
    else:
        tid = node.findtext("id", default="")
        tasks[tid] = {
            "name": node.findtext("name", default=""),
            "assignee": node.findtext("assignee/username", default=""),
            "size": node.findtext("size", default="0"),
        }

    updated_raw = node.findtext("updated", default="")
    updated_date = updated_raw.split(" ")[0] if updated_raw else ""

    return ProjectMeta(
        project_id=project_id,
        project_name=project_name,
        labels=labels,
        label_types=label_types,
        tasks=tasks,
        updated_date=updated_date,
    )


def _parse_images(root: ET.Element, meta: ProjectMeta) -> List[ImageRecord]:
    images = []
    for img in root.findall("image"):
        img_id = img.get("id", "")
        img_name = img.get("name", "")
        task_id = img.get("task_id", "")
        if not task_id and len(meta.tasks) == 1:
            task_id = next(iter(meta.tasks))

        task_info = meta.tasks.get(task_id, {})
        task_name = task_info.get("name", "")
        assignee = task_info.get("assignee", "")

        applied_labels: Set[str] = {tag.get("label", "") for tag in img.findall("tag") if tag.get("label")}

        bbox_counts: Dict[str, int] = {}
        orient_by_label: Dict[str, Dict[str, int]] = {}
        horizontal_count = 0
        vertical_count = 0
        diagonal_count = 0

        for box in img.findall("box"):
            lbl = box.get("label", "")
            if lbl:
                applied_labels.add(lbl)
                bbox_counts[lbl] = bbox_counts.get(lbl, 0) + 1
                w = float(box.get("xbr", 0)) - float(box.get("xtl", 0))
                h = float(box.get("ybr", 0)) - float(box.get("ytl", 0))
                orient = "vertical" if h > w else "horizontal"
                obl = orient_by_label.setdefault(lbl, {"horizontal": 0, "vertical": 0, "diagonal": 0})
                obl[orient] += 1
                if orient == "vertical":
                    vertical_count += 1
                else:
                    horizontal_count += 1

        for elem in (*img.findall("polygon"), *img.findall("polyline")):
            lbl = elem.get("label", "")
            if lbl:
                applied_labels.add(lbl)
                bbox_counts[lbl] = bbox_counts.get(lbl, 0) + 1
                orient_by_label.setdefault(lbl, {"horizontal": 0, "vertical": 0, "diagonal": 0})["diagonal"] += 1
                diagonal_count += 1

        images.append(
            ImageRecord(
                img_id=img_id,
                img_name=img_name,
                task_id=task_id,
                task_name=task_name,
                assignee=assignee,
                applied_labels=applied_labels,
                bbox_counts=bbox_counts,
                orient_by_label=orient_by_label,
                horizontal_count=horizontal_count,
                vertical_count=vertical_count,
                diagonal_count=diagonal_count,
            )
        )

    return images


def to_image_df(parsed: ParsedAnnotation) -> pd.DataFrame:
    labels = parsed.meta.labels
    rows = []
    for img in parsed.images:
        row = {
            "img_name": img.img_name,
            "task_name": img.task_name,
            "assignee": img.assignee,
            "is_labeled": int(img.is_labeled),
        }
        for lbl in labels:
            row[lbl] = 1 if lbl in img.applied_labels else 0
            row[f"bbox_{lbl}"] = img.bbox_counts.get(lbl, 0)
            obl = img.orient_by_label.get(lbl, {"horizontal": 0, "vertical": 0, "diagonal": 0})
            row[f"horizontal_{lbl}"] = obl["horizontal"]
            row[f"vertical_{lbl}"]   = obl["vertical"]
            row[f"diagonal_{lbl}"]   = obl["diagonal"]
        row["horizontal_count"] = img.horizontal_count
        row["vertical_count"]   = img.vertical_count
        row["diagonal_count"]   = img.diagonal_count
        rows.append(row)

    return pd.DataFrame(rows)
