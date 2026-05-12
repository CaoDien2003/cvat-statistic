import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvat_stats.parser import parse_xml, to_image_df, read_xml_date

BEAM = Path(__file__).parent.parent / "example" / "beam" / "annotations.xml"
JOB1 = Path(__file__).parent.parent / "example" / "beam_job1" / "annotations.xml"
JOB2 = Path(__file__).parent.parent / "example" / "beam_job2" / "annotations.xml"


class TestProjectLevelMeta:
    def setup_method(self):
        self.p = parse_xml(BEAM)

    def test_project_name(self):
        assert self.p.meta.project_name == "Beam_Header_4_May"

    def test_project_id(self):
        assert self.p.meta.project_id == "41"

    def test_label_count(self):
        assert len(self.p.meta.labels) == 17

    def test_label_names_present(self):
        for lbl in ("beam_dash", "beam_solid", "beam_dash_diagonal", "beam_solid_diagonal"):
            assert lbl in self.p.meta.labels

    def test_label_types(self):
        assert self.p.meta.label_types["beam_dash"] == "rectangle"
        assert self.p.meta.label_types["beam_dash_diagonal"] == "polygon"

    def test_task_count(self):
        assert len(self.p.meta.tasks) == 94

    def test_task_names_and_assignees(self):
        assert self.p.meta.tasks["719"]["name"] == "B1_Job1"
        assert self.p.meta.tasks["719"]["assignee"] == "hingo"
        assert self.p.meta.tasks["720"]["name"] == "B1_Job2"
        assert self.p.meta.tasks["720"]["assignee"] == "anguyen6"

    def test_updated_date(self):
        assert self.p.meta.updated_date.startswith("2026-05-")


class TestProjectLevelImages:
    def setup_method(self):
        self.p = parse_xml(BEAM)

    def test_image_count(self):
        assert len(self.p.images) == 2600

    def test_first_image_task_mapping(self):
        img = self.p.images[0]
        assert img.task_id == "719"
        assert img.task_name == "B1_Job1"
        assert img.assignee == "hingo"

    def test_no_image_has_empty_task_name(self):
        for img in self.p.images:
            assert img.task_name != "", f"image {img.img_name} has empty task_name"

    def test_some_images_have_assignees(self):
        assignees = {img.assignee for img in self.p.images}
        assert "hingo" in assignees
        assert "anguyen6" in assignees

    def test_labeled_count(self):
        labeled = sum(1 for img in self.p.images if img.is_labeled)
        assert labeled == 212

    def test_tag_images_have_applied_labels(self):
        img = self.p.images[0]
        assert img.is_labeled
        assert len(img.applied_labels) > 0


class TestProjectLevelDataFrame:
    def setup_method(self):
        p = parse_xml(BEAM)
        self.p = p
        self.df = to_image_df(p)

    def test_shape_rows(self):
        assert len(self.df) == 2600

    def test_assignees_has_known_users(self):
        users = set(self.df["assignee"].unique())
        assert "hingo" in users
        assert "anguyen6" in users

    def test_task_names_not_empty(self):
        assert (self.df["task_name"] != "").all()

    def test_expected_assignees(self):
        users = set(self.df["assignee"].unique())
        for u in ("hingo", "anguyen6", "phtran", "hpham3", "tnguyen16"):
            assert u in users

    def test_label_columns_present(self):
        for lbl in self.p.meta.labels:
            assert lbl in self.df.columns
            assert f"bbox_{lbl}" in self.df.columns

    def test_is_labeled_sum(self):
        assert int(self.df["is_labeled"].sum()) == 212

    def test_no_negative_counts(self):
        for lbl in self.p.meta.labels:
            assert (self.df[f"bbox_{lbl}"] >= 0).all()


class TestTaskLevelMetaJob1:
    def setup_method(self):
        self.p = parse_xml(JOB1)

    def test_project_name(self):
        assert self.p.meta.project_name == "job1"

    def test_project_id(self):
        assert self.p.meta.project_id == "21"

    def test_label_count(self):
        assert len(self.p.meta.labels) == 11

    def test_label_names(self):
        for lbl in ("dash", "solid", "text", "schedule", "val"):
            assert lbl in self.p.meta.labels

    def test_single_task(self):
        assert len(self.p.meta.tasks) == 1
        assert "21" in self.p.meta.tasks
        assert self.p.meta.tasks["21"]["name"] == "job1"


class TestTaskLevelImagesJob1:
    def setup_method(self):
        self.p = parse_xml(JOB1)

    def test_image_count(self):
        assert len(self.p.images) == 484

    def test_task_id_fallback(self):
        for img in self.p.images:
            assert img.task_id == "21", f"{img.img_name} has task_id={img.task_id!r}"

    def test_task_name_populated(self):
        for img in self.p.images:
            assert img.task_name == "job1", f"{img.img_name} has task_name={img.task_name!r}"

    def test_labeled_count(self):
        labeled = sum(1 for img in self.p.images if img.is_labeled)
        assert labeled == 311

    def test_first_image_has_boxes(self):
        img = self.p.images[0]
        assert img.is_labeled
        total_boxes = sum(img.bbox_counts.values())
        assert total_boxes > 0


class TestTaskLevelDataFrameJob1:
    def setup_method(self):
        p = parse_xml(JOB1)
        self.p = p
        self.df = to_image_df(p)

    def test_shape_rows(self):
        assert len(self.df) == 484

    def test_task_name_column(self):
        assert (self.df["task_name"] == "job1").all()

    def test_is_labeled_sum(self):
        assert int(self.df["is_labeled"].sum()) == 311

    def test_label_columns_present(self):
        for lbl in self.p.meta.labels:
            assert lbl in self.df.columns
            assert f"bbox_{lbl}" in self.df.columns

    def test_dash_bbox_count(self):
        assert int(self.df["bbox_dash"].sum()) == 335

    def test_solid_bbox_count(self):
        assert int(self.df["bbox_solid"].sum()) == 135


class TestTaskLevelMetaJob2:
    def setup_method(self):
        self.p = parse_xml(JOB2)

    def test_project_name(self):
        assert self.p.meta.project_name == "job2"

    def test_project_id(self):
        assert self.p.meta.project_id == "119"

    def test_label_count(self):
        assert len(self.p.meta.labels) == 11

    def test_single_task(self):
        assert len(self.p.meta.tasks) == 1
        assert "119" in self.p.meta.tasks


class TestTaskLevelImagesJob2:
    def setup_method(self):
        self.p = parse_xml(JOB2)

    def test_image_count(self):
        assert len(self.p.images) == 491

    def test_task_id_fallback(self):
        for img in self.p.images:
            assert img.task_id == "119"

    def test_task_name_populated(self):
        for img in self.p.images:
            assert img.task_name == "job2"


class TestReadXmlDate:
    def test_project_date(self):
        d = read_xml_date(BEAM)
        assert d is not None
        assert str(d).startswith("2026-05-")

    def test_task_date_job1(self):
        d = read_xml_date(JOB1)
        assert d is not None
        assert str(d).startswith("2026-05-")

    def test_nonexistent_file(self):
        d = read_xml_date("nonexistent.xml")
        assert d is None
