import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvat_stats.parser import parse_xml, to_image_df
from cvat_stats.stats import summary_stats, label_stats, group_stats, user_progress, sorted_labels

BEAM = Path(__file__).parent.parent / "example" / "beam" / "annotations.xml"
JOB1 = Path(__file__).parent.parent / "example" / "beam_job1" / "annotations.xml"
JOB2 = Path(__file__).parent.parent / "example" / "beam_job2" / "annotations.xml"


class TestSortedLabels:
    def test_tag_before_polygon_before_rectangle(self):
        labels = ["rect_a", "tag_a", "poly_a"]
        types = {"rect_a": "rectangle", "tag_a": "tag", "poly_a": "polygon"}
        result = sorted_labels(labels, types)
        assert result.index("tag_a") < result.index("poly_a")
        assert result.index("poly_a") < result.index("rect_a")

    def test_unknown_type_treated_as_tag(self):
        labels = ["x", "y"]
        types = {"x": "unknown", "y": "rectangle"}
        result = sorted_labels(labels, types)
        assert result.index("x") < result.index("y")

    def test_same_type_stable(self):
        labels = ["a", "b", "c"]
        types = {"a": "rectangle", "b": "rectangle", "c": "rectangle"}
        result = sorted_labels(labels, types)
        assert set(result) == {"a", "b", "c"}


class TestSummaryStatsBeam:
    def setup_method(self):
        p = parse_xml(BEAM)
        self.p = p
        self.df = to_image_df(p)
        self.summary = summary_stats(p, self.df)

    def _get(self, metric):
        row = self.summary[self.summary["metric"] == metric]
        assert not row.empty, f"metric '{metric}' not found"
        return row.iloc[0]["value"]

    def test_project_name(self):
        assert self._get("project_name") == "Beam_Header_4_May"

    def test_total_images(self):
        assert self._get("total_images") == 2600

    def test_labeled_images(self):
        assert self._get("labeled_images") == 212

    def test_unlabeled_images(self):
        assert self._get("unlabeled_images") == 2388

    def test_labeled_pct_format(self):
        pct = self._get("labeled_pct")
        assert pct.endswith("%")
        assert float(pct.strip("%")) == pytest.approx(212 / 2600 * 100, abs=0.1)

    def test_selected_label_classes(self):
        assert self._get("selected_label_classes") == 17

    def test_label_subset_filter(self):
        subset = ["beam_dash", "beam_solid"]
        s = summary_stats(self.p, self.df, labels=subset)
        assert int(s[s["metric"] == "selected_label_classes"].iloc[0]["value"]) == 2
        metrics = set(s["metric"])
        assert "images_with_beam_dash" in metrics
        assert "images_with_beam_solid" in metrics
        assert "images_with_other_dash" not in metrics


class TestSummaryStatsJob1:
    def setup_method(self):
        p = parse_xml(JOB1)
        self.p = p
        self.df = to_image_df(p)
        self.summary = summary_stats(p, self.df)

    def _get(self, metric):
        row = self.summary[self.summary["metric"] == metric]
        assert not row.empty, f"metric '{metric}' not found"
        return row.iloc[0]["value"]

    def test_project_name(self):
        assert self._get("project_name") == "job1"

    def test_total_images(self):
        assert self._get("total_images") == 484

    def test_labeled_images(self):
        assert self._get("labeled_images") == 311

    def test_selected_label_classes(self):
        assert self._get("selected_label_classes") == 11

    def test_dash_bbox_total(self):
        assert self._get("total_bbox_dash") == 335

    def test_solid_bbox_total(self):
        assert self._get("total_bbox_solid") == 135


class TestLabelStatsJob1:
    def setup_method(self):
        p = parse_xml(JOB1)
        self.p = p
        self.df = to_image_df(p)
        labeled_count = int(self.df["is_labeled"].sum())
        self.ls = label_stats(self.df, p.meta.labels, p.meta.label_types, labeled_count)

    def test_columns(self):
        assert set(self.ls.columns) >= {"type", "class", "images_with_label", "total_bboxes", "pct_of_labeled_imgs"}

    def test_row_count(self):
        assert len(self.ls) == 11

    def test_dash_row(self):
        row = self.ls[self.ls["class"] == "dash"].iloc[0]
        assert row["total_bboxes"] == 335
        assert row["type"] == "rectangle"

    def test_solid_row(self):
        row = self.ls[self.ls["class"] == "solid"].iloc[0]
        assert row["total_bboxes"] == 135

    def test_pct_format(self):
        for pct in self.ls["pct_of_labeled_imgs"]:
            assert pct.endswith("%")

    def test_type_ordering(self):
        types = list(self.ls["type"])
        tag_indices = [i for i, t in enumerate(types) if t == "tag"]
        rect_indices = [i for i, t in enumerate(types) if t == "rectangle"]
        if tag_indices and rect_indices:
            assert max(tag_indices) < min(rect_indices)


class TestGroupStatsJob1:
    def setup_method(self):
        p = parse_xml(JOB1)
        self.df = to_image_df(p)
        self.gs = group_stats(self.df, p.meta.labels, p.meta.label_types)

    def test_columns(self):
        assert set(self.gs.columns) == {"line_type", "orientation", "total_imgs", "total_bbox"}

    def test_line_types(self):
        assert set(self.gs["line_type"]) == {"dash", "solid"}

    def test_orientations(self):
        assert set(self.gs["orientation"]) == {"horizontal", "vertical", "diagonal"}

    def test_non_negative(self):
        assert (self.gs["total_imgs"] >= 0).all()
        assert (self.gs["total_bbox"] >= 0).all()

    def test_six_rows(self):
        assert len(self.gs) == 6


class TestGroupStatsBeam:
    def setup_method(self):
        p = parse_xml(BEAM)
        self.df = to_image_df(p)
        self.gs = group_stats(self.df, p.meta.labels, p.meta.label_types)

    def test_six_rows(self):
        assert len(self.gs) == 6

    def test_non_negative(self):
        assert (self.gs["total_imgs"] >= 0).all()
        assert (self.gs["total_bbox"] >= 0).all()


class TestUserProgressBeam:
    def setup_method(self):
        p = parse_xml(BEAM)
        self.p = p
        self.df = to_image_df(p)
        self.up = user_progress(self.df, p.meta.labels, p.meta.label_types)

    def test_columns(self):
        assert "assignee" in self.up.columns
        assert "assigned" in self.up.columns
        assert "labeled" in self.up.columns
        assert "unlabeled" in self.up.columns

    def test_expected_users(self):
        users = set(self.up["assignee"])
        for u in ("hingo", "anguyen6", "phtran"):
            assert u in users

    def test_assigned_sum(self):
        assert int(self.up["assigned"].sum()) == 2600

    def test_labeled_unlabeled_consistent(self):
        for _, row in self.up.iterrows():
            assert row["labeled"] + row["unlabeled"] == row["assigned"]

    def test_known_assignees_present(self):
        users = set(self.up["assignee"])
        assert "hingo" in users
        assert "anguyen6" in users


class TestUserProgressJob1:
    def setup_method(self):
        p = parse_xml(JOB1)
        self.p = p
        self.df = to_image_df(p)
        self.up = user_progress(self.df, p.meta.labels, p.meta.label_types)

    def test_assigned_sum(self):
        assert int(self.up["assigned"].sum()) == 484

    def test_labeled_sum(self):
        assert int(self.up["labeled"].sum()) == 311

    def test_labeled_unlabeled_consistent(self):
        for _, row in self.up.iterrows():
            assert row["labeled"] + row["unlabeled"] == row["assigned"]

    def test_label_columns_present(self):
        for lbl in self.p.meta.labels:
            assert lbl in self.up.columns
