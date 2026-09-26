import pytest

from tb_afb.inference.who_grader import WHOGrader


def test_who_zn_grading_thresholds():
    grader = WHOGrader()
    assert grader.calculate_grade(0, 100)["grade"] == "Negative"
    assert grader.calculate_grade(7, 100)["grade"] == "Scanty"
    assert grader.calculate_grade(50, 100)["grade"] == "1+"
    assert grader.calculate_grade(50, 50)["grade"] == "2+"
    assert grader.calculate_grade(500, 20)["grade"] == "3+"


def test_inadequate_field_count_is_not_called_negative():
    assert WHOGrader().calculate_grade(0, 20)["grade"] == "Insufficient fields"


def test_invalid_counts_rejected():
    grader = WHOGrader()
    with pytest.raises(ValueError):
        grader.calculate_grade(-1, 100)
    with pytest.raises(ValueError):
        grader.calculate_grade(1, 0)
