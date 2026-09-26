from typing import Any, Dict


class WHOGrader:
    """Grade Ziehl-Neelsen smear microscopy counts using WHO/IUATLD rules.

    The grading rules are defined in terms of AFB observed across microscope fields.
    A slide area cannot be converted reliably into an equivalent field count without
    microscope-specific field geometry and an explicit acquisition protocol, so this
    class requires the number of fields actually examined.
    """

    def calculate_grade(self, afb_count: int, fields_examined: int) -> Dict[str, Any]:
        if isinstance(afb_count, bool) or not isinstance(afb_count, int) or afb_count < 0:
            raise ValueError("afb_count must be a non-negative integer.")
        if isinstance(fields_examined, bool) or not isinstance(fields_examined, int) or fields_examined <= 0:
            raise ValueError("fields_examined must be a positive integer.")

        afb_per_field = afb_count / fields_examined
        afb_per_100_fields = afb_per_field * 100.0

        if fields_examined >= 20 and afb_per_field > 10:
            grade = "3+"
        elif fields_examined >= 50 and 1 <= afb_per_field <= 10:
            grade = "2+"
        elif fields_examined >= 100 and afb_count > 0:
            if afb_per_100_fields < 10:
                grade = "Scanty"
            else:
                grade = "1+"
        elif fields_examined >= 100 and afb_count == 0:
            grade = "Negative"
        else:
            grade = "Insufficient fields"

        return {
            "grade": grade,
            "afb_per_field": afb_per_field,
            "afb_per_100_fields": afb_per_100_fields,
            "fields_counted": fields_examined,
            "report_string": f"{grade} ({afb_count} AFB in {fields_examined} fields)",
        }
