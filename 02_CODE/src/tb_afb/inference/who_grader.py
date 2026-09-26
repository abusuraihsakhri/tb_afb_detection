from typing import Any, Dict


class WHOGrader:
    """Grade Ziehl-Neelsen smear microscopy from counted AFB and fields.

    This class implements the conventional WHO/IUATLD reporting thresholds for
    oil-immersion smear microscopy. It deliberately requires an observed field
    count; slide area or detector tile counts are not interchangeable with
    microscopist-examined high-power fields.
    """

    def calculate_grade(self, afb_count: int, fields_examined: int) -> Dict[str, Any]:
        if afb_count < 0:
            raise ValueError("AFB count cannot be negative.")
        if fields_examined <= 0:
            raise ValueError("fields_examined must be a positive integer.")

        fields = int(fields_examined)
        count = int(afb_count)
        afb_per_field = count / fields
        afb_per_100_fields = afb_per_field * 100.0

        grade = "Not reportable"
        reportable = False
        reason = "Observed field count does not meet a WHO/IUATLD reporting threshold."

        if fields >= 20 and afb_per_field > 10:
            grade = "3+"
            reportable = True
            reason = ">10 AFB per field with at least 20 fields examined."
        elif fields >= 50 and afb_per_field >= 1:
            grade = "2+"
            reportable = True
            reason = "1-10 AFB per field with at least 50 fields examined."
        elif fields >= 100:
            if count == 0:
                grade = "Negative"
                reportable = True
                reason = "No AFB observed with at least 100 fields examined."
            elif afb_per_100_fields < 10:
                grade = "Scanty"
                reportable = True
                reason = "1-9 AFB per 100 fields; report the observed count."
            elif afb_per_100_fields < 100:
                grade = "1+"
                reportable = True
                reason = "10-99 AFB per 100 fields."

        if grade == "Scanty" and fields == 100:
            report_string = f"Scanty ({count} AFB / 100 fields)"
        elif reportable:
            report_string = f"{grade} ({count} AFB / {fields} fields)"
        else:
            report_string = f"Not reportable ({count} AFB / {fields} fields)"

        return {
            "grade": grade,
            "reportable": reportable,
            "reason": reason,
            "afb_count": count,
            "fields_counted": fields,
            "afb_per_hpf": afb_per_field,
            "afb_per_100_hpf": afb_per_100_fields,
            "report_string": report_string,
        }
