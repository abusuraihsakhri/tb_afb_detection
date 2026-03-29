from typing import Optional, Dict, Any

class WHOGrader:
    """
    Calculate WHO/IUATLD smear grading from AFB detections.
    Protected against Mathematical anomalies and Divide-by-Zero exploits.
    """
    HPF_AREA_MM2 = 0.196  
    
    def __init__(self, magnification: float = 40.0, fov_microns: float = 400.0):
        # 🛡️ SECURITY CONTROL: Mag validation
        if magnification <= 0:
            raise ValueError("Magnification must be strictly positive.")
        self.magnification = magnification
        self.fov_microns = fov_microns
        
    def calculate_grade(self, afb_count: int, slide_area_mm2: float, fields_examined: Optional[int] = None) -> Dict[str, Any]:
        """Calculate WHO grade from AFB count."""
        # 🛡️ SECURITY CONTROL: Absolute checks to avoid Zero-division causing NaN backend crash cascades
        if slide_area_mm2 <= 1e-6:
             raise ValueError(f"Slide computed area {slide_area_mm2} is illegally small/negative.")
             
        fields = fields_examined if fields_examined else self._estimate_hpf_equivalent(slide_area_mm2, self.magnification)
        
        # 🛡️ SECURITY CONTROL: Avoid dividing by 0 fields
        fields = max(1, fields)
        afb_per_hpf = afb_count / fields
        afb_per_100_hpf = afb_per_hpf * 100.0
        
        grade = "Negative"
        if afb_per_hpf >= 10:
             grade = "3+"
        elif afb_per_hpf >= 1:
             grade = "2+"
        elif afb_per_100_hpf >= 10:
             grade = "1+"
        elif 0 < afb_count < 10:
             grade = "Scanty"

        return {
            "grade": grade,
            "afb_per_100_hpf": afb_per_100_hpf,
            "afb_per_hpf": afb_per_hpf,
            "fields_counted": fields,
            "report_string": f"{grade} ({afb_count} AFB / {fields} HPF)"
        }
    
    def _estimate_hpf_equivalent(self, slide_area_mm2: float, magnification: float) -> int:
        return int((slide_area_mm2 / self.HPF_AREA_MM2) * ((1000/magnification)**2))
