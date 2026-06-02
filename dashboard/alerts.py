"""
Threshold rule-engine analyzer for Air Quality Index alerts.
Translates numeric predictions to actionable environmental safety messages.
"""


def get_epa_tier_details(aqi_value: float) -> dict:
    """Returns official health classifications and theme styling for given AQI values."""
    if aqi_value <= 50:
        return {"label": "Good", "color": "#00e400", "bg_color": "#1e3a1e"}
    elif aqi_value <= 100:
        return {"label": "Moderate", "color": "#ffff00", "bg_color": "#3a3a1e"}
    elif aqi_value <= 150:
        return {"label": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "bg_color": "#4a321a"}
    elif aqi_value <= 200:
        return {"label": "Unhealthy", "color": "#ff0000", "bg_color": "#4a1a1a"}
    elif aqi_value <= 300:
        return {"label": "Very Unhealthy", "color": "#8f3f97", "bg_color": "#331a4a"}
    else:
        return {"label": "Hazardous", "color": "#7e0023", "bg_color": "#300a14"}


def generate_bounded_alert_banner(pred_aqi: float, upper_bound: float) -> tuple[str, str]:
    """Generates situational status summaries for live monitoring updates."""
    if upper_bound > 200 or pred_aqi > 150:
        return (
            "Simulated Warning: Elevated AQI values detected.",
            "warning"
        )
    return (
        "Simulated Success: Atmospheric clearance metrics confirmed.",
        "success"
    )