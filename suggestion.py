from typing import Dict


def _severity_label(rate: float) -> str:
    """Map a violation percentage to the same severity tiers used in app.py's cards."""
    if rate >= 50:
        return "Critical"
    elif rate >= 25:
        return "High"
    elif rate >= 10:
        return "Medium"
    return "Low"


def generate_safety_suggestions(
    metrics: Dict[str, int],
    percentages: Dict[str, float]
) -> str:
    """
    Generate PPE compliance recommendations from the detection numbers directly.

    This is a rule-based (no external API, no cost, no rate limits) drop-in
    replacement for the old OpenAI-powered version — same signature, same
    call site in app.py, nothing else needs to change. It works instantly
    and offline, which also matters once this is deployed publicly: anyone
    clicking the button on a public Streamlit Cloud link no longer burns
    through a shared API budget.

    Args:
        metrics: Detection counts by label (e.g. {"NO-Hardhat": 12, ...})
        percentages: Non-compliance rates (e.g. {"no_hardhat_rate": 24.5, ...})

    Returns:
        Formatted safety recommendations as plain text.
    """
    rate_info = [
        ("no_hardhat_rate", "Hardhat", metrics.get("NO-Hardhat", 0)),
        ("no_mask_rate", "Mask", metrics.get("NO-Mask", 0)),
        ("no_safety_vest_rate", "Safety Vest", metrics.get("NO-Safety Vest", 0)),
    ]

    # Sort by severity (highest violation rate first) so the biggest risk leads
    active_rates = [
        (key, label, count, percentages[key])
        for key, label, count in rate_info
        if key in percentages
    ]
    active_rates.sort(key=lambda x: x[3], reverse=True)

    if not active_rates:
        return "No PPE compliance data available yet. Run a batch, image, or video analysis first."

    total_violations = sum(v for k, v in metrics.items() if k.startswith("NO-"))
    total_detections = sum(metrics.values())
    overall_rate = round((total_violations / total_detections) * 100, 1) if total_detections else 0.0

    lines = []
    lines.append(
        f"Overall PPE violation rate: {overall_rate}% "
        f"({total_violations} violations out of {total_detections} total detections)."
    )
    lines.append("")

    recommendations_by_item = {
        "Hardhat": {
            "Critical": "Hardhat non-compliance is critical — stop work in the affected area until every worker is fitted with a hardhat, and station a spotter at entry points to enforce it.",
            "High": "Hardhat violations are high — run an immediate toolbox talk on head protection and check that enough hardhats are actually available on site.",
            "Medium": "Hardhat compliance needs attention — spot-check the crew during shift changes and remind supervisors to enforce the rule at entry points.",
            "Low": "Hardhat compliance is mostly good — keep spot-checks in place to maintain it.",
        },
        "Mask": {
            "Critical": "Mask non-compliance is critical — halt tasks that generate dust or fumes until masks are worn, and check the mask supply isn't the bottleneck.",
            "High": "Mask violations are high — reissue masks at the start of each shift and add signage at dusty/high-exposure zones.",
            "Medium": "Mask compliance needs attention — add reminders at zone entrances and confirm masks are comfortable enough for full-shift wear.",
            "Low": "Mask compliance is mostly good — keep monitoring exposure-heavy zones.",
        },
        "Safety Vest": {
            "Critical": "Safety vest non-compliance is critical — this is a high-visibility risk around moving vehicles/machinery; pause vehicle movement in the zone until vests are on.",
            "High": "Safety vest violations are high — check vest availability and sizing, and brief drivers/operators to watch for unvested workers.",
            "Medium": "Safety vest compliance needs attention — reinforce the rule near vehicle and machinery traffic lanes.",
            "Low": "Safety vest compliance is mostly good — keep it up, especially near active machinery.",
        },
    }

    lines.append("Priority actions:")
    for key, label, count, rate in active_rates:
        severity = _severity_label(rate)
        lines.append(f"- [{severity} — {rate}%] {recommendations_by_item[label][severity]}")

    lines.append("")
    lines.append(
        "General: log repeat offenders by zone/shift if the camera setup allows it, "
        "and review these numbers weekly to see whether training or enforcement changes are working."
    )

    return "\n".join(lines)