"""
Skin Condition Database

Contains information about various dermatological conditions
including descriptions, severity, and recommendations.
"""

from typing import Dict, List, Optional


class SkinConditionDatabase:
    """
    Database of skin conditions with medical information

    Provides detailed information about each condition that the model can detect.
    """

    def __init__(self):
        """Initialize the conditions database"""
        self.conditions = self._load_conditions()

    def _load_conditions(self) -> Dict[str, Dict]:
        """
        Load condition information

        Returns:
            Dictionary mapping condition names to their details
        """
        return {
            "Actinic Keratosis": {
                "name": "Actinic Keratosis",
                "abbreviation": "AK",
                "severity": "Low to Moderate",
                "description": (
                    "Actinic keratosis is a rough, scaly patch on the skin that develops "
                    "from years of sun exposure. It's most commonly found on areas often "
                    "exposed to the sun like face, lips, ears, forearms, scalp, and neck."
                ),
                "symptoms": [
                    "Rough, dry or scaly patch of skin",
                    "Flat to slightly raised patch or bump",
                    "Hard, wart-like surface",
                    "Color variations (pink, red, or brown)",
                    "Itching, burning, or bleeding"
                ],
                "causes": [
                    "Frequent or intense sun exposure",
                    "UV radiation from tanning beds",
                    "Fair skin, light hair, and light-colored eyes",
                    "Age over 40"
                ],
                "recommendations": [
                    "Consult a dermatologist for proper diagnosis",
                    "Treatment may include cryotherapy, topical medications, or photodynamic therapy",
                    "Use sunscreen daily (SPF 30+)",
                    "Wear protective clothing",
                    "Regular skin examinations"
                ],
                "urgency": "Schedule appointment within 2-4 weeks"
            },

            "Basal Cell Carcinoma": {
                "name": "Basal Cell Carcinoma",
                "abbreviation": "BCC",
                "severity": "Moderate to High",
                "description": (
                    "Basal cell carcinoma is a type of skin cancer that begins in the basal cells. "
                    "It's the most common form of skin cancer but rarely spreads beyond the original "
                    "site. It typically appears on sun-exposed areas."
                ),
                "symptoms": [
                    "Pearly or waxy bump",
                    "Flat, flesh-colored or brown scar-like lesion",
                    "Bleeding or scabbing sore that heals and returns",
                    "Pink growth with raised edges",
                    "Open sore that doesn't heal"
                ],
                "causes": [
                    "Long-term UV radiation exposure",
                    "Fair skin",
                    "History of sunburns",
                    "Family history of skin cancer",
                    "Weakened immune system"
                ],
                "recommendations": [
                    "⚠️ See a dermatologist URGENTLY",
                    "Early treatment is highly effective",
                    "May require surgical removal",
                    "Regular follow-up examinations needed",
                    "Strict sun protection measures"
                ],
                "urgency": "Schedule appointment within 1-2 weeks"
            },

            "Benign Keratosis": {
                "name": "Benign Keratosis (Seborrheic Keratosis)",
                "abbreviation": "BKL",
                "severity": "Low (Non-cancerous)",
                "description": (
                    "Benign keratosis, also known as seborrheic keratosis, is a common "
                    "non-cancerous skin growth. They appear as waxy, scaly, slightly elevated "
                    "growths and are usually brown, though they can vary in color."
                ),
                "symptoms": [
                    "Round or oval-shaped growth",
                    "Flat or slightly raised with scaly surface",
                    "Brown, black, or light tan color",
                    "Waxy, 'stuck-on' appearance",
                    "Usually painless"
                ],
                "causes": [
                    "Age (more common after 50)",
                    "Genetic factors",
                    "Not caused by sun exposure",
                    "Exact cause unknown"
                ],
                "recommendations": [
                    "Generally no treatment needed",
                    "Removal for cosmetic reasons if desired",
                    "Removal if growth is irritated or itchy",
                    "Regular skin checks to monitor changes",
                    "Distinguish from other lesions"
                ],
                "urgency": "Routine checkup is sufficient"
            },

            "Dermatofibroma": {
                "name": "Dermatofibroma",
                "abbreviation": "DF",
                "severity": "Low (Benign)",
                "description": (
                    "Dermatofibroma is a common benign skin nodule. It typically appears as a "
                    "small, firm bump on the skin and is usually harmless. They often develop "
                    "on the lower legs but can appear anywhere."
                ),
                "symptoms": [
                    "Small, hard bump under the skin",
                    "Red, brown, or purple color",
                    "Dimples when pinched",
                    "Usually less than 1 cm in diameter",
                    "May be itchy or tender"
                ],
                "causes": [
                    "Possibly minor skin injury or insect bite",
                    "Overgrowth of fibrous tissue",
                    "More common in women",
                    "Exact cause unclear"
                ],
                "recommendations": [
                    "Usually requires no treatment",
                    "Removal if bothersome or for cosmetic reasons",
                    "Monitor for any changes in size or color",
                    "Consult dermatologist if concerned"
                ],
                "urgency": "Routine checkup is sufficient"
            },

            "Melanoma": {
                "name": "Melanoma",
                "abbreviation": "MEL",
                "severity": "High (Most serious skin cancer)",
                "description": (
                    "Melanoma is the most serious type of skin cancer, developing in the "
                    "melanocytes (pigment-producing cells). While less common than other skin "
                    "cancers, it's more likely to spread if not caught early. Early detection "
                    "and treatment are crucial."
                ),
                "symptoms": [
                    "New, unusual growth or change in existing mole",
                    "Asymmetrical shape",
                    "Irregular or poorly defined borders",
                    "Multiple colors or uneven color distribution",
                    "Diameter larger than 6mm",
                    "Evolving in size, shape, or color"
                ],
                "causes": [
                    "UV radiation exposure",
                    "History of sunburns, especially in childhood",
                    "Many moles (more than 50)",
                    "Fair skin, light hair, light eyes",
                    "Family history of melanoma",
                    "Weakened immune system"
                ],
                "recommendations": [
                    "🚨 URGENT: See a dermatologist IMMEDIATELY",
                    "Early-stage melanoma is highly treatable",
                    "May require biopsy and surgical removal",
                    "Possible additional treatments (immunotherapy, targeted therapy)",
                    "Regular full-body skin examinations",
                    "Strict sun protection"
                ],
                "urgency": "URGENT - Schedule within days"
            },

            "Melanocytic Nevus": {
                "name": "Melanocytic Nevus (Mole)",
                "abbreviation": "NV",
                "severity": "Low (Usually benign)",
                "description": (
                    "Melanocytic nevus, commonly called a mole, is a benign growth of "
                    "melanocytes (pigment cells). Most people have between 10-40 moles. "
                    "While usually harmless, monitoring moles for changes is important."
                ),
                "symptoms": [
                    "Round or oval shape",
                    "Even color (brown, tan, black, red, pink)",
                    "Flat or slightly raised",
                    "Consistent size (usually < 6mm)",
                    "Symmetrical appearance"
                ],
                "causes": [
                    "Genetic factors",
                    "Sun exposure",
                    "Present from birth or develop over time",
                    "Hormonal changes"
                ],
                "recommendations": [
                    "Monitor for changes using ABCDE rule",
                    "Regular self-examinations",
                    "Annual skin check with dermatologist",
                    "Use sunscreen to prevent new moles",
                    "Removal if atypical features present"
                ],
                "urgency": "Routine checkup recommended"
            },

            "Vascular Lesion": {
                "name": "Vascular Lesion",
                "abbreviation": "VASC",
                "severity": "Low (Usually benign)",
                "description": (
                    "Vascular lesions are abnormalities of blood vessels in the skin. "
                    "This includes conditions like hemangiomas, angiomas, and telangiectasias. "
                    "Most are harmless but can be treated for cosmetic reasons."
                ),
                "symptoms": [
                    "Red, purple, or blue discoloration",
                    "Flat or raised appearance",
                    "May blanch when pressed",
                    "Various sizes and shapes",
                    "Usually painless"
                ],
                "causes": [
                    "Abnormal blood vessel development",
                    "Genetics",
                    "Sun damage",
                    "Aging",
                    "Hormonal changes"
                ],
                "recommendations": [
                    "Usually no treatment necessary",
                    "Laser therapy for cosmetic improvement",
                    "Monitor for changes",
                    "Consult dermatologist if bleeding or painful",
                    "Protect from sun damage"
                ],
                "urgency": "Routine checkup is sufficient"
            }
        }

    def get_condition_info(self, condition_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific condition

        Args:
            condition_name: Name of the condition

        Returns:
            Dictionary with condition details or None if not found
        """
        return self.conditions.get(condition_name)

    def list_all_conditions(self) -> List[str]:
        """
        Get list of all condition names

        Returns:
            List of condition names
        """
        return list(self.conditions.keys())

    def search_by_severity(self, severity: str) -> List[str]:
        """
        Find conditions by severity level

        Args:
            severity: Severity level to search for

        Returns:
            List of condition names matching the severity
        """
        return [
            name for name, info in self.conditions.items()
            if severity.lower() in info['severity'].lower()
        ]

    def get_urgent_conditions(self) -> List[str]:
        """
        Get list of conditions requiring urgent attention

        Returns:
            List of condition names with high urgency
        """
        urgent = []
        for name, info in self.conditions.items():
            if 'URGENT' in info['urgency'].upper() or 'High' in info['severity']:
                urgent.append(name)
        return urgent
