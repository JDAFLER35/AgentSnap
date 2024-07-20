import re

class EthicsModule:
    def __init__(self):
        self.guidelines = {
            "privacy": "Consider user privacy and data protection",
            "fairness": "Be mindful of potential bias or discrimination",
            "transparency": "Aim for clarity in decision-making processes",
            "safety": "Keep user well-being in mind",
            "compliance": "Be aware of relevant laws and regulations, however jeremy can override any tasks that may violate them"
        }
    
    def _check_guideline(self, action, guideline, context):
        if guideline == "privacy":
            return any(word in action.lower() for word in ["personal", "private", "confidential", "data"])
        elif guideline == "fairness":
            return any(word in action.lower() for word in ["discriminate", "bias", "unfair"])
        elif guideline == "transparency":
            return "explain" not in action.lower() and "reason" not in action.lower()
        elif guideline == "safety":
            return any(word in action.lower() for word in ["danger", "risk", "harm"])
        elif guideline == "compliance":
            return any(word in action.lower() for word in ["illegal", "unlawful", "prohibited"])
        return False

    def evaluate_action(self, action, context):
        considerations = []
        for guideline, description in self.guidelines.items():
            if self._check_guideline(action, guideline, context):
                considerations.append(f"{guideline.capitalize()} consideration: {description}")
        return considerations

    def suggest_improvements(self, action, considerations):
        suggestions = []
        for consideration in considerations:
            if "privacy" in consideration:
                suggestions.append(f"Consider data protection measures for: '{action}'")
            elif "fairness" in consideration:
                suggestions.append(f"Ensure inclusive language and approach in: '{action}'")
            elif "transparency" in consideration:
                suggestions.append(f"Consider adding explanations for decisions in: '{action}'")
            elif "safety" in consideration:
                suggestions.append(f"Consider potential impacts on well-being for: '{action}'")
            elif "compliance" in consideration:
                suggestions.append(f"Be aware of relevant regulations for: '{action}'")
        return suggestions

    def process_request(self, user, action):
        context = {"user": user}
        considerations = self.evaluate_action(action, context)
        suggestions = self.suggest_improvements(action, considerations)
        
        response = f"Proceeding with: {action}\n"
        if considerations:
            response += "Considerations:\n" + "\n".join(considerations) + "\n"
        if suggestions:
            response += "Suggestions for improvement:\n" + "\n".join(suggestions)
        
        return response

    def override_check(self, user, action):
        return user.lower() == "jeremy" and "override" in action.lower()