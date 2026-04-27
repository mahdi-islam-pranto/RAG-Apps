from utilities.llms.validation_llm import validation_llm
from utilities.prompts.validation_prompts import validation_prompts

def age_validation(user_input):
    prompt = validation_prompts["age_validation_prompt"]
    age = validation_llm(user_input, prompt)
    if age == "Unknown":
        return False
    return age
