from utilities.llms.validation_llm import validation_llm
from utilities.prompts.validation_prompts import validation_prompts

def dob_validation(user_input):
    prompt = validation_prompts["date_of_birth_validation_prompt"]
    dob = validation_llm(user_input, prompt)
    if dob == "Unknown":
        return False
    return dob