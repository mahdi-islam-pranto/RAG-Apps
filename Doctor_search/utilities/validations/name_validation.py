from utilities.llms.validation_llm import validation_llm
from utilities.prompts.validation_prompts import validation_prompts

# name validation
def name_validation(user_input):

    promt = validation_prompts["name_validation_prompt"]
    # call the llm 
    name = validation_llm(user_input, promt)
    if name == "Unknown":
        return False
    return name