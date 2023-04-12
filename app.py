######
# Project       : GPT4PI
# Author        : ParisNeo
# Licence       : Apache 2.0
# Description   : 
# What if we bring the power of LLMS to Raspberry PI?
######
from pyllamacpp import model
def create_model(self, model_path):
    return model.Model(
        ggml_model=model_path, 
        n_ctx=512, 
        )
def callback(text):
    print(text)

if __name__=="__main__":
    model_path = f"./models/gpt4all-lora-quantized-ggml.bin"
    model = create_model(model_path)
    print("Testing GPT4All on Raspberry pi")
    while True:
        prompt = input(">")
        model.generate(prompt,new_text_callback=callback, n_predict=55)

        