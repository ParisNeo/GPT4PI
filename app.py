######
# Project       : GPT4PI
# Author        : ParisNeo
# Licence       : Apache 2.0
# Description   : 
# What if we bring the power of LLMS to Raspberry PI?
######
from pyllamacpp import model
def create_model(self):
    return model.Model(
        ggml_model=f"./models/{self.args.model}", 
        n_ctx=512, 
        seed=self.args.seed,
        )
def callback(text):
    print(text)

if __name__=="__main__":
    model = create_model()
    print("Testing GPT4All on Raspberry pi")
    while True:
        prompt = input(">")
        model.generate(prompt,new_text_callback=callback, n_predict=55)