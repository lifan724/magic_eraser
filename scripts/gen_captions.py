from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration
)


model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt_list = [
    "USER: <image>\nWhat's the content of the image? ASSISTANT:",
    "USER: <image>\nWhat's the {} like in the image? ASSISTANT:",
    "USER: <image>\nDescribe the background {}? ASSISTANT:",
]


def gen_captions(img_dir, background_tag):
    image = Image.open(img_dir)
    prompt = prompt_list[-1].format(background_tag)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=77)
    cap = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return cap


if __name__ == '__main__':
    gen_captions(
        img_dir="./imgs/sheep/input.png",
        background_tag = "gravel, grass"
    )