from openai import OpenAI
import base64
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


generation_key = "xxxxx"  # GPT key
client = OpenAI(
    api_key=generation_key,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def gpt_infer(system, text, image_list, model="gpt-4-vision-preview", max_tokens=600, response_format=None):

    user_content = []
    for i, image in enumerate(image_list):
        if image is not None:
            user_content.append(
                {
                    "type": "text",
                    "text": f"Image {i}:"
                },
            )

            with open(image, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            image_message = {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{image_base64}",
                         "detail": "low"
                     }
                 }
            user_content.append(image_message)

    user_content.append(
        {
            "type": "text",
            "text": text
        }
    )

    messages = [
        {"role": "system",
         "content": system
         },
        {"role": "user",
         "content": user_content
         }
    ]

    if response_format:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens, response_format=response_format)
    else:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens)

    # print(chat_message)
    answer = chat_message.choices[0].message.content
    tokens = chat_message.usage

    return answer, tokens


