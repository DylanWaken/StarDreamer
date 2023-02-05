import base64
import io

import cv2
import tornado.web
import tornado.ioloop
import openai
import torch
import json

from PIL import Image

import pipeline
from io import BytesIO
import numpy
from matplotlib import pyplot as plt

openai.api_key = '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'
model = 'text-davinci-003'
max_words = 50
max_characters = 400
max_tokens = 150
ppl = pipeline.generatePipelinee()


def get_info(galaxy):
    prompt = 'describe ' + galaxy + ' scientifically (in ' + str(max_characters) + ' characters)'
    response = openai.Completion.create(model=model, prompt=prompt, max_tokens=max_tokens)
    print(response)
    return response['choices'][0]['text'].strip()


def get_transformed_image(raw_sketch):
    print(type(raw_sketch))

    image_stream = BytesIO(raw_sketch)

    # open the image using PIL
    image = Image.open(image_stream)

    return ppl.generate(image)


class basicRequestHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("""
            <html>
                <body>
                    <form action="/" method="post" enctype="multipart/form-data">
                        choose file: <input type="file" name="sketch">
                        <input type="submit" value="Submit">
                    </form>
                </body>
            </html>
        """)

    def post(self):
        print("received request")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

        # convert input sketch for future processing
        raw_sketch = json.loads(self.request.body.decode('utf-8'))
        # raw_sketch = self.request.files["sketch"][0]["body"]
        transformed_img, galaxy_class = get_transformed_image(base64.b64decode(raw_sketch["image"]))
        # transformed_img, galaxy_class = get_transformed_image(raw_sketch)

        print(type(transformed_img))
        # plt.imshow(transformed_img)
        # plt.show()
        # output = {"name": galaxy_class, "info": get_info(galaxy_class),
        #           "image": str(base64.b64encode(transformed_img.tobytes()))[2:-1]}
        transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
        tmp_img = Image.fromarray(numpy.array(transformed_img))
        buffered = io.BytesIO()
        tmp_img.save(buffered, format="JPEG")

        base64_bytes = base64.b64encode(buffered.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        output = {"name": galaxy_class, "info": get_info(galaxy_class),
                  "image": base64_string}
        with open("test.txt", "w") as f:
            f.write(base64_string)
        print(output["name"])
        print(output["info"])

        self.write(json.dumps(output))

        print("success!")

    def options(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.set_header("Access-Control-Max-Age", "3600")
        self.finish()
