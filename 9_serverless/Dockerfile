FROM --platform=amd64 agrigorev/model-2024-hairstyle:v3

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install numpy==1.23.1
COPY [ "lambda_function.py", "./" ]

CMD [ "lambda_function.lambda_handler" ]