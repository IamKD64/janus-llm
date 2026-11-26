# Inference image for the fine-tuned adapter. Needs an NVIDIA GPU at run time (4-bit bitsandbytes).
# The adapter is NOT baked in; mount it:
#   docker run --gpus all -p 8000:8000 -v $PWD/outputs/qlora-banking77:/adapter \
#     -e ADAPTER_PATH=/adapter ghcr.io/<you>/<repo>:latest
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[train,serve]"
ENV MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct ADAPTER_PATH=/adapter
EXPOSE 8000
CMD ["uvicorn", "p3ft.serve:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
