# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app


RUN pip install uv
RUN uv venv

COPY pyproject.toml uv.lock ./

RUN uv sync

COPY src ./src
COPY configs ./configs
COPY helper_function ./helper_function
COPY checkpoints ./checkpoints


# For the container to use the 'uv' virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Run the forward pass
CMD ["uv", "run", "python", "-m", "src.forward_pass_ex"]
