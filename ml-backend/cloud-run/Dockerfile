FROM python:3.10 as build-stage
WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10
RUN apt-get update

WORKDIR /app
COPY --from=build-stage /tmp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY . .
RUN ls -la
EXPOSE 8080
CMD ["python", "main.py"]