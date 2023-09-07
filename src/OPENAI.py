#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 17:53
# @Author  : 雷雨
# @File    : openai.py
# @Desc    :
import openai


def init(
        base_url: str = None,
        api_key: str = None,
        proxy: str = None,
):
    for attr, value in locals().items():
        if value:
            setattr(openai, attr, value)


def upload(training_file_name, validation_file_name=None):
    training_response = openai.File.create(
        file=open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response["id"]
    print("Training file ID:", training_file_id)

    validation_file_id = None
    if validation_file_name:
        validation_response = openai.File.create(
            file=open(validation_file_name, "rb"), purpose="fine-tune"
        )
        validation_file_id = validation_response["id"]
        print("Validation file ID:", validation_file_id)

    return training_file_id, validation_file_id


def tune(suffix, training_file_id, validation_file_id=None):
    response = openai.FineTuningJob.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-3.5-turbo",
        suffix=suffix,
    )

    job_id = response["id"]

    print("Job ID:", response["id"])
    print("Status:", response["status"])
    return job_id


def cancel_job(job_id):
    openai.FineTuningJob.cancel(job_id)


def check_job_status(job_id):
    response = openai.FineTuningJob.retrieve(job_id)

    print("Job ID:", response["id"])
    print("Status:", response["status"])
    print("Trained Tokens:", response["trained_tokens"])


def check_job_progress(job_id):
    response = openai.FineTuningJob.list_events(id=job_id, limit=50)

    events = response["data"]
    events.reverse()

    for event in events:
        print(event["message"])


def get_fine_tuned_model(job_id):
    response = openai.FineTuningJob.retrieve(job_id)
    fine_tuned_model_id = response["fine_tuned_model"]
    print("Fine-tuned model ID:", fine_tuned_model_id)
    return fine_tuned_model_id


def chat_fine_tuned_model(fine_tuned_model_id, messages):
    response = openai.ChatCompletion.create(
        model=fine_tuned_model_id, messages=messages, temperature=0, max_tokens=500
    )
    return response["choices"][0]["message"]["content"]
