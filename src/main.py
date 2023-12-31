import json
from collections import defaultdict

from src import (
    validation,
    utils,
    OPENAI
)

# step 0 准备数据集
...
dataset_path = 'dataset/'
data_path = dataset_path + 'toy_chat_fine_tuning.jsonl'

# step 1 加载数据集
dataset = utils.load_json(data_path)

# step2: 验证数据
validation.verification_format(dataset)

# step3: 验证token
convo_lens = validation.verification_token(dataset)

# step4: 计算花费
utils.get_cost(dataset, convo_lens)

# step5: 划分数据集(可选)
train_file_name = 'toy_chat_fine_tuning.jsonl'
validation_file_name = None
utils.split_dataset(dataset, dataset_path, train_file_name, validation_file_name, scale=3/1)

# step6: 上传数据
OPENAI.init()  # change base_url api_key proxy and python path
training_file_id, validation_file_id = OPENAI.upload(train_file_name, validation_file_name)

# step7: 开始训练
suffix = ...
job_id = OPENAI.tune(suffix, training_file_id, validation_file_id)
OPENAI.check_job_progress(job_id)  # or check_job_status(job_id)

# step8: 训练进度
fine_tuned_model_id = OPENAI.get_fine_tuned_model(job_id)

# step9: 使用
messages = [{'content': ..., 'role': 'system'}, {'content': ..., 'role': 'user'}]
print(OPENAI.chat_fine_tuned_model(fine_tuned_model_id, messages=messages))
