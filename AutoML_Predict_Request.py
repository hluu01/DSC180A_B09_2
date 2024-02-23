# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_tabular_classification_sample]
from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))


# [END aiplatform_predict_tabular_classification_sample]

predict_tabular_classification_sample(
    project="749709878834",
    endpoint_id="9211226831414362112",
    location="us-central1",
    instances=[{ "article": "o give the president of the #Ukraine #billions of dollars to buy these #yachts and live an extravagant lifestyle. I mean, to fight a #war against #Russia. (Screengrab from Instagram) The post was flagged as part of Meta’s efforts to combat false news and misinformation on its News Feed. (Read more about our partnership with Meta, which owns Facebook and Instagram.) The purchase documents are inauthentic as they include outdated information. The yachts purportedly bought by Zelenskyy’s associates are still for sale. The documents are titled Memorandum of Agreement approved by the Mediterranean Yacht Brokers Association. However, the organization that supposedly approved this 2023 sale changed its name to MYBA The Worldwide Yachting Association in 2008. MYBA also uses a different logo now than the one shown on the documents."}]
)