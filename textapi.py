from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2


PAT = '8fe469dde8b44a059342086cdf8fa12d'
USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = 'GPT-4'
MODEL_VERSION_ID = '5d7a50b44aec4a01a9c492c5a5fcf387'
with open("detected_alphabets.txt", "r") as file:
    text_from_file = file.read()
RAW_TEXT = text_from_file + 'Convert the alphabets into a proper sentence'

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (('authorization', 'Key ' + PAT),)

userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

post_model_outputs_response = stub.PostModelOutputs(
    service_pb2.PostModelOutputsRequest(
        user_app_id=userDataObject,
        model_id=MODEL_ID,
        version_id=MODEL_VERSION_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    text=resources_pb2.Text(
                        raw=RAW_TEXT
                    )
                )
            )
        ]
    ),
    metadata=metadata
)
if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
    print(post_model_outputs_response.status)
    raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

output = post_model_outputs_response.outputs[0]

output_text = output.data.text.raw


# Save output to a text file, removing apostrophes
with open("input.txt", "w") as f:
    f.write(output_text)

print("Completion:\n")
print(output_text)

# Read text from the text file
with open("input.txt", "r") as file:
    text_with_apostrophes = file.read()

# Remove apostrophes from the text
text_without_apostrophes = text_with_apostrophes.replace('"', '')

# Write the modified text back to the file
with open("input.txt", "w") as file:
    file.write(text_without_apostrophes)
print("Apostrophes removed successfully.")


