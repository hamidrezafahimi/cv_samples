import onnxruntime as rt

# Path to your ONNX model file
# model_path = '/home/hamid/model.onnx'  # Update with your actual path
# model_path = '/media/hamid/Workspace/autonoccv/yolov8-onnx-cpp/checkpoints/yolov8n.onnx'  # Update with your actual path
model_path = '/media/hamid/Workspace/Zenith/BiLSTM_ANPR/deployment/models/ModelPlate.onnx'  # Update with your actual path

# Load the ONNX model
sess = rt.InferenceSession(model_path)

# Get the model metadata
meta = sess.get_modelmeta()

# Print the metadata
print(f"Custom metadata map: {meta.custom_metadata_map}")
print(f"Description: {meta.description}")
print(f"Domain: {meta.domain}")
print(f"Graph name: {meta.graph_name}")
print(f"Producer name: {meta.producer_name}")
print(f"Version: {meta.version}")
