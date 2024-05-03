import onnxruntime as rt

# Path to your ONNX model file
# model_path = '/home/hamid/model.onnx'  # Update with your actual path
# model_path = '/media/hamid/Workspace/autonoccv/yolov8-onnx-cpp/checkpoints/yolov8n.onnx'  # Update with your actual path
model_path = '/media/hamid/Workspace/Zenith/BiLSTM_ANPR/deployment/models/ModelPlate.onnx'  # Update with your actual path

# Load the ONNX model
sess = rt.InferenceSession(model_path)

# Get input names and properties
print("Input details:")
for input_info in sess.get_inputs():
    print(f"  Name: {input_info.name}")
    print(f"  Shape: {input_info.shape}")
    print(f"  Data type: {input_info.type}")

# Get output names
print("\nOutput details:")
for output_info in sess.get_outputs():
    print(f"  Name: {output_info.name}")