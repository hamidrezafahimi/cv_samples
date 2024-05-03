import onnx

# Load the ONNX model (replace 'model.onnx' with your actual model path)
model_path = '/media/hamid/Workspace/Zenith/BiLSTM_ANPR/deployment/models/ModelPlate.onnx'  # Update with your actual path
# model_path = '/media/hamid/Workspace/autonoccv/yolov8-onnx-cpp/checkpoints/yolov8n.onnx'  # Update with your actual path
model = onnx.load(model_path)

# Iterate through inputs of the graph
for input_info in model.graph.input:
    print(f"Input Name: {input_info.name}")
    if input_info.type.tensor_type.HasField("shape"):
        shape = input_info.type.tensor_type.shape
        dims = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in shape.dim]
        print(f"  Shape: {dims}")
    else:
        print("  Unknown rank")
