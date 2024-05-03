import onnxruntime as rt

# Path to your ONNX model file
# model_path = '/home/hamid/model.onnx'  # Update with your actual path
# model_path = '/media/hamid/Workspace/autonoccv/yolov8-onnx-cpp/checkpoints/yolov8n.onnx'  # Update with your actual path
model_path = '/media/hamid/Workspace/Zenith/BiLSTM_ANPR/deployment/models/ModelPlate.onnx'  # Update with your actual path

# Load the ONNX model
sess = rt.InferenceSession(model_path)

# Get output details
output_details = sess.get_outputs()

# Print detailed information for each output
# for output_info in output_details:
#     print(output_info)
#     print('---------')
for output_info in output_details:
    print(output_info)
    print(f"Output Name: {output_info.name}")
    print(f"  Shape: {output_info.shape}")
    print(f"  Data Type: {output_info.type}")
    print('---------')
    # print(f"  Element Type: {output_info.element_type}")
    # print(f"  Quantization Parameters: {output_info.quantization_parameters}\n")
