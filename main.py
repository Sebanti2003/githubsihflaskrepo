from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Load the PyTorch model
model = torch.load('model.pt')
model.eval()  # Set the model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['inputData']
    input_tensor = torch.tensor(input_data)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return jsonify(output.tolist())

if __name__ == '__main__':
    app.run(port=5000)
