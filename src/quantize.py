import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import os

class QuantizedLinearRegression(nn.Module):
    def __init__(self, input_dim, weight, bias):
        super(QuantizedLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.linear.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32).unsqueeze(0))
        self.linear.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

def quantize_float16(params, param_name="parameter"):
    params = np.array(params, dtype=np.float32)
    print(f"\n Quantizing {param_name} with float16 precision...")
    quantized = params.astype(np.float16)
    dequantized = quantized.astype(np.float32)
    error = np.mean(np.abs(params - dequantized))
    print(f"  MAE after float16 quantization: {error:.8f}")
    
    return {
        'quantized_data': quantized,
        'is_constant': False,
    }

def dequantize(quant_info):
    if quant_info.get('is_constant', False):
        return np.full(len(quant_info['quantized_data']), quant_info['original_value'], dtype=np.float32)
    return quant_info['quantized_data'].astype(np.float32)

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        predictions = model(inputs)
        r2 = r2_score(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
    return r2, rmse

def main():
    print(" Quantization pipeline started...")

    # Load the trained sklearn model
    model_sklearn = joblib.load("models/linear_model.joblib")
    weights = model_sklearn.coef_
    bias = model_sklearn.intercept_
    print(f" Loaded model: {type(model_sklearn)} | Coef shape: {weights.shape}")
    print(f" Extracted weights shape: {weights.shape}, Bias: {bias}")

    # Quantize weights and bias using float16
    q_weights_info = quantize_float16(weights, "weights")
    q_bias_info = quantize_float16([bias], "bias")

    # Dequantize
    dq_weights = dequantize(q_weights_info)
    dq_bias = dequantize(q_bias_info)[0]

    # Save quantized PyTorch model
    input_dim = weights.shape[0]
    quantized_model = QuantizedLinearRegression(input_dim, dq_weights, dq_bias)
    os.makedirs("models", exist_ok=True)
    torch.save(quantized_model.state_dict(), "models/quantized_pytorch_model.pt")
    print(f"\n Quantized PyTorch model saved: models/quantized_pytorch_model.pt")

    # Run inference on dummy input
    dummy_input = torch.tensor([[0.5] * input_dim], dtype=torch.float32)
    output = quantized_model(dummy_input)
    print(f" PyTorch inference: {output.item():.4f}")

    # Load dataset
    X_test, y_test = joblib.load("models/test_data.joblib")

    # Evaluate original sklearn model
    y_pred_original = model_sklearn.predict(X_test)
    r2_original = r2_score(y_test, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))

    # Evaluate quantized PyTorch model
    r2_quant, rmse_quant = evaluate_model(quantized_model, X_test, y_test)

    # File size comparison
    size_orig = os.path.getsize("models/linear_model.joblib") / 1024
    size_quant = os.path.getsize("models/quantized_pytorch_model.pt") / 1024

    print("\n Evaluation:")
    print(f"  R² original: {r2_original:.6f} | RMSE original: {rmse_original:.4f}")
    print(f"  R² quantized: {r2_quant:.6f} | RMSE quantized: {rmse_quant:.4f}")

    print("\n FINAL COMPARISON TABLE")
    print("-" * 60)
    print(f"{'Metric':<20} {'Original':<15} {'Quantized':<15}")
    print(f"{'-'*60}")
    print(f"{'R² Score':<20} {r2_original:<15.6f} {r2_quant:<15.6f}")
    print(f"{'File Size (KB)':<20} {size_orig:<15.3f} {size_quant:<15.3f}")
    print(f"{'Theoretical Size (bytes)':<20} {weights.nbytes + 4:<15} {dq_weights.nbytes + 4:<15}")
    print(f"{'Theoretical Size (KB)':<25} {(weights.nbytes + 4)/1024:<15.2f} {(dq_weights.nbytes + 4)/1024:<15.2f}")
    print(f"{'Compression Ratio':<20} {(size_orig / size_quant):.2f}x")

    # Summary
    print("\n Summary:")
    print(f"  Theoretical compression: {(weights.nbytes + 4)/(dq_weights.nbytes + 4):.2f}x")
    print(f"  R² preserved: {r2_original - r2_quant:.6f} loss")

    # Save comparison
    comparison = {
        "r2_original": r2_original,
        "r2_quantized": r2_quant,
        "rmse_original": rmse_original,
        "rmse_quantized": rmse_quant,
        "size_original_kb": size_orig,
        "size_quantized_kb": size_quant,
    }
    joblib.dump(comparison, "models/comparison_results.joblib")

    print("\n Quantization pipeline completed successfully!")

if __name__ == "__main__":
    main()
