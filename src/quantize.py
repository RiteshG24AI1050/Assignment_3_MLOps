import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error


def load_model(model_path="models/linear_model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Loaded model: {type(model)} | Coef shape: {model.coef_.shape}")
    return model


def extract_parameters(model):
    weights = model.coef_.astype(np.float32)
    bias = np.float32(model.intercept_)
    print(f"Extracted weights shape: {weights.shape}, Bias: {bias}")
    return weights, bias


def quantize_uint8(params, param_name="parameter"):
    params = np.array(params, dtype=np.float32)
    min_val, max_val = float(np.min(params)), float(np.max(params))
    print(f"\n Quantizing {param_name} — Range: [{min_val:.6f}, {max_val:.6f}]")

    if np.isclose(max_val, min_val, atol=1e-8):
        return {
            'quantized_data': np.array([128], dtype=np.uint8),
            'scale': np.float32(1.0),
            'zero_point': np.uint8(128),
            'min_val': np.float32(min_val),
            'max_val': np.float32(max_val),
            'is_constant': True,
            'original_value': np.float32(min_val)
        }

    scale = np.float32((max_val - min_val) / 255.0)
    quantized = np.clip(np.round((params - min_val) / scale), 0, 255).astype(np.uint8)
    dequantized = (quantized.astype(np.float32) * scale) + min_val
    error = np.mean(np.abs(params - dequantized))

    print(f"  Scale: {scale:.8f} | Quantization MAE: {error:.8f}")
    return {
        'quantized_data': quantized,
        'scale': scale,
        'zero_point': np.uint8(0),
        'min_val': np.float32(min_val),
        'max_val': np.float32(max_val),
        'is_constant': False
    }


def dequantize(quant_info):
    if quant_info.get('is_constant', False):
        return np.full(len(quant_info['quantized_data']), quant_info['original_value'], dtype=np.float32)
    return (quant_info['quantized_data'].astype(np.float32) * quant_info['scale']) + quant_info['min_val']


class QuantizedLinearModel(nn.Module):
    def __init__(self, weights, bias):
        super(QuantizedLinearModel, self).__init__()
        self.linear = nn.Linear(weights.shape[0], 1)
        weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)  # shape: (1, in_features)
        bias_tensor = torch.tensor([bias], dtype=torch.float32)  # shape: (1,)
        self.linear.weight = nn.Parameter(weight_tensor)
        self.linear.bias = nn.Parameter(bias_tensor)

    def forward(self, x):
        return self.linear(x)



def evaluate(original_w, original_b, quant_w, quant_b, test_data_path="models/test_data.joblib"):
    X_test, y_test = joblib.load(test_data_path)

    def _metrics(y_true, y_pred):
        return r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

    orig_pred = X_test @ original_w.T + original_b
    quant_pred = X_test @ quant_w.T + quant_b

    r2_orig, rmse_orig = _metrics(y_test, orig_pred)
    r2_quant, rmse_quant = _metrics(y_test, quant_pred)

    print(f"\n Evaluation:")
    print(f"  R² original:   {r2_orig:.6f} | RMSE original:   {rmse_orig:.6f}")
    print(f"  R² quantized:  {r2_quant:.6f} | RMSE quantized:  {rmse_quant:.6f}")

    return r2_orig, r2_quant, rmse_orig, rmse_quant


def file_size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0


def theoretical_sizes(weights, bias, quant_weights_info):
    orig_size = weights.nbytes + np.array([bias], dtype=np.float32).nbytes
    quant_size = (
        quant_weights_info['quantized_data'].nbytes +
        3 * 4  # scale, min_val, max_val (float32)
    )
    return orig_size, quant_size


def save_results(results_dict, out_path="models/comparison_results.joblib"):
    joblib.dump(results_dict, out_path)
    print(f"Comparison saved to: {out_path}")


def main():
    print("===Sklearn to Quantized PyTorch Model Conversion===")
    os.makedirs("models", exist_ok=True)

    # Step 1: Load model
    model = load_model()
    weights, bias = extract_parameters(model)

    # Step 2: Save unquantized parameters
    joblib.dump({'weights': weights, 'bias': bias}, "models/unquant_params.joblib")

    # Step 3: Quantization
    q_weights_info = quantize_uint8(weights, "weights")
    q_bias_info = quantize_uint8([bias], "bias")
    joblib.dump({'weights': q_weights_info, 'bias': q_bias_info}, "models/quant_params.joblib")

    dq_weights = dequantize(q_weights_info).reshape(weights.shape)
    dq_bias = dequantize(q_bias_info)[0]

    # Step 5: Rebuild & Save PyTorch Model
    model_torch = QuantizedLinearModel(dq_weights, dq_bias)
    torch.save(model_torch.state_dict(), "models/quantized_pytorch_model.pt")
    print("Quantized PyTorch model saved: models/quantized_pytorch_model.pt")

    # Step 6: PyTorch Inference Example
    input_dim = weights.shape[0]
    dummy_input = torch.tensor([[0.5] * input_dim], dtype=torch.float32)  # Note the double brackets
    model_torch.eval()
    with torch.no_grad():
        y_pred = model_torch(dummy_input)
    print(f"PyTorch inference: {y_pred.item():.4f}")

    # Step 7: Evaluate
    r2_orig, r2_quant, rmse_orig, rmse_quant = evaluate(weights, bias, dq_weights, dq_bias)

    # Step 8: File size and compression analysis
    size_orig_kb = file_size_kb("models/unquant_params.joblib")
    size_quant_kb = file_size_kb("models/quant_params.joblib")
    theor_orig, theor_quant = theoretical_sizes(weights, bias, q_weights_info)
    compression_ratio = theor_orig / theor_quant if theor_quant > 0 else 0

    print("\n FINAL COMPARISON TABLE")
    print("-" * 60)
    print(f"{'Metric':<25} {'Original':<15} {'Quantized':<15}")
    print(f"{'R² Score':<25} {r2_orig:<15.6f} {r2_quant:<15.6f}")
    print(f"{'File Size (KB)':<25} {size_orig_kb:<15.3f} {size_quant_kb:<15.3f}")
    print(f"{'Theoretical Size (bytes)':<25} {theor_orig:<15} {theor_quant:<15}")
    print(f"{'Theoretical Size (KB)':<25} {theor_orig/1024:<15.3f} {theor_quant/1024:<15.3f}")
    print(f"{'Compression Ratio':<25} {'':<15} {compression_ratio:.2f}x")
    print("\n Summary:")
    print(f" Theoretical compression: {compression_ratio:.2f}x ({(1 - 1/compression_ratio)*100:.1f}% reduction)")
    print(f" R² preserved: {abs(r2_orig - r2_quant):.6f}")

    # Step 9: Save Results
    save_results({
        "r2_original": r2_orig,
        "r2_quantized": r2_quant,
        "rmse_original": rmse_orig,
        "rmse_quantized": rmse_quant,
        "file_size_orig_kb": size_orig_kb,
        "file_size_quant_kb": size_quant_kb,
        "theoretical_size_orig": theor_orig,
        "theoretical_size_quant": theor_quant,
        "compression_ratio": compression_ratio,
    })


if __name__ == "__main__":
    try:
        main()
        print("\n Quantization pipeline completed successfully!")
    except Exception as e:
        print(f"\n Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()