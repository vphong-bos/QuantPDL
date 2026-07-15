# QuantPDL doc

The repository implements quantization for the PDL model using PyTorch, AIMET and further libraries like ONNX, INC ONNX. Model detail can be found in the paper: https://arxiv.org/pdf/1911.10194

This document explains the three main quantization/export entrypoints in `QuantPDL`:

* **AIMET**: `build_sim_quantized_pdl.py`
* **ONNX Runtime PTQ**: `build_quantized_pdl.py`
* **INC for ONNX**: `build_neural_compressed_pdl.py`

The repo's flow is: start from the FP32 PDL checkpoint, prepare a representative calibration set, and export an INT8-ready model for deployment or further evaluation. In our current setting, AIMET is enough, other quantize library is for experiment only. This doc explains how to run the QuantPDL AIMET flow on a regular GPU server, from environment setup to generating a quantized model.

## AIMET workflow summary

1. Load the pre-trained FP32 PDL model.
   - `build_sim_quantized_pdl.py` loads weights from `--weights_path` (default: `weights/model_final_bd324a.pkl`).
2. Prepare calibration dataset.
   - Use `--calib_images`, `--num_calib`, `--calib_size`, and `--seed`.
3. Apply Cross-Layer Equalization (CLE) for better quantization.
   - Enable with `--enable_cle`.
4. Create a QuantizationSimModel (Post-Training Quantization).
   - Built internally after model wrapping/folding.
5. Calibrate the quantized model.
   - Encodings are computed using calibration forward pass.
6. Compare FP32 vs INT8 outputs.
   - Run evaluation using `run_eval.py` with both `--fp32_weights` and `--quant_weights`.

7. Experiment on advanced quantization techniques.
   - Optional switches: `--enable_bn_fold`, `--enable_adaround`, `--enable_seq_mse`, `--enable_bn_reestimation`, `--run_quant_analyzer`.
8. Export the quantized model.
   - ONNX and AIMET export artifacts are written to `--export_path` unless `--no_export` is used.

## 1) Paths used in this doc

Use these defaults from your current workspace:

- Repo root: `/workspace/quant_pipeline/QuantPDL`
- Working data dir: `/workspace/quant_pipeline`
- Cityscapes root created by these steps: `/workspace/quant_pipeline/cityscapes`
- Quantized export dir: `/workspace/quant_pipeline/quantized_model`
- Checkpoint dir: `/workspace/quant_pipeline/checkpoint`

If you want different locations, update the variables in Step 2.

## 2) Set environment variables

```bash
export REPO_ROOT="/workspace/quant_pipeline/QuantPDL"
export CITYSCAPES_DIR="${REPO_ROOT}/cityscapes"
export QUANT_OUT_DIR="${REPO_ROOT}/quantized_model"
export CKPT_DIR="${REPO_ROOT}/checkpoint"
```

## 3) Setup dependencies and baseline FP32 weights

```bash
bash "${REPO_ROOT}/setup.sh"
```

This installs Python dependencies, Cityscapes scripts, neural-compressor, and downloads:

- `${REPO_ROOT}/weights/model_final_bd324a.pkl`

## 4) Download Cityscapes packages

Important:

- You need a Cityscapes account: https://www.cityscapes-dataset.com/register/
- The downloader may prompt for username/password in terminal on first run.

```bash
python "${REPO_ROOT}/quantization/downloader.py" -d "${REPO_ROOT}" leftImg8bit_trainvaltest.zip
python "${REPO_ROOT}/quantization/downloader.py" -d "${REPO_ROOT}" gtFine_trainvaltest.zip
```

## 5) Extract required Cityscapes folders

```bash
mkdir -p "${CITYSCAPES_DIR}"

unzip -q -o "${REPO_ROOT}/leftImg8bit_trainvaltest.zip" "leftImg8bit/train/*" -d "${CITYSCAPES_DIR}"
unzip -q -o "${REPO_ROOT}/leftImg8bit_trainvaltest.zip" "leftImg8bit/val/*" -d "${CITYSCAPES_DIR}"
rm -f "${REPO_ROOT}/leftImg8bit_trainvaltest.zip"

unzip -q -o "${REPO_ROOT}/gtFine_trainvaltest.zip" "gtFine/val/*" -d "${CITYSCAPES_DIR}"
rm -f "${REPO_ROOT}/gtFine_trainvaltest.zip"
```

After extraction, calibration images should exist under:

- `${CITYSCAPES_DIR}/leftImg8bit/train`

## 6) Create output directories

```bash
mkdir -p "${QUANT_OUT_DIR}" "${CKPT_DIR}"
```

## 7) Run AIMET quantization (equivalent of your Kaggle command, converted to local paths)

```bash
cd "${REPO_ROOT}"

python build_sim_quantized_pdl.py \
  --calib_images "${CITYSCAPES_DIR}/leftImg8bit/train" \
  --num_calib 50 \
  --calib_size 50 \
  --export_path "${QUANT_OUT_DIR}" \
  --device cuda \
  --enable_custom_conv_bn_fold \
  --config_file "${REPO_ROOT}/config/fully_symmetric.json" \
  --seed 16 \
  --enable_bn_fold
```

## 8) Expected outputs

Main artifacts are written to:

- Quantized export: `${QUANT_OUT_DIR}`

Run evaluate by running

```bash
python run_eval.py \
  --cityscapes_root "${CITYSCAPES_DIR}" \
  --fp32_weights "${REPO_ROOT}/weights/model_final_bd324a.pkl" \
  --quant_weights "${QUANT_OUT_DIR}/panoptic_deeplab_int8.onnx" \
  --model_category PANOPTIC_DEEPLAB \
  --split val
```

## 9) Common issues

1. `Bad credentials` from downloader:
   Use valid Cityscapes credentials; rerun the downloader command.

2. No CUDA available:
   Replace `--device cuda` with `--device cpu` (slower).

3. Calibration path errors:
   Verify `${CITYSCAPES_DIR}/leftImg8bit/train` exists and contains images.

4. If you were planning Bias Correction:
   The current `build_sim_quantized_pdl.py` in this workspace has the bias-correction execution path commented out.
