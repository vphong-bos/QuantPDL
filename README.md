# QuantPDL doc

The repository implements quantization for the PDL model using PyTorch, AIMET and further libraries like ONNX, INC ONNX. Model detail can be found in the paper: https://arxiv.org/pdf/1911.10194

This document explains the three main quantization/export entrypoints in `QuantPDL`:

* **AIMET**: `build_sim_quantized_pdl.py`
* **ONNX Runtime PTQ**: `build_quantized_pdl.py`
* **INC for ONNX**: `build_neural_compressed_pdl.py`

The repo's flow is: start from the FP32 PDL checkpoint, prepare a representative calibration set, and export an INT8-ready model for deployment or further evaluation. In our current setting, AIMET is enough, other quantize library is for experiment only. This doc explains how to run the QuantPDL AIMET flow on a regular GPU server, from environment setup to generating a quantized model.

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
export WORK_DIR="/workspace/quant_pipeline"
export CITYSCAPES_DIR="${WORK_DIR}/cityscapes"
export QUANT_OUT_DIR="${WORK_DIR}/quantized_model"
export CKPT_DIR="${WORK_DIR}/checkpoint"
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
python "${REPO_ROOT}/quantization/downloader.py" -d "${WORK_DIR}" leftImg8bit_trainvaltest.zip
python "${REPO_ROOT}/quantization/downloader.py" -d "${WORK_DIR}" gtFine_trainvaltest.zip
```

## 5) Extract required Cityscapes folders

```bash
mkdir -p "${CITYSCAPES_DIR}"

unzip -q -o "${WORK_DIR}/leftImg8bit_trainvaltest.zip" "leftImg8bit/train/*" -d "${CITYSCAPES_DIR}"
unzip -q -o "${WORK_DIR}/leftImg8bit_trainvaltest.zip" "leftImg8bit/val/*" -d "${CITYSCAPES_DIR}"
rm -f "${WORK_DIR}/leftImg8bit_trainvaltest.zip"

unzip -q -o "${WORK_DIR}/gtFine_trainvaltest.zip" "gtFine/val/*" -d "${CITYSCAPES_DIR}"
rm -f "${WORK_DIR}/gtFine_trainvaltest.zip"
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
  --save_quant_checkpoint "${CKPT_DIR}/panoptic_deeplab_int8_state_dict.pkl" \
  --device cuda \
  --enable_custom_conv_bn_fold \
  --config_file "${REPO_ROOT}/config/fully_symmetric.json" \
  --seed 16 \
  --enable_bn_fold
```

## 8) Expected outputs

Main artifacts are written to:

- Quantized export: `${QUANT_OUT_DIR}`
- Sim export artifacts: `${QUANT_OUT_DIR}/sim_export`
- Serialized AIMET checkpoint: `${CKPT_DIR}/panoptic_deeplab_int8_state_dict.pkl`

## 9) Common issues

1. `Bad credentials` from downloader:
   Use valid Cityscapes credentials; rerun the downloader command.

2. No CUDA available:
   Replace `--device cuda` with `--device cpu` (slower).

3. Calibration path errors:
   Verify `${CITYSCAPES_DIR}/leftImg8bit/train` exists and contains images.

4. If you were planning Bias Correction:
   The current `build_sim_quantized_pdl.py` in this workspace has the bias-correction execution path commented out.
