<div align="center">

# DAM4SAM_opt
</div>
# grounded_sam2_opt
optimize grounded_sam2 with tensorrt


## Download models

```bash
cd checkpoints
bash download_ckpts.sh 
```

## Download onnx models

```bash
cd sam2_opt/sam2/checkpoints
bash download_opt.sh
```
## demo: run_auto.py

### how to run and speedup
how to run
```bash
SAM2_VERSION_TRACK=dam4sam python run_auto.py --dir frames-dir --ext jpg --output_dir output-dir --model_path sam2_opt/sam2/checkpoints/opts
```
note: if add --model_path, speed up with tensorrt, else use pytorch raw version, the logic is below.
```python
    def run_sequence(dir_path, file_extension, output_dir, model_path=None):
        ...
        if not model_path:
        # torch
        tracker = DAM4SAMTracker(tracker_name="sam21pp-L")
    else:
        # speedup with tensorrt
        tracker = DAM4SAMTracker(tracker_name="sam21pp-L", backend="tensorrt", model_root_path=model_path)
```