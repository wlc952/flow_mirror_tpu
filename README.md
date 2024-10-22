# flow_mirror_tpu

Flow_mirror on Sophgo SG2300x

## Prepare environment and  Download models

```bash
python3 -m venv fm_venv
source fm_venv/bin/activate
sudo chmod +x prepare.sh
sudo chmod +x download.sh
bash prepare.sh
bash download.sh
```

## Inference on Sophgo SG2300x

```python
## webui version
python src_sail/app.py
```
