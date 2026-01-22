# Generative AI : DDPM, CFG and DDIM
Conditional Diffusion to restore a masked image (inpainting) on MNIST/Fashion-MNIST

## Authors
Baptiste ARNOLD \
Kahled MILI \
Aurélien DAUDIN \
Angela SAADE \
Pierre SCHWEITZER \
Maxime RUFF

## Usage

Make sure you have download all depedencies inside `requirements.txt` before
tring to execute the notebook. You can also create a virtual python environment
by doing this :
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Run le UI

```bash
source .venv/bin/activate
cd src/
streamlit run ui.py
```
