# Generative AI : DDPM, CFG and DDIM
The project aims to explore the field of conditional diffusion models to restore a masked image (inpainting) on MNIST/Fashion-MNIST (with some extended tests on the CelebA dataset).

## Authors
Baptiste ARNOLD \
Kahled MILI \
Aurélien DAUDIN \
Angela SAADE \
Pierre SCHWEITZER \
Maxime RUFF

## Notebooks

To understand what we have done in this work, follow the next notebooks (located in the `src/` directory):
- `stable_diffusion_inpainting.ipynb`: aims to use Stable Diffusion as a baseline model to see how it performs for inpainting tasks
- `diffusion_network.ipynb`: implementation and training of a diffusion model using CFG from scratch over the Fashion MNIST dataset
- `inference.ipynb`: use of the later model to generate images using DDPM and DDIM as well as applying the DPS method to make image inpainting.
- `celeba.ipynb` and `celeba_inf.ipynb`: notebooks that aims to run inpainting over pictures of people using the CelebA dataset

## How to run

Make sure you have download all depedencies inside `requirements.txt` before
tring to execute the notebook. You can also create a virtual python environment
by doing this :
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To open the UI, run:
```bash
source .venv/bin/activate
cd src/
streamlit run ui.py
```
