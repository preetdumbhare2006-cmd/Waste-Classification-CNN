# Waste Classification using CNN

Binary image classification:
- Degradable = paper + cardboard
- Non-Degradable = plastic + glass + metal
- `trash` is excluded because it is a mixed/ambiguous category.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Put these five folders inside `dataset/`:
cardboard, glass, metal, paper, plastic

## Run
python src/dataset_check.py
python src/prepare_dataset.py
python src/train.py
python src/evaluate.py
python src/predict.py path/to/image.jpg
streamlit run app/app.py
