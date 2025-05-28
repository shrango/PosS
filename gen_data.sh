pip install --upgrade pip
pip install setuptools==69.5.1
pip install -r requirements.txt --use-pep517

python -m ge_data.allocation \
--outdir \
--data_path \
--model_path
