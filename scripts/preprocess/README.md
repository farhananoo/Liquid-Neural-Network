# Preprocess data

Run preprocessing:

`python main.py`

Data should be already loaded into `lungtissues/data` folder. To change the behaviour update `./configs/data/default/load_dir` parameter.

`load_dir` folder should contain `raw/images` folder with images (folders), `raw/biospecimen.cart.*.json` and `raw/metadata.cart.*.json`. Data in the required format can be directly downloaded from `https://portal.gdc.cancer.gov/cart`.

Example:

![Alt text](download_data_example.jpg?raw=true "Example")

# Configuration

TODO: Describe this section