# Data Availability for DiEmph

## Prerequisites

Please use the following command to install `python3` dependencies.
```
pip install -r requirements.txt
```

We use public available datasets to train our models.
Thus they are not included in this repository.
Please refer to our paper for details.

We upload at Google Drive the models we trained, the models improved by DiEmph, and the testing datasets (both the in-distribution one and the out-of-distribution one).

Please use the following command to download the data.
```
python3 download_data.py
```
The script is expected to download two files, i.e., `dataset.zip` and `models.zip`.
Please unzip them to the root directory of this repository.
```shell
unzip dataset.zip
unzip models.zip
```

## File Structure

After downloading the aforementioned data, the file structure of this repository is as follows:
```
.
├── README.md
├── dataset
├── dataset.zip
├── download_data.py
├── jtrans
├── models
├── models.zip
├── requirements.txt
└── trex
```

`README.md` is this file.

`jtrans/` contains the code for JTrans. A copy of DiEmph's implementation (adapted to JTrans) is also in this directory. (See the following section for details.)

`trex/` contains the code for Trex. A copy of DiEmph's implementation (adapted to Trex) is also in this directory. (See the following section for details.)

`dataset/` contains the testing datasets.

`models/` contains the original models and models improved by DiEmph.

`download_data.py` is the script to download the data.

`requirements.txt` contains the dependencies.

The following part describes (1) how to run DiEmph on the models to find problematic instructions, and (2) how to evaluate the models improved by DiEmph.


## Run DiEmph on Models

In this step, we use DiEmph to analyze finetuned models and the related training data.
Specifically, DiEmph takes as input a model and its training data,
and outputs a list of instructions that may lead to the model's suboptimal performance
on the out-of-distribution data.

There are two copies of DiEmph's code in this repository.
They are almost the same. The only difference is how they invoke the model.
Recall that DiEmph needs to invoke the model to compute classifiction importance.

Since the logic of these copies are the same, 
here we only describe the copy for the Trex model.

The main logic of DiEmph is in `trex/diemph_analyze_training_data.py`.
Function `analyze_one_cfg(model, current_func)` computes the classification importance
and semantics importance for all instructions in a given function.

Then the following snippet of code (at around line 100 in `trex/diemph_analyze_training_data.py`)
uses KDE to detect instructions with exceptionally high classification importance.
```python
  # kde
  sensitive_instrs = []
  non_sensitive_instrs = []
  from scipy import stats
  from scipy import integrate
  for variance in tqdm(results_all):
    all_variances = [i[4] for i in variance]
    kde = stats.gaussian_kde(all_variances)
    for entry in variance:
      if integrate.quad(kde, a=-np.inf, b=entry[4])[0] < 0.1:
        sensitive_instrs.append(entry)
      else:
        non_sensitive_instrs.append(entry)
```

After that, DiEmph computes the instructions with exceptionally high classification importance
but low semantics importance, and prints the top-10 frequent instructions.

Following sections describe how to run DiEmph on the models in this repository.


### Trex

This section contains commands to run DiEmph on the Trex models.

First, please `cd` to the `trex/` directory.

The following command runs DiEmph on the Trex model trained on the BinaryCorp dataset.

**Note that the `<absolute-path-to-data-availability-packb>` in the command should 
be replaced with the absolute path to the data availability pack.**

```shell
python3 diemph_analyze_training_data.py --model ../models/ori/trex-bc/checkpoint.pt --sample-in ../dataset/data-static-analysis/bincorp-sample-200.pkl --fout trex-bc-static --post True --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin
```

The following command runs DiEmph on the Trex model trained on the Binkit dataset.
```shell
python3 diemph_analyze_training_data.py --model ../models/ori/trex-bk/checkpoint.pt --sample-in ../dataset/data-static-analysis/binkit-sample-30.pkl --fout trex-bk-static --post True --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin
```

The following command runs DiEmph on the Trex model trained on the Hows dataset.
```
python3 diemph_analyze_training_data.py --model ../models/ori/trex-hs/checkpoint.pt --sample-in ../dataset/data-static-analysis/hows-sample-all.pkl --fout trex-hs-static --post True --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin
```


### JTrans

This section contains commands to run DiEmph on the JTrans models.

First, please `cd` to the `jtrans/` directory.

The following command runs DiEmph on the JTrans model trained on the BinaryCorp dataset.
```
python3 diemph_analyze_training_data.py --sample-in ../dataset/data-static-analysis/bincorp-sample-200.pkl --model ../models/ori/jtrans-bc --fout static-jtrans-bc --post
```

The following command runs DiEmph on the JTrans model trained on the Binkit dataset.
```
python3 diemph_analyze_training_data.py --sample-in ../dataset/data-static-analysis/binkit-sample-30.pkl --model ../models/ori/jtrans-bk --fout static-jtrans-bk --post
```

The following command runs DiEmph on the JTrans model trained on the Hows dataset.
```
python3 diemph_analyze_training_data.py --sample-in ../dataset/data-static-analysis/hows-sample-all.pkl --model ../models/ori/jtrans-hs --fout static-jtrans-hs --post
```

## Evaluation

After getting the instructions listed by DiEmph, 
we deemphsize the top-5 instructions in the list 
by removing them from the training data and retrain the model.
The retrained models are stored at `../models/diemph/`.
This section contains commands to evaluate the retrained models.

### Trex

The evaluation script for Trex is `trex/diemph_eval.py`.
Its usage is as follows:
```
python3 diemph_eval.py --model <path-to-model> --data <path-to-data-bin> --sample <path-to-sample> [--bf-removal] --rewrite-strategy <strategy>
```
where `<strategy>` can be `binarycorp`, `binkit`, or `hows`, denoting three datasets used to train the models.

`--bf-removal` is a flag used to indicate whether the script should deemphsize problematic instructions in the testing inputs. 
When evaluate models improved by DiEmph, this flag should be set.
When evaluate original models, this flag should not be set.

For example, the following command evaluates the `Trex models improved by DiEmph` on out-of-distribution testing samples from Coreutils.
```
# trained on BinaryCorp
python3 diemph_eval.py --model ../models/diemph/trex-bc/checkpoint.pt --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin --sample ../dataset/out-dist/coreutils-500-500.instance.pkl --bf-removal --rewrite-strategy binarycorp

# trained on Binkit
python3 diemph_eval.py --model ../models/diemph/trex-bk/checkpoint.pt --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin --sample ../dataset/out-dist/coreutils-500-500.instance.pkl --bf-removal --rewrite-strategy binkit

# trained on How-Solve
python3 diemph_eval.py --model ../models/diemph/trex-hs/checkpoint.pt --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin --sample ../dataset/out-dist/coreutils-500-500.instance.pkl --bf-removal --rewrite-strategy how
```

Similarly, the following command evaluates the original Trex models on out-of-distribution testing samples from Coreutils.
```
# trained on BinaryCorp
python3 diemph_eval.py --model ../models/ori/trex-bc/checkpoint.pt --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin --sample ../dataset/out-dist/coreutils-500-500.instance.pkl

# trained on Binkit
python3 diemph_eval.py --model ../models/ori/trex-bk/checkpoint.pt --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin --sample ../dataset/out-dist/coreutils-500-500.instance.pkl

# trained on How-Solve
python3 diemph_eval.py --model ../models/ori/trex-hs/checkpoint.pt --data <absolute-path-to-data-availability-pack>/trex/../dataset/trex-data-bin --sample ../dataset/out-dist/coreutils-500-500.instance.pkl
```

To test Trex models on other datasets, please simply change the `--sample` argument to the path of the testing samples.
Specifically, they are located at:
```
# In-distribution testing samples
dataset/in-dist/libsql-500-500.instance.pkl
dataset/in-dist/openssl-500-500.instance.pkl
dataset/in-dist/binutils-500-500.instance.pkl
dataset/in-dist/libcurl-500-500.instance.pkl
dataset/in-dist/libmagick-500-500.instance.pkl
dataset/in-dist/putty-500-500.instance.pkl
dataset/in-dist/coreutils-500-500.instance.pkl

# Out-of-distribution testing samples
dataset/out-dist/libsql-500-500.instance.pkl
dataset/out-dist/openssl-500-500.instance.pkl
dataset/out-dist/binutils-500-500.instance.pkl
dataset/out-dist/libcurl-500-500.instance.pkl
dataset/out-dist/libmagick-500-500.instance.pkl
dataset/out-dist/putty-500-500.instance.pkl
dataset/out-dist/coreutils-500-500.instance.pkl
```

### JTrans

Evaluating JTrans models is a bit more complicated than Trex models.
For each evaluation, there are three steps:
1. Encoding functions in the `O0` binaries.
2. Encoding functions in the `O3` binaries.
3. Evaluating the model on the encoded functions.

This section use Coreutils as an example to illustrate the evaluation process.

The following commands encode functions in the `O0` binaries.

```bash
python3 diemph_encode_binary_batch.py --mix-baseline True --model-path ../models/diemph/jtrans-bc/ --baseline-model ../models/ori/jtrans-bc --rewrite-strategy binarycorp --sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl --binary-entry-path ../dataset/out-dist-jtrans/coreutils-O0.prep.pkl  --fout coreutils.diemph.jtrans-bc.O0.pkl
```
It contains the following arguments:
- `--mix-baseline True`: indicates that the script should encode functions in the `O0` binaries using both the original JTrans model and the improved JTrans model.
It is used to handle the extremely small functions.
The flag should be set when evaluating the improved JTrans model.
- `--model-path ../models/diemph/jtrans-bc/`: the path to the improved JTrans model.
- `--baseline-model ../models/ori/jtrans-bc`: the path to the original JTrans model.
- `--rewrite-strategy binarycorp`: the rewrite strategy used to encode functions.
Similar to Trex, there are three rewrite strategies: `binarycorp`, `binkit`, and `how`.
- `--sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl`: the path to the testing samples.
- `--binary-entry-path ../dataset/out-dist-jtrans/coreutils-O0.prep.pkl`: the path to the raw functions extracted from the `O0` binary.
- `--fout coreutils.diemph.jtrans-bc.O0.pkl`: the path to the output file. It contains the embeddings of functions in the `O0` binary.

The above command stores the encoded functions in `coreutils.diemph.jtrans-bc.O0.pkl`.

Similarly, the following commands encode functions in the `O3` binaries.
```bash
python3 diemph_encode_binary_batch.py --mix-baseline True --model-path ../models/diemph/jtrans-bc/ --baseline-model ../models/ori/jtrans-bc --rewrite-strategy binarycorp --sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl --binary-entry-path ../dataset/out-dist-jtrans/coreutils-O3.prep.pkl  --fout coreutils.diemph.jtrans-bc.O3.pkl
```
It stores the encoded functions in `coreutils.diemph.jtrans-bc.O3.pkl`.

After encoding functions in both `O0` and `O3` binaries, 
we query the encoded functions in `O0` binaries to find the corresponding functions in `O3` binaries. 
The following command conducts the query and calculates the results.
```bash
python3 diemph_eval_batch.py --sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl --O0 coreutils.diemph.jtrans-bc.O0.pkl --O3 coreutils.diemph.jtrans-bc.O3.pkl
```

This command contains the following arguments:
- `--sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl`: the path to the testing samples.
- `--O0 coreutils.diemph.jtrans-bc.O0.pkl`: the path to the encoded functions in `O0` binaries.
- `--O3 coreutils.diemph.jtrans-bc.O3.pkl`: the path to the encoded functions in `O3` binaries.



Similarly, the following commands are used to evaluate the original JTrans model.
Note that to evaluate the original JTrans model,
we do not need to specify the `--mix-baseline` argument and the `--baseline-model` argument.
In addition, we need to specify `--ori True` to indicate that we are evaluating the original JTrans model.
```bash
python3 diemph_encode_binary_batch.py --ori True --model-path ../models/ori/jtrans-bc/ --sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl --binary-entry-path ../dataset/out-dist-jtrans/coreutils-O0.prep.pkl  --fout coreutils.ori.jtrans-bc.O0.pkl

python3 diemph_encode_binary_batch.py --ori True --model-path ../models/ori/jtrans-bc/ --sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl --binary-entry-path ../dataset/out-dist-jtrans/coreutils-O3.prep.pkl  --fout coreutils.ori.jtrans-bc.O3.pkl

python3 diemph_eval_batch.py --sample ../dataset/out-dist-jtrans/coreutils-sample-500-500.pkl --O0 coreutils.ori.jtrans-bc.O0.pkl --O3 coreutils.ori.jtrans-bc.O3.pkl
```

To evaluate on other projects, please simply replace `coreutils` with the name of the project.
The files containing testing samples and raw functions are stored in the `dataset/out-dist-jtrans` and `dataset/in-dist-jtrans` directory.