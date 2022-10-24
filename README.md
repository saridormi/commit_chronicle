# Commits dataset

![GitHub](https://img.shields.io/github/license/saridormi/commits_dataset?style=for-the-badge)

This repository contains code for collecting and processing diffs, messages and metadata from commits from open-source GitHub repositories.

## Table of contents
- [Ready-to-use dataset](#ready-to-use-dataset)
- [Data collection](#data-collection)
- [Data processing](#data-processing)
- [Training tokenizer](#training-tokenizer)
- [Data tokenization](#data-tokenization)

## Ready-to-use dataset 

> :star2: work in progress: this section will contain a link to access a multilingual commits dataset

## Data collection

### How to use

Follow these steps:

1. **Clone repo**
    ```
    git clone https://github.com/saridormi/commits_dataset.git
    ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ``` 

3. **Provide repos to collect data from**

    We used [GitHub Search](https://arxiv.org/abs/2103.04682) to select repositories that meet several criteria *(you can look through [`choosing_repos.ipynb`](notebooks/choosing_repos.ipynb) for more information on our specific criteria)*.

    <details>
    <summary>:yellow_heart: click here for more information about expected data format</summary>
   
    > :exclamation: Repositories are pre-split on parts *(in our case, train/val/test)*.
    > 
    > It doesn't matter for collection script, but having part called `train` is **necessary for correct work of processing script**.
   
    The script expects data to be stored in the following way:

   ```
         ├── ...  # data directory
         │   ├── part_1
         │   │    ├── repo_1.json
         │   │    ├── ...
         │   │    └── repo_n.json
         │   ├── ...
         │   └── part_k
         └── ...
   ```

   Information about each repo is stored in its own json file and should include the following keys:
   - `"repo"`: full repository name
   - `"url"`: repository URL
   - `"hashes"`: hashes of specific commits; only these commits are collected
   
   An example:

   ```
      {
       'repo': 'saridormi/commits_dataset',
       'url': 'https://github.com/saridormi/commits_dataset.git',
       'hashes': ['a7fb3b64184f0af5b08285cce14b9139baa94049']
      }
   ```
   </details>

5. **Define configuration**

      Configuration is defined at [`configs/collect_data.yaml`](configs/collect_data.yaml).

      <details>
      <summary>:yellow_heart: click here for more information about possible options</summary>
   
      Basically, config looks like that:

      ```
      data_format: ...
      n_workers: ...
      org_repo_sep: ...
   
      repo_processor:
         chunksize: ...
   
      pydriller_kwargs:
        ...
   
      paths: ...
          temp_clone_dir: ...
          input_dir: ...
          output_dir: ...
      ```

      * `data_format`: format to use for reading & writing data; currently, only `jsonl` is supported
      * `n_workers`: # of threads for data collection
      * `org_repo_sep`: smth to replace `/` in `"org/repo"`      

      * `repo_processor`
        * `chunksize`: # of examples in single data chunk (large files are processed in chunks)

      * `pydriller_kwargs`:
      
        All keyword arguments under this key are passed to PyDriller's `RepositoryMining`. 
      See [PyDriller documentation](https://pydriller.readthedocs.io/en/1.15/reference.html#pydriller.repository_mining.RepositoryMining) for more information.
      
      * `paths`:
      
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `temp_clone_dir`: directory remote repos will be cloned to
        * `input_dir`: directory to read data about repos from
        * `output_dir`: directory to save gathered data to
        </details>

6. **Collect data**

    To start collecting data, run the following command:
    ```
    python -m src.collect_data
    ```

### Data format

<details>
   <summary>:yellow_heart: click here for more information about collected data format</summary>
     
   Currently, data is saved in JSON Lines format. Information about each commit includes the following keys:

   - `"author"`: commit author (name, email)
   - `"date"`: commit timestamp (in format `"%d.%m.%Y %H:%M:%S"`)
   - `"hash"`: commit hash
   - `"message"`: commit message
   - `"mods"`: list of files modifications in commit
     - Each modification is a dictionary itself and includes the following keys:
       - `"change_type"`: one of `"ADD"`, `"COPY"`, `"RENAME"`, `"DELETE"`, `"MODIFY"` or `"UNKNOWN"`
       - `"old_path"`: old path to file
       - `"new_path"`: new path to file
       - `"diff"`: file diff
   - `"repo"`: full repository name
   
   [An example:](https://github.com/saridormi/commits_dataset/commit/a7fb3b64184f0af5b08285cce14b9139baa94049)

   ```
   {
     'author': ['Aleksandra Eliseeva', 'xxx@email.com'],
     'date': '05.07.2021 15:10:07',
     'hash': 'a7fb3b64184f0af5b08285cce14b9139baa94049',
     'message': 'Add license badge to readme',
     'mods': [{'change_type': 'MODIFY',
               'diff': '@@ -1,6 +1,6 @@\n'
                       ' # Commits dataset\n'
                       ' \n'
                       '-> :heavy_exclamation_mark: **TODO:** license\n'
                       '+![GitHub](https://img.shields.io/github/license/saridormi/commits_dataset?style=for-the-badge)\n'
               'new_path': 'README.md',
               'old_path': 'README.md'}],
     'repo': 'saridormi/commits_dataset'
   }
   ```

   First, commits from each repo are saved to its own file and zipped, so folder structure looks like this:
 
   ```
      ├── ...  # output folder
      │   ├── part_1
      │   │    ├── repo_1.jsonl.gz
      │   │    ├── ...
      │   │    └── repo_n.jsonl.gz
      │   ├── ...
      │   └── part_k
      └── ...
   ```

   At the end commits from each part are united to single files, so folder structure looks like this:
   ```
      ├── ...  # output folder
      │   ├── part_1.jsonl
      │   ├── ...
      │   └── part_k.jsonl
      └── ...
   ```

   Currently, script doesn't remove the former version, you should do it manually if you don't need raw data.
</details>

## Data processing

### How to use

> :star2: Start from step 4 if you've used the script for data collection.

Follow these steps:

1. **Clone repo**
    ```
    git clone https://github.com/saridormi/commits_dataset.git
    ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Provide data**

    > :exclamation: Several processing stages treat `train` part different from others,
    so having part called `train` is necessary for correct work of processing script.
   
    Processing script expects input data to be stored in the same format collection script saves it. See all the details above,
    at [data format](#data-format) section.

4. **Define configuration**

    Configuration is defined at [`configs/process_data.yaml`](configs/process_data.yaml).

    <details>
      <summary>:yellow_heart: click here for more information about possible options</summary>
      
      Basically, config looks like that:

      ```
      data_format: ...
      line_sep: ...
   
      parts:
         ...
   
      outliers_processor:
         args: ...
         ...
   
      message_processor:
         args: ...
         ...
   
      diff_processor:
         args: ...
         ...
   
      lexer:
         args: ...
         ...
   
      pre_deduplication_processor:
         args: ...
         ...
   
      post_deduplication_processor:
         args: ...
         ...
   
      metadata_processor:
         args: ...
         ...
   
      paths:
         input_dir: ...
         tokens_percentile_dir: ...
         literals_percentile_dir: ...
         deduplication_dir: ...
         metadata_dir: ...
      ```
   
      * `data_format`: String, format to use for reading & writing data; currently, only `jsonl` is supported.
      * `line_sep`: String, will be used as line separator.
      * `parts`: List of strings, dataset parts.
      * `paths`:
      
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `input_dir`: Directory to read data from.
        * `tokens_percentile_dir`: Directory to save percentiles for # tokens.
        * `literals_percentile_dir`: Directory to save percentiles for literals lengths.
        * `deduplication_dir`: Directory to save clone search results.
        * `metadata_dir`: Directory to read/save metadata about authors, licenses, etc.

      Every processor has `args` subkey for the same keyword arguments:
      * `chunksize`: Number of examples in single data chunk (large files are processed in chunks) (optional, default value is 1000).
      * `n_workers`: Number of workers for data processing (optional, default value is 1 => sequential).
   
      Some processors also accept specific keywords arguments:
      * `outliers_processor`:
        * `lower_percentile`: Percentile of # tokens to use as lower bound (should be in (0, 1) range).
        * `upper_percentile`: Percentile of # tokens to use as upper bound (should be in (0, 1) range).
        * `diff_upper_bound`: Constant upper bound for # tokens in diffs (optional).
      * `message_processor`:
        * `replace_patterns`: True to replace unwanted patterns in messages with special tokens, False to just delete them. 
      * `lexer`:
        * `upper_percentile`: Percentile of lexemes' lengths to use as upper bound (should be in (0, 1) range).
      * `post_deduplication_processor`:
        * `only_full_inner_clones`: True to drop clones both in terms of diffs and in terms of messages, False to drop clones either in terms of diffs or in terms of messages.
        * `only_train_inner_clones`: True to drop inner clones (clones within the same dataset part) only for train, False to do it for all dataset parts.
        * `only_train_outer_clones`: True to drop outer clones (clones between different dataset parts) only for train, False to do it for all dataset parts.
        * `identical_clones`: True to use logic for 100% clones, False to use logic for similar clones.
   </details>
    
5. **Process data**

    To start processing data, run the following command:

    ```
    python -m src.process_data
    ```

    > :star2: Note that you can skip any processing stage by setting corresponding config key to `False`. 
    For example, here is how you can skip deduplication stage with [hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/):
    > ```
    >  python -m src.process_data post_deduplication_processing=False
    >  ```

### Stages

> :star2: work in progress: this section will contain a more detailed description of processing stages

## Training tokenizer

This repo also contains code for training tokenizer on diffs from collected data
via [🤗Tokenizers](https://huggingface.co/tokenizers/) library. 

Currently, you can either train byte-level BPE tokenizer
or define all components from [🤗Tokenizers](https://huggingface.co/tokenizers/) manually.

### How to use

> :star2: Start from step 4 if you've used the script for data collection and/or processing.

Follow these steps:

1. **Clone repo**
    ```
    git clone https://github.com/saridormi/commits_dataset.git
    ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Provide data**

    > :exclamation: Having part called `train` is necessary for correct work of tokenizer training script.

    Tokenizer training script expects input data to be stored in the same format collection script saves it. See all the details above,
    at [data format](#data-format) section.

4. **Define configuration**

   Configuration is defined at [`configs/train_tokenizer.yaml`](configs/train_tokenizer.yaml).   

    <details>
      <summary>:yellow_heart: click here for more information about possible options</summary>
      
      Basically, config looks like that:

      ```
      data_format: ...
      line_sep: ...
      msg_tokens: ...

      diff_extractor:
        chunksize: ...
        n_workers: ...
        upper_percentile: ...
        n_train_examples: ...
   
      tokenizer:
         configuration: ...
         byte_level: 
            tokenizer: ...
            train: ...
         custom:
           tokenizer: ...
           normalizer: ...
           pre_tokenizer: ...
           decoder: ...
           trainer: ...
   
      paths:
        input_dir: ...
        tokenizer_dir: ...
      ```
   
      * `data_format`: String, format to use for reading & writing data; currently, only `jsonl` is supported.
      * `line_sep`: String, will be used as line separator.
      * `msg_tokens`: True to add special tokens to replace unwanted patterns to tokenizer, False otherwise.
      
      * `diff_extractor`
   
        This class is used to extract given number of diffs from train part of dataset. It accepts the following arguments:
        * `chunksize`: Number of examples in single data chunk (large files are processed in chunks) (optional, default value is 1000).
        * `n_workers`: Number of workers for data processing (optional, default value is 1 => sequential).
        * `upper_percentile`: Percentile of diffs' lengths to use as upper bound (should be in (0, 1) range).
        * `n_train_examples`: A number of examples from train to use for tokenizer training (optional, if this key is empty or not present, all examples will be used).        

      * `tokenizer`:
        * `configuration`: Tokenizer configuration to use. Currently, `byte_level` and `custom` are supported.
        * `byte_level`:

           [ByteLevelBPETokenizer from 🤗Tokenizers](https://github.com/huggingface/tokenizers/blob/v0.12.0/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py) will be used.
           * `tokenizer`: All arguments are passed to [ByteLevelBPETokenizer](https://github.com/huggingface/tokenizers/blob/v0.12.0/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py) class.
           * `train`: All arguments are passed to `train` method of  [ByteLevelBPETokenizer](https://github.com/huggingface/tokenizers/blob/v0.12.0/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py).
         * `custom`: 
           
           Define all tokenizer components from [🤗Tokenizers](https://huggingface.co/docs/tokenizers/components): normalizer, pre_tokenizer, tokenizer, normalizer, decoder.
           Hydra's instantiate semantic is used.
      * `paths`:
      
          Paths are moved to separate key to convert them to absolute paths via hydra.
          * `input_dir`: Directory to read data from.
          * `tokenizer_dir`: Directory to save tokenizer to.
   </details>

6. **Train tokenizer**

    To start training tokenizer, run the following command:

    ```
    python -m src.train_tokenizer
    ```

## Data tokenization

This repository also contains code for processing data to specific format 
required by [our pipeline for training & evaluation of Transformer models for commit message completion task](https://github.com/JetBrains-Research/commit_message_generation).
   
### How to use

> :star2: Start from step 4 if you've used the script for data collection and/or processing.

Follow these steps:

1. **Clone repo**
    ```
    git clone https://github.com/saridormi/commits_dataset.git
    ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Provide data**

    > :exclamation: Having part called `train` is necessary for correct work of tokenization script.

    Data tokenization script expects input data to be stored in the same format collection script saves it. See all the details above,
    at [data format](#data-format) section.

4. **Define configuration**

   Configuration is defined at [`configs/tokenize_data.yaml`](configs/tokenize_data.yaml).   

    <details>
      <summary>:yellow_heart: click here for more information about possible options</summary>
      
      Basically, config looks like that:

      ```
      data_format: ...
      line_sep: ...

      training_processor:
         chunksize: ...
   
         diff_tokenizer_name_or_path: ...
         msg_tokenizer_name_or_path: ...
   
         diff_kwargs:
           ...
         msg_kwargs:
           ...
      
      only_messages: ...
      only_diffs: ...

      preprocess_data: ...
      tokenize_data: ...
      truncate_diffs: ...
      context_len: ...
   
      paths:
         diff_tokenizer_path: ...
         input_dir: ...
         output_dir: ...
      ```
   
      * `data_format`: String, format to use for reading & writing data; currently, only `jsonl` is supported.
      * `line_sep`: String, will be used as line separator.
      
      * `training_processor`:
        * `chunksize`: Number of examples in single data chunk (large files are processed in chunks) (optional, default value is 1000).

        * `diff_tokenizer_name_or_path`: Name on HuggingFace Hub for diff tokenizer, keep empty if you want to use local path.
           (see [🤗 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained) for more information)
        * `msg_tokenizer_name_or_path`: Name on HuggingFace Hub for message tokenizer, keep empty if you want to use local path.
           (see [🤗 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained) for more information)
        * `diff_kwargs`: 
        
          All keyword arguments under this key are passed to diff tokenizer. 
           See [🤗 Transformers documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__) for more information.
      
        * `msg_kwargs`:
        
          All keyword arguments under this key are passed to message tokenizer. 
           See [🤗 Transformers documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__) for more information.
      * There are several keys that allow to tweak the logic for this script.
        * `only_messages`/`only_diffs`: True to process only the corresponding data type with given tokenizer. Might be useful for cases when we want to initialize encoder and decoder with different pretrained models, hence, with different tokenizers.
        * `preprocess_data`: True to run data preprocessing (aggregating commit message history, etc., everything that comes before tokenization), False to search for preprocessed files in `paths/temp_dir`.
        * `tokenize_data`: True to run data tokenization.
        * `truncate_diffs`: True to iterate over tokenized diffs and create additional version, where each example is trimmed to `context_len` tokens.
        * `context_len`: Maximum number of tokens if `truncate_diffs` is set to True.
   
      * `paths`:
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `input_dir`: Directory to read data from.
        * `output_dir`: Directory to save tokenized data to.
        * `temp_dir`: Directory to save preprocessed version of input data (there are several steps like aggregating message history for each author; these temp versions might be reused with different tokenizers to save time).
        * `diff_tokenizer_name_or_path`: Local path to diff tokenizer, keep empty if you want to use pretrained tokenizer from HuggingFace Hub.
        * `msg_tokenizer_name_or_path`: Local path to message tokenizer, keep empty if you want to use pretrained tokenizer from HuggingFace Hub.
   </details>
    
5. **Tokenize data**

   To start tokenizing data, run the following command:

    ```
    python -m src.tokenize_data
    ```
