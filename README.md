# Commits dataset

![GitHub](https://img.shields.io/github/license/saridormi/commits_dataset?style=for-the-badge)

This repository contains code for collecting and processing diffs, messages and metadata from commits from open-source GitHub repositories.

## Table of contents
- [Ready-to-use dataset](#ready-to-use-dataset)
- [Requirements](#data-collection)
- [Data collection](#data-collection)
- [Data processing](#data-processing)
- [Training tokenizer](#training-tokenizer)

## Ready-to-use dataset 

> :star2: work in progress: this section will contain a link to access a multilingual commits dataset

## Requirements

* :snake: Python
* :floppy_disk: Dependencies
  * This project provides dependencies for two Python dependency managers:
    * [Poetry](https://python-poetry.org/): [`poetry.lock`](poetry.lock), [`pyproject.toml`](pyproject.toml)
    * [pip](https://pip.pypa.io/en/stable/): [`requirements.txt`](requirements.txt) (obtained through `poetry export`)

## Data collection

### How to use

Follow these steps:

1. **Provide repos to collect data from**

    We used [GitHub Search](https://arxiv.org/abs/2103.04682) to select repositories that meet several criteria
    and queried GitHub API for additional info like `full_name` property. 

    You can look through [`choosing_repos.ipynb`](notebooks/choosing_repos.ipynb)
    for an overview of the whole process and some statistics on repositories used for our dataset.

    <details>
    <summary>:yellow_heart: click here for more information about expected data format</summary>
   
    > :exclamation: Repositories are pre-split on parts *(in our case, train/val/test)*.
    > 
    > It doesn't matter for collection script, but having part called `train` is **necessary for correct work of processing script**.
   
    The script expects repositories for each part to be stored in separate JSONLines file:

   ```
         ├── ...  # data directory
         │   ├── part_1.jsonl
         │   ├── ...
         │   └── part_k.jsonl
         └── ...
   ```
   
   Each file should have the following keys:
   - `"name"`: repository name
   - `"github_url"`: repository URL

   An example:

   ```
      {
       "name": "saridormi/commits_dataset",
       "github_url": "git://github.com/saridormi/commits_dataset.git",
        ...  # all other keys are not necessary
      }
   ```
   </details>

2. **Define configuration**

      Configuration is defined at [`configs/collect_data.yaml`](configs/collect_data.yaml).

      <details>
      <summary>:yellow_heart: click here for more information about possible options</summary>
   
      Basically, config looks like that:

      ```
      data_format: ...
      n_workers: ...   
      
      parts: ...
   
      repo_processor:
         chunksize: ...
   
      pydriller_kwargs:
        ...
   
      paths: ...
          temp_clone_dir: ...
          input_dir: ...
          output_dir: ...
      ```

      * `data_format`: String, format to use for reading & writing data; currently, only `jsonl` is supported.
      * `n_workers`: Number of workers for data processing (optional, default value is 1 => sequential).
      * `parts`: List of strings, dataset parts.
      * `repo_processor`
        * `chunksize`: Number of examples in single data chunk (large files are processed in chunks) (optional, default value is 1000).

      * `pydriller_kwargs`:
      
        All keyword arguments under this key are passed to PyDriller's `RepositoryMining`. 
      See [PyDriller documentation](https://pydriller.readthedocs.io/en/1.15/reference.html#pydriller.repository_mining.RepositoryMining) for more information.
      
        If you want to provide date-related arguments (`since`, `to`), write them in `%d-%m-%Y` format. 
      
      * `paths`:
      
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `temp_clone_dir`: directory remote repos will be cloned to
        * `input_dir`: directory to read data about repos from
        * `output_dir`: directory to save gathered data to
        </details>

3. **Collect data**

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

   At the end commits from each part are united to single file, so folder structure looks like this:
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

### Stages

> :star2: work in progress: this section will contain a more detailed description of processing stages

### How to use

> :star2: Start from step 2 if you've used the script for data collection.

Follow these steps:

1. **Provide data**

    > :exclamation: Several processing stages treat `train` part different from others,
    so having part called `train` is necessary for correct work of processing script.
   
    Processing script expects input data to be stored in the same format collection script saves it. See all the details above,
    at [data format](#data-format) section.

2. **Define configuration**

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
         stats_percentile_dir: ...
         deduplication_dir: ...
         metadata_dir: ...
      ```
   
      * `data_format`: String, format to use for reading & writing data; currently, only `jsonl` is supported.
      * `line_sep`: String, will be used as line separator.
      * `parts`: List of strings, dataset parts.
      * `paths`:
      
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `input_dir`: Directory to read data from.
        * `stats_percentile_dir`: Directory to save percentiles for # tokens, # characters, # modified files (outliers processing).
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
      * `post_deduplication_processor`:
        * `only_full_inner_clones`: True to drop clones both in terms of diffs and in terms of messages, False to drop clones either in terms of diffs or in terms of messages.
        * `only_train_inner_clones`: True to drop inner clones (clones within the same dataset part) only for train, False to do it for all dataset parts.
        * `only_train_outer_clones`: True to drop outer clones (clones between different dataset parts) only for train, False to do it for all dataset parts.
        * `identical_clones`: True to use logic for 100% clones, False to use logic for similar clones.
   </details>
    
3. **Process data**

    To start processing data, run the following command:

    ```
    python -m src.process_data
    ```

    > :star2: Note that you can skip any processing stage by setting corresponding config key to `False`. 
    For example, here is how you can skip deduplication stage with [hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/):
    > ```
    >  python -m src.process_data post_deduplication_processor=False
    >  ```
   
## Training tokenizer

This repo also contains code for training tokenizer on diffs from collected data
via [🤗Tokenizers](https://huggingface.co/tokenizers/) library. 

Currently, you can either train byte-level BPE tokenizer
or define all components from [🤗Tokenizers](https://huggingface.co/tokenizers/) manually.

### How to use

> :star2: Start from step 2 if you've used the script for data collection and/or processing.

Follow these steps:

1. **Provide data**

    > :exclamation: Having part called `train` is necessary for correct work of tokenizer training script.

    Tokenizer training script expects input data to be stored in the same format collection script saves it. See all the details above,
    at [data format](#data-format) section.

2. **Define configuration**

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

3. **Train tokenizer**

    To start training tokenizer, run the following command:

    ```
    python -m src.train_tokenizer
    ```
