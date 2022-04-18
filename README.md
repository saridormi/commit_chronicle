# Commits dataset

![GitHub](https://img.shields.io/github/license/saridormi/commits_dataset?style=for-the-badge)

This repository contains code for collecting and processing diffs, messages and metadata of commits from open source GitHub repositories.

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
   
      outliers_processor:
         ...
   
      message_processor:
         ...
   
      diff_processor:
         ...
   
      lexer:
         ...
      
      pre_deduplication_processor:
         ...
   
      post_deduplication_processor:
         ...
   
      final_processor:
         ...
   
      paths:
         input_dir: ...
         licenses_dir: ...
         tokens_percentile_dir: ...
         literals_percentile_dir: ...
         deduplication_dir: ...
      ```
   
      * `data_format`: format to use for reading & writing data; currently, only `jsonl` is supported
      * `clones_ready`: boolean, stops after `pre_deduplication_processor` stage if set to `False`
      * `paths`:
      
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `input_dir`: directory to read data from
        * `licenses_dir`: directory to read repository-license mapping from
        * `tokens_percentile_dir`: directory to save percentiles for # tokens
        * `literals_percentile_dir`: directory to save percentiles for literals lengths
        * `deduplication_dir`: directory to save clone search results

      Each processor accepts two keyword arguments:
      * `chunksize`: # of examples in single data chunk (large files are processed in chunks) (optional, default value is 1000)
      * `n_workers`: # of threads for data processing (optional, default value is 1)
   
      Some processors also accept other keywords arguments:
      * `outliers_processor`:
        * `lower_percentile`: # tokens percentile to use as lower bound (should be in (0, 1) range)
        * `upper_percentile`: # tokens percentile to use as upper bound (should be in (0, 1) range)
        * `diff_upper_bound`: constant upper bound for # tokens in diffs (optional)
      * `lexer`:
        * `upper_percentile`: literals' lengths percentile to use as upper bound (should be in (0, 1) range)
        * `sep_char`: character to use as a delimiter between lexemes (important: should be a single character)
   </details>
    
5. **Process data**

    To start processing data, run the following command:

    ```
    python -m src.process_data
    ```
   
    Note that the script above doesn't include deduplication. When you have files with ids of duplicated entries ready, 
    run the following command:
    
    ```
    python -m src.drop_clones
    ```

### Stages

> :star2: work in progress: this section will contain a more detailed description of processing stages

## Training tokenizer

This repo also contains code for training custom tokenizer on diffs from collected data
via [🤗 Tokenizers](https://huggingface.co/tokenizers/) library.

> :exclamation: Custom lexer class is used for pre-tokenization. Currently, instead of actually splitting input 
> on tokens, it inserts special delimiter.
> Tokenizer is then trained (and saved) with 
> [CharDelimiterSplit](https://huggingface.co/docs/tokenizers/python/v0.9.4/components.html#pre-tokenizers) pre-tokenizer
> on this delimiter.

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
      n_train_examples: ...
   
      diff_extractor:
        chunksize: ...
        n_workers: ...
   
      tokenizer:
         _target_: tokenizers.models.BPE
         ...
   
      pre_tokenizer:
          _target_: tokenizers.pre_tokenizers.CharDelimiterSplit
          sep_token: ...
   
      trainer:
         _target_: tokenizers.trainers.BpeTrainer
         ...
   
      paths:
        input_dir: ...
        tokenizer_dir: ...
        percentile_dir: ...
      ```
   
      * `data_format`: format to use for reading & writing data; currently, only `jsonl` is supported
      * `n_train_examples`: how many diffs from train will be used for tokenizer training     
      * `diff_extractor`
   
        This class is used to extract given number of diffs from train part of dataset. It accepts the following arguments:
        * `chunksize`: # of examples in single data chunk (large files are processed in chunks) (optional, default value is 1000)
        * `n_workers`: # of threads for data processing (optional, default value is 1)
   
      * `tokenizer`/`pre_tokenizer`/`trainer`
        > :exclamation: Make sure that `sep_token` for `pre_tokenizer` is the same as `sep_char` used for `lexer` at [`configs/process_data.yaml`](configs/process_data.yaml).
      
        All arguments except `_target_` are passed to corresponding target class. 
      See [🤗 Tokenizers documentation](https://huggingface.co/docs/tokenizers/python/v0.9.4/) for more information.
      
      * `paths`:
      
          Paths are moved to separate key to convert them to absolute paths via hydra.
          * `input_dir`: directory to read data from
          * `tokenizer_dir`: directory to save tokenizer to
          * `percentile_dir`: directory to save percentiles for literals` lengths
   </details>

6. **Train tokenizer**

    To start training tokenizer, run the following command:

    ```
    python -m src.train_tokenizer
    ```

## Data tokenization

This repository also contains code for tokenization of collected data.

> :exclamation: Tokenized data is saved to specific format 
required by [our pipeline for training & evaluation of Transformer models 
> for commit message completion task](https://github.com/JetBrains-Research/commit_message_generation).
   
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

    Tokenizer training script expects input data to be stored in the same format collection script saves it. See all the details above,
    at [data format](#data-format) section.

4. **Define configuration**

   Configuration is defined at [`configs/tokenize_data.yaml`](configs/tokenize_data.yaml).   

    <details>
      <summary>:yellow_heart: click here for more information about possible options</summary>
      
      Basically, config looks like that:

      ```
      data_format: ...

      training_processor:
         blocksize: ...
         chunksize: ...
         clean_temp_files: ...
         msg_tokenizer_name: ...
         diff_kwargs:
           ...
         msg_kwargs:
           ...
    
      paths:
         diff_tokenizer_path: ...
         input_dir: ...
         output_dir: ...
      ```
   
      * `data_format`: format to use for reading & writing data; currently, only `jsonl` is supported
   
      * `training_processor`:
        * `blocksize`: # of bytes in single block (in that case, [`dask`](https://docs.dask.org/en/stable/) is used to process whole large file lazily)
        * `chunksize`: # of examples in single data chunk (large files are processed in chunks)
        * `clean_temp_files`: True to remove temporary files, False to keep (optional, default value is True)
        * `msg_tokenizer_name`: pretrained name for message tokenizer
        (see [🤗 Transformers documentation](https://huggingface.co/transformers/v4.2.2/model_doc/auto.html#transformers.AutoTokenizer.from_pretrained) for more information)
        * `diff_kwargs`: 
        
          All keyword arguments under this key are passed to diff tokenizer. 
           See [🤗 Transformers documentation](https://huggingface.co/transformers/v4.2.2/main_classes/tokenizer.html#transformers.PreTrainedTokenizerFast.__call__) for more information.
      
        * `msg_kwargs`:
        
          All keyword arguments under this key are passed to message tokenizer. 
           See [🤗 Transformers documentation](https://huggingface.co/transformers/v4.2.2/main_classes/tokenizer.html#transformers.PreTrainedTokenizerFast.__call__) for more information.
      
      * `paths`:
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `diff_tokenizer_path`: path to load diff tokenizer from
        * `input_dir`: directory to read data from
        * `output_dir`: directory to save tokenized data to
   </details>
    
5. **Train tokenizer**

   To start tokenizing data, run the following command:

    ```
    python -m src.tokenize_data
    ```
