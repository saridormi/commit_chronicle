# Commits dataset

![GitHub](https://img.shields.io/github/license/saridormi/commits_dataset?style=for-the-badge)

This repository contains code for collecting and filtering data about commits from open source GitHub repos.

## Table of contents
- [Ready-to-use dataset](#ready-to-use-dataset)
- [How to use data collection code](#how-to-use-data-collection-code)
- [How to use data processing code](#how-to-use-data-processing-code)

## Ready-to-use dataset 

> :exclamation: This section is about **old** Java dataset, current data format is different.

Dataset is currently available **only to JetBrains employees** at [Google Drive](https://drive.google.com/drive/folders/1Z3LgzG23KcZGln53ta4WVKBPp0S8_XhZ?usp=sharing).

There are two options:
* data right after collection with minimum processing is stored in **raw_data** folder
<details>
<summary>:yellow_heart: click here for more information on data format </summary>

> :floppy_disk: At this point dataset takes around 16GB of disk space.

Data from each repo is saved as separate `.gz` archive. Inside it is a `.jsonl` file where each line has keys `author`, `date`, `message` and `diff`. 
* `author` is a list with information (name/nickname and email) about person who made commit
* `date` is a date (with time) when commit was made
* `message` is a commit message
* `diff` is a list of diffs for each file modified in commit

Example:
| author | date | message | diff |
|:-:|:-:|:-:|:-:|
|[name, email] | 2021-01-01 00:00:00 | cool commit message | [changes in file1, changes in file2, ...]|	

Diff for each file is basically `git diff` output string but special git heading like `@@ -6,22 +6,24 @@` is omitted and it additionally contains special token `<FILE>` in line with filenames. 

Preprocessing at this point includes separating input lines with `<nl>` token and adding whitespaces around punctuation marks (both in messages and in diffs).

</details>

* data after all filtering is stored as **filtered_and_deduplicated_df.csv** file
<details>
<summary>:yellow_heart: click here for more information on data format</summary>

> :floppy_disk: At this point dataset takes around 4GB of disk space.

Data from all repos is stored in one file. It has the following columns:
* `author`: unique integer for each (name, email) pair in dataset
* `date`: date (with time) when commit was made
* `message`: commit message
* `diff`: single diff for all modified files
* `num_mods`: number of modified files
* `repo`: GitHub repository name
* `sample_id`: service column for deduplication *(maybe I should drop it)*
* `project_id`: 1 if commit is in train part of dataset, 2 - if in validation, 3 - if in test 

Diff is basically `git diff` output string but some special info like `index e345a66..f841d45` or `@@ -6,22 +6,24 @@` is omitted and it additionally contains special token `<FILE>` in lines with filenames. 

Message is, well, commit message. 

Note that in both cases input lines are separated with `<nl>` token and punctuation marks are additionally separated by whitespaces.

Super simple examples of data format in cases of modifying, adding, deleting or renaming file:
|author|date|diff|message|num_mods|repo|sample_id|project_id|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1|2021-01-01 00:00:00| <FILE> conf / config . yaml \<nl\> - batch_size : 4 \<nl\> + batch_size : 8|Modify config|1|organization/repo|1|1|
|2|2021-01-01 00:00:00| new file \<nl\> <FILE> conf / config . yaml \<nl\> + batch_size : 8|Add config|1|organization/repo|2|1|
|1|2021-01-01 00:00:00| deleted file \<nl\> <FILE> conf / config . yaml \<nl\> - batch_size : 4|Delete config|1|organization/repo|3|1|
|2|2021-01-01 00:00:00| rename from conf / config . yaml \<nl\> rename to conf / conf . yaml|Rename config|1|organization/repo|4|1|
</details>

## How to use data collection code

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
   Repositories are pre-split on parts *(in our case, train/val/test)*.

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

4. **Define configuration**

      Configuration is defined at [`src/collection/config.yaml`](src/collection/config.yaml).

      <details>
      <summary>:yellow_heart: click here for more information about possible options</summary>
   
      Basically, config looks like that:

      ```
      repo_processor:
         chunksize: ...
   
      pydriller_kwargs:
        ...
   
      n_workers: ...
      org_repo_sep: ...
   
      paths: ...
          temp_clone_dir: ...
          input_dir: ...
          output_dir: ...
      ```
   
      * `repo_processor`
        * `chunksize`: # of examples in single chunk

      * `pydriller_kwargs`
      
        All options from here are passed to PyDriller's `RepositoryMining` as kwargs. See [PyDriller docs](https://pydriller.readthedocs.io/en/1.15/reference.html#pydriller.repository_mining.RepositoryMining) for more information.
      
      * `n_workers`: # of threads for parallel data gathering
      * `org_repo_sep`: symbol to replace `/` in `"org/repo"`
      * `paths`:
      
        Paths are moved to separate key to convert them all to absolute paths via hydra.
        * `temp_clone_dir`: directory remote repos will be cloned to
        * `input_dir`: directory to read data about repos from
        * `output_dir`: directory to save gathered data to
        </details>

5. **Collect data**

    To start collecting data, run the following command:
    ```
    python -m src.collection.collect_data
    ```
   
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

## How to use data processing code

> :star2: work in progress
