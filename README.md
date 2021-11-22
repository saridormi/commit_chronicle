# Commits dataset

![GitHub](https://img.shields.io/github/license/saridormi/commits_dataset?style=for-the-badge)

> :exclamation: WARNING: I've changed a lot recently and readme is not really up to date yet

This repository contains code for collecting and filtering data about commits from open source GitHub repos.

## Table of contents
- [Ready-to-use dataset](#ready-to-use-dataset)
- [How to use data collection code](#how-to-use-data-collection-code)

## Ready-to-use dataset 

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

    For data collection you need Python 3.8 and [PyDriller](https://github.com/ishepard/pydriller)

      *(Note: latest version seems to differ a lot from the one I used, so make sure to install `1.15.5`)*

3. **Choose repos to collect data from**

    We used [GitHub Search](https://arxiv.org/abs/2103.04682) to select repositories that meet several criteria *(you can look through `choosing_repos.ipynb` for any language in `notebooks` folder for more information on our specific criteria)*.

    You can choose repositories however you like, just provide two files: with repos URLs *(e.g. `https://github.com/saridormi/commits_dataset.git`)* and with repos names *(or something that you want to be used to name directories)*.
    
4. **Collect data**

    To collect data, run the following command:
    ```
    python collect_data.py
    ```
    You can (and most likely should) use several command line arguments, add `-h` flag to command above to see more information.
