This repository contains dataset related to the paper: "Similar but Patched Code Considered Harmful -- The Impact of Similar but Patched Code on Recurring Vulnerability Detection and How to Remove Them", which is currently anonymous for double-blind review.

**vulnerability feature DB/**: The vulnerability feature database, which contains the vulnerabilities and patch logs.

**vulnerability feature DB/vulnerability.csv**: The vulnerability dataset, containing fix commits collected from multiple vulnerability sources. The dataset contains the following columns:

* pdetail_id: data ID
* pdetail_level: 'file' or 'function'
* pdetail_name: file name or function name
* pdetail_content: the content of the file or function
* pdetail_type:  'before' or 'after'
* lang: which language the file or function belongs to
* pdetail_file_ext: the file extension
* vuln_id: CVE ID of the vulnerability
* commit_id: commit ID of the fix commit
* repo_name: repository name
* repo_url: repository URL
* commit_date: commit date
* start_line: if pdetail_level is 'function', the start line of the function
* end_line: if pdetail_level is 'function', the end line of the function
* path: the relative path of the vulnerability file
* group_name: the group name of the project

**vulnerability feature DB/vul_fix_log_and_indexs.csv**: The fix log generated from the vulnerability dataset. The dataset contains the following columns:

* CVE ID: CVE ID of the vulnerability
* project: the project name
* commit_id: commit ID of the fix commit
* file_path: the relative path of the vulnerability file
* file_name: the file name
* func_name: the function name
* fix_log: the fix log

**CVE_list.csv**: the filtered CVE lists, which contains fixing commits that contain C/C++ file changes

**SBP Dataset/**: The *Similar-but-Patched Dataset*

**SBP Dataset/sbp_dataset.csv**: The final *SBP dataset*.The dataset contains the following columns:

* id: the ID of the function, in the format `source_commit_ID:before_or_after:file_path:line_number`.
* code: the function code
* code_hash: the MD5 hash of the function code
* label: 1-vulnerable code, 0-SBP code
* project: (only for SBP code) the source project of the function
* branch: (only for SBP code) the source branch of the function
* vuln_code_hash: (only for SBP code) the MD5 hash of the matched vulnerable function

**SBP Dataset/raw/**: The unprocessed raw data.

**DL Model Eval/**: The evaluation code and results of the four DL-based models on SBP dataset.

**FVF Source/**: The source code of the baseline tools and the FVF.

**FVF Source/hash_based/**: The source code of the hash-based baseline detection tool.

* Note: The hash-based tool stores the vulnerability and target function signatures in MySQL database;
* and uses SQL query to search for the matched functions.

**FVF Source/redebug/**: The source code of the redebug baseline detection tool.

* FVF Source/run_redebug.py: The script to run the redebug tool.

**FVF Source/vuddy/**: The source code of the vuddy baseline detection tool.

* FVF Source/run_vuddy.py: The script to run the vuddy tool.
* FVF Source/run_vuddy_util.py: The utility for running the vuddy tool.

**FVF Source/mvp_fast_matcher/**: The implementation of the matching algorithms of mvp tool.

**FVF Source/fvf.py**: The source code of FVF false alarm reduction tool.

* FVF Source/utils/: The utility functions for the FVF tool.
* FVF Source/run_fvf_demo.py: A show case of the FVF tool.
