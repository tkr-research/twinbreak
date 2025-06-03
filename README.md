# TwinBreak

This repository contains the code for the USENIX Security 2025 paper ''TwinBreak: Jailbreaking LLM Security Alignments
based on Twin Prompts''.

## 📝 Cite this Work

To reference this work, the following citations should be used.

### Plain Text

```
Krauß, T., Dashtbani, H., & Dmitrienko, A. (2025). TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts. 34rd USENIX Security Symposium (USENIX Security 25).
```

### BibTeX

```bibtex
@inproceedings {krauss2025,
    author = {Torsten Krau{\ss} and Hamid Dashtbani and Alexandra Dmitrienko},
    title = {TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts},
    booktitle = {34rd USENIX Security Symposium (USENIX Security 25)},
    year = {2025},
    publisher = {USENIX Association}
}
```

### 📁 Project Structure

```text
├── .gitignore
├── README.md                       <-- This readme file 
├── configs                       <-- This folder contains all configurations for one experiment running twinbreak
│   ├── experiments                       <-- This folder contains all experiment configurations, including configuration files, enums, and a Python class as data transfer object.
│   │   ├── ExperimentConfig.py                       <-- This class is the data transfer object for the experiment configurations (which also holds a twinbreak configuration)
│   │   ├── dtype                       <-- This folder holds an enum and a factory for the precision in which the models are loaded
│   │   │   ├── DTypeConfigFactory.py
│   │   │   └── PrecisionConfig.py
│   │   ├── experiment_1.yaml                       <-- The configuration for experiment 1 of the USENIX artifact.
│   │   ├── experiment_2_1.yaml                       <-- The configuration for experiment 2.1 of the USENIX artifact.
│   │   ├── experiment_2_2.yaml                       <-- The configuration for experiment 2.2 of the USENIX artifact.
│   │   ├── experiment_2_3.yaml                       <-- The configuration for experiment 2.3 of the USENIX artifact.
│   │   ├── experiment_3_1.yaml                       <-- The configuration for experiment 3.1 of the USENIX artifact.
│   │   ├── experiment_3_2.yaml                       <-- The configuration for experiment 3.2 of the USENIX artifact.
│   │   └── experiment_default_settings.yaml                       <-- The default configuration of an experiment. This is the same as for experiment 1
│   └── twinbreak                       <-- This folder contains all twinbreak configurations, including configuration files, enums, and a Python class as data transfer object
│       ├── BatchModeConfig.py                       <-- This class provides an enum for a setting of twinbreak
│       ├── TargetInputTokenConfig.py                       <-- This class provides an enum for a setting of twinbreak
│       ├── TargetModuleConfig.py                       <-- This class provides an enum for a setting of twinbreak
│       ├── TokenMeanConfig.py                       <-- This class provides an enum for a setting of twinbreak
│       ├── TwinBreakConfig.py                       <-- This class is the data transfer object for the twinbreak configurations
│       ├── hyperparameter_study                       <-- This folder contains the twinbreak configuration files for the hyperparameter stuy
│       │   ├── H*.yaml 
│       └── twinbreak_default_settings.yaml                       <-- This is the default setting of TwinBreak, which is the same as for experiment 1
├── dataset                       <-- This folder contains all datasets and classes to handle the datasets
│   ├── DatasetBucket.py                       <-- A class that holds our datasets and provides all methods to access the data including tokenizer functionality
│   ├── DatasetBucketFactory.py                       <-- Factory to create a DatasetBucket with the respective dataset
│   ├── DatasetsIdentifier.py                       <-- Enum identifiers for our datasets located under json folder
│   ├── UtilityDatasetIdentifier.py                       <-- This Enum holds the identifiers used by lm_eval for the utility benchmarks
│   └── json                       <-- This folder contains the safety alignment benchmark datasets in the required JSON format
│       ├── ablation_non_similar.json                       <-- This dataset contains harmless prompt pairs and is used to validate if twin prompt pairs perform better
│       ├── advbench.json                       <-- Full advbench dataset
│       ├── harmbench_validation.json                       <-- We used 200 prompts from harmbench. 100 of them are part of TwinPrompt. The other 100 prompts are stored here.
│       ├── jailbreakbench.json                       <-- Full jailbreakbench dataset
│       ├── strongreject.json                       <-- Full strongreject dataset
│       └── twinprompt.json                       <-- The TwinPrompt dataset introduced in this work.
├── environment.yml                       <-- A full configuration of the conda environment used in this experiment
├── experiments 
│   ├── ExperimentExecutor.py                       <-- The class executes an experiment.
│   ├── ExperimentVisualizer.py                       <-- This class visualizes the results in the log file nicely in form of tables.
│   ├── experiment_1.py                       <-- The script executes experiment 1 of the USENIX artifact.
│   ├── experiment_2_1.py                       <-- The script executes experiment 2.1 of the USENIX artifact.
│   ├── experiment_2_2.py                       <-- The script executes experiment 2.2 of the USENIX artifact.
│   ├── experiment_2_3.py                       <-- The script executes experiment 2.3 of the USENIX artifact.
│   ├── experiment_3_1.py                       <-- The script executes experiment 3.1 of the USENIX artifact.
│   ├── experiment_3_2.py                       <-- The script executes experiment 3.2 of the USENIX artifact.
│   ├── experiment_default.py                       <-- The script is the template for an experiment. It is the same as experiment 1 of the USENIX artifact.
│   └── experiment_test.py                       <-- The script executes functionality test of the USENIX artifact.
├── helper                       <-- This folder contains helper classes used during experiment execution
│   ├── FileHandler.py                       <-- This class handles file accesses
│   ├── LlamaGuardHandler.py                       <-- This class evaluates the safety of a llm response using llamaguard
│   ├── LoggingHandler.py                       <-- This class is responsible for logging to a log file
│   ├── RandomHandler.py                       <-- This class can set the randomness
│   └── StrongRejectHandler.py                       <-- This class evaluates the safety of a llm response using strongreject
├── models                       <-- This folder contains an abstract model class, a model factory, and individual model classes for each model used in the project.
│   ├── AbstractModel.py                       <-- The abstract model class that provides all functionality necessary in this project.
│   ├── ModelConfig.py                       <-- The configurations needed to create a model
│   ├── ModelFactory.py                       <-- The factory to create a model
│   ├── ModelIdentifier.py                       <-- The enum to identify models
│   ├── deepseek
│   │   ├── AbstractDeepseekModel.py
│   │   └── Deepseek_7b.py
│   ├── gemma2
│   │   ├── AbstractGemma2Model.py
│   │   ├── Gemma2_27b.py
│   │   ├── Gemma2_2b.py
│   │   └── Gemma2_9b.py
│   ├── gemma3
│   │   ├── AbstractGemma3Model.py
│   │   └── Gemma3_1b.py
│   ├── llama2
│   │   ├── AbstractLlama2Model.py                       <-- Abstract class with functionality for all llama 2 models
│   │   ├── Llama2_13b.py                       <-- One model instance holding the hugging face identifier
│   │   ├── Llama2_70b.py
│   │   └── Llama2_7b.py
│   ├── llama3
│   │   ├── AbstractLlama3Model.py
│   │   ├── Llama31_8b.py
│   │   └── Llama33_70b.py
│   ├── mistral
│   │   ├── AbstractMistralModel.py
│   │   └── Mistral_7b.py
│   └── qwen25
│       ├── AbstractQwen25Model.py
│       ├── Qwen25_14b.py
│       ├── Qwen25_32b.py
│       ├── Qwen25_3b.py
│       ├── Qwen25_72b.py
│       └── Qwen25_7b.py
├── requirements.txt                       <-- A full list of pip packages used in the experiments
├── results                       <-- Folder to log results
│   ├── experiment_1                       <-- Result folder for experiment 1 of the artifact (created during execution of experiment 1)
│   │   ├── config.txt                       <-- The config of experiment 1
│   │   ├── log                       <-- Folder for log files of this experiment
│   │   │   └── log0.txt                       <-- The log file for experiment 1
│   │   └── twinbreak                       <-- This folder contains twinbreak intermediate output of this experiment, like generated responses and results.
├── sample.env                       <-- A sample .env file that is used by the setup.sh file to create a real .env file on the server
├── setup.sh                       <-- Setup script of the USENIX artifact.
└── twinbreak                       <-- Folder containing the twinbreak functionality
    ├── TwinBreak.py                       <-- The core twinbreak functionality
    ├── TwinBreakAndEval.py                       <-- Wrapper around the core twinbreak functionality allowing utility and safety evaluation
    ├── TwinBreakResult.py                       <-- Data transfer object for results produced by twinbreak
    └── TwinBreakResultBucket.py                       <-- Multiple instances of this class are used in TwinBreakResult to store the identified parameters by twinbreak
```

## 🖥️ Hardware Setup

The experiments in this project run reliably on Unix-based servers equipped with a sufficiently powerful NVIDIA GPU with
at least 48 GB of memory.
More powerful GPUs and greater memory capacities will lead to faster runtimes.
In addition, approximately 120 GB of disk space is required for downloading large language models (LLMs) from Hugging
Face.
The codebase and experimental outputs occupy roughly 100 MB.
We also recommend using a Conda environment for managing dependencies, which consumes around 7 GB of disk space.

### Minimum System Requirements

- NVIDIA GPU with at least 48 GB of memory
- 130 GB of available disk space

💡 Hint 1: Reproducing the full experimental suite from the paper, including models with up to 70 billion
parameters, demands significantly more resources, both GPU memory and disk space. For reference, our experimental setup
used an Intel Xeon Gold 6526Y CPU (16 cores, 64 threads), 256 GB of RAM, four NVIDIA L40S GPUs (each with 48 GB of GDDR6
memory), and a 7 TB HDD. Scripts related to these large-scale experiments are included for completeness but are not
required to run the artifact’s core demonstration. Users may safely skip them if constrained by hardware.

💡 Hint 2: The most memory-intensive aspect of the artifact is the generation of LLM responses during
utility and safety evaluations, not the TwinBreak attack itself. By default, a batch size of 20 is used. To reduce
runtime, this can be increased up to 100, provided sufficient GPU memory is available. If the available GPU memory is
less than 48 GB, the batch size may need to be reduced to fit the model, though this will come at the cost of longer
runtimes.

## 🛠️ Software Setup

The experiments are designed to run on a Unix-based operating system, such as Debian, which serves as our reference
platform.
The environment requires Python 3.10 and PyTorch 2.7.0, with CUDA-enabled access to NVIDIA GPUs.
While we recommend using CUDA 12.6, we also provide instructions for alternative CUDA versions.

We suggest using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to
create an isolated Conda environment, which simplifies setup and ensures proper dependency management. All required
packages can be installed via the [pip package manager](https://pip.pypa.io/en/stable/).

### Required Python Packages

- torch 2.7.0
- transformers 4.44.2
- lm\_eval 0.4.7
- [StrongREJECT](https://github.com/dsbowen/strong_reject)
- python-dotenv 0.9.9
- pyyaml 6.0.2

💡 Hint: We provide specific package versions to ensure a smooth and consistent reproduction process. In
particular, newer versions of _transformers_ may lead to issues related to _torch.\_dynamo_, which
attempts to compile Python model code into a single optimized computation graph. If the model's code involves frequent
changes in shape, type, or control flow, such as from dynamic _forward()_ logic or the use of hooks, Dynamo may
recompile the graph repeatedly, eventually reaching the default limit of 8 recompilations. This issue arises due to the
pruning implementation, which relies on forward hooks. While it is possible to circumvent the problem by disabling
TorchDynamo Just-In-Time (JIT) Compilation using the _TORCHDYNAMO\_DISABLE=1_ environment flag, doing so can
significantly increase execution time. Therefore, we recommend using the package versions and settings specified in the
paper.

### Hugging Face Access

To download the required LLMs from [Hugging Face](https://huggingface.co/),
users must have a Hugging Face account that has accepted the respective model license agreements. An access token
associated with this account is necessary and must be used within the project to authenticate and enable model
downloads.

The following models, used in the core experiments of this artifact, require license agreement on the
respective Hugging Face website and subsequent access token authorization:

- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/google/gemma-2-9b-it
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- https://huggingface.co/meta-llama/Llama-Guard-3-8B
- https://huggingface.co/google/gemma-2b

## 📁 Datasets

The artifact contains three dataset categories: one for assessing safety alignment, another for measuring utility, and
the newly introduced TwinPrompt dataset used during the TwinBreak attack.

### Safety Alignment Benchmarks

The artifact uses four datasets comprising harmful prompts to assess LLMs' security and resilience against potential
misuse.
These prompts are designed to mimic malicious interactions.
All datasets are open-source and are already part of the repository formatted as JSON files.
The four datasets and respective links to the original dataset files are:

1) [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv), which contains
   520 harmful prompts.
2) [HarmBench](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv),
   an improved version of AdvBench with 400 harmful prompts. Following related work (as discussed in the paper), we only
   use 200 prompts from HarmBench.
3) [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), which
   contains 100 pairs of harmful and harmless prompts. However, the pairs are not twins as in our dataset.
4) [StrongREJECT](https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv),
   which contains 313 harmful prompts trying to address shortcomings of the previous datasets.

### Utility Benchmarks

We use five dataset to utility analysis. All datasets can be easily downloaded and
are open-source. The five datasets are:

1) OpenBookQA, which tests an LLM's reasoning and knowledge absorption
   capability with a focus on preliminary scientific topics.
2) ARC-Challenge that targets more complex science questions.
3) HellaSwag which asks the LLM to choose the most plausible continuation scenario given a partial sentence or scenario.
4) RTE, which evaluates with whether a hypothesis can be inferred from a premise.
5) WinoGrande evaluates an LLM's
   common sense and contextual understanding.

The datasets are downloaded and used on the fly via the [lm\_eval Python package](https://pypi.org/project/lm-eval/).

### TwinPrompt Dataset

Finally, we publish our new dataset, TwinPrompt, which consists of one hundred
prompt pairs with one harmful and one harmless prompt that yield high structural and content similarity. The dataset is
also part of this repository and in JSON format and can be found under _twinbreak/dataset/json/twinprompt.json_.

## 🚀 Getting Started

It is assumed that the setup begins with a newly created user account on a Unix-based server, such as a Debian system.

To ensure reproducibility and consistent package management, we use _Miniconda_ to manage the Python environment. While
it is possible to install the dependencies globally without Miniconda, we recommend using Miniconda as described below.
If Miniconda is skipped, users must manually install Python 3.10 and ensure all dependencies are correctly resolved.

💡 Hint:  When copying multi-line commands from the README file, we recommend first pasting them into a plain text
editor (e.g., Notepad) to remove any unintended line breaks before executing them in the terminal.

### Hugging Face Token

If you already have access to a Hugging Face access token with read permissions, you can use it directly. Otherwise,
follow the steps below to create your own token:

- Create an [Hugging Face](https://huggingface.co/) account. Such an account can be created for free.
- To generate an access token, visit https://huggingface.co/settings/tokens}{https://huggingface.co/settings/tokens,
  click ''Create new token'', select the ''Read'' token type, and assign a name to the token. Once created, copy the
  token for later use.
- To get access to the models, please follow the license agreement instructions provided on the Hugging Face links
  provided under Section ''Hugging Face Access''. After confirming the agreements, it may take up to an hour for your
  access status to change from “pending” to “accepted.”

### Miniconda Installation

If Miniconda or Anaconda is not already installed on the server, follow the steps below to install Miniconda.

1) Download the Miniconda installer:

  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2) Run the installer:

  ```bash
bash Miniconda3-latest-Linux-x86_64.sh
  ```

During installation, follow these steps:

- Press _ENTER} to view the license terms.
- Press _SPACE} to scroll to the end of the license terms.
- Type _yes} to accept the license agreement.
- Press _ENTER} to confirm the default location for conda environments.
- Type _yes} to enable automatic activation of conda on shell startup.

Afterward, restart your shell, such that conda is activated. The shell should look like this showing that the base conda
environment is activated:

 ```text
(base) user@server:/home/user$
  ```

### Environment Setup

1) Create and activate the Conda environment:

  ```bash
conda create --name twinbreak python=3.10
  ```

During the installation process enter y to install the new packages.

  ```bash
conda activate twinbreak
  ```

Afterward, your shell should look like this, showing that the new twinbreak environment is activated:

```text
(twinbreak) user@server:/home/user$

  ```

2) Identify your CUDA version (if applicable):

  ```bash
nvidia-smi
  ```

The CUDA version appears in the top-right corner of the table output.

3) Install PyTorch corresponding to your CUDA version:
    - __For CUDA 11.8:__
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```
    - __For CUDA 12.6 (default if compatible):__

    ```bash
    pip install torch
    ```

    - __For CUDA 12.8 or later:__
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    ```

If you have a different CUDA version, you should try out the next smaller PyTorch version.

4) Install the remaining dependencies:

  ```bash
pip install transformers==4.44.2
pip install lm_eval==0.4.7
pip install git+https://github.com/dsbowen/strong_reject.git@e286f0da86d92c929a6fda20a9992f28c5969044
pip install dotenv==0.9.9
pip install pyyaml==6.0.2
  ```

### Code Retrieval and Initialization

1) Navigate to the desired directory where the project should be stored and clone the repository:

  ```bash
git clone https://github.com/tkr-research/twinbreak.git
  ```

2) Enter the project root:

  ```bash
cd twinbreak
  ```

3) Run the setup script, providing your Hugging Face access token and the model storage path. Therefore, replace
   _<HUGGING\_FACE\_TOKEN>_ and _<STORE\_MODEL\_DISK\_PATH>_ with the real values, e.g.,
   _hf\_xxxYOURTOKENxxx_ and _\home\user\.cache\huggingface_. Note, that the terminal user needs read and write
   permissions to the selected directory. The two values
   will be saved in a _.env_ file and used to set necessary environment variables. Additionally, the script will
   configure the project root as part of the Python path.

  ```bash
source setup.sh --hf-token <HUGGING_FACE_TOKEN> --store-model-disk-path <STORE_MODEL_DISK_PATH>
  ```

The terminal should output the following:

  ```text
.env file created and values injected:
- HF_TOKEN
- STORE_MODEL_DISK_PATH

  PYTHONPATH set to /home/user/twinbreak
  ```

### Basic Functionality Test

To run a simple functionality test, execute the following commands.

1) Navigate to the experiments directory:

  ```bash
cd experiments
  ```

2) Execute the test script to verify the setup:

  ```bash
python -u experiment_test.py
  ```

3) If successful, the following message will be displayed:

  ```text
The system is set up to run experiments!
  ```

## 📊 Experiments & Results

### Major Claims of this Project

- __C1__: TwinBreak effectively removes safety alignment from open-source large language models (LLMs) with minimal
  impact on their utility, as demonstrated on the LLaMA 2 (7B) model. This is proven by experiment (E1) in this
  artifact, which is also reported in Section 4.2 of the paper, with results reported in the first rows of Tables 2–5
  for safety alignment removal, and Tables 15–19 for utility preservation.

- __C2__: TwinBreak demonstrates effectiveness across diverse model architectures and vendors, as shown on Llama 3.1 (
  8B), Gemma 2 (9B), and Qwen 2.5 (7B). This is proven by experiment (E2) in this artifact, which is also reported in
  Section 4.2 of the paper, with results reported in the remaining rows of Tables 2–5 for safety alignment removal, and
  Tables 15–19 for utility preservation.

- __C3__: TwinBreak also proves effective across varying model sizes, as demonstrated on the larger LLaMA 2 (13B) and
  the smaller Qwen 2.5 (3B). This is proven by experiment (E3) in this artifact, which is also reported in Section 4.6
  of the paper, with results reported in rows two and six of Table 8.

### Experiments

The reported experiment runtimes were obtained using a single NVIDIA L40s GPU with 48 GB of memory.
Actual runtimes may vary depending on the number of GPUs, GPU model, and available memory.
By default, the provided code will automatically utilize all available GPUs on the server.

If you wish to restrict execution to a specific subset of GPUs, you will need to modify the command used to run the
Python scripts.
For example, to use only the GPUs with indices 1 and 2, the command would be:

```bash
 CUDA_VISIBLE_DEVICES=1,2 python -u experiment_test.py
```

Given the potentially long runtime of the experiments, we recommend using the _screen_ tool to prevent interruptions in
case the connection to the server is lost. This helps ensure that experiments continue running even if the terminal
session is disconnected. For usage instructions, please refer to
the [screen manual page](https://wiki.ubuntuusers.de/Screen/). However, this step is optional for executing the project.

- __E1__: _General Functionality_ [1 human-minute + 2 compute-hour + 35GB disk]: In this experiment, we attack the LLaMA
  2 (7B) model with TwinBreak and evaluate the jailbreak success as well as the utility with all benchmarks. The results
  prove the major claim (C1).
    - Preparation: Follow all instructions under the _Getting Started_ section. The terminal should reside in the
      _twinbreak\experiments_ folder.
    - Execution: Execute the script for the first experiment.
      ```bash
      python -u experiment_1.py
      ```
    - Results: The experiment logs its results both to the terminal and to a log file located at
      _twinbreak\results\experiment\_1\log\log0.txt_. At the end of the output, a summary of the results is presented in
      two tables. The first table reports the utility benchmark results and corresponds to the first rows of Tables
      15–19 in the paper. The second table presents the safety benchmark results and can be compared to the first rows
      of Tables 2–5. For clarity, the relevant reference table from the paper is also indicated alongside each output
      table in the terminal.

    - 💡 Hint: We expect the reported values to fall within a 1–5\% range of those published in the paper. Additionally,
      the runtime of the attack is displayed beneath the tables and can be compared to the values in Table 7. Note that
      runtime is highly dependent on hardware. For example, while the paper reports a runtime of 162 seconds, our server
      achieved significantly faster results (16 seconds), as the experiment was run on Kaggle.

- __E2__: _Model Architecture Independence_ [3 human-minute + 7 compute-hour + 114GB disk]: In this experiment, we
  repeat
  experiment (E1) with different model architectures from different vendors, namely Llama 3.1 (8B), Gemma 2 (9B), and
  Qwen 2.5 (7B). The results prove the major claim (C2).
    - Preparation: Follow all instructions under _Getting Started_ section. The terminal should reside in the
      _twinbreak\experiments} folder.
    - Execution: We provide one script for each of the models. Execute the scripts individually.
        - To execute TwinBreak for Qwen 2.5 (7b), execute the following
          command. [1 human-minute + 2 compute-hour + 37GB disk]
            ```bash
             python -u experiment_2_1.py
            ```
        - To execute TwinBreak for Gemma 2 (9b), execute the following
          command. [1 human-minute + 3 compute-hour + 40GB disk]
            ```bash
             python -u experiment_2_2.py
            ```
        - To execute TwinBreak for LLaMA 3.1 (8b), execute the following
          command. [1 human-minute + 2 compute-hour + 37GB disk]
            ```bash
             python -u experiment_2_3.py
            ```
    - Results: As with experiment (E1), each script generates terminal output and corresponding log files stored in the
      appropriate results folder, for example,
      _twinbreak\results\experiment\_2\_1\log\log0.txt_ for _experiment\_2\_1.py_.
      The output can be compared to the utility and safety benchmarks in Tables 15–19 and Tables 2–5, respectively.
    - 💡 Hint: As before, we expect the reported values to be within a 1–5\% range of those presented in the paper.

- __E2__: _E3: _Model Size Independence_ [2 human-minute + 4 compute-hour + 75GB disk]: In this experiment we attack
  models with different model sizes, namely LLaMA 2 (13b) and Qwen 2.5 (3b), using TwinBreak. We evaluate the jailbreak
  success with the newest benchmark (StrongREJECT) and the utility with HellaSwag. The results prove the major claim (
  C3).
    - Preparation: Preparation: Follow all instructions under _Getting Started_ section. The terminal should reside in
      the _twinbreak\experiments_ folder.
    - Execution: We provide one script for each of the models. Execute the scripts individually.
        - To execute TwinBreak for LLaMA 2 (13b), execute the following
          command. [1 human-minute + 3 compute-hour + 47GB disk]
            ```bash
             python -u experiment_3_1.py
            ```
        - To execute TwinBreak for Qwen 2.5 (3b), execute the following
          command. [1 human-minute + 1 compute-hour + 28GB disk]
            ```bash
             python -u experiment_3_2.py
            ```
    - Results: As with experiment (E1), each script generates terminal output and corresponding log files stored in the
      appropriate results folder, for example,
      _twinbreak\results\experiment\_3\_1\log\log0.txt_ for _experiment\_3\_1.py_. The output can be compared to the
      utility and safety benchmark results in Table 8. Specifically, the _AVG Degradation_ value in the utility
      benchmarks corresponds to the _TwinBreak Utility_ column for the respective model in Table 8, while the
      StrongREJECT score in the _Iteration 5_ column reflects the _TwinBreak ASR_ column in Table 8.
    - 💡 Hint: As with previous experiments, we expect the reported values to fall within a 1–5\% range of those
      presented in the paper.

## ♻️ Reusability

To evaluate _TwinBreak_ with different models or alternative hyperparameters, the file _experiment\_default.py_ in the
_experiments_ folder can be duplicated and modified accordingly.
Corresponding configuration files can be generated by copying and adjusting _experiment\_default\_settings.yaml_ and
_twinbreak\_default\_settings.yaml_.

The configuration files used for the hyperparameter study (corresponding to Table 20 in the paper) are provided in
_twinbreak/configs/twinbreak/hyperparameter\_study_.

To apply _TwinBreak_ to new models that are not yet supported, a new subclass of the _AbstractModel_ class should be
implemented in _twinbreak/models_.
