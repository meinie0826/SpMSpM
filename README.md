

## 1. Clone this project.
```bash
git clone https://github.com/HPMLL/SpInfer_EuroSys25.git
cd SpInfer_EuroSys25
git submodule update --init --recursive
source Init_SpInfer.sh
cd $SpInfer_HOME/third_party/FasterTransformer && git apply ../ft_spinfer.patch
cd $SpInfer_HOME/third_party/sputnik && git apply ../sputnik.patch
```

+ **Requirements**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.3`
> + `cmake >= 3.30.3`
> + `CUDA >= 12.2` and `nvcc >= 12.0`
> + NVIDIA GPU with `sm >= 80` (i.e., Ampere-A6000 and  Ada -RTX4090).

## 2. Environment Setup. (Install via Conda)
+ 2.1 Install **`conda`** on system **[Toturial](https://docs.anaconda.com/miniconda/)**.
+ 2.2 Create a **`conda`** environment: 
```
cd $SpInfer_HOME
conda env create -f spinfer.yml
conda activate spinfer
```

## 3. Install **`SpInfer`**.
The libSpMM_API.so and SpMM_API.cuh will be available for easy integration after:
```
cd $SpInfer_HOME/build && make -j
```

## 4. Running **SpInfer** in kernel benchmark (Figure 10).
- Build Sputnik.

```bash
cd $SpInfer_HOME/third_party/
source build_sputnik.sh
```

- Build SparTA.

```bash
cd $SpInfer_HOME/third_party/
source preparse_cusparselt.sh
```

- Reproduce Figure 10.

```bash
cd $SpInfer_HOME/kernel_benchmark
source test_env
make -j
source benchmark.sh
```

Check the results in raw csv files and the reproduced Figure10.png (Fig. 10).

## 5. Running End-to-end model.
#### 5.1 Building
Follow the steps in **[SpInfer/docs/LLMInferenceExample](https://github.com/HPMLL/SpInfer_EuroSys25/blob/main/docs/LLMInferenceExample.md#llm-inference-example)**
+ Building Faster-Transformer with (SpInfer, Flash-llm or Standard) integration
+ Downloading & Converting OPT models
+ Configuration
Note: Model_dir is different for SpInfer, Flash-llm and Faster-Transformer.
#### 5.2 Running **SpInfer** Inference
> + `cd $SpInfer_HOME/third_party/`
> + `bash run_1gpu_loop.sh`
> + Check the results (Fig.13/14) in `$SpInfer_HOME/third_party/FasterTransformer/OutputFile_1gpu_our_60_inlen64/`
> + Test tensor_para_size=2 using `bash run_2gpu_loop.sh`
> + Test tensor_para_size=4 using `bash run_4gpu_loop.sh`
#### 5.3 Running **Flash-llm** Inference
> + `cd $FlashLLM_HOME/third_party/`
> + `bash run_1gpu_loop.sh`
> + Check the results in `$FlashLLM_HOME/third_party/FasterTransformer/OutputFile_1gpu_our_60_inlen64/`
> + Test tensor_para_size=1 using `bash run_1gpu_loop.sh`
#### 5.4 Running **Faster-transformer** Inference
> + `cd $FT_HOME/third_party/`
> + `bash run_2gpu_loop.sh`
> + Check the results in `$FT_HOME/FasterTransformer/OutputFile_2gpu_our_60_inlen64/`
#### 5.5 Runing **DeepSpeed** Inference
> + `cd $SpInfer_HOME/end2end_inference/ds_scripts`
> + `pip install -r requirements.txt`
> + `bash run_ds_loop.sh`
> + Check the results in `$SpInfer_HOME/end2end_inference/ds_scripts/ds_result/`

---


```
{ BasedOnStyle: LLVM, IndentWidth: 4, TabWidth: 4, UseTab: Never, AlignAfterOpenBracket: Align, AlignOperands: Align, AlignConsecutiveAssignments: Consecutive, AlignConsecutiveDeclarations: Consecutive, SpaceBeforeAssignmentOperators: true, SpaceAfterCStyleCast: true, SpaceBeforeParens: Never, SpaceInEmptyParentheses: false, SpacesInAngles: Never, SpacesInContainerLiterals: false, PointerAlignment: Left, ColumnLimit: 130, BreakBeforeBraces: Attach, AllowShortFunctionsOnASingleLine: false }
```
