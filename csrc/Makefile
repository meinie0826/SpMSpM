# Copyright 2025 The SpInfer Authors. All rights reserved.
# Copyright 2023 The FLash-LLM Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -O3 -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     := -fPIC
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/
LIBRARIES := -lcublas -lcusparse

################################################################################

# Gencode arguments
SMS ?= 89
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS +=  --threads 0 --std=c++11
ALL_CCFLAGS += -maxrregcount=255 
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################

HEAD_FILES = SpMM_API.cuh \
				Reduction_Kernel.cuh SpMM_Kernel.cuh \
			 MatMulUtilities.cuh \
			 MMA_PTX.cuh AsyncCopy_PTX.cuh \
			 TilingConfig.h
# 修改目标规则
all: spmm_test

# 删除libSpMM_API.a相关规则

SpMM_API.o: $(HEAD_FILES)  
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

spmm_test.o: spmm_test.cu spmm_test_utils.h 
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -I.. -o $@ -c $<


spmm_test: spmm_test.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(ALL_CCFLAGS)  $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

clean:
	rm -f SpMM_API.o spmm_test.o spmm_test

run : clean
	$(MAKE) -j90
	./run_all_main.sh > out4