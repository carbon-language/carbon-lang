# Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(LINK_WITH_FIR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINK_WITH_FIR")
  set(FORTRAN_FIR_LIB "FortranFIR")
endif()

add_executable(f18
  f18.cc
  dump.cc
)

target_link_libraries(f18
  FortranParser
  FortranEvaluate
  FortranSemantics
  ${FORTRAN_FIR_LIB}
  ${LLVM_COMMON_LIBS}
)

add_executable(f18-parse-demo
  f18-parse-demo.cc
  stub-evaluate.cc
)

target_link_libraries(f18-parse-demo
  FortranParser
)
