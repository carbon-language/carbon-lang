# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

add_library(FortranEvaluateTesting
  testing.cc
  fp-testing.cc
)

add_executable(leading-zero-bit-count-test
  leading-zero-bit-count.cc
)

target_link_libraries(leading-zero-bit-count-test
  FortranEvaluate
  FortranEvaluateTesting
)

add_executable(bit-population-count-test
  bit-population-count.cc
)

target_link_libraries(bit-population-count-test
  FortranEvaluate
  FortranEvaluateTesting
)

add_executable(integer-test
  integer.cc
)

target_link_libraries(integer-test
  FortranEvaluate
  FortranEvaluateTesting
)

add_executable(logical-test
  logical.cc
)

target_link_libraries(logical-test
  FortranEvaluate
  FortranEvaluateTesting
)

add_executable(real-test
  real.cc
)

target_link_libraries(real-test
  FortranEvaluate
  FortranEvaluateTesting
)
