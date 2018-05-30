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
)

add_executable(leading-zero-bit-count-test
  leading-zero-bit-count-test.cc
)

target_link_libraries(leading-zero-bit-count-test
  FortranEvaluate
  FortranEvaluateTesting
)

add_executable(fixed-point-test
  fixed-point-test.cc
)

target_link_libraries(fixed-point-test
  FortranEvaluate
  FortranEvaluateTesting
)
