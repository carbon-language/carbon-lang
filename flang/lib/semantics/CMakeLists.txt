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

add_library(FortranSemantics
  assignment.cc
  attr.cc
  canonicalize-do.cc
  check-allocate.cc
  check-arithmeticif.cc
  check-block.cc
  check-coarray.cc
  check-deallocate.cc
  check-do.cc
  check-if-stmt.cc
  check-io.cc
  check-nullify.cc
  check-omp-structure.cc
  check-return.cc
  check-stop.cc
  expression.cc
  mod-file.cc
  program-tree.cc
  resolve-labels.cc
  resolve-names.cc
  resolve-names-utils.cc
  rewrite-parse-tree.cc
  scope.cc
  semantics.cc
  symbol.cc
  tools.cc
  type.cc
  unparse-with-symbols.cc
)

target_link_libraries(FortranSemantics
  FortranCommon
  FortranEvaluate
)

install (TARGETS FortranSemantics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
