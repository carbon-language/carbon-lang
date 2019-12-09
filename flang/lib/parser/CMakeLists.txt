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

add_library(FortranParser
  Fortran-parsers.cc
  char-buffer.cc
  char-set.cc
  characters.cc
  debug-parser.cc
  executable-parsers.cc
  expr-parsers.cc
  instrumented-parser.cc
  io-parsers.cc
  message.cc
  openmp-parsers.cc
  parse-tree.cc
  parsing.cc
  preprocessor.cc
  prescan.cc
  program-parsers.cc
  provenance.cc
  source.cc
  token-sequence.cc
  tools.cc
  unparse.cc
  user-state.cc
)

target_link_libraries(FortranParser
  FortranCommon
)

install (TARGETS FortranParser
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
