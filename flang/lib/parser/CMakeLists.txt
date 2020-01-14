#===-- lib/parser/CMakeLists.txt -------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_library(FortranParser
  Fortran-parsers.cc
  char-buffer.cc
  char-block.cc
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
