#===-- lib/semantics/CMakeLists.txt ----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_library(FortranSemantics
  assignment.cc
  attr.cc
  canonicalize-do.cc
  canonicalize-omp.cc
  check-allocate.cc
  check-arithmeticif.cc
  check-call.cc
  check-coarray.cc
  check-deallocate.cc
  check-declarations.cc
  check-do.cc
  check-if-stmt.cc
  check-io.cc
  check-nullify.cc
  check-omp-structure.cc
  check-purity.cc
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
