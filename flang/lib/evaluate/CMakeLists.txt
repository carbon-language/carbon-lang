#===-- lib/evaluate/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_library(FortranEvaluate
  call.cc
  characteristics.cc
  check-expression.cc
  common.cc
  complex.cc
  constant.cc
  expression.cc
  fold.cc
  fold-character.cc
  fold-complex.cc
  fold-integer.cc
  fold-logical.cc
  fold-real.cc
  formatting.cc
  host.cc
  integer.cc
  intrinsics.cc
  intrinsics-library.cc
  logical.cc
  real.cc
  shape.cc
  static-data.cc
  tools.cc
  type.cc
  variable.cc
)

target_link_libraries(FortranEvaluate
  FortranCommon
  FortranDecimal
  FortranParser
  m
)

install (TARGETS FortranEvaluate
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if (LIBPGMATH_DIR)
  # If pgmath library is found, it can be used for constant folding.
  find_library(LIBPGMATH pgmath PATHS ${LIBPGMATH_DIR})
  if(LIBPGMATH)
    add_compile_definitions(LINK_WITH_LIBPGMATH)
    target_link_libraries(FortranEvaluate ${LIBPGMATH})
    message(STATUS "Found libpgmath: ${LIBPGMATH}")
  else()
    message(STATUS "Libpgmath not found in: ${LIBPGMATH_DIR}")
  endif()
endif()
