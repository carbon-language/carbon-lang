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

add_library(FortranEvaluate
  call.cc
  characteristics.cc
  check-call.cc
  check-expression.cc
  common.cc
  complex.cc
  constant.cc
  expression.cc
  fold.cc
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
