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

# Run tests with test_errors.sh. It compiles the test with f18 and compares
# actual errors produced with expected ones listed in the source.

# These test files have expected errors in the source
set(ERROR_TESTS
  implicit01.f90
  implicit02.f90
  implicit03.f90
  implicit04.f90
  implicit05.f90
  implicit06.f90
  implicit07.f90
  resolve01.f90
  resolve02.f90
  resolve03.f90
  resolve04.f90
  resolve05.f90
  resolve06.f90
  resolve07.f90
  resolve08.f90
  resolve09.f90
  resolve10.f90
  resolve11.f90
  resolve12.f90
  resolve13.f90
  resolve14.f90
  resolve15.f90
  resolve16.f90
  resolve17.f90
  resolve18.f90
  resolve19.f90
  resolve20.f90
  resolve21.f90
  resolve22.f90
  resolve23.f90
  resolve24.f90
  resolve25.f90
  resolve26.f90
  resolve27.f90
  resolve28.f90
  resolve29.f90
)

# These test files have expected symbols in the source
set(SYMBOL_TESTS
  symbol01.f90
  symbol02.f90
  symbol03.f90
)

# These test files have expected .mod file contents in the source
set(MODFILE_TESTS
  modfile01.f90
  modfile02.f90
  modfile03.f90
  modfile04.f90
  modfile05.f90
  modfile06.f90
  modfile07.f90
  modfile08.f90
  modfile09-*.f90
)

foreach(test ${ERROR_TESTS})
  add_test(NAME ${test} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_errors.sh ${test})
endforeach()

foreach(test ${SYMBOL_TESTS})
  add_test(NAME ${test} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_symbols.sh ${test})
endforeach()

foreach(test ${MODFILE_TESTS})
  add_test(NAME ${test} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_modfile.sh ${test})
endforeach()
