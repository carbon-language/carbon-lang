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
  implicit08.f90
  int-literals.f90
  io01.f90
  io02.f90
  io03.f90
  io04.f90
  io05.f90
  io06.f90
  io07.f90
  io08.f90
  io09.f90
  io10.f90
  kinds02.f90
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
  resolve30.f90
  resolve31.f90
  resolve32.f90
  resolve33.f90
  resolve34.f90
  resolve35.f90
  resolve36.f90
  resolve37.f90
  resolve38.f90
  resolve39.f90
  resolve40.f90
  resolve41.f90
  resolve42.f90
  resolve43.f90
  resolve44.f90
  resolve45.f90
  resolve46.f90
  resolve47.f90
  resolve48.f90
  resolve49.f90
  resolve50.f90
  resolve51.f90
  resolve52.f90
  resolve53.f90
  resolve54.f90
  resolve55.f90
  resolve56.f90
  resolve57.f90
  stop01.f90
  structconst01.f90
  structconst02.f90
  structconst03.f90
  structconst04.f90
  assign01.f90
  if_arith02.f90
  if_arith03.f90
  if_arith04.f90
  if_construct02.f90
  if_stmt02.f90
  if_stmt03.f90
  computed-goto01.f90
  computed-goto02.f90
  nullify01.f90
  nullify02.f90
  deallocate01.f90
  deallocate04.f90
  deallocate05.f90
  coarrays01.f90
  altreturn01.f90
# Issue 407
#  altreturn02.f90
#  altreturn03.f90
  altreturn04.f90
  allocate01.f90
  allocate02.f90
  allocate03.f90
  allocate04.f90
  allocate05.f90
  allocate06.f90
  allocate07.f90
  allocate08.f90
  allocate09.f90
  allocate10.f90
  allocate11.f90
  allocate12.f90
  allocate13.f90
  dosemantics01.f90
  dosemantics02.f90
  dosemantics03.f90
  dosemantics04.f90
  expr-errors01.f90
  null01.f90
  omp-clause-validity01.f90
#  omp-nested01.f90
  omp-declarative-directive.f90
  equivalence01.f90
  init01.f90
  if_arith01.f90
  if_construct01.f90
  if_stmt01.f90
)

# These test files have expected symbols in the source
set(SYMBOL_TESTS
  symbol01.f90
  symbol02.f90
  symbol03.f90
  symbol05.f90
  symbol06.f90
  symbol07.f90
  symbol08.f90
  symbol09.f90
  symbol10.f90
  symbol11.f90
  symbol12.f90
  symbol13.f90
  symbol14.f90
  kinds01.f90
  kinds03.f90
  procinterface01.f90
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
  modfile10.f90
  modfile11.f90
  modfile12.f90
  modfile13.f90
  modfile14.f90
  modfile15.f90
  modfile16.f90
  modfile17.f90
  modfile18.f90
  modfile19.f90
  modfile20.f90
  modfile21.f90
  modfile22.f90
  modfile23.f90
  modfile24.f90
  modfile25.f90
  modfile26.f90
  modfile27.f90
  modfile28.f90
  modfile29.f90
)

set(LABEL_TESTS
  label*.[Ff]90
)

set(DOCONCURRENT_TESTS
  doconcurrent*.[Ff]90
)

set(CANONDO_TESTS
  canondo*.[Ff]90
)

set(FORALL_TESTS
  forall*.[Ff]90
)

set(F18 $<TARGET_FILE:f18>)

foreach(test ${ERROR_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_errors.sh ${test} ${F18})
endforeach()

foreach(test ${SYMBOL_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_symbols.sh ${test} ${F18})
endforeach()

foreach(test ${MODFILE_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_modfile.sh ${test} ${F18})
endforeach()

foreach(test ${LABEL_TESTS} ${CANONDO_TESTS} ${DOCONCURRENT_TESTS} ${FORALL_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_any.sh ${test} ${F18})
endforeach()
