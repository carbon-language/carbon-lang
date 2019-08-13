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

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

add_executable(f18
  f18.cc
  dump.cc
)

target_link_libraries(f18
  FortranParser
  FortranEvaluate
  FortranSemantics
)

add_executable(f18-parse-demo
  f18-parse-demo.cc
  stub-evaluate.cc
)

target_link_libraries(f18-parse-demo
  FortranParser
)

set_target_properties(f18 f18-parse-demo
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

set(MODULES
  "ieee_arithmetic"
  "ieee_exceptions"
  "ieee_features"
  "iso_c_binding"
  "iso_fortran_env"
)

# Create module files directly from the top-level module source directory
foreach(filename ${MODULES})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/${filename}.mod
    COMMAND f18 -fparse-only -fdebug-semantics ${PROJECT_SOURCE_DIR}/module/${filename}.f90
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include"
    DEPENDS f18 ${PROJECT_SOURCE_DIR}/module/${filename}.f90
  )
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/${filename}.f18.mod
    COMMAND f18 -fparse-only -fdebug-semantics -module-suffix .f18.mod ${PROJECT_SOURCE_DIR}/module/${filename}.f90
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include"
    DEPENDS f18 ${PROJECT_SOURCE_DIR}/module/${filename}.f90
  )
  list(APPEND MODULE_FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${filename}.mod")
  list(APPEND MODULE_FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${filename}.f18.mod")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${filename}.mod DESTINATION include)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${filename}.f18.mod DESTINATION include)
endforeach()

add_custom_target(module_files ALL DEPENDS ${MODULE_FILES})

install(TARGETS f18 f18-parse-demo DESTINATION bin)

file(COPY flang.sh
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bin"
  FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)
file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/bin/flang.sh" "${CMAKE_CURRENT_BINARY_DIR}/bin/flang")

install(PROGRAMS flang.sh DESTINATION bin RENAME flang)
