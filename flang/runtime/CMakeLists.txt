#===-- runtime/CMakeLists.txt ----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Allow an unresolved external reference to the entry point of the
# Fortran main PROGRAM from main.cc.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -u __FortranProgram")

add_library(FortranRuntime
  ISO_Fortran_binding.cc
  derived-type.cc
  descriptor.cc
  main.cc
  stop.cc
  terminator.cc
  transformational.cc
  type-code.cc
)

target_link_libraries(FortranRuntime
  FortranEvaluate
)
