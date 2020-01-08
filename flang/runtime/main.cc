//===-- runtime/main.cc -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#include "main.h"
#include "terminator.h"
#include <cfenv>
#include <cstdlib>

namespace Fortran::runtime {
int argc;
const char **argv;
const char **envp;
}

extern "C" {

void __FortranProgram();  // PROGRAM statement

int main(int argc, const char *argv[], const char *envp[]) {
  Fortran::runtime::argc = argc;
  Fortran::runtime::argv = argv;
  Fortran::runtime::envp = envp;
  std::feclearexcept(FE_ALL_EXCEPT);
  std::fesetround(FE_TONEAREST);
  std::atexit(Fortran::runtime::NotifyOtherImagesOfNormalEnd);
  // TODO: Runtime configuration settings from environment
  __FortranProgram();
  return EXIT_SUCCESS;
}
}
