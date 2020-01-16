//===-- runtime/main.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_MAIN_H_
#define FORTRAN_RUNTIME_MAIN_H_

#include "entry-names.h"

namespace Fortran::runtime {
struct ExecutionEnvironment {
  void Configure(int argc, const char *argv[], const char *envp[]);

  int argc;
  const char **argv;
  const char **envp;
  int listDirectedOutputLineLengthLimit;
};
extern ExecutionEnvironment executionEnvironment;
}

extern "C" {
void RTNAME(ProgramStart)(int, const char *[], const char *[]);
}

#endif  // FORTRAN_RUNTIME_MAIN_H_
