//===-- runtime/main.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "main.h"
#include "io-stmt.h"
#include "terminator.h"
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace Fortran::runtime {
ExecutionEnvironment executionEnvironment;

void ExecutionEnvironment::Configure(
    int ac, const char *av[], const char *env[]) {
  argc = ac;
  argv = av;
  envp = env;
  listDirectedOutputLineLengthLimit = 79;  // PGI default

  if (auto *x{std::getenv("FORT_FMT_RECL")}) {
    char *end;
    auto n{std::strtol(x, &end, 10)};
    if (n > 0 && n < std::numeric_limits<int>::max() && *end == '\0') {
      listDirectedOutputLineLengthLimit = n;
    } else {
      std::fprintf(
          stderr, "Fortran runtime: FORT_FMT_RECL=%s is invalid; ignored\n", x);
    }
  }
}
}

static void ConfigureFloatingPoint() {
#ifdef feclearexcept  // a macro in some environments; omit std::
  feclearexcept(FE_ALL_EXCEPT);
#else
  std::feclearexcept(FE_ALL_EXCEPT);
#endif
#ifdef fesetround
  fesetround(FE_TONEAREST);
#else
  std::fesetround(FE_TONEAREST);
#endif
}

extern "C" {

void RTNAME(ProgramStart)(int argc, const char *argv[], const char *envp[]) {
  std::atexit(Fortran::runtime::NotifyOtherImagesOfNormalEnd);
  Fortran::runtime::executionEnvironment.Configure(argc, argv, envp);
  ConfigureFloatingPoint();
}
}
