//===-- runtime/io.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-api.h"
#include "format.h"
#include "io-stmt.h"
#include "memory.h"
#include "terminator.h"
#include <cstdlib>
#include <memory>

namespace Fortran::runtime::io {

Cookie IONAME(BeginInternalFormattedOutput)(char *internal,
    std::size_t internalLength, const char *format, std::size_t formatLength,
    void ** /*scratchArea*/, std::size_t /*scratchBytes*/,
    const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InternalFormattedIoStatementState<false>>{}(oom, internal,
      internalLength, format, formatLength, sourceFile, sourceLine);
}

enum Iostat IONAME(EndIoStatement)(Cookie io) {
  return static_cast<enum Iostat>(io->EndIoStatement());
}
}
