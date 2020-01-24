//===-- runtime/io.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the I/O statement API

#include "io-api.h"
#include "format.h"
#include "io-stmt.h"
#include "memory.h"
#include "numeric-output.h"
#include "terminator.h"
#include "unit.h"
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

Cookie IONAME(BeginExternalFormattedOutput)(const char *format,
    std::size_t formatLength, ExternalUnit unitNumber, const char *sourceFile,
    int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  int unit{unitNumber == DefaultUnit ? 6 : unitNumber};
  ExternalFile &file{ExternalFile::LookUpOrCrash(unit, terminator)};
  return &file.BeginIoStatement<ExternalFormattedIoStatementState<false>>(
      file, format, formatLength, sourceFile, sourceLine);
}

bool IONAME(OutputInteger64)(Cookie cookie, std::int64_t n) {
  IoStatementState &io{*cookie};
  DataEdit edit;
  io.GetNext(edit);
  return EditIntegerOutput(io, edit, n);
}

bool IONAME(OutputReal64)(Cookie cookie, double x) {
  IoStatementState &io{*cookie};
  DataEdit edit;
  io.GetNext(edit);
  return RealOutputEditing<double, 15, 53, 1024>{io, x}.Edit(edit);
}

bool IONAME(OutputAscii)(Cookie cookie, const char *x, std::size_t length) {
  IoStatementState &io{*cookie};
  DataEdit edit;
  io.GetNext(edit);
  if (edit.descriptor != 'A' && edit.descriptor != 'G') {
    io.Crash(
        "Data edit descriptor '%c' may not be used with a CHARACTER data item",
        edit.descriptor);
    return false;
  }
  int len{static_cast<int>(length)};
  int width{edit.width.value_or(len)};
  return EmitRepeated(io, ' ', std::max(0, width - len)) &&
      io.Emit(x, std::min(width, len));
}

bool IONAME(OutputLogical)(Cookie cookie, bool truth) {
  IoStatementState &io{*cookie};
  DataEdit edit;
  io.GetNext(edit);
  if (edit.descriptor != 'L' && edit.descriptor != 'G') {
    io.Crash(
        "Data edit descriptor '%c' may not be used with a LOGICAL data item",
        edit.descriptor);
    return false;
  }
  return EmitRepeated(io, ' ', std::max(0, edit.width.value_or(1) - 1)) &&
      io.Emit(truth ? "T" : "F", 1);
}

enum Iostat IONAME(EndIoStatement)(Cookie cookie) {
  IoStatementState &io{*cookie};
  return static_cast<enum Iostat>(io.EndIoStatement());
}
}
