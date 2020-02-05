//===-- runtime/io-error.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Distinguishes I/O error conditions; fatal ones lead to termination,
// and those that the user program has chosen to handle are recorded
// so that the highest-priority one can be returned as IOSTAT=.

#ifndef FORTRAN_RUNTIME_IO_ERROR_H_
#define FORTRAN_RUNTIME_IO_ERROR_H_

#include "terminator.h"
#include <cinttypes>

namespace Fortran::runtime::io {

class IoErrorHandler : public Terminator {
public:
  using Terminator::Terminator;
  explicit IoErrorHandler(const Terminator &that) : Terminator{that} {}
  void Begin(const char *sourceFileName, int sourceLine);
  void HasIoStat() { flags_ |= hasIoStat; }
  void HasErrLabel() { flags_ |= hasErr; }
  void HasEndLabel() { flags_ |= hasEnd; }
  void HasEorLabel() { flags_ |= hasEor; }

  void SignalError(int iostatOrErrno);
  void SignalErrno();
  void SignalEnd();
  void SignalEor();

  int GetIoStat() const { return ioStat_; }

private:
  enum Flag : std::uint8_t {
    hasIoStat = 1,  // IOSTAT=
    hasErr = 2,  // ERR=
    hasEnd = 4,  // END=
    hasEor = 8,  // EOR=
  };
  std::uint8_t flags_{0};
  int ioStat_{0};
};

}
#endif  // FORTRAN_RUNTIME_IO_ERROR_H_
