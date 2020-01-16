//===-- runtime/io-error.cc -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-error.h"
#include "magic-numbers.h"
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace Fortran::runtime::io {

void IoErrorHandler::Begin(const char *sourceFileName, int sourceLine) {
  flags_ = 0;
  ioStat_ = 0;
  hitEnd_ = false;
  hitEor_ = false;
  SetLocation(sourceFileName, sourceLine);
}

void IoErrorHandler::SignalError(int iostatOrErrno) {
  if (iostatOrErrno != 0) {
    if (flags_ & hasIoStat) {
      if (!ioStat_) {
        ioStat_ = iostatOrErrno;
      }
    } else if (iostatOrErrno == FORTRAN_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT) {
      Crash("INQUIRE on internal unit");
    } else {
      Crash("I/O error %d: %s", iostatOrErrno, std::strerror(iostatOrErrno));
    }
  }
}

void IoErrorHandler::SignalEnd() {
  if (flags_ & hasEnd) {
    hitEnd_ = true;
  } else {
    Crash("End of file");
  }
}

void IoErrorHandler::SignalEor() {
  if (flags_ & hasEor) {
    hitEor_ = true;
  } else {
    Crash("End of record");
  }
}

int IoErrorHandler::GetIoStat() const {
  if (ioStat_) {
    return ioStat_;
  } else if (hitEnd_) {
    return FORTRAN_RUNTIME_IOSTAT_END;
  } else if (hitEor_) {
    return FORTRAN_RUNTIME_IOSTAT_EOR;
  } else {
    return 0;
  }
}

}
