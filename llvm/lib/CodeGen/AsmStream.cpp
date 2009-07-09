//===-- llvm/CodeGen/AsmStream.cpp - AsmStream Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instantiations of "standard" AsmOStreams.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/AsmStream.h"

#include <unistd.h>

namespace llvm {
  AsmStreambuf asmoutbuf(STDOUT_FILENO);
  AsmStreambuf asmerrbuf(STDERR_FILENO);

  //===----------------------------------------------------------------------===//
  //  raw_asm_ostream
  //===----------------------------------------------------------------------===//

  raw_asm_ostream::~raw_asm_ostream() {
    flush();
  }

  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  void raw_asm_ostream::flush_impl() {
    if (OutBufCur-OutBufStart)
      OS.write(OutBufStart, OutBufCur-OutBufStart);

    HandleFlush();
  }

  namespace {
    AsmOStream AsmOut(&asmoutbuf);
    AsmOStream AsmErr(&asmerrbuf);
  }

  raw_asm_ostream asmout(AsmOut);
  raw_asm_ostream asmerr(AsmErr);
}
