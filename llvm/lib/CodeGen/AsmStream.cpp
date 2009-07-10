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

namespace llvm {
  raw_asm_fd_ostream asmouts(STDOUT_FILENO, false);
  raw_asm_fd_ostream asmerrs(STDERR_FILENO, false);
}
