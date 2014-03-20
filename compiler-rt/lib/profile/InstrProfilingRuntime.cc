//===- InstrProfilingRuntime.cpp - PGO runtime initialization -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

extern "C" {

#include "InstrProfilingExtras.h"

extern int __llvm_pgo_runtime;
int __llvm_pgo_runtime;

}

namespace {

class RegisterAtExit {
public:
  RegisterAtExit() { __llvm_pgo_register_write_file_atexit(); }
};

RegisterAtExit Registration;

}
