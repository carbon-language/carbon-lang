//===- InstrProfilingRuntime.cpp - PGO runtime initialization -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

extern "C" {

#include "InstrProfiling.h"

COMPILER_RT_VISIBILITY int __llvm_profile_runtime;
}

namespace {

class RegisterRuntime {
public:
  RegisterRuntime() {
    __llvm_profile_register_write_file_atexit();
    __llvm_profile_initialize_file();
  }
};

RegisterRuntime Registration;

}
