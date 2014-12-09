//===-libllvm.cpp - LLVM Shared Library -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is empty and serves only the purpose of making CMake happy because
// you can't define a target with no sources.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"

#if defined(DISABLE_LLVM_DYLIB_ATEXIT)
extern "C" int __cxa_atexit();
extern "C" int __cxa_atexit() { return 0; }
#endif
