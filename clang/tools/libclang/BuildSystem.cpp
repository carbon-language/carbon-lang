//===- BuildSystem.cpp - Utilities for use by build systems ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements various utilities for use by build systems.
//
//===----------------------------------------------------------------------===//

#include "clang-c/BuildSystem.h"
#include "llvm/Support/TimeValue.h"

extern "C" {
unsigned long long clang_getBuildSessionTimestamp(void) {
  return llvm::sys::TimeValue::now().toEpochTime();
}
} // extern "C"

