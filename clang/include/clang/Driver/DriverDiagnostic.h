//===--- DiagnosticDriver.h - Diagnostics for the driver --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIAGNOSTICDRIVER_H
#define LLVM_CLANG_DIAGNOSTICDRIVER_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
// FIXME: REMOVE??
      NUM_BUILTIN_DRIVER_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif

