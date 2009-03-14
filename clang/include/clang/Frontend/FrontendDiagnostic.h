//===--- DiagnosticFrontend.h - Diagnostics for frontend --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTENDDIAGNOSTIC_H
#define LLVM_CLANG_FRONTENDDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#define FRONTENDSTART
#include "clang/Basic/DiagnosticFrontendKinds.def"
#undef DIAG
      NUM_BUILTIN_FRONTEND_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
