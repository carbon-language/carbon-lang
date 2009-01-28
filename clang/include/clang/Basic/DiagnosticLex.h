//===--- DiagnosticLex.h - Diagnostics for liblex ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIAGNOSTICLEX_H
#define LLVM_CLANG_DIAGNOSTICLEX_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
#define LEXSTART
#include "DiagnosticLexKinds.def"
      NUM_BUILTIN_LEX_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
