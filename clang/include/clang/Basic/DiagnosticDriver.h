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
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
      NUM_BUILTIN_DRIVER_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
#ifndef LLVM_CLANG_DIAGNOSTICDRIVER_H
#define LLVM_CLANG_DIAGNOSTICDRIVER_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
      NUM_BUILTIN_DRIVER_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
#ifndef LLVM_CLANG_DIAGNOSTICDRIVER_H
#define LLVM_CLANG_DIAGNOSTICDRIVER_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
      NUM_BUILTIN_DRIVER_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
