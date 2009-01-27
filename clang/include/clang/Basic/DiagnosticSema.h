#ifndef LLVM_CLANG_DIAGNOSTICSEMA_H
#define LLVM_CLANG_DIAGNOSTICSEMA_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
#define SEMASTART
#include "DiagnosticSemaKinds.def"
      NUM_BUILTIN_SEMA_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
