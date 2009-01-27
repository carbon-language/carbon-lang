#ifndef LLVM_CLANG_DIAGNOSTICPARSE_H
#define LLVM_CLANG_DIAGNOSTICPARSE_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
#define PARSESTART
#include "DiagnosticParseKinds.def"
      NUM_BUILTIN_PARSE_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
