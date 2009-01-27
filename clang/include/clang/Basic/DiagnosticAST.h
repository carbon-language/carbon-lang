#ifndef LLVM_CLANG_DIAGNOSTICAST_H
#define LLVM_CLANG_DIAGNOSTICAST_H

#include "clang/Basic/DiagnosticAST.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
#define ASTSTART
#include "DiagnosticASTKinds.def"
      NUM_BUILTIN_AST_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
