#ifndef LLVM_CLANG_DIAGNOSTICANALYSIS_H
#define LLVM_CLANG_DIAGNOSTICANALYSIS_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag { 
    enum {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticCommonKinds.def"
#define ANALYSISSTART
#include "DiagnosticAnalysisKinds.def"
      NUM_BUILTIN_ANALYSIS_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
