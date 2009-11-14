//===--- ChainedDiagnosticClient.h - Chain Diagnostic Clients ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_CHAINEDDIAGNOSTICCLIENT_H
#define LLVM_CLANG_FRONTEND_CHAINEDDIAGNOSTICCLIENT_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
class LangOptions;

/// ChainedDiagnosticClient - Chain two diagnostic clients so that diagnostics
/// go to the first client and then the second. The first diagnostic client
/// should be the "primary" client, and will be used for computing whether the
/// diagnostics should be included in counts.
class ChainedDiagnosticClient : public DiagnosticClient {
  llvm::OwningPtr<DiagnosticClient> Primary;
  llvm::OwningPtr<DiagnosticClient> Secondary;

public:
  ChainedDiagnosticClient(DiagnosticClient *_Primary,
                          DiagnosticClient *_Secondary) {
    Primary.reset(_Primary);
    Secondary.reset(_Secondary);
  }

  virtual void BeginSourceFile(const LangOptions &LO,
                               const Preprocessor *PP) {
    Primary->BeginSourceFile(LO, PP);
    Secondary->BeginSourceFile(LO, PP);
  }

  virtual void EndSourceFile() {
    Secondary->EndSourceFile();
    Primary->EndSourceFile();
  }

  virtual bool IncludeInDiagnosticCounts() const {
    return Primary->IncludeInDiagnosticCounts();
  }

  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info) {
    Primary->HandleDiagnostic(DiagLevel, Info);
    Secondary->HandleDiagnostic(DiagLevel, Info);
  }
};

} // end namspace clang

#endif
