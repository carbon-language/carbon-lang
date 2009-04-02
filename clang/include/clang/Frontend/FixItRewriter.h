//===--- FixItRewriter.h - Fix-It Rewriter Diagnostic Client ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a diagnostic client adaptor that performs rewrites as
// suggested by code modification hints attached to diagnostics. It
// then forwards any diagnostics to the adapted diagnostic client.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_FRONTEND_FIX_IT_REWRITER_H
#define LLVM_CLANG_FRONTEND_FIX_IT_REWRITER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Rewrite/Rewriter.h"

namespace clang {

class SourceManager;

class FixItRewriter : public DiagnosticClient {
  /// \brief The diagnostics machinery.
  Diagnostic &Diags;

  /// \brief The rewriter used to perform the various code
  /// modifications.
  Rewriter Rewrite;

  /// \brief The diagnostic client that performs the actual formatting
  /// of error messages.
  DiagnosticClient *Client;

  /// \brief The number of rewriter failures.
  unsigned NumFailures;

public:
  /// \brief Initialize a new fix-it rewriter.
  FixItRewriter(Diagnostic &Diags, SourceManager &SourceMgr);

  /// \brief Destroy the fix-it rewriter.
  ~FixItRewriter();

  /// \brief Write the modified source file.
  ///
  /// \returns true if there was an error, false otherwise.
  bool WriteFixedFile(const std::string &InFileName, 
                      const std::string &OutFileName = std::string());

  /// IncludeInDiagnosticCounts - This method (whose default implementation
  ///  returns true) indicates whether the diagnostics handled by this
  ///  DiagnosticClient should be included in the number of diagnostics
  ///  reported by Diagnostic.
  virtual bool IncludeInDiagnosticCounts() const;

  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or
  /// capturing it to a log as needed.
  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info);

  /// \brief Emit a diagnostic via the adapted diagnostic client.
  void Diag(FullSourceLoc Loc, unsigned DiagID);
};

}

#endif // LLVM_CLANG_FRONTEND_FIX_IT_REWRITER_H
