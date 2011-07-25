//===-- VerifyDiagnosticsClient.h - Verifying Diagnostic Client -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_VERIFYDIAGNOSTICSCLIENT_H
#define LLVM_CLANG_FRONTEND_VERIFYDIAGNOSTICSCLIENT_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {

class Diagnostic;
class TextDiagnosticBuffer;

/// VerifyDiagnosticsClient - Create a diagnostic client which will use markers
/// in the input source to check that all the emitted diagnostics match those
/// expected.
///
/// USING THE DIAGNOSTIC CHECKER:
///
/// Indicating that a line expects an error or a warning is simple. Put a
/// comment on the line that has the diagnostic, use:
///
///     expected-{error,warning,note}
///
/// to tag if it's an expected error or warning, and place the expected text
/// between {{ and }} markers. The full text doesn't have to be included, only
/// enough to ensure that the correct diagnostic was emitted.
///
/// Here's an example:
///
///   int A = B; // expected-error {{use of undeclared identifier 'B'}}
///
/// You can place as many diagnostics on one line as you wish. To make the code
/// more readable, you can use slash-newline to separate out the diagnostics.
///
/// The simple syntax above allows each specification to match exactly one
/// error.  You can use the extended syntax to customize this. The extended
/// syntax is "expected-<type> <n> {{diag text}}", where <type> is one of
/// "error", "warning" or "note", and <n> is a positive integer. This allows the
/// diagnostic to appear as many times as specified. Example:
///
///   void f(); // expected-note 2 {{previous declaration is here}}
///
/// Regex matching mode may be selected by appending '-re' to type. Example:
///
///   expected-error-re
///
/// Examples matching error: "variable has incomplete type 'struct s'"
///
///   // expected-error {{variable has incomplete type 'struct s'}}
///   // expected-error {{variable has incomplete type}}
///
///   // expected-error-re {{variable has has type 'struct .'}}
///   // expected-error-re {{variable has has type 'struct .*'}}
///   // expected-error-re {{variable has has type 'struct (.*)'}}
///   // expected-error-re {{variable has has type 'struct[[:space:]](.*)'}}
///
class VerifyDiagnosticsClient : public DiagnosticClient {
public:
  Diagnostic &Diags;
  llvm::OwningPtr<DiagnosticClient> PrimaryClient;
  llvm::OwningPtr<TextDiagnosticBuffer> Buffer;
  Preprocessor *CurrentPreprocessor;

private:
  FileID FirstErrorFID; // FileID of first diagnostic
  void CheckDiagnostics();

public:
  /// Create a new verifying diagnostic client, which will issue errors to \arg
  /// PrimaryClient when a diagnostic does not match what is expected (as
  /// indicated in the source file). The verifying diagnostic client takes
  /// ownership of \arg PrimaryClient.
  VerifyDiagnosticsClient(Diagnostic &Diags, DiagnosticClient *PrimaryClient);
  ~VerifyDiagnosticsClient();

  virtual void BeginSourceFile(const LangOptions &LangOpts,
                               const Preprocessor *PP);

  virtual void EndSourceFile();

  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info);
};

} // end namspace clang

#endif
