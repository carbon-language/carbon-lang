//===--- Diagnostic.h - C Language Family Diagnostic Handling ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Diagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIAGNOSTIC_H
#define LLVM_CLANG_DIAGNOSTIC_H

#include <string>

namespace llvm {
namespace clang {
  class DiagnosticClient;
  class SourceBuffer;
  class SourceLocation;
  
  // Import the diagnostic enums themselves.
  namespace diag {
    enum kind {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticKinds.def"
      NUM_DIAGNOSTICS
    };
  }
  
/// Diagnostic - This concrete class is used by the front-end to report
/// problems and issues.  It massages the diagnostics (e.g. handling things like
/// "report warnings as errors" and passes them off to the DiagnosticClient for
/// reporting to the user.
class Diagnostic {
  bool WarningsAsErrors;      // Treat warnings like errors: 
  bool WarnOnExtensions;      // Enables warnings for gcc extensions: -pedantic.
  bool ErrorOnExtensions;     // Error on extensions: -pedantic-errors.
  DiagnosticClient &Client;
public:
  Diagnostic(DiagnosticClient &client) : Client(client) {
    WarningsAsErrors = false;
    WarnOnExtensions = false;
    ErrorOnExtensions = false;
  }
  
  //===--------------------------------------------------------------------===//
  //  Diagnostic characterization methods, used by a client to customize how
  //

  /// setWarningsAsErrors - When set to true, any warnings reported are issued
  /// as errors.
  void setWarningsAsErrors(bool Val) { WarningsAsErrors = Val; }
  bool getWarningsAsErrors() const { return WarningsAsErrors; }
  
  /// setWarnOnExtensions - When set to true, issue warnings on GCC extensions,
  /// the equivalent of GCC's -pedantic.
  void setWarnOnExtensions(bool Val) { WarnOnExtensions = Val; }
  bool getWarnOnExtensions() const { return WarnOnExtensions; }
  
  /// setErrorOnExtensions - When set to true issue errors for GCC extensions
  /// instead of warnings.  This is the equivalent to GCC's -pedantic-errors.
  void setErrorOnExtensions(bool Val) { ErrorOnExtensions = Val; }
  bool getErrorOnExtensions() const { return ErrorOnExtensions; }

  
  //===--------------------------------------------------------------------===//
  // Diagnostic classification and reporting interfaces.
  //

  /// getDescription - Given a diagnostic ID, return a description of the
  /// issue.
  static const char *getDescription(unsigned DiagID);
  
  /// Level - The level of the diagnostic 
  enum Level {
    // FIXME: Anachronism?
    Ignored, Note, Warning, Error, Fatal, Sorry
  };
  
  /// isNoteWarningOrExtension - Return true if the unmapped diagnostic level of
  /// the specified diagnostic ID is a Note, Warning, or Extension.
  static bool isNoteWarningOrExtension(unsigned DiagID);

  /// getDiagnosticLevel - Based on the way the client configured the Diagnostic
  /// object, classify the specified diagnostic ID into a Level, consumable by
  /// the DiagnosticClient.
  Level getDiagnosticLevel(unsigned DiagID) const;
  
  /// Report - Issue the message to the client. If the client wants us to stop
  /// compilation, return true, otherwise return false.  DiagID is a member of
  /// the diag::kind enum.  
  bool Report(SourceLocation Pos, unsigned DiagID,
              const std::string &Extra = "");
};

/// DiagnosticClient - This is an abstract interface implemented by clients of
/// the front-end, which formats and prints fully processed diagnostics.
class DiagnosticClient {
public:
  
  virtual ~DiagnosticClient();
  
  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or 
  /// capturing it to a log as needed.  If this returns true, compilation will
  /// be gracefully terminated, otherwise compilation will continue.
  virtual bool HandleDiagnostic(Diagnostic::Level DiagLevel, SourceLocation Pos,
                                diag::kind ID, const std::string &Msg) = 0;
};

}  // end namespace clang
}  // end namespace llvm

#endif
