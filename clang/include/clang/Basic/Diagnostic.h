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

namespace clang {
  class DiagnosticClient;
  class SourceLocation;
  class SourceRange;
  
  // Import the diagnostic enums themselves.
  namespace diag {
    /// diag::kind - All of the diagnostics that can be emitted by the frontend.
    enum kind {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticKinds.def"
      NUM_DIAGNOSTICS
    };
    
    /// Enum values that allow the client to map NOTEs, WARNINGs, and EXTENSIONs
    /// to either MAP_IGNORE (nothing), MAP_WARNING (emit a warning), MAP_ERROR
    /// (emit as an error), or MAP_DEFAULT (handle the default way).
    enum Mapping {
      MAP_DEFAULT = 0,     //< Do not map this diagnostic.
      MAP_IGNORE  = 1,     //< Map this diagnostic to nothing, ignore it.
      MAP_WARNING = 2,     //< Map this diagnostic to a warning.
      MAP_ERROR   = 3      //< Map this diagnostic to an error.
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

  /// DiagMappings - Mapping information for diagnostics.  Mapping info is
  /// packed into two bits per diagnostic.
  unsigned char DiagMappings[(diag::NUM_DIAGNOSTICS+3)/4];
  
  /// ErrorOccurred - This is set to true when an error is emitted, and is
  /// sticky.
  bool ErrorOccurred;

  unsigned NumDiagnostics;    // Number of diagnostics reported
  unsigned NumErrors;         // Number of diagnostics that are errors
public:
  explicit Diagnostic(DiagnosticClient &client);
  
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

  /// setDiagnosticMapping - This allows the client to specify that certain
  /// warnings are ignored.  Only NOTEs, WARNINGs, and EXTENSIONs can be mapped.
  void setDiagnosticMapping(diag::kind Diag, diag::Mapping Map) {
    assert(isNoteWarningOrExtension(Diag) && "Cannot map errors!");
    unsigned char &Slot = DiagMappings[Diag/4];
    unsigned Bits = (Diag & 3)*2;
    Slot &= ~(3 << Bits);
    Slot |= Map << Bits;
  }

  /// getDiagnosticMapping - Return the mapping currently set for the specified
  /// diagnostic.
  diag::Mapping getDiagnosticMapping(diag::kind Diag) const {
    return (diag::Mapping)((DiagMappings[Diag/4] >> (Diag & 3)*2) & 3);
  }
  
  bool hasErrorOccurred() const { return ErrorOccurred; }

  unsigned getNumErrors() const { return NumErrors; }
  unsigned getNumDiagnostics() const { return NumDiagnostics; }
  
  //===--------------------------------------------------------------------===//
  // Diagnostic classification and reporting interfaces.
  //

  /// getDescription - Given a diagnostic ID, return a description of the
  /// issue.
  static const char *getDescription(unsigned DiagID);
  
  /// Level - The level of the diagnostic 
  enum Level {
    Ignored, Note, Warning, Error, Fatal, Sorry
  };
  
  /// isNoteWarningOrExtension - Return true if the unmapped diagnostic level of
  /// the specified diagnostic ID is a Note, Warning, or Extension.
  static bool isNoteWarningOrExtension(unsigned DiagID);

  /// getDiagnosticLevel - Based on the way the client configured the Diagnostic
  /// object, classify the specified diagnostic ID into a Level, consumable by
  /// the DiagnosticClient.
  Level getDiagnosticLevel(unsigned DiagID) const;
  
  /// Report - Issue the message to the client.  DiagID is a member of the
  /// diag::kind enum.  
  void Report(SourceLocation Pos, unsigned DiagID,
              const std::string *Strs = 0, unsigned NumStrs = 0,
              const SourceRange *Ranges = 0, unsigned NumRanges = 0);
};

/// DiagnosticClient - This is an abstract interface implemented by clients of
/// the front-end, which formats and prints fully processed diagnostics.
class DiagnosticClient {
public:
  virtual ~DiagnosticClient();

  /// IgnoreDiagnostic - If the client wants to ignore this diagnostic, then
  /// return true.
  virtual bool IgnoreDiagnostic(Diagnostic::Level DiagLevel,
                                SourceLocation Pos) = 0;

  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or
  /// capturing it to a log as needed.
  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel, SourceLocation Pos,
                                diag::kind ID, const std::string *Strs,
                                unsigned NumStrs, const SourceRange *Ranges, 
                                unsigned NumRanges) = 0;
};

}  // end namespace clang

#endif
