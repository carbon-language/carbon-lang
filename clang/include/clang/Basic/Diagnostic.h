//===--- Diagnostic.h - C Language Family Diagnostic Handling ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Diagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIAGNOSTIC_H
#define LLVM_CLANG_DIAGNOSTIC_H

#include "clang/Basic/SourceLocation.h"
#include <string>
#include <cassert>

namespace clang {
  class DiagnosticClient;
  class SourceRange;
  class SourceManager;
  
  // Import the diagnostic enums themselves.
  namespace diag {
    class CustomDiagInfo;
    
    /// diag::kind - All of the diagnostics that can be emitted by the frontend.
    enum kind {
#define DIAG(ENUM,FLAGS,DESC) ENUM,
#include "DiagnosticKinds.def"
      NUM_BUILTIN_DIAGNOSTICS
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
public:
  /// Level - The level of the diagnostic, after it has been through mapping.
  enum Level {
    Ignored, Note, Warning, Error, Fatal
  };
  
private:  
  bool IgnoreAllWarnings;     // Ignore all warnings: -w
  bool WarningsAsErrors;      // Treat warnings like errors: 
  bool WarnOnExtensions;      // Enables warnings for gcc extensions: -pedantic.
  bool ErrorOnExtensions;     // Error on extensions: -pedantic-errors.
  DiagnosticClient &Client;

  /// DiagMappings - Mapping information for diagnostics.  Mapping info is
  /// packed into two bits per diagnostic.
  unsigned char DiagMappings[(diag::NUM_BUILTIN_DIAGNOSTICS+3)/4];
  
  /// ErrorOccurred - This is set to true when an error is emitted, and is
  /// sticky.
  bool ErrorOccurred;

  unsigned NumDiagnostics;    // Number of diagnostics reported
  unsigned NumErrors;         // Number of diagnostics that are errors

  /// CustomDiagInfo - Information for uniquing and looking up custom diags.
  diag::CustomDiagInfo *CustomDiagInfo;
public:
  explicit Diagnostic(DiagnosticClient &client);
  ~Diagnostic();
  
  //===--------------------------------------------------------------------===//
  //  Diagnostic characterization methods, used by a client to customize how
  //
  
  DiagnosticClient &getClient() { return Client; };
  
  const DiagnosticClient &getClient() const { return Client; };

  /// setIgnoreAllWarnings - When set to true, any unmapped warnings are
  /// ignored.  If this and WarningsAsErrors are both set, then this one wins.
  void setIgnoreAllWarnings(bool Val) { IgnoreAllWarnings = Val; }
  bool getIgnoreAllWarnings() const { return IgnoreAllWarnings; }
  
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
    assert(Diag < diag::NUM_BUILTIN_DIAGNOSTICS &&
           "Can only map builtin diagnostics");
    assert(isBuiltinNoteWarningOrExtension(Diag) && "Cannot map errors!");
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
  
  /// getCustomDiagID - Return an ID for a diagnostic with the specified message
  /// and level.  If this is the first request for this diagnosic, it is
  /// registered and created, otherwise the existing ID is returned.
  unsigned getCustomDiagID(Level L, const char *Message);
  
  //===--------------------------------------------------------------------===//
  // Diagnostic classification and reporting interfaces.
  //

  /// getDescription - Given a diagnostic ID, return a description of the
  /// issue.
  const char *getDescription(unsigned DiagID);
  
  /// isBuiltinNoteWarningOrExtension - Return true if the unmapped diagnostic
  /// level of the specified diagnostic ID is a Note, Warning, or Extension.
  /// Note that this only works on builtin diagnostics, not custom ones.
  static bool isBuiltinNoteWarningOrExtension(unsigned DiagID);

  /// getDiagnosticLevel - Based on the way the client configured the Diagnostic
  /// object, classify the specified diagnostic ID into a Level, consumable by
  /// the DiagnosticClient.
  Level getDiagnosticLevel(unsigned DiagID) const;
  
  /// Report - Issue the message to the client.  DiagID is a member of the
  /// diag::kind enum.  
  void Report(FullSourceLoc Pos, unsigned DiagID,
              const std::string *Strs = 0, unsigned NumStrs = 0,
              const SourceRange *Ranges = 0, unsigned NumRanges = 0) {
    Report(NULL, Pos, DiagID, Strs, NumStrs, Ranges, NumRanges);
  }                                                                      
  
  /// Report - Issue the message to the client.  DiagID is a member of the
  /// diag::kind enum.  
  void Report(unsigned DiagID,
              const std::string *Strs = 0, unsigned NumStrs = 0,
              const SourceRange *Ranges = 0, unsigned NumRanges = 0) {
    Report(FullSourceLoc(), DiagID, Strs, NumStrs, Ranges, NumRanges);
  }
  
  /// Report - Issue the message to the specified client. 
  ///  DiagID is a member of the diag::kind enum.
  void Report(DiagnosticClient* C, FullSourceLoc Pos, unsigned DiagID,
              const std::string *Strs = 0, unsigned NumStrs = 0,
              const SourceRange *Ranges = 0, unsigned NumRanges = 0);
};

/// DiagnosticClient - This is an abstract interface implemented by clients of
/// the front-end, which formats and prints fully processed diagnostics.
class DiagnosticClient {
public:
  virtual ~DiagnosticClient();

  /// isInSystemHeader - If the client can tell that this is a system header,
  /// return true.
  virtual bool isInSystemHeader(FullSourceLoc Pos) const { return false; }

  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or
  /// capturing it to a log as needed.
  virtual void HandleDiagnostic(Diagnostic &Diags, 
                                Diagnostic::Level DiagLevel,
                                FullSourceLoc Pos,
                                diag::kind ID,
                                const std::string *Strs,
                                unsigned NumStrs,
                                const SourceRange *Ranges, 
                                unsigned NumRanges) = 0;
};

}  // end namespace clang

#endif
