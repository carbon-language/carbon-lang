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
  class DiagnosticInfo;
  
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
  bool SuppressSystemWarnings;// Suppress warnings in system headers.
  DiagnosticClient *Client;

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
  explicit Diagnostic(DiagnosticClient *client = 0);
  ~Diagnostic();
  
  //===--------------------------------------------------------------------===//
  //  Diagnostic characterization methods, used by a client to customize how
  //
  
  DiagnosticClient *getClient() { return Client; };
  const DiagnosticClient *getClient() const { return Client; };
    
  void setClient(DiagnosticClient* client) { Client = client; }

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

  /// setSuppressSystemWarnings - When set to true mask warnings that
  /// come from system headers.
  void setSuppressSystemWarnings(bool Val) { SuppressSystemWarnings = Val; }
  bool getSuppressSystemWarnings() const { return SuppressSystemWarnings; }

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
  const char *getDescription(unsigned DiagID) const;
  
  /// isBuiltinNoteWarningOrExtension - Return true if the unmapped diagnostic
  /// level of the specified diagnostic ID is a Note, Warning, or Extension.
  /// Note that this only works on builtin diagnostics, not custom ones.
  static bool isBuiltinNoteWarningOrExtension(unsigned DiagID);

  /// getDiagnosticLevel - Based on the way the client configured the Diagnostic
  /// object, classify the specified diagnostic ID into a Level, consumable by
  /// the DiagnosticClient.
  Level getDiagnosticLevel(unsigned DiagID) const;
  
  
  /// Report - Issue the message to the client.  DiagID is a member of the
  /// diag::kind enum.  This actually returns a new instance of DiagnosticInfo
  /// which emits the diagnostics (through ProcessDiag) when it is destroyed.
  inline DiagnosticInfo Report(FullSourceLoc Pos, unsigned DiagID);
  
private:
  // This is private state used by DiagnosticInfo.  We put it here instead of
  // in DiagnosticInfo in order to keep DiagnosticInfo a small light-weight
  // object.  This implementation choice means that we can only have one
  // diagnostic "in flight" at a time, but this seems to be a reasonable
  // tradeoff to keep these objects small.  Assertions verify that only one
  // diagnostic is in flight at a time.
  friend class DiagnosticInfo;
  
  /// DiagArguments - The values for the various substitution positions.  It
  /// currently only support 10 arguments (%0-%9).
  std::string DiagArguments[10];
  /// DiagRanges - The list of ranges added to this diagnostic.  It currently
  /// only support 10 ranges, could easily be extended if needed.
  const SourceRange *DiagRanges[10];
  
  /// NumDiagArgs - This is set to -1 when no diag is in flight.  Otherwise it
  /// is the number of entries in Arguments.
  signed char NumDiagArgs;
  /// NumRanges - This is the number of ranges in the DiagRanges array.
  unsigned char NumDiagRanges;
  
  /// ProcessDiag - This is the method used to report a diagnostic that is
  /// finally fully formed.
  void ProcessDiag(const DiagnosticInfo &Info);
};
  
/// DiagnosticInfo - This is a little helper class used to produce diagnostics.
/// This is constructed with an ID and location, and then has some number of
/// arguments (for %0 substitution) and SourceRanges added to it with the
/// overloaded operator<<.  Once it is destroyed, it emits the diagnostic with
/// the accumulated information.
///
/// Note that many of these will be created as temporary objects (many call
/// sites), so we want them to be small to reduce stack space usage etc.  For
/// this reason, we stick state in the Diagnostic class, see the comment there
/// for more info.
class DiagnosticInfo {
  mutable Diagnostic *DiagObj;
  FullSourceLoc Loc;
  unsigned DiagID;
  void operator=(const DiagnosticInfo&); // DO NOT IMPLEMENT
public:
  DiagnosticInfo(Diagnostic *diagObj, FullSourceLoc loc, unsigned diagID) :
    DiagObj(diagObj), Loc(loc), DiagID(diagID) {
    if (DiagObj == 0) return;
    assert(DiagObj->NumDiagArgs == -1 &&
           "Multiple diagnostics in flight at once!");
    DiagObj->NumDiagArgs = DiagObj->NumDiagRanges = 0;
  }
  
  /// Copy constructor.  When copied, this "takes" the diagnostic info from the
  /// input and neuters it.
  DiagnosticInfo(const DiagnosticInfo &D) {
    DiagObj = D.DiagObj;
    Loc = D.Loc;
    DiagID = D.DiagID;
    D.DiagObj = 0;
  }
  
  /// Destructor - The dtor emits the diagnostic.
  ~DiagnosticInfo() {
    // If DiagObj is null, then its soul was stolen by the copy ctor.
    if (!DiagObj) return;
    
    DiagObj->ProcessDiag(*this);

    // This diagnostic is no longer in flight.
    DiagObj->NumDiagArgs = -1;
  }
  
  const Diagnostic *getDiags() const { return DiagObj; }
  unsigned getID() const { return DiagID; }
  const FullSourceLoc &getLocation() const { return Loc; }
  
  /// Operator bool: conversion of DiagnosticInfo to bool always returns true.
  /// This allows is to be used in boolean error contexts like:
  /// return Diag(...);
  operator bool() const { return true; }

  unsigned getNumArgs() const { return DiagObj->NumDiagArgs; }
  
  /// getArgStr - Return the provided argument string specified by Idx.
  const std::string &getArgStr(unsigned Idx) const {
    assert((signed char)Idx < DiagObj->NumDiagArgs &&
           "Argument out of range!");
    return DiagObj->DiagArguments[Idx];
  }
  
  /// getNumRanges - Return the number of source ranges associated with this
  /// diagnostic.
  unsigned getNumRanges() const {
    return DiagObj->NumDiagRanges;
  }
  
  const SourceRange &getRange(unsigned Idx) const {
    assert(Idx < DiagObj->NumDiagRanges && "Invalid diagnostic range index!");
    return *DiagObj->DiagRanges[Idx];
  }
  
  DiagnosticInfo &operator<<(const std::string &S) {
    assert((unsigned)DiagObj->NumDiagArgs < 
           sizeof(DiagObj->DiagArguments)/sizeof(DiagObj->DiagArguments[0]) &&
           "Too many arguments to diagnostic!");
    DiagObj->DiagArguments[DiagObj->NumDiagArgs++] = S;
    return *this;
  }
  
  DiagnosticInfo &operator<<(const SourceRange &R) {
    assert((unsigned)DiagObj->NumDiagArgs < 
           sizeof(DiagObj->DiagRanges)/sizeof(DiagObj->DiagRanges[0]) &&
           "Too many arguments to diagnostic!");
    DiagObj->DiagRanges[DiagObj->NumDiagRanges++] = &R;
    return *this;
  }
  
};


/// Report - Issue the message to the client.  DiagID is a member of the
/// diag::kind enum.  This actually returns a new instance of DiagnosticInfo
/// which emits the diagnostics (through ProcessDiag) when it is destroyed.
inline DiagnosticInfo Diagnostic::Report(FullSourceLoc Pos, unsigned DiagID) {
  DiagnosticInfo D(this, Pos, DiagID);
  return D;
}
  

/// DiagnosticClient - This is an abstract interface implemented by clients of
/// the front-end, which formats and prints fully processed diagnostics.
class DiagnosticClient {
protected:
  std::string FormatDiagnostic(const DiagnosticInfo &Info);
public:
  virtual ~DiagnosticClient();

  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or
  /// capturing it to a log as needed.
  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info) = 0;
};

}  // end namespace clang

#endif
