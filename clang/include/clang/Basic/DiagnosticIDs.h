//===--- DiagnosticIDs.h - Diagnostic IDs Handling --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Diagnostic IDs-related interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIAGNOSTICIDS_H
#define LLVM_CLANG_DIAGNOSTICIDS_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
  class Diagnostic;
  class SourceLocation;

  // Import the diagnostic enums themselves.
  namespace diag {
    // Start position for diagnostics.
    enum {
      DIAG_START_DRIVER   =                        300,
      DIAG_START_FRONTEND = DIAG_START_DRIVER   +  100,
      DIAG_START_LEX      = DIAG_START_FRONTEND +  100,
      DIAG_START_PARSE    = DIAG_START_LEX      +  300,
      DIAG_START_AST      = DIAG_START_PARSE    +  300,
      DIAG_START_SEMA     = DIAG_START_AST      +  100,
      DIAG_START_ANALYSIS = DIAG_START_SEMA     + 1500,
      DIAG_UPPER_LIMIT    = DIAG_START_ANALYSIS +  100
    };

    class CustomDiagInfo;

    /// diag::kind - All of the diagnostics that can be emitted by the frontend.
    typedef unsigned kind;

    // Get typedefs for common diagnostics.
    enum {
#define DIAG(ENUM,FLAGS,DEFAULT_MAPPING,DESC,GROUP,SFINAE,CATEGORY) ENUM,
#include "clang/Basic/DiagnosticCommonKinds.inc"
      NUM_BUILTIN_COMMON_DIAGNOSTICS
#undef DIAG
    };

    /// Enum values that allow the client to map NOTEs, WARNINGs, and EXTENSIONs
    /// to either MAP_IGNORE (nothing), MAP_WARNING (emit a warning), MAP_ERROR
    /// (emit as an error).  It allows clients to map errors to
    /// MAP_ERROR/MAP_DEFAULT or MAP_FATAL (stop emitting diagnostics after this
    /// one).
    enum Mapping {
      // NOTE: 0 means "uncomputed".
      MAP_IGNORE  = 1,     //< Map this diagnostic to nothing, ignore it.
      MAP_WARNING = 2,     //< Map this diagnostic to a warning.
      MAP_ERROR   = 3,     //< Map this diagnostic to an error.
      MAP_FATAL   = 4,     //< Map this diagnostic to a fatal error.

      /// Map this diagnostic to "warning", but make it immune to -Werror.  This
      /// happens when you specify -Wno-error=foo.
      MAP_WARNING_NO_WERROR = 5,
      /// Map this diagnostic to "error", but make it immune to -Wfatal-errors.
      /// This happens for -Wno-fatal-errors=foo.
      MAP_ERROR_NO_WFATAL = 6
    };
  }

/// \brief Used for handling and querying diagnostic IDs. Can be used and shared
/// by multiple Diagnostics for multiple translation units.
class DiagnosticIDs : public llvm::RefCountedBase<DiagnosticIDs> {
public:
  /// Level - The level of the diagnostic, after it has been through mapping.
  enum Level {
    Ignored, Note, Warning, Error, Fatal
  };

private:
  /// CustomDiagInfo - Information for uniquing and looking up custom diags.
  diag::CustomDiagInfo *CustomDiagInfo;

public:
  DiagnosticIDs();
  ~DiagnosticIDs();

  /// getCustomDiagID - Return an ID for a diagnostic with the specified message
  /// and level.  If this is the first request for this diagnosic, it is
  /// registered and created, otherwise the existing ID is returned.
  unsigned getCustomDiagID(Level L, llvm::StringRef Message);

  //===--------------------------------------------------------------------===//
  // Diagnostic classification and reporting interfaces.
  //

  /// getDescription - Given a diagnostic ID, return a description of the
  /// issue.
  const char *getDescription(unsigned DiagID) const;

  /// isNoteWarningOrExtension - Return true if the unmapped diagnostic
  /// level of the specified diagnostic ID is a Warning or Extension.
  /// This only works on builtin diagnostics, not custom ones, and is not legal to
  /// call on NOTEs.
  static bool isBuiltinWarningOrExtension(unsigned DiagID);

  /// \brief Determine whether the given built-in diagnostic ID is a
  /// Note.
  static bool isBuiltinNote(unsigned DiagID);

  /// isBuiltinExtensionDiag - Determine whether the given built-in diagnostic
  /// ID is for an extension of some sort.
  ///
  static bool isBuiltinExtensionDiag(unsigned DiagID) {
    bool ignored;
    return isBuiltinExtensionDiag(DiagID, ignored);
  }
  
  /// isBuiltinExtensionDiag - Determine whether the given built-in diagnostic
  /// ID is for an extension of some sort.  This also returns EnabledByDefault,
  /// which is set to indicate whether the diagnostic is ignored by default (in
  /// which case -pedantic enables it) or treated as a warning/error by default.
  ///
  static bool isBuiltinExtensionDiag(unsigned DiagID, bool &EnabledByDefault);
  

  /// getWarningOptionForDiag - Return the lowest-level warning option that
  /// enables the specified diagnostic.  If there is no -Wfoo flag that controls
  /// the diagnostic, this returns null.
  static const char *getWarningOptionForDiag(unsigned DiagID);

  /// getWarningOptionForDiag - Return the category number that a specified
  /// DiagID belongs to, or 0 if no category.
  static unsigned getCategoryNumberForDiag(unsigned DiagID);

  /// getCategoryNameFromID - Given a category ID, return the name of the
  /// category.
  static const char *getCategoryNameFromID(unsigned CategoryID);
  
  /// \brief Enumeration describing how the the emission of a diagnostic should
  /// be treated when it occurs during C++ template argument deduction.
  enum SFINAEResponse {
    /// \brief The diagnostic should not be reported, but it should cause
    /// template argument deduction to fail.
    ///
    /// The vast majority of errors that occur during template argument 
    /// deduction fall into this category.
    SFINAE_SubstitutionFailure,
    
    /// \brief The diagnostic should be suppressed entirely.
    ///
    /// Warnings generally fall into this category.
    SFINAE_Suppress,
    
    /// \brief The diagnostic should be reported.
    ///
    /// The diagnostic should be reported. Various fatal errors (e.g., 
    /// template instantiation depth exceeded) fall into this category.
    SFINAE_Report
  };
  
  /// \brief Determines whether the given built-in diagnostic ID is
  /// for an error that is suppressed if it occurs during C++ template
  /// argument deduction.
  ///
  /// When an error is suppressed due to SFINAE, the template argument
  /// deduction fails but no diagnostic is emitted. Certain classes of
  /// errors, such as those errors that involve C++ access control,
  /// are not SFINAE errors.
  static SFINAEResponse getDiagnosticSFINAEResponse(unsigned DiagID);

private:
  /// setDiagnosticGroupMapping - Change an entire diagnostic group (e.g.
  /// "unknown-pragmas" to have the specified mapping.  This returns true and
  /// ignores the request if "Group" was unknown, false otherwise.
  bool setDiagnosticGroupMapping(const char *Group, diag::Mapping Map,
                                 SourceLocation Loc, Diagnostic &Diag) const;

  /// \brief Based on the way the client configured the Diagnostic
  /// object, classify the specified diagnostic ID into a Level, consumable by
  /// the DiagnosticClient.
  ///
  /// \param Loc The source location we are interested in finding out the
  /// diagnostic state. Can be null in order to query the latest state.
  DiagnosticIDs::Level getDiagnosticLevel(unsigned DiagID, SourceLocation Loc,
                                          const Diagnostic &Diag) const;

  /// getDiagnosticLevel - This is an internal implementation helper used when
  /// DiagClass is already known.
  DiagnosticIDs::Level getDiagnosticLevel(unsigned DiagID,
                                          unsigned DiagClass,
                                          SourceLocation Loc,
                                          const Diagnostic &Diag) const;

  /// ProcessDiag - This is the method used to report a diagnostic that is
  /// finally fully formed.
  ///
  /// \returns true if the diagnostic was emitted, false if it was
  /// suppressed.
  bool ProcessDiag(Diagnostic &Diag) const;

  friend class Diagnostic;
};

}  // end namespace clang

#endif
