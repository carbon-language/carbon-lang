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
#include <vector>
#include <cassert>

namespace llvm {
  template <typename T> class SmallVectorImpl;
}

namespace clang {
  class DiagnosticBuilder;
  class DiagnosticClient;
  class IdentifierInfo;
  class LangOptions;
  class PartialDiagnostic;
  class SourceRange;
  
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
      DIAG_START_ANALYSIS = DIAG_START_SEMA     + 1000,
      DIAG_UPPER_LIMIT    = DIAG_START_ANALYSIS +  100
    };

    class CustomDiagInfo;
    
    /// diag::kind - All of the diagnostics that can be emitted by the frontend.
    typedef unsigned kind;

    // Get typedefs for common diagnostics.
    enum {
#define DIAG(ENUM,FLAGS,DEFAULT_MAPPING,DESC,GROUP,SFINAE) ENUM,
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
      MAP_WARNING_NO_WERROR = 5
    };
  }
  
/// \brief Annotates a diagnostic with some code that should be
/// inserted, removed, or replaced to fix the problem.
///
/// This kind of hint should be used when we are certain that the
/// introduction, removal, or modification of a particular (small!)
/// amount of code will correct a compilation error. The compiler
/// should also provide full recovery from such errors, such that
/// suppressing the diagnostic output can still result in successful
/// compilation.
class CodeModificationHint {
public:
  /// \brief Tokens that should be removed to correct the error.
  SourceRange RemoveRange;

  /// \brief The location at which we should insert code to correct
  /// the error.
  SourceLocation InsertionLoc;

  /// \brief The actual code to insert at the insertion location, as a
  /// string.
  std::string CodeToInsert;

  /// \brief Empty code modification hint, indicating that no code
  /// modification is known.
  CodeModificationHint() : RemoveRange(), InsertionLoc() { }

  /// \brief Create a code modification hint that inserts the given
  /// code string at a specific location.
  static CodeModificationHint CreateInsertion(SourceLocation InsertionLoc, 
                                              const std::string &Code) {
    CodeModificationHint Hint;
    Hint.InsertionLoc = InsertionLoc;
    Hint.CodeToInsert = Code;
    return Hint;
  }

  /// \brief Create a code modification hint that removes the given
  /// source range.
  static CodeModificationHint CreateRemoval(SourceRange RemoveRange) {
    CodeModificationHint Hint;
    Hint.RemoveRange = RemoveRange;
    return Hint;
  }

  /// \brief Create a code modification hint that replaces the given
  /// source range with the given code string.
  static CodeModificationHint CreateReplacement(SourceRange RemoveRange, 
                                                const std::string &Code) {
    CodeModificationHint Hint;
    Hint.RemoveRange = RemoveRange;
    Hint.InsertionLoc = RemoveRange.getBegin();
    Hint.CodeToInsert = Code;
    return Hint;
  }
};

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
  
  /// ExtensionHandling - How do we handle otherwise-unmapped extension?  This
  /// is controlled by -pedantic and -pedantic-errors.
  enum ExtensionHandling {
    Ext_Ignore, Ext_Warn, Ext_Error
  };
  
  enum ArgumentKind {
    ak_std_string,      // std::string
    ak_c_string,        // const char *
    ak_sint,            // int
    ak_uint,            // unsigned
    ak_identifierinfo,  // IdentifierInfo
    ak_qualtype,        // QualType
    ak_declarationname, // DeclarationName
    ak_nameddecl,       // NamedDecl *
    ak_nestednamespec   // NestedNameSpecifier *
  };

private: 
  unsigned char AllExtensionsSilenced; // Used by __extension__
  bool IgnoreAllWarnings;        // Ignore all warnings: -w
  bool WarningsAsErrors;         // Treat warnings like errors: 
  bool SuppressSystemWarnings;   // Suppress warnings in system headers.
  ExtensionHandling ExtBehavior; // Map extensions onto warnings or errors?
  DiagnosticClient *Client;

  /// DiagMappings - Mapping information for diagnostics.  Mapping info is
  /// packed into four bits per diagnostic.  The low three bits are the mapping
  /// (an instance of diag::Mapping), or zero if unset.  The high bit is set
  /// when the mapping was established as a user mapping.  If the high bit is
  /// clear, then the low bits are set to the default value, and should be
  /// mapped with -pedantic, -Werror, etc.

  typedef std::vector<unsigned char> DiagMappings;
  mutable std::vector<DiagMappings> DiagMappingsStack;

  /// ErrorOccurred / FatalErrorOccurred - This is set to true when an error or
  /// fatal error is emitted, and is sticky.
  bool ErrorOccurred;
  bool FatalErrorOccurred;
  
  /// LastDiagLevel - This is the level of the last diagnostic emitted.  This is
  /// used to emit continuation diagnostics with the same level as the
  /// diagnostic that they follow.
  Diagnostic::Level LastDiagLevel;

  unsigned NumDiagnostics;    // Number of diagnostics reported
  unsigned NumErrors;         // Number of diagnostics that are errors

  /// CustomDiagInfo - Information for uniquing and looking up custom diags.
  diag::CustomDiagInfo *CustomDiagInfo;

  /// ArgToStringFn - A function pointer that converts an opaque diagnostic
  /// argument to a strings.  This takes the modifiers and argument that was
  /// present in the diagnostic.
  /// This is a hack to avoid a layering violation between libbasic and libsema.
  typedef void (*ArgToStringFnTy)(ArgumentKind Kind, intptr_t Val,
                                  const char *Modifier, unsigned ModifierLen,
                                  const char *Argument, unsigned ArgumentLen,
                                  llvm::SmallVectorImpl<char> &Output,
                                  void *Cookie);
  void *ArgToStringCookie;
  ArgToStringFnTy ArgToStringFn;
public:
  explicit Diagnostic(DiagnosticClient *client = 0);
  ~Diagnostic();
  
  //===--------------------------------------------------------------------===//
  //  Diagnostic characterization methods, used by a client to customize how
  //
  
  DiagnosticClient *getClient() { return Client; };
  const DiagnosticClient *getClient() const { return Client; };
    

  /// pushMappings - Copies the current DiagMappings and pushes the new copy 
  /// onto the top of the stack.
  void pushMappings();

  /// popMappings - Pops the current DiagMappings off the top of the stack
  /// causing the new top of the stack to be the active mappings. Returns
  /// true if the pop happens, false if there is only one DiagMapping on the
  /// stack.
  bool popMappings();

  void setClient(DiagnosticClient* client) { Client = client; }

  /// setIgnoreAllWarnings - When set to true, any unmapped warnings are
  /// ignored.  If this and WarningsAsErrors are both set, then this one wins.
  void setIgnoreAllWarnings(bool Val) { IgnoreAllWarnings = Val; }
  bool getIgnoreAllWarnings() const { return IgnoreAllWarnings; }
  
  /// setWarningsAsErrors - When set to true, any warnings reported are issued
  /// as errors.
  void setWarningsAsErrors(bool Val) { WarningsAsErrors = Val; }
  bool getWarningsAsErrors() const { return WarningsAsErrors; }
  
  /// setSuppressSystemWarnings - When set to true mask warnings that
  /// come from system headers.
  void setSuppressSystemWarnings(bool Val) { SuppressSystemWarnings = Val; }
  bool getSuppressSystemWarnings() const { return SuppressSystemWarnings; }

  /// setExtensionHandlingBehavior - This controls whether otherwise-unmapped
  /// extension diagnostics are mapped onto ignore/warning/error.  This
  /// corresponds to the GCC -pedantic and -pedantic-errors option.
  void setExtensionHandlingBehavior(ExtensionHandling H) {
    ExtBehavior = H;
  }
  
  /// AllExtensionsSilenced - This is a counter bumped when an __extension__
  /// block is encountered.  When non-zero, all extension diagnostics are
  /// entirely silenced, no matter how they are mapped.
  void IncrementAllExtensionsSilenced() { ++AllExtensionsSilenced; }
  void DecrementAllExtensionsSilenced() { --AllExtensionsSilenced; }
  bool hasAllExtensionsSilenced() { return AllExtensionsSilenced != 0; }
  
  /// setDiagnosticMapping - This allows the client to specify that certain
  /// warnings are ignored.  Notes can never be mapped, errors can only be
  /// mapped to fatal, and WARNINGs and EXTENSIONs can be mapped arbitrarily.
  void setDiagnosticMapping(diag::kind Diag, diag::Mapping Map) {
    assert(Diag < diag::DIAG_UPPER_LIMIT &&
           "Can only map builtin diagnostics");
    assert((isBuiltinWarningOrExtension(Diag) || Map == diag::MAP_FATAL) &&
           "Cannot map errors!");
    setDiagnosticMappingInternal(Diag, Map, true);
  }
  
  /// setDiagnosticGroupMapping - Change an entire diagnostic group (e.g.
  /// "unknown-pragmas" to have the specified mapping.  This returns true and
  /// ignores the request if "Group" was unknown, false otherwise.
  bool setDiagnosticGroupMapping(const char *Group, diag::Mapping Map);

  bool hasErrorOccurred() const { return ErrorOccurred; }
  bool hasFatalErrorOccurred() const { return FatalErrorOccurred; }

  unsigned getNumErrors() const { return NumErrors; }
  unsigned getNumDiagnostics() const { return NumDiagnostics; }
  
  /// getCustomDiagID - Return an ID for a diagnostic with the specified message
  /// and level.  If this is the first request for this diagnosic, it is
  /// registered and created, otherwise the existing ID is returned.
  unsigned getCustomDiagID(Level L, const char *Message);
  
  
  /// ConvertArgToString - This method converts a diagnostic argument (as an
  /// intptr_t) into the string that represents it.
  void ConvertArgToString(ArgumentKind Kind, intptr_t Val,
                          const char *Modifier, unsigned ModLen,
                          const char *Argument, unsigned ArgLen,
                          llvm::SmallVectorImpl<char> &Output) const {
    ArgToStringFn(Kind, Val, Modifier, ModLen, Argument, ArgLen, Output,
                  ArgToStringCookie);
  }
  
  void SetArgToStringFn(ArgToStringFnTy Fn, void *Cookie) {
    ArgToStringFn = Fn;
    ArgToStringCookie = Cookie;
  }
  
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
  static bool isBuiltinExtensionDiag(unsigned DiagID);
  
  /// getWarningOptionForDiag - Return the lowest-level warning option that
  /// enables the specified diagnostic.  If there is no -Wfoo flag that controls
  /// the diagnostic, this returns null.
  static const char *getWarningOptionForDiag(unsigned DiagID);

  /// \brief Determines whether the given built-in diagnostic ID is
  /// for an error that is suppressed if it occurs during C++ template
  /// argument deduction.
  ///
  /// When an error is suppressed due to SFINAE, the template argument
  /// deduction fails but no diagnostic is emitted. Certain classes of
  /// errors, such as those errors that involve C++ access control,
  /// are not SFINAE errors.
  static bool isBuiltinSFINAEDiag(unsigned DiagID);

  /// getDiagnosticLevel - Based on the way the client configured the Diagnostic
  /// object, classify the specified diagnostic ID into a Level, consumable by
  /// the DiagnosticClient.
  Level getDiagnosticLevel(unsigned DiagID) const;  
  
  /// Report - Issue the message to the client.  @c DiagID is a member of the
  /// @c diag::kind enum.  This actually returns aninstance of DiagnosticBuilder
  /// which emits the diagnostics (through @c ProcessDiag) when it is destroyed.
  /// @c Pos represents the source location associated with the diagnostic,
  /// which can be an invalid location if no position information is available.
  inline DiagnosticBuilder Report(FullSourceLoc Pos, unsigned DiagID);

  /// \brief Clear out the current diagnostic.
  void Clear() { CurDiagID = ~0U; }
  
private:
  /// getDiagnosticMappingInfo - Return the mapping info currently set for the
  /// specified builtin diagnostic.  This returns the high bit encoding, or zero
  /// if the field is completely uninitialized.
  unsigned getDiagnosticMappingInfo(diag::kind Diag) const {
    const DiagMappings &currentMappings = DiagMappingsStack.back();
    return (diag::Mapping)((currentMappings[Diag/2] >> (Diag & 1)*4) & 15);
  }
  
  void setDiagnosticMappingInternal(unsigned DiagId, unsigned Map,
                                    bool isUser) const {
    if (isUser) Map |= 8;  // Set the high bit for user mappings.
    unsigned char &Slot = DiagMappingsStack.back()[DiagId/2];
    unsigned Shift = (DiagId & 1)*4;
    Slot &= ~(15 << Shift);
    Slot |= Map << Shift;
  }
  
  /// getDiagnosticLevel - This is an internal implementation helper used when
  /// DiagClass is already known.
  Level getDiagnosticLevel(unsigned DiagID, unsigned DiagClass) const;

  // This is private state used by DiagnosticBuilder.  We put it here instead of
  // in DiagnosticBuilder in order to keep DiagnosticBuilder a small lightweight
  // object.  This implementation choice means that we can only have one
  // diagnostic "in flight" at a time, but this seems to be a reasonable
  // tradeoff to keep these objects small.  Assertions verify that only one
  // diagnostic is in flight at a time.
  friend class DiagnosticBuilder;
  friend class DiagnosticInfo;

  /// CurDiagLoc - This is the location of the current diagnostic that is in
  /// flight.
  FullSourceLoc CurDiagLoc;
  
  /// CurDiagID - This is the ID of the current diagnostic that is in flight.
  /// This is set to ~0U when there is no diagnostic in flight.
  unsigned CurDiagID;

  enum {
    /// MaxArguments - The maximum number of arguments we can hold. We currently
    /// only support up to 10 arguments (%0-%9).  A single diagnostic with more
    /// than that almost certainly has to be simplified anyway.
    MaxArguments = 10
  };
  
  /// NumDiagArgs - This contains the number of entries in Arguments.
  signed char NumDiagArgs;
  /// NumRanges - This is the number of ranges in the DiagRanges array.
  unsigned char NumDiagRanges;
  /// \brief The number of code modifications hints in the
  /// CodeModificationHints array.
  unsigned char NumCodeModificationHints;

  /// DiagArgumentsKind - This is an array of ArgumentKind::ArgumentKind enum
  /// values, with one for each argument.  This specifies whether the argument
  /// is in DiagArgumentsStr or in DiagArguments.
  unsigned char DiagArgumentsKind[MaxArguments];
  
  /// DiagArgumentsStr - This holds the values of each string argument for the
  /// current diagnostic.  This value is only used when the corresponding
  /// ArgumentKind is ak_std_string.
  std::string DiagArgumentsStr[MaxArguments];

  /// DiagArgumentsVal - The values for the various substitution positions. This
  /// is used when the argument is not an std::string.  The specific value is
  /// mangled into an intptr_t and the intepretation depends on exactly what
  /// sort of argument kind it is.
  intptr_t DiagArgumentsVal[MaxArguments];
  
  /// DiagRanges - The list of ranges added to this diagnostic.  It currently
  /// only support 10 ranges, could easily be extended if needed.
  const SourceRange *DiagRanges[10];
  
  enum { MaxCodeModificationHints = 3 };

  /// CodeModificationHints - If valid, provides a hint with some code
  /// to insert, remove, or modify at a particular position.
  CodeModificationHint CodeModificationHints[MaxCodeModificationHints];

  /// ProcessDiag - This is the method used to report a diagnostic that is
  /// finally fully formed.
  ///
  /// \returns true if the diagnostic was emitted, false if it was
  /// suppressed.
  bool ProcessDiag();
};

//===----------------------------------------------------------------------===//
// DiagnosticBuilder
//===----------------------------------------------------------------------===//

/// DiagnosticBuilder - This is a little helper class used to produce
/// diagnostics.  This is constructed by the Diagnostic::Report method, and
/// allows insertion of extra information (arguments and source ranges) into the
/// currently "in flight" diagnostic.  When the temporary for the builder is
/// destroyed, the diagnostic is issued.
///
/// Note that many of these will be created as temporary objects (many call
/// sites), so we want them to be small and we never want their address taken.
/// This ensures that compilers with somewhat reasonable optimizers will promote
/// the common fields to registers, eliminating increments of the NumArgs field,
/// for example.
class DiagnosticBuilder {
  mutable Diagnostic *DiagObj;
  mutable unsigned NumArgs, NumRanges, NumCodeModificationHints;
  
  void operator=(const DiagnosticBuilder&); // DO NOT IMPLEMENT
  friend class Diagnostic;
  explicit DiagnosticBuilder(Diagnostic *diagObj)
    : DiagObj(diagObj), NumArgs(0), NumRanges(0), 
      NumCodeModificationHints(0) {}

public:  
  /// Copy constructor.  When copied, this "takes" the diagnostic info from the
  /// input and neuters it.
  DiagnosticBuilder(const DiagnosticBuilder &D) {
    DiagObj = D.DiagObj;
    D.DiagObj = 0;
    NumArgs = D.NumArgs;
    NumRanges = D.NumRanges;
    NumCodeModificationHints = D.NumCodeModificationHints;
  }

  /// \brief Simple enumeration value used to give a name to the
  /// suppress-diagnostic constructor.
  enum SuppressKind { Suppress };

  /// \brief Create an empty DiagnosticBuilder object that represents
  /// no actual diagnostic.
  explicit DiagnosticBuilder(SuppressKind) 
    : DiagObj(0), NumArgs(0), NumRanges(0), NumCodeModificationHints(0) { }

  /// \brief Force the diagnostic builder to emit the diagnostic now.
  ///
  /// Once this function has been called, the DiagnosticBuilder object
  /// should not be used again before it is destroyed.
  ///
  /// \returns true if a diagnostic was emitted, false if the
  /// diagnostic was suppressed.
  bool Emit() {
    // If DiagObj is null, then its soul was stolen by the copy ctor
    // or the user called Emit().
    if (DiagObj == 0) return false;

    // When emitting diagnostics, we set the final argument count into
    // the Diagnostic object.
    DiagObj->NumDiagArgs = NumArgs;
    DiagObj->NumDiagRanges = NumRanges;
    DiagObj->NumCodeModificationHints = NumCodeModificationHints;

    // Process the diagnostic, sending the accumulated information to the
    // DiagnosticClient.
    bool Emitted = DiagObj->ProcessDiag();

    // Clear out the current diagnostic object.
    DiagObj->Clear();

    // This diagnostic is dead.
    DiagObj = 0;

    return Emitted;
  }

  /// Destructor - The dtor emits the diagnostic if it hasn't already
  /// been emitted.
  ~DiagnosticBuilder() { Emit(); }
  
  /// Operator bool: conversion of DiagnosticBuilder to bool always returns
  /// true.  This allows is to be used in boolean error contexts like:
  /// return Diag(...);
  operator bool() const { return true; }

  void AddString(const std::string &S) const {
    assert(NumArgs < Diagnostic::MaxArguments &&
           "Too many arguments to diagnostic!");
    if (DiagObj) {
      DiagObj->DiagArgumentsKind[NumArgs] = Diagnostic::ak_std_string;
      DiagObj->DiagArgumentsStr[NumArgs++] = S;
    }
  }
  
  void AddTaggedVal(intptr_t V, Diagnostic::ArgumentKind Kind) const {
    assert(NumArgs < Diagnostic::MaxArguments &&
           "Too many arguments to diagnostic!");
    if (DiagObj) {
      DiagObj->DiagArgumentsKind[NumArgs] = Kind;
      DiagObj->DiagArgumentsVal[NumArgs++] = V;
    }
  }
  
  void AddSourceRange(const SourceRange &R) const {
    assert(NumRanges < 
           sizeof(DiagObj->DiagRanges)/sizeof(DiagObj->DiagRanges[0]) &&
           "Too many arguments to diagnostic!");
    if (DiagObj)
      DiagObj->DiagRanges[NumRanges++] = &R;
  }    

  void AddCodeModificationHint(const CodeModificationHint &Hint) const {
    assert(NumCodeModificationHints < Diagnostic::MaxCodeModificationHints &&
           "Too many code modification hints!");
    if (DiagObj)
      DiagObj->CodeModificationHints[NumCodeModificationHints++] = Hint;
  }
};

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const std::string &S) {
  DB.AddString(S);
  return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const char *Str) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(Str),
                  Diagnostic::ak_c_string);
  return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB, int I) {
  DB.AddTaggedVal(I, Diagnostic::ak_sint);
  return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,bool I) {
  DB.AddTaggedVal(I, Diagnostic::ak_sint);
  return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           unsigned I) {
  DB.AddTaggedVal(I, Diagnostic::ak_uint);
  return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const IdentifierInfo *II) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(II),
                  Diagnostic::ak_identifierinfo);
  return DB;
}
  
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const SourceRange &R) {
  DB.AddSourceRange(R);
  return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const CodeModificationHint &Hint) {
  DB.AddCodeModificationHint(Hint);
  return DB;
}

/// Report - Issue the message to the client.  DiagID is a member of the
/// diag::kind enum.  This actually returns a new instance of DiagnosticBuilder
/// which emits the diagnostics (through ProcessDiag) when it is destroyed.
inline DiagnosticBuilder Diagnostic::Report(FullSourceLoc Loc, unsigned DiagID){
  assert(CurDiagID == ~0U && "Multiple diagnostics in flight at once!");
  CurDiagLoc = Loc;
  CurDiagID = DiagID;
  return DiagnosticBuilder(this);
}

//===----------------------------------------------------------------------===//
// DiagnosticInfo
//===----------------------------------------------------------------------===//
  
/// DiagnosticInfo - This is a little helper class (which is basically a smart
/// pointer that forward info from Diagnostic) that allows clients to enquire
/// about the currently in-flight diagnostic.
class DiagnosticInfo {
  const Diagnostic *DiagObj;
public:
  explicit DiagnosticInfo(const Diagnostic *DO) : DiagObj(DO) {}
  
  const Diagnostic *getDiags() const { return DiagObj; }
  unsigned getID() const { return DiagObj->CurDiagID; }
  const FullSourceLoc &getLocation() const { return DiagObj->CurDiagLoc; }
  
  unsigned getNumArgs() const { return DiagObj->NumDiagArgs; }
  
  /// getArgKind - Return the kind of the specified index.  Based on the kind
  /// of argument, the accessors below can be used to get the value.
  Diagnostic::ArgumentKind getArgKind(unsigned Idx) const {
    assert(Idx < getNumArgs() && "Argument index out of range!");
    return (Diagnostic::ArgumentKind)DiagObj->DiagArgumentsKind[Idx];
  }
  
  /// getArgStdStr - Return the provided argument string specified by Idx.
  const std::string &getArgStdStr(unsigned Idx) const {
    assert(getArgKind(Idx) == Diagnostic::ak_std_string &&
           "invalid argument accessor!");
    return DiagObj->DiagArgumentsStr[Idx];
  }
  
  /// getArgCStr - Return the specified C string argument.
  const char *getArgCStr(unsigned Idx) const {
    assert(getArgKind(Idx) == Diagnostic::ak_c_string &&
           "invalid argument accessor!");
    return reinterpret_cast<const char*>(DiagObj->DiagArgumentsVal[Idx]);
  }
  
  /// getArgSInt - Return the specified signed integer argument.
  int getArgSInt(unsigned Idx) const {
    assert(getArgKind(Idx) == Diagnostic::ak_sint &&
           "invalid argument accessor!");
    return (int)DiagObj->DiagArgumentsVal[Idx];
  }
  
  /// getArgUInt - Return the specified unsigned integer argument.
  unsigned getArgUInt(unsigned Idx) const {
    assert(getArgKind(Idx) == Diagnostic::ak_uint &&
           "invalid argument accessor!");
    return (unsigned)DiagObj->DiagArgumentsVal[Idx];
  }
  
  /// getArgIdentifier - Return the specified IdentifierInfo argument.
  const IdentifierInfo *getArgIdentifier(unsigned Idx) const {
    assert(getArgKind(Idx) == Diagnostic::ak_identifierinfo &&
           "invalid argument accessor!");
    return reinterpret_cast<IdentifierInfo*>(DiagObj->DiagArgumentsVal[Idx]);
  }
  
  /// getRawArg - Return the specified non-string argument in an opaque form.
  intptr_t getRawArg(unsigned Idx) const {
    assert(getArgKind(Idx) != Diagnostic::ak_std_string &&
           "invalid argument accessor!");
    return DiagObj->DiagArgumentsVal[Idx];
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
  
  unsigned getNumCodeModificationHints() const {
    return DiagObj->NumCodeModificationHints;
  }

  const CodeModificationHint &getCodeModificationHint(unsigned Idx) const {
    return DiagObj->CodeModificationHints[Idx];
  }

  const CodeModificationHint *getCodeModificationHints() const {
    return DiagObj->NumCodeModificationHints? 
             &DiagObj->CodeModificationHints[0] : 0;
  }

  /// FormatDiagnostic - Format this diagnostic into a string, substituting the
  /// formal arguments into the %0 slots.  The result is appended onto the Str
  /// array.
  void FormatDiagnostic(llvm::SmallVectorImpl<char> &OutStr) const;
};
  
/// DiagnosticClient - This is an abstract interface implemented by clients of
/// the front-end, which formats and prints fully processed diagnostics.
class DiagnosticClient {
public:
  virtual ~DiagnosticClient();
  
  /// setLangOptions - This is set by clients of diagnostics when they know the
  /// language parameters of the diagnostics that may be sent through.  Note
  /// that this can change over time if a DiagClient has multiple languages sent
  /// through it.  It may also be set to null (e.g. when processing command line
  /// options).
  virtual void setLangOptions(const LangOptions *LO) {}
  
  /// IncludeInDiagnosticCounts - This method (whose default implementation
  ///  returns true) indicates whether the diagnostics handled by this
  ///  DiagnosticClient should be included in the number of diagnostics
  ///  reported by Diagnostic.
  virtual bool IncludeInDiagnosticCounts() const;

  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or
  /// capturing it to a log as needed.
  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info) = 0;
};

}  // end namespace clang

#endif
