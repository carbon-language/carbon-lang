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

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/type_traits.h"

#include <vector>
#include <list>

namespace clang {
  class DiagnosticClient;
  class DiagnosticBuilder;
  class IdentifierInfo;
  class DeclContext;
  class LangOptions;
  class Preprocessor;
  class DiagnosticErrorTrap;
  class StoredDiagnostic;

/// \brief Annotates a diagnostic with some code that should be
/// inserted, removed, or replaced to fix the problem.
///
/// This kind of hint should be used when we are certain that the
/// introduction, removal, or modification of a particular (small!)
/// amount of code will correct a compilation error. The compiler
/// should also provide full recovery from such errors, such that
/// suppressing the diagnostic output can still result in successful
/// compilation.
class FixItHint {
public:
  /// \brief Code that should be replaced to correct the error. Empty for an
  /// insertion hint.
  CharSourceRange RemoveRange;

  /// \brief The actual code to insert at the insertion location, as a
  /// string.
  std::string CodeToInsert;

  /// \brief Empty code modification hint, indicating that no code
  /// modification is known.
  FixItHint() : RemoveRange() { }

  bool isNull() const {
    return !RemoveRange.isValid();
  }
  
  /// \brief Create a code modification hint that inserts the given
  /// code string at a specific location.
  static FixItHint CreateInsertion(SourceLocation InsertionLoc,
                                   StringRef Code) {
    FixItHint Hint;
    Hint.RemoveRange =
      CharSourceRange(SourceRange(InsertionLoc, InsertionLoc), false);
    Hint.CodeToInsert = Code;
    return Hint;
  }

  /// \brief Create a code modification hint that removes the given
  /// source range.
  static FixItHint CreateRemoval(CharSourceRange RemoveRange) {
    FixItHint Hint;
    Hint.RemoveRange = RemoveRange;
    return Hint;
  }
  static FixItHint CreateRemoval(SourceRange RemoveRange) {
    return CreateRemoval(CharSourceRange::getTokenRange(RemoveRange));
  }
  
  /// \brief Create a code modification hint that replaces the given
  /// source range with the given code string.
  static FixItHint CreateReplacement(CharSourceRange RemoveRange,
                                     StringRef Code) {
    FixItHint Hint;
    Hint.RemoveRange = RemoveRange;
    Hint.CodeToInsert = Code;
    return Hint;
  }
  
  static FixItHint CreateReplacement(SourceRange RemoveRange,
                                     StringRef Code) {
    return CreateReplacement(CharSourceRange::getTokenRange(RemoveRange), Code);
  }
};

/// Diagnostic - This concrete class is used by the front-end to report
/// problems and issues.  It massages the diagnostics (e.g. handling things like
/// "report warnings as errors" and passes them off to the DiagnosticClient for
/// reporting to the user. Diagnostic is tied to one translation unit and
/// one SourceManager.
class Diagnostic : public llvm::RefCountedBase<Diagnostic> {
public:
  /// Level - The level of the diagnostic, after it has been through mapping.
  enum Level {
    Ignored = DiagnosticIDs::Ignored,
    Note = DiagnosticIDs::Note,
    Warning = DiagnosticIDs::Warning,
    Error = DiagnosticIDs::Error,
    Fatal = DiagnosticIDs::Fatal
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
    ak_nestednamespec,  // NestedNameSpecifier *
    ak_declcontext      // DeclContext *
  };

  /// Specifies which overload candidates to display when overload resolution
  /// fails.
  enum OverloadsShown {
    Ovl_All,  ///< Show all overloads.
    Ovl_Best  ///< Show just the "best" overload candidates.
  };

  /// ArgumentValue - This typedef represents on argument value, which is a
  /// union discriminated by ArgumentKind, with a value.
  typedef std::pair<ArgumentKind, intptr_t> ArgumentValue;

private:
  unsigned char AllExtensionsSilenced; // Used by __extension__
  bool IgnoreAllWarnings;        // Ignore all warnings: -w
  bool WarningsAsErrors;         // Treat warnings like errors.
  bool EnableAllWarnings;        // Enable all warnings.
  bool ErrorsAsFatal;            // Treat errors like fatal errors.
  bool SuppressSystemWarnings;   // Suppress warnings in system headers.
  bool SuppressAllDiagnostics;   // Suppress all diagnostics.
  OverloadsShown ShowOverloads;  // Which overload candidates to show.
  unsigned ErrorLimit;           // Cap of # errors emitted, 0 -> no limit.
  unsigned TemplateBacktraceLimit; // Cap on depth of template backtrace stack,
                                   // 0 -> no limit.
  ExtensionHandling ExtBehavior; // Map extensions onto warnings or errors?
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> Diags;
  DiagnosticClient *Client;
  bool OwnsDiagClient;
  SourceManager *SourceMgr;

  /// \brief Mapping information for diagnostics.  Mapping info is
  /// packed into four bits per diagnostic.  The low three bits are the mapping
  /// (an instance of diag::Mapping), or zero if unset.  The high bit is set
  /// when the mapping was established as a user mapping.  If the high bit is
  /// clear, then the low bits are set to the default value, and should be
  /// mapped with -pedantic, -Werror, etc.
  ///
  /// A new DiagState is created and kept around when diagnostic pragmas modify
  /// the state so that we know what is the diagnostic state at any given
  /// source location.
  class DiagState {
    llvm::DenseMap<unsigned, unsigned> DiagMap;

  public:
    typedef llvm::DenseMap<unsigned, unsigned>::const_iterator iterator;

    void setMapping(diag::kind Diag, unsigned Map) { DiagMap[Diag] = Map; }

    diag::Mapping getMapping(diag::kind Diag) const {
      iterator I = DiagMap.find(Diag);
      if (I != DiagMap.end())
        return (diag::Mapping)I->second;
      return diag::Mapping();
    }

    iterator begin() const { return DiagMap.begin(); }
    iterator end() const { return DiagMap.end(); }
  };

  /// \brief Keeps and automatically disposes all DiagStates that we create.
  std::list<DiagState> DiagStates;

  /// \brief Represents a point in source where the diagnostic state was
  /// modified because of a pragma. 'Loc' can be null if the point represents
  /// the diagnostic state modifications done through the command-line.
  struct DiagStatePoint {
    DiagState *State;
    FullSourceLoc Loc;
    DiagStatePoint(DiagState *State, FullSourceLoc Loc)
      : State(State), Loc(Loc) { } 
    
    bool operator<(const DiagStatePoint &RHS) const {
      // If Loc is invalid it means it came from <command-line>, in which case
      // we regard it as coming before any valid source location.
      if (RHS.Loc.isInvalid())
        return false;
      if (Loc.isInvalid())
        return true;
      return Loc.isBeforeInTranslationUnitThan(RHS.Loc);
    }
  };

  /// \brief A vector of all DiagStatePoints representing changes in diagnostic
  /// state due to diagnostic pragmas. The vector is always sorted according to
  /// the SourceLocation of the DiagStatePoint.
  typedef std::vector<DiagStatePoint> DiagStatePointsTy;
  mutable DiagStatePointsTy DiagStatePoints;

  /// \brief Keeps the DiagState that was active during each diagnostic 'push'
  /// so we can get back at it when we 'pop'.
  std::vector<DiagState *> DiagStateOnPushStack;

  DiagState *GetCurDiagState() const {
    assert(!DiagStatePoints.empty());
    return DiagStatePoints.back().State;
  }

  void PushDiagStatePoint(DiagState *State, SourceLocation L) {
    FullSourceLoc Loc(L, *SourceMgr);
    // Make sure that DiagStatePoints is always sorted according to Loc.
    assert((Loc.isValid() || DiagStatePoints.empty()) &&
           "Adding invalid loc point after another point");
    assert((Loc.isInvalid() || DiagStatePoints.empty() ||
            DiagStatePoints.back().Loc.isInvalid() ||
            DiagStatePoints.back().Loc.isBeforeInTranslationUnitThan(Loc)) &&
           "Previous point loc comes after or is the same as new one");
    DiagStatePoints.push_back(DiagStatePoint(State,
                                             FullSourceLoc(Loc, *SourceMgr)));
  }

  /// \brief Finds the DiagStatePoint that contains the diagnostic state of
  /// the given source location.
  DiagStatePointsTy::iterator GetDiagStatePointForLoc(SourceLocation Loc) const;

  /// ErrorOccurred / FatalErrorOccurred - This is set to true when an error or
  /// fatal error is emitted, and is sticky.
  bool ErrorOccurred;
  bool FatalErrorOccurred;

  /// \brief Indicates that an unrecoverable error has occurred.
  bool UnrecoverableErrorOccurred;
  
  /// \brief Counts for DiagnosticErrorTrap to check whether an error occurred
  /// during a parsing section, e.g. during parsing a function.
  unsigned TrapNumErrorsOccurred;
  unsigned TrapNumUnrecoverableErrorsOccurred;

  /// LastDiagLevel - This is the level of the last diagnostic emitted.  This is
  /// used to emit continuation diagnostics with the same level as the
  /// diagnostic that they follow.
  DiagnosticIDs::Level LastDiagLevel;

  unsigned NumWarnings;       // Number of warnings reported
  unsigned NumErrors;         // Number of errors reported
  unsigned NumErrorsSuppressed; // Number of errors suppressed

  /// ArgToStringFn - A function pointer that converts an opaque diagnostic
  /// argument to a strings.  This takes the modifiers and argument that was
  /// present in the diagnostic.
  ///
  /// The PrevArgs array (whose length is NumPrevArgs) indicates the previous
  /// arguments formatted for this diagnostic.  Implementations of this function
  /// can use this information to avoid redundancy across arguments.
  ///
  /// This is a hack to avoid a layering violation between libbasic and libsema.
  typedef void (*ArgToStringFnTy)(
      ArgumentKind Kind, intptr_t Val,
      const char *Modifier, unsigned ModifierLen,
      const char *Argument, unsigned ArgumentLen,
      const ArgumentValue *PrevArgs,
      unsigned NumPrevArgs,
      SmallVectorImpl<char> &Output,
      void *Cookie,
      SmallVectorImpl<intptr_t> &QualTypeVals);
  void *ArgToStringCookie;
  ArgToStringFnTy ArgToStringFn;

  /// \brief ID of the "delayed" diagnostic, which is a (typically
  /// fatal) diagnostic that had to be delayed because it was found
  /// while emitting another diagnostic.
  unsigned DelayedDiagID;

  /// \brief First string argument for the delayed diagnostic.
  std::string DelayedDiagArg1;

  /// \brief Second string argument for the delayed diagnostic.
  std::string DelayedDiagArg2;

public:
  explicit Diagnostic(const llvm::IntrusiveRefCntPtr<DiagnosticIDs> &Diags,
                      DiagnosticClient *client = 0,
                      bool ShouldOwnClient = true);
  ~Diagnostic();

  const llvm::IntrusiveRefCntPtr<DiagnosticIDs> &getDiagnosticIDs() const {
    return Diags;
  }

  DiagnosticClient *getClient() { return Client; }
  const DiagnosticClient *getClient() const { return Client; }

  /// \brief Return the current diagnostic client along with ownership of that
  /// client.
  DiagnosticClient *takeClient() {
    OwnsDiagClient = false;
    return Client;
  }

  bool hasSourceManager() const { return SourceMgr != 0; }
  SourceManager &getSourceManager() const {
    assert(SourceMgr && "SourceManager not set!");
    return *SourceMgr;
  }
  void setSourceManager(SourceManager *SrcMgr) { SourceMgr = SrcMgr; }

  //===--------------------------------------------------------------------===//
  //  Diagnostic characterization methods, used by a client to customize how
  //  diagnostics are emitted.
  //

  /// pushMappings - Copies the current DiagMappings and pushes the new copy
  /// onto the top of the stack.
  void pushMappings(SourceLocation Loc);

  /// popMappings - Pops the current DiagMappings off the top of the stack
  /// causing the new top of the stack to be the active mappings. Returns
  /// true if the pop happens, false if there is only one DiagMapping on the
  /// stack.
  bool popMappings(SourceLocation Loc);

  /// \brief Set the diagnostic client associated with this diagnostic object.
  ///
  /// \param ShouldOwnClient true if the diagnostic object should take
  /// ownership of \c client.
  void setClient(DiagnosticClient *client, bool ShouldOwnClient = true);

  /// setErrorLimit - Specify a limit for the number of errors we should
  /// emit before giving up.  Zero disables the limit.
  void setErrorLimit(unsigned Limit) { ErrorLimit = Limit; }
  
  /// \brief Specify the maximum number of template instantiation
  /// notes to emit along with a given diagnostic.
  void setTemplateBacktraceLimit(unsigned Limit) {
    TemplateBacktraceLimit = Limit;
  }
  
  /// \brief Retrieve the maximum number of template instantiation
  /// nodes to emit along with a given diagnostic.
  unsigned getTemplateBacktraceLimit() const {
    return TemplateBacktraceLimit;
  }
  
  /// setIgnoreAllWarnings - When set to true, any unmapped warnings are
  /// ignored.  If this and WarningsAsErrors are both set, then this one wins.
  void setIgnoreAllWarnings(bool Val) { IgnoreAllWarnings = Val; }
  bool getIgnoreAllWarnings() const { return IgnoreAllWarnings; }

  /// setEnableAllWarnings - When set to true, any unmapped ignored warnings
  /// are no longer ignored.  If this and IgnoreAllWarnings are both set,
  /// then that one wins.
  void setEnableAllWarnings(bool Val) { EnableAllWarnings = Val; }
  bool getEnableAllWarnngs() const { return EnableAllWarnings; }
  
  /// setWarningsAsErrors - When set to true, any warnings reported are issued
  /// as errors.
  void setWarningsAsErrors(bool Val) { WarningsAsErrors = Val; }
  bool getWarningsAsErrors() const { return WarningsAsErrors; }

  /// setErrorsAsFatal - When set to true, any error reported is made a
  /// fatal error.
  void setErrorsAsFatal(bool Val) { ErrorsAsFatal = Val; }
  bool getErrorsAsFatal() const { return ErrorsAsFatal; }

  /// setSuppressSystemWarnings - When set to true mask warnings that
  /// come from system headers.
  void setSuppressSystemWarnings(bool Val) { SuppressSystemWarnings = Val; }
  bool getSuppressSystemWarnings() const { return SuppressSystemWarnings; }

  /// \brief Suppress all diagnostics, to silence the front end when we 
  /// know that we don't want any more diagnostics to be passed along to the
  /// client
  void setSuppressAllDiagnostics(bool Val = true) { 
    SuppressAllDiagnostics = Val; 
  }
  bool getSuppressAllDiagnostics() const { return SuppressAllDiagnostics; }
  
  /// \brief Specify which overload candidates to show when overload resolution
  /// fails.  By default, we show all candidates.
  void setShowOverloads(OverloadsShown Val) {
    ShowOverloads = Val;
  }
  OverloadsShown getShowOverloads() const { return ShowOverloads; }
  
  /// \brief Pretend that the last diagnostic issued was ignored. This can
  /// be used by clients who suppress diagnostics themselves.
  void setLastDiagnosticIgnored() {
    LastDiagLevel = DiagnosticIDs::Ignored;
  }
  
  /// setExtensionHandlingBehavior - This controls whether otherwise-unmapped
  /// extension diagnostics are mapped onto ignore/warning/error.  This
  /// corresponds to the GCC -pedantic and -pedantic-errors option.
  void setExtensionHandlingBehavior(ExtensionHandling H) {
    ExtBehavior = H;
  }
  ExtensionHandling getExtensionHandlingBehavior() const { return ExtBehavior; }

  /// AllExtensionsSilenced - This is a counter bumped when an __extension__
  /// block is encountered.  When non-zero, all extension diagnostics are
  /// entirely silenced, no matter how they are mapped.
  void IncrementAllExtensionsSilenced() { ++AllExtensionsSilenced; }
  void DecrementAllExtensionsSilenced() { --AllExtensionsSilenced; }
  bool hasAllExtensionsSilenced() { return AllExtensionsSilenced != 0; }

  /// \brief This allows the client to specify that certain
  /// warnings are ignored.  Notes can never be mapped, errors can only be
  /// mapped to fatal, and WARNINGs and EXTENSIONs can be mapped arbitrarily.
  ///
  /// \param Loc The source location that this change of diagnostic state should
  /// take affect. It can be null if we are setting the latest state.
  void setDiagnosticMapping(diag::kind Diag, diag::Mapping Map,
                            SourceLocation Loc);

  /// setDiagnosticGroupMapping - Change an entire diagnostic group (e.g.
  /// "unknown-pragmas" to have the specified mapping.  This returns true and
  /// ignores the request if "Group" was unknown, false otherwise.
  ///
  /// 'Loc' is the source location that this change of diagnostic state should
  /// take affect. It can be null if we are setting the state from command-line.
  bool setDiagnosticGroupMapping(StringRef Group, diag::Mapping Map,
                                 SourceLocation Loc = SourceLocation()) {
    return Diags->setDiagnosticGroupMapping(Group, Map, Loc, *this);
  }

  bool hasErrorOccurred() const { return ErrorOccurred; }
  bool hasFatalErrorOccurred() const { return FatalErrorOccurred; }
  
  /// \brief Determine whether any kind of unrecoverable error has occurred.
  bool hasUnrecoverableErrorOccurred() const {
    return FatalErrorOccurred || UnrecoverableErrorOccurred;
  }
  
  unsigned getNumWarnings() const { return NumWarnings; }

  void setNumWarnings(unsigned NumWarnings) {
    this->NumWarnings = NumWarnings;
  }

  /// getCustomDiagID - Return an ID for a diagnostic with the specified message
  /// and level.  If this is the first request for this diagnosic, it is
  /// registered and created, otherwise the existing ID is returned.
  unsigned getCustomDiagID(Level L, StringRef Message) {
    return Diags->getCustomDiagID((DiagnosticIDs::Level)L, Message);
  }

  /// ConvertArgToString - This method converts a diagnostic argument (as an
  /// intptr_t) into the string that represents it.
  void ConvertArgToString(ArgumentKind Kind, intptr_t Val,
                          const char *Modifier, unsigned ModLen,
                          const char *Argument, unsigned ArgLen,
                          const ArgumentValue *PrevArgs, unsigned NumPrevArgs,
                          SmallVectorImpl<char> &Output,
                          SmallVectorImpl<intptr_t> &QualTypeVals) const {
    ArgToStringFn(Kind, Val, Modifier, ModLen, Argument, ArgLen,
                  PrevArgs, NumPrevArgs, Output, ArgToStringCookie,
                  QualTypeVals);
  }

  void SetArgToStringFn(ArgToStringFnTy Fn, void *Cookie) {
    ArgToStringFn = Fn;
    ArgToStringCookie = Cookie;
  }

  /// \brief Reset the state of the diagnostic object to its initial 
  /// configuration.
  void Reset();
  
  //===--------------------------------------------------------------------===//
  // Diagnostic classification and reporting interfaces.
  //

  /// \brief Based on the way the client configured the Diagnostic
  /// object, classify the specified diagnostic ID into a Level, consumable by
  /// the DiagnosticClient.
  ///
  /// \param Loc The source location we are interested in finding out the
  /// diagnostic state. Can be null in order to query the latest state.
  Level getDiagnosticLevel(unsigned DiagID, SourceLocation Loc,
                           diag::Mapping *mapping = 0) const {
    return (Level)Diags->getDiagnosticLevel(DiagID, Loc, *this, mapping);
  }

  /// Report - Issue the message to the client.  @c DiagID is a member of the
  /// @c diag::kind enum.  This actually returns aninstance of DiagnosticBuilder
  /// which emits the diagnostics (through @c ProcessDiag) when it is destroyed.
  /// @c Pos represents the source location associated with the diagnostic,
  /// which can be an invalid location if no position information is available.
  inline DiagnosticBuilder Report(SourceLocation Pos, unsigned DiagID);
  inline DiagnosticBuilder Report(unsigned DiagID);

  void Report(const StoredDiagnostic &storedDiag);

  /// \brief Determine whethere there is already a diagnostic in flight.
  bool isDiagnosticInFlight() const { return CurDiagID != ~0U; }

  /// \brief Set the "delayed" diagnostic that will be emitted once
  /// the current diagnostic completes.
  ///
  ///  If a diagnostic is already in-flight but the front end must
  ///  report a problem (e.g., with an inconsistent file system
  ///  state), this routine sets a "delayed" diagnostic that will be
  ///  emitted after the current diagnostic completes. This should
  ///  only be used for fatal errors detected at inconvenient
  ///  times. If emitting a delayed diagnostic causes a second delayed
  ///  diagnostic to be introduced, that second delayed diagnostic
  ///  will be ignored.
  ///
  /// \param DiagID The ID of the diagnostic being delayed.
  ///
  /// \param Arg1 A string argument that will be provided to the
  /// diagnostic. A copy of this string will be stored in the
  /// Diagnostic object itself.
  ///
  /// \param Arg2 A string argument that will be provided to the
  /// diagnostic. A copy of this string will be stored in the
  /// Diagnostic object itself.
  void SetDelayedDiagnostic(unsigned DiagID, StringRef Arg1 = "",
                            StringRef Arg2 = "");
  
  /// \brief Clear out the current diagnostic.
  void Clear() { CurDiagID = ~0U; }

private:
  /// \brief Report the delayed diagnostic.
  void ReportDelayed();


  /// getDiagnosticMappingInfo - Return the mapping info currently set for the
  /// specified builtin diagnostic.  This returns the high bit encoding, or zero
  /// if the field is completely uninitialized.
  diag::Mapping getDiagnosticMappingInfo(diag::kind Diag,
                                         DiagState *State) const {
    return State->getMapping(Diag);
  }

  void setDiagnosticMappingInternal(unsigned DiagId, unsigned Map,
                                    DiagState *State,
                                    bool isUser, bool isPragma) const {
    if (isUser) Map |= 8;  // Set the high bit for user mappings.
    if (isPragma) Map |= 0x10;  // Set the bit for diagnostic pragma mappings.
    State->setMapping((diag::kind)DiagId, Map);
  }

  // This is private state used by DiagnosticBuilder.  We put it here instead of
  // in DiagnosticBuilder in order to keep DiagnosticBuilder a small lightweight
  // object.  This implementation choice means that we can only have one
  // diagnostic "in flight" at a time, but this seems to be a reasonable
  // tradeoff to keep these objects small.  Assertions verify that only one
  // diagnostic is in flight at a time.
  friend class DiagnosticIDs;
  friend class DiagnosticBuilder;
  friend class DiagnosticInfo;
  friend class PartialDiagnostic;
  friend class DiagnosticErrorTrap;
  
  /// CurDiagLoc - This is the location of the current diagnostic that is in
  /// flight.
  SourceLocation CurDiagLoc;

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
  /// FixItHints array.
  unsigned char NumFixItHints;

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
  /// mangled into an intptr_t and the interpretation depends on exactly what
  /// sort of argument kind it is.
  intptr_t DiagArgumentsVal[MaxArguments];

  /// DiagRanges - The list of ranges added to this diagnostic.  It currently
  /// only support 10 ranges, could easily be extended if needed.
  CharSourceRange DiagRanges[10];

  enum { MaxFixItHints = 6 };

  /// FixItHints - If valid, provides a hint with some code
  /// to insert, remove, or modify at a particular position.
  FixItHint FixItHints[MaxFixItHints];

  /// ProcessDiag - This is the method used to report a diagnostic that is
  /// finally fully formed.
  ///
  /// \returns true if the diagnostic was emitted, false if it was
  /// suppressed.
  bool ProcessDiag() {
    return Diags->ProcessDiag(*this);
  }

  friend class ASTReader;
  friend class ASTWriter;
};

/// \brief RAII class that determines when any errors have occurred
/// between the time the instance was created and the time it was
/// queried.
class DiagnosticErrorTrap {
  Diagnostic &Diag;
  unsigned NumErrors;
  unsigned NumUnrecoverableErrors;

public:
  explicit DiagnosticErrorTrap(Diagnostic &Diag)
    : Diag(Diag) { reset(); }

  /// \brief Determine whether any errors have occurred since this
  /// object instance was created.
  bool hasErrorOccurred() const {
    return Diag.TrapNumErrorsOccurred > NumErrors;
  }

  /// \brief Determine whether any unrecoverable errors have occurred since this
  /// object instance was created.
  bool hasUnrecoverableErrorOccurred() const {
    return Diag.TrapNumUnrecoverableErrorsOccurred > NumUnrecoverableErrors;
  }

  // Set to initial state of "no errors occurred".
  void reset() {
    NumErrors = Diag.TrapNumErrorsOccurred;
    NumUnrecoverableErrors = Diag.TrapNumUnrecoverableErrorsOccurred;
  }
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
  mutable unsigned NumArgs, NumRanges, NumFixItHints;

  void operator=(const DiagnosticBuilder&); // DO NOT IMPLEMENT
  friend class Diagnostic;
  explicit DiagnosticBuilder(Diagnostic *diagObj)
    : DiagObj(diagObj), NumArgs(0), NumRanges(0), NumFixItHints(0) {}

  friend class PartialDiagnostic;

protected:
  void FlushCounts();
  
public:
  /// Copy constructor.  When copied, this "takes" the diagnostic info from the
  /// input and neuters it.
  DiagnosticBuilder(const DiagnosticBuilder &D) {
    DiagObj = D.DiagObj;
    D.DiagObj = 0;
    NumArgs = D.NumArgs;
    NumRanges = D.NumRanges;
    NumFixItHints = D.NumFixItHints;
  }

  /// \brief Simple enumeration value used to give a name to the
  /// suppress-diagnostic constructor.
  enum SuppressKind { Suppress };

  /// \brief Create an empty DiagnosticBuilder object that represents
  /// no actual diagnostic.
  explicit DiagnosticBuilder(SuppressKind)
    : DiagObj(0), NumArgs(0), NumRanges(0), NumFixItHints(0) { }

  /// \brief Force the diagnostic builder to emit the diagnostic now.
  ///
  /// Once this function has been called, the DiagnosticBuilder object
  /// should not be used again before it is destroyed.
  ///
  /// \returns true if a diagnostic was emitted, false if the
  /// diagnostic was suppressed.
  bool Emit();

  /// Destructor - The dtor emits the diagnostic if it hasn't already
  /// been emitted.
  ~DiagnosticBuilder() { Emit(); }

  /// isActive - Determine whether this diagnostic is still active.
  bool isActive() const { return DiagObj != 0; }

  /// \brief Retrieve the active diagnostic ID.
  ///
  /// \pre \c isActive()
  unsigned getDiagID() const {
    assert(isActive() && "Diagnostic is inactive");
    return DiagObj->CurDiagID;
  }
  
  /// \brief Clear out the current diagnostic.
  void Clear() { DiagObj = 0; }
  
  /// Operator bool: conversion of DiagnosticBuilder to bool always returns
  /// true.  This allows is to be used in boolean error contexts like:
  /// return Diag(...);
  operator bool() const { return true; }

  void AddString(StringRef S) const {
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

  void AddSourceRange(const CharSourceRange &R) const {
    assert(NumRanges <
           sizeof(DiagObj->DiagRanges)/sizeof(DiagObj->DiagRanges[0]) &&
           "Too many arguments to diagnostic!");
    if (DiagObj)
      DiagObj->DiagRanges[NumRanges++] = R;
  }

  void AddFixItHint(const FixItHint &Hint) const {
    assert(NumFixItHints < Diagnostic::MaxFixItHints &&
           "Too many fix-it hints!");
    if (DiagObj)
      DiagObj->FixItHints[NumFixItHints++] = Hint;
  }
};

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           StringRef S) {
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

// Adds a DeclContext to the diagnostic. The enable_if template magic is here
// so that we only match those arguments that are (statically) DeclContexts;
// other arguments that derive from DeclContext (e.g., RecordDecls) will not
// match.
template<typename T>
inline
typename llvm::enable_if<llvm::is_same<T, DeclContext>, 
                         const DiagnosticBuilder &>::type
operator<<(const DiagnosticBuilder &DB, T *DC) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(DC),
                  Diagnostic::ak_declcontext);
  return DB;
}
  
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const SourceRange &R) {
  DB.AddSourceRange(CharSourceRange::getTokenRange(R));
  return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const CharSourceRange &R) {
  DB.AddSourceRange(R);
  return DB;
}
  
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const FixItHint &Hint) {
  DB.AddFixItHint(Hint);
  return DB;
}

/// Report - Issue the message to the client.  DiagID is a member of the
/// diag::kind enum.  This actually returns a new instance of DiagnosticBuilder
/// which emits the diagnostics (through ProcessDiag) when it is destroyed.
inline DiagnosticBuilder Diagnostic::Report(SourceLocation Loc,
                                            unsigned DiagID){
  assert(CurDiagID == ~0U && "Multiple diagnostics in flight at once!");
  CurDiagLoc = Loc;
  CurDiagID = DiagID;
  return DiagnosticBuilder(this);
}
inline DiagnosticBuilder Diagnostic::Report(unsigned DiagID) {
  return Report(SourceLocation(), DiagID);
}

//===----------------------------------------------------------------------===//
// DiagnosticInfo
//===----------------------------------------------------------------------===//

/// DiagnosticInfo - This is a little helper class (which is basically a smart
/// pointer that forward info from Diagnostic) that allows clients to enquire
/// about the currently in-flight diagnostic.
class DiagnosticInfo {
  const Diagnostic *DiagObj;
  StringRef StoredDiagMessage;
public:
  explicit DiagnosticInfo(const Diagnostic *DO) : DiagObj(DO) {}
  DiagnosticInfo(const Diagnostic *DO, StringRef storedDiagMessage)
    : DiagObj(DO), StoredDiagMessage(storedDiagMessage) {}

  const Diagnostic *getDiags() const { return DiagObj; }
  unsigned getID() const { return DiagObj->CurDiagID; }
  const SourceLocation &getLocation() const { return DiagObj->CurDiagLoc; }
  bool hasSourceManager() const { return DiagObj->hasSourceManager(); }
  SourceManager &getSourceManager() const { return DiagObj->getSourceManager();}

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

  const CharSourceRange &getRange(unsigned Idx) const {
    assert(Idx < DiagObj->NumDiagRanges && "Invalid diagnostic range index!");
    return DiagObj->DiagRanges[Idx];
  }

  unsigned getNumFixItHints() const {
    return DiagObj->NumFixItHints;
  }

  const FixItHint &getFixItHint(unsigned Idx) const {
    return DiagObj->FixItHints[Idx];
  }

  const FixItHint *getFixItHints() const {
    return DiagObj->NumFixItHints?
             &DiagObj->FixItHints[0] : 0;
  }

  /// FormatDiagnostic - Format this diagnostic into a string, substituting the
  /// formal arguments into the %0 slots.  The result is appended onto the Str
  /// array.
  void FormatDiagnostic(SmallVectorImpl<char> &OutStr) const;

  /// FormatDiagnostic - Format the given format-string into the
  /// output buffer using the arguments stored in this diagnostic.
  void FormatDiagnostic(const char *DiagStr, const char *DiagEnd,
                        SmallVectorImpl<char> &OutStr) const;
};

/**
 * \brief Represents a diagnostic in a form that can be retained until its 
 * corresponding source manager is destroyed. 
 */
class StoredDiagnostic {
  unsigned ID;
  Diagnostic::Level Level;
  FullSourceLoc Loc;
  std::string Message;
  std::vector<CharSourceRange> Ranges;
  std::vector<FixItHint> FixIts;

public:
  StoredDiagnostic();
  StoredDiagnostic(Diagnostic::Level Level, const DiagnosticInfo &Info);
  StoredDiagnostic(Diagnostic::Level Level, unsigned ID, 
                   StringRef Message);
  StoredDiagnostic(Diagnostic::Level Level, unsigned ID, 
                   StringRef Message, FullSourceLoc Loc,
                   ArrayRef<CharSourceRange> Ranges,
                   ArrayRef<FixItHint> Fixits);
  ~StoredDiagnostic();

  /// \brief Evaluates true when this object stores a diagnostic.
  operator bool() const { return Message.size() > 0; }

  unsigned getID() const { return ID; }
  Diagnostic::Level getLevel() const { return Level; }
  const FullSourceLoc &getLocation() const { return Loc; }
  StringRef getMessage() const { return Message; }

  void setLocation(FullSourceLoc Loc) { this->Loc = Loc; }

  typedef std::vector<CharSourceRange>::const_iterator range_iterator;
  range_iterator range_begin() const { return Ranges.begin(); }
  range_iterator range_end() const { return Ranges.end(); }
  unsigned range_size() const { return Ranges.size(); }

  typedef std::vector<FixItHint>::const_iterator fixit_iterator;
  fixit_iterator fixit_begin() const { return FixIts.begin(); }
  fixit_iterator fixit_end() const { return FixIts.end(); }
  unsigned fixit_size() const { return FixIts.size(); }
};

/// DiagnosticClient - This is an abstract interface implemented by clients of
/// the front-end, which formats and prints fully processed diagnostics.
class DiagnosticClient {
protected:
  unsigned NumWarnings;       // Number of warnings reported
  unsigned NumErrors;         // Number of errors reported
  
public:
  DiagnosticClient() : NumWarnings(0), NumErrors(0) { }

  unsigned getNumErrors() const { return NumErrors; }
  unsigned getNumWarnings() const { return NumWarnings; }

  virtual ~DiagnosticClient();

  /// BeginSourceFile - Callback to inform the diagnostic client that processing
  /// of a source file is beginning.
  ///
  /// Note that diagnostics may be emitted outside the processing of a source
  /// file, for example during the parsing of command line options. However,
  /// diagnostics with source range information are required to only be emitted
  /// in between BeginSourceFile() and EndSourceFile().
  ///
  /// \arg LO - The language options for the source file being processed.
  /// \arg PP - The preprocessor object being used for the source; this optional
  /// and may not be present, for example when processing AST source files.
  virtual void BeginSourceFile(const LangOptions &LangOpts,
                               const Preprocessor *PP = 0) {}

  /// EndSourceFile - Callback to inform the diagnostic client that processing
  /// of a source file has ended. The diagnostic client should assume that any
  /// objects made available via \see BeginSourceFile() are inaccessible.
  virtual void EndSourceFile() {}

  /// IncludeInDiagnosticCounts - This method (whose default implementation
  /// returns true) indicates whether the diagnostics handled by this
  /// DiagnosticClient should be included in the number of diagnostics reported
  /// by Diagnostic.
  virtual bool IncludeInDiagnosticCounts() const;

  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or
  /// capturing it to a log as needed.
  ///
  /// Default implementation just keeps track of the total number of warnings
  /// and errors.
  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info);
};

}  // end namespace clang

#endif
