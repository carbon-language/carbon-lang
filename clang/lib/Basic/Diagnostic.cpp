//===--- Diagnostic.cpp - C Language Family Diagnostic Handling -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Diagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

static void DummyArgToStringFn(Diagnostic::ArgumentKind AK, intptr_t QT,
                               const char *Modifier, unsigned ML,
                               const char *Argument, unsigned ArgLen,
                               const Diagnostic::ArgumentValue *PrevArgs,
                               unsigned NumPrevArgs,
                               llvm::SmallVectorImpl<char> &Output,
                               void *Cookie) {
  const char *Str = "<can't format argument>";
  Output.append(Str, Str+strlen(Str));
}


Diagnostic::Diagnostic(const llvm::IntrusiveRefCntPtr<DiagnosticIDs> &diags,
                       DiagnosticClient *client, bool ShouldOwnClient)
  : Diags(diags), Client(client), OwnsDiagClient(ShouldOwnClient),
    SourceMgr(0) {
  ArgToStringFn = DummyArgToStringFn;
  ArgToStringCookie = 0;

  AllExtensionsSilenced = 0;
  IgnoreAllWarnings = false;
  WarningsAsErrors = false;
  ErrorsAsFatal = false;
  SuppressSystemWarnings = false;
  SuppressAllDiagnostics = false;
  ShowOverloads = Ovl_All;
  ExtBehavior = Ext_Ignore;

  ErrorLimit = 0;
  TemplateBacktraceLimit = 0;

  // Set all mappings to 'unset'.
  DiagMappingsStack.clear();
  DiagMappingsStack.push_back(DiagMappings());

  Reset();
}

Diagnostic::~Diagnostic() {
  if (OwnsDiagClient)
    delete Client;
}


void Diagnostic::pushMappings() {
  // Avoids undefined behavior when the stack has to resize.
  DiagMappingsStack.reserve(DiagMappingsStack.size() + 1);
  DiagMappingsStack.push_back(DiagMappingsStack.back());
}

bool Diagnostic::popMappings() {
  if (DiagMappingsStack.size() == 1)
    return false;

  DiagMappingsStack.pop_back();
  return true;
}

void Diagnostic::Reset() {
  ErrorOccurred = false;
  FatalErrorOccurred = false;
  
  NumWarnings = 0;
  NumErrors = 0;
  NumErrorsSuppressed = 0;
  CurDiagID = ~0U;
  // Set LastDiagLevel to an "unset" state. If we set it to 'Ignored', notes
  // using a Diagnostic associated to a translation unit that follow
  // diagnostics from a Diagnostic associated to anoter t.u. will not be
  // displayed.
  LastDiagLevel = (DiagnosticIDs::Level)-1;
  DelayedDiagID = 0;
}

void Diagnostic::SetDelayedDiagnostic(unsigned DiagID, llvm::StringRef Arg1,
                                      llvm::StringRef Arg2) {
  if (DelayedDiagID)
    return;

  DelayedDiagID = DiagID;
  DelayedDiagArg1 = Arg1.str();
  DelayedDiagArg2 = Arg2.str();
}

void Diagnostic::ReportDelayed() {
  Report(DelayedDiagID) << DelayedDiagArg1 << DelayedDiagArg2;
  DelayedDiagID = 0;
  DelayedDiagArg1.clear();
  DelayedDiagArg2.clear();
}

void DiagnosticBuilder::FlushCounts() {
  DiagObj->NumDiagArgs = NumArgs;
  DiagObj->NumDiagRanges = NumRanges;
  DiagObj->NumFixItHints = NumFixItHints;
}

bool DiagnosticBuilder::Emit() {
  // If DiagObj is null, then its soul was stolen by the copy ctor
  // or the user called Emit().
  if (DiagObj == 0) return false;

  // When emitting diagnostics, we set the final argument count into
  // the Diagnostic object.
  FlushCounts();

  // Process the diagnostic, sending the accumulated information to the
  // DiagnosticClient.
  bool Emitted = DiagObj->ProcessDiag();

  // Clear out the current diagnostic object.
  unsigned DiagID = DiagObj->CurDiagID;
  DiagObj->Clear();

  // If there was a delayed diagnostic, emit it now.
  if (DiagObj->DelayedDiagID && DiagObj->DelayedDiagID != DiagID)
    DiagObj->ReportDelayed();

  // This diagnostic is dead.
  DiagObj = 0;

  return Emitted;
}


DiagnosticClient::~DiagnosticClient() {}

void DiagnosticClient::HandleDiagnostic(Diagnostic::Level DiagLevel,
                                        const DiagnosticInfo &Info) {
  if (!IncludeInDiagnosticCounts())
    return;

  if (DiagLevel == Diagnostic::Warning)
    ++NumWarnings;
  else if (DiagLevel >= Diagnostic::Error)
    ++NumErrors;
}

/// ModifierIs - Return true if the specified modifier matches specified string.
template <std::size_t StrLen>
static bool ModifierIs(const char *Modifier, unsigned ModifierLen,
                       const char (&Str)[StrLen]) {
  return StrLen-1 == ModifierLen && !memcmp(Modifier, Str, StrLen-1);
}

/// ScanForward - Scans forward, looking for the given character, skipping
/// nested clauses and escaped characters.
static const char *ScanFormat(const char *I, const char *E, char Target) {
  unsigned Depth = 0;

  for ( ; I != E; ++I) {
    if (Depth == 0 && *I == Target) return I;
    if (Depth != 0 && *I == '}') Depth--;

    if (*I == '%') {
      I++;
      if (I == E) break;

      // Escaped characters get implicitly skipped here.

      // Format specifier.
      if (!isdigit(*I) && !ispunct(*I)) {
        for (I++; I != E && !isdigit(*I) && *I != '{'; I++) ;
        if (I == E) break;
        if (*I == '{')
          Depth++;
      }
    }
  }
  return E;
}

/// HandleSelectModifier - Handle the integer 'select' modifier.  This is used
/// like this:  %select{foo|bar|baz}2.  This means that the integer argument
/// "%2" has a value from 0-2.  If the value is 0, the diagnostic prints 'foo'.
/// If the value is 1, it prints 'bar'.  If it has the value 2, it prints 'baz'.
/// This is very useful for certain classes of variant diagnostics.
static void HandleSelectModifier(const DiagnosticInfo &DInfo, unsigned ValNo,
                                 const char *Argument, unsigned ArgumentLen,
                                 llvm::SmallVectorImpl<char> &OutStr) {
  const char *ArgumentEnd = Argument+ArgumentLen;

  // Skip over 'ValNo' |'s.
  while (ValNo) {
    const char *NextVal = ScanFormat(Argument, ArgumentEnd, '|');
    assert(NextVal != ArgumentEnd && "Value for integer select modifier was"
           " larger than the number of options in the diagnostic string!");
    Argument = NextVal+1;  // Skip this string.
    --ValNo;
  }

  // Get the end of the value.  This is either the } or the |.
  const char *EndPtr = ScanFormat(Argument, ArgumentEnd, '|');

  // Recursively format the result of the select clause into the output string.
  DInfo.FormatDiagnostic(Argument, EndPtr, OutStr);
}

/// HandleIntegerSModifier - Handle the integer 's' modifier.  This adds the
/// letter 's' to the string if the value is not 1.  This is used in cases like
/// this:  "you idiot, you have %4 parameter%s4!".
static void HandleIntegerSModifier(unsigned ValNo,
                                   llvm::SmallVectorImpl<char> &OutStr) {
  if (ValNo != 1)
    OutStr.push_back('s');
}

/// HandleOrdinalModifier - Handle the integer 'ord' modifier.  This
/// prints the ordinal form of the given integer, with 1 corresponding
/// to the first ordinal.  Currently this is hard-coded to use the
/// English form.
static void HandleOrdinalModifier(unsigned ValNo,
                                  llvm::SmallVectorImpl<char> &OutStr) {
  assert(ValNo != 0 && "ValNo must be strictly positive!");

  llvm::raw_svector_ostream Out(OutStr);

  // We could use text forms for the first N ordinals, but the numeric
  // forms are actually nicer in diagnostics because they stand out.
  Out << ValNo;

  // It is critically important that we do this perfectly for
  // user-written sequences with over 100 elements.
  switch (ValNo % 100) {
  case 11:
  case 12:
  case 13:
    Out << "th"; return;
  default:
    switch (ValNo % 10) {
    case 1: Out << "st"; return;
    case 2: Out << "nd"; return;
    case 3: Out << "rd"; return;
    default: Out << "th"; return;
    }
  }
}


/// PluralNumber - Parse an unsigned integer and advance Start.
static unsigned PluralNumber(const char *&Start, const char *End) {
  // Programming 101: Parse a decimal number :-)
  unsigned Val = 0;
  while (Start != End && *Start >= '0' && *Start <= '9') {
    Val *= 10;
    Val += *Start - '0';
    ++Start;
  }
  return Val;
}

/// TestPluralRange - Test if Val is in the parsed range. Modifies Start.
static bool TestPluralRange(unsigned Val, const char *&Start, const char *End) {
  if (*Start != '[') {
    unsigned Ref = PluralNumber(Start, End);
    return Ref == Val;
  }

  ++Start;
  unsigned Low = PluralNumber(Start, End);
  assert(*Start == ',' && "Bad plural expression syntax: expected ,");
  ++Start;
  unsigned High = PluralNumber(Start, End);
  assert(*Start == ']' && "Bad plural expression syntax: expected )");
  ++Start;
  return Low <= Val && Val <= High;
}

/// EvalPluralExpr - Actual expression evaluator for HandlePluralModifier.
static bool EvalPluralExpr(unsigned ValNo, const char *Start, const char *End) {
  // Empty condition?
  if (*Start == ':')
    return true;

  while (1) {
    char C = *Start;
    if (C == '%') {
      // Modulo expression
      ++Start;
      unsigned Arg = PluralNumber(Start, End);
      assert(*Start == '=' && "Bad plural expression syntax: expected =");
      ++Start;
      unsigned ValMod = ValNo % Arg;
      if (TestPluralRange(ValMod, Start, End))
        return true;
    } else {
      assert((C == '[' || (C >= '0' && C <= '9')) &&
             "Bad plural expression syntax: unexpected character");
      // Range expression
      if (TestPluralRange(ValNo, Start, End))
        return true;
    }

    // Scan for next or-expr part.
    Start = std::find(Start, End, ',');
    if (Start == End)
      break;
    ++Start;
  }
  return false;
}

/// HandlePluralModifier - Handle the integer 'plural' modifier. This is used
/// for complex plural forms, or in languages where all plurals are complex.
/// The syntax is: %plural{cond1:form1|cond2:form2|:form3}, where condn are
/// conditions that are tested in order, the form corresponding to the first
/// that applies being emitted. The empty condition is always true, making the
/// last form a default case.
/// Conditions are simple boolean expressions, where n is the number argument.
/// Here are the rules.
/// condition  := expression | empty
/// empty      :=                             -> always true
/// expression := numeric [',' expression]    -> logical or
/// numeric    := range                       -> true if n in range
///             | '%' number '=' range        -> true if n % number in range
/// range      := number
///             | '[' number ',' number ']'   -> ranges are inclusive both ends
///
/// Here are some examples from the GNU gettext manual written in this form:
/// English:
/// {1:form0|:form1}
/// Latvian:
/// {0:form2|%100=11,%10=0,%10=[2,9]:form1|:form0}
/// Gaeilge:
/// {1:form0|2:form1|:form2}
/// Romanian:
/// {1:form0|0,%100=[1,19]:form1|:form2}
/// Lithuanian:
/// {%10=0,%100=[10,19]:form2|%10=1:form0|:form1}
/// Russian (requires repeated form):
/// {%100=[11,14]:form2|%10=1:form0|%10=[2,4]:form1|:form2}
/// Slovak
/// {1:form0|[2,4]:form1|:form2}
/// Polish (requires repeated form):
/// {1:form0|%100=[10,20]:form2|%10=[2,4]:form1|:form2}
static void HandlePluralModifier(const DiagnosticInfo &DInfo, unsigned ValNo,
                                 const char *Argument, unsigned ArgumentLen,
                                 llvm::SmallVectorImpl<char> &OutStr) {
  const char *ArgumentEnd = Argument + ArgumentLen;
  while (1) {
    assert(Argument < ArgumentEnd && "Plural expression didn't match.");
    const char *ExprEnd = Argument;
    while (*ExprEnd != ':') {
      assert(ExprEnd != ArgumentEnd && "Plural missing expression end");
      ++ExprEnd;
    }
    if (EvalPluralExpr(ValNo, Argument, ExprEnd)) {
      Argument = ExprEnd + 1;
      ExprEnd = ScanFormat(Argument, ArgumentEnd, '|');

      // Recursively format the result of the plural clause into the
      // output string.
      DInfo.FormatDiagnostic(Argument, ExprEnd, OutStr);
      return;
    }
    Argument = ScanFormat(Argument, ArgumentEnd - 1, '|') + 1;
  }
}


/// FormatDiagnostic - Format this diagnostic into a string, substituting the
/// formal arguments into the %0 slots.  The result is appended onto the Str
/// array.
void DiagnosticInfo::
FormatDiagnostic(llvm::SmallVectorImpl<char> &OutStr) const {
  const char *DiagStr = getDiags()->getDiagnosticIDs()->getDescription(getID());
  const char *DiagEnd = DiagStr+strlen(DiagStr);

  FormatDiagnostic(DiagStr, DiagEnd, OutStr);
}

void DiagnosticInfo::
FormatDiagnostic(const char *DiagStr, const char *DiagEnd,
                 llvm::SmallVectorImpl<char> &OutStr) const {

  /// FormattedArgs - Keep track of all of the arguments formatted by
  /// ConvertArgToString and pass them into subsequent calls to
  /// ConvertArgToString, allowing the implementation to avoid redundancies in
  /// obvious cases.
  llvm::SmallVector<Diagnostic::ArgumentValue, 8> FormattedArgs;
  
  while (DiagStr != DiagEnd) {
    if (DiagStr[0] != '%') {
      // Append non-%0 substrings to Str if we have one.
      const char *StrEnd = std::find(DiagStr, DiagEnd, '%');
      OutStr.append(DiagStr, StrEnd);
      DiagStr = StrEnd;
      continue;
    } else if (ispunct(DiagStr[1])) {
      OutStr.push_back(DiagStr[1]);  // %% -> %.
      DiagStr += 2;
      continue;
    }

    // Skip the %.
    ++DiagStr;

    // This must be a placeholder for a diagnostic argument.  The format for a
    // placeholder is one of "%0", "%modifier0", or "%modifier{arguments}0".
    // The digit is a number from 0-9 indicating which argument this comes from.
    // The modifier is a string of digits from the set [-a-z]+, arguments is a
    // brace enclosed string.
    const char *Modifier = 0, *Argument = 0;
    unsigned ModifierLen = 0, ArgumentLen = 0;

    // Check to see if we have a modifier.  If so eat it.
    if (!isdigit(DiagStr[0])) {
      Modifier = DiagStr;
      while (DiagStr[0] == '-' ||
             (DiagStr[0] >= 'a' && DiagStr[0] <= 'z'))
        ++DiagStr;
      ModifierLen = DiagStr-Modifier;

      // If we have an argument, get it next.
      if (DiagStr[0] == '{') {
        ++DiagStr; // Skip {.
        Argument = DiagStr;

        DiagStr = ScanFormat(DiagStr, DiagEnd, '}');
        assert(DiagStr != DiagEnd && "Mismatched {}'s in diagnostic string!");
        ArgumentLen = DiagStr-Argument;
        ++DiagStr;  // Skip }.
      }
    }

    assert(isdigit(*DiagStr) && "Invalid format for argument in diagnostic");
    unsigned ArgNo = *DiagStr++ - '0';

    Diagnostic::ArgumentKind Kind = getArgKind(ArgNo);
    
    switch (Kind) {
    // ---- STRINGS ----
    case Diagnostic::ak_std_string: {
      const std::string &S = getArgStdStr(ArgNo);
      assert(ModifierLen == 0 && "No modifiers for strings yet");
      OutStr.append(S.begin(), S.end());
      break;
    }
    case Diagnostic::ak_c_string: {
      const char *S = getArgCStr(ArgNo);
      assert(ModifierLen == 0 && "No modifiers for strings yet");

      // Don't crash if get passed a null pointer by accident.
      if (!S)
        S = "(null)";

      OutStr.append(S, S + strlen(S));
      break;
    }
    // ---- INTEGERS ----
    case Diagnostic::ak_sint: {
      int Val = getArgSInt(ArgNo);

      if (ModifierIs(Modifier, ModifierLen, "select")) {
        HandleSelectModifier(*this, (unsigned)Val, Argument, ArgumentLen,
                             OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "s")) {
        HandleIntegerSModifier(Val, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "plural")) {
        HandlePluralModifier(*this, (unsigned)Val, Argument, ArgumentLen,
                             OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "ordinal")) {
        HandleOrdinalModifier((unsigned)Val, OutStr);
      } else {
        assert(ModifierLen == 0 && "Unknown integer modifier");
        llvm::raw_svector_ostream(OutStr) << Val;
      }
      break;
    }
    case Diagnostic::ak_uint: {
      unsigned Val = getArgUInt(ArgNo);

      if (ModifierIs(Modifier, ModifierLen, "select")) {
        HandleSelectModifier(*this, Val, Argument, ArgumentLen, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "s")) {
        HandleIntegerSModifier(Val, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "plural")) {
        HandlePluralModifier(*this, (unsigned)Val, Argument, ArgumentLen,
                             OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "ordinal")) {
        HandleOrdinalModifier(Val, OutStr);
      } else {
        assert(ModifierLen == 0 && "Unknown integer modifier");
        llvm::raw_svector_ostream(OutStr) << Val;
      }
      break;
    }
    // ---- NAMES and TYPES ----
    case Diagnostic::ak_identifierinfo: {
      const IdentifierInfo *II = getArgIdentifier(ArgNo);
      assert(ModifierLen == 0 && "No modifiers for strings yet");

      // Don't crash if get passed a null pointer by accident.
      if (!II) {
        const char *S = "(null)";
        OutStr.append(S, S + strlen(S));
        continue;
      }

      llvm::raw_svector_ostream(OutStr) << '\'' << II->getName() << '\'';
      break;
    }
    case Diagnostic::ak_qualtype:
    case Diagnostic::ak_declarationname:
    case Diagnostic::ak_nameddecl:
    case Diagnostic::ak_nestednamespec:
    case Diagnostic::ak_declcontext:
      getDiags()->ConvertArgToString(Kind, getRawArg(ArgNo),
                                     Modifier, ModifierLen,
                                     Argument, ArgumentLen,
                                     FormattedArgs.data(), FormattedArgs.size(),
                                     OutStr);
      break;
    }
    
    // Remember this argument info for subsequent formatting operations.  Turn
    // std::strings into a null terminated string to make it be the same case as
    // all the other ones.
    if (Kind != Diagnostic::ak_std_string)
      FormattedArgs.push_back(std::make_pair(Kind, getRawArg(ArgNo)));
    else
      FormattedArgs.push_back(std::make_pair(Diagnostic::ak_c_string,
                                        (intptr_t)getArgStdStr(ArgNo).c_str()));
    
  }
}

StoredDiagnostic::StoredDiagnostic() { }

StoredDiagnostic::StoredDiagnostic(Diagnostic::Level Level, 
                                   llvm::StringRef Message)
  : Level(Level), Loc(), Message(Message) { }

StoredDiagnostic::StoredDiagnostic(Diagnostic::Level Level, 
                                   const DiagnosticInfo &Info)
  : Level(Level) {
  assert((Info.getLocation().isInvalid() || Info.hasSourceManager()) &&
       "Valid source location without setting a source manager for diagnostic");
  if (Info.getLocation().isValid())
    Loc = FullSourceLoc(Info.getLocation(), Info.getSourceManager());
  llvm::SmallString<64> Message;
  Info.FormatDiagnostic(Message);
  this->Message.assign(Message.begin(), Message.end());

  Ranges.reserve(Info.getNumRanges());
  for (unsigned I = 0, N = Info.getNumRanges(); I != N; ++I)
    Ranges.push_back(Info.getRange(I));

  FixIts.reserve(Info.getNumFixItHints());
  for (unsigned I = 0, N = Info.getNumFixItHints(); I != N; ++I)
    FixIts.push_back(Info.getFixItHint(I));
}

StoredDiagnostic::~StoredDiagnostic() { }

/// IncludeInDiagnosticCounts - This method (whose default implementation
///  returns true) indicates whether the diagnostics handled by this
///  DiagnosticClient should be included in the number of diagnostics
///  reported by Diagnostic.
bool DiagnosticClient::IncludeInDiagnosticCounts() const { return true; }

PartialDiagnostic::StorageAllocator::StorageAllocator() {
  for (unsigned I = 0; I != NumCached; ++I)
    FreeList[I] = Cached + I;
  NumFreeListEntries = NumCached;
}

PartialDiagnostic::StorageAllocator::~StorageAllocator() {
  assert(NumFreeListEntries == NumCached && "A partial is on the lamb");
}
