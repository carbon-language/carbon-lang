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
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <vector>
#include <map>
#include <cstring>
using namespace clang;

//===----------------------------------------------------------------------===//
// Builtin Diagnostic information
//===----------------------------------------------------------------------===//

/// Flag values for diagnostics.
enum {
  // Diagnostic classes.
  NOTE       = 0x01,
  WARNING    = 0x02,
  EXTENSION  = 0x03,
  EXTWARN    = 0x04,
  ERROR      = 0x05,
  FATAL      = 0x06,
  class_mask = 0x07
};

/// DiagnosticFlags - A set of flags, or'd together, that describe the
/// diagnostic.
#define DIAG(ENUM,FLAGS,DESC) FLAGS,
static unsigned char DiagnosticFlagsCommon[] = {
#include "clang/Basic/DiagnosticCommonKinds.def"
  0
};
static unsigned char DiagnosticFlagsLex[] = {
#include "clang/Basic/DiagnosticLexKinds.def"
  0
};
static unsigned char DiagnosticFlagsParse[] = {
#include "clang/Basic/DiagnosticParseKinds.def"
  0
};
static unsigned char DiagnosticFlagsAST[] = {
#include "clang/Basic/DiagnosticASTKinds.def"
  0
};
static unsigned char DiagnosticFlagsSema[] = {
#include "clang/Basic/DiagnosticSemaKinds.def"
  0
};
static unsigned char DiagnosticFlagsAnalysis[] = {
#include "clang/Basic/DiagnosticAnalysisKinds.def"
  0
};
#undef DIAG

/// getDiagClass - Return the class field of the diagnostic.
///
static unsigned getBuiltinDiagClass(unsigned DiagID) {
  assert(DiagID < diag::DIAG_UPPER_LIMIT &&
         "Diagnostic ID out of range!");
  unsigned res;
  if (DiagID < diag::DIAG_START_LEX)
    res = DiagnosticFlagsCommon[DiagID];
  else if (DiagID < diag::DIAG_START_PARSE)
    res = DiagnosticFlagsLex[DiagID - diag::DIAG_START_LEX - 1];
  else if (DiagID < diag::DIAG_START_AST)
    res = DiagnosticFlagsParse[DiagID - diag::DIAG_START_PARSE - 1];
  else if (DiagID < diag::DIAG_START_SEMA)
    res = DiagnosticFlagsAST[DiagID - diag::DIAG_START_AST - 1];
  else if (DiagID < diag::DIAG_START_ANALYSIS)
    res = DiagnosticFlagsSema[DiagID - diag::DIAG_START_SEMA - 1];
  else
    res = DiagnosticFlagsAnalysis[DiagID - diag::DIAG_START_ANALYSIS - 1];
  return res & class_mask;
}

/// DiagnosticText - An english message to print for the diagnostic.  These
/// should be localized.
#define DIAG(ENUM,FLAGS,DESC) DESC,
static const char * const DiagnosticTextCommon[] = {
#include "clang/Basic/DiagnosticCommonKinds.def"
  0
};
static const char * const DiagnosticTextLex[] = {
#include "clang/Basic/DiagnosticLexKinds.def"
  0
};
static const char * const DiagnosticTextParse[] = {
#include "clang/Basic/DiagnosticParseKinds.def"
  0
};
static const char * const DiagnosticTextAST[] = {
#include "clang/Basic/DiagnosticASTKinds.def"
  0
};
static const char * const DiagnosticTextSema[] = {
#include "clang/Basic/DiagnosticSemaKinds.def"
  0
};
static const char * const DiagnosticTextAnalysis[] = {
#include "clang/Basic/DiagnosticAnalysisKinds.def"
  0
};
#undef DIAG

//===----------------------------------------------------------------------===//
// Custom Diagnostic information
//===----------------------------------------------------------------------===//

namespace clang {
  namespace diag {
    class CustomDiagInfo {
      typedef std::pair<Diagnostic::Level, std::string> DiagDesc;
      std::vector<DiagDesc> DiagInfo;
      std::map<DiagDesc, unsigned> DiagIDs;
    public:
      
      /// getDescription - Return the description of the specified custom
      /// diagnostic.
      const char *getDescription(unsigned DiagID) const {
        assert(this && DiagID-DIAG_UPPER_LIMIT < DiagInfo.size() &&
               "Invalid diagnosic ID");
        return DiagInfo[DiagID-DIAG_UPPER_LIMIT].second.c_str();
      }
      
      /// getLevel - Return the level of the specified custom diagnostic.
      Diagnostic::Level getLevel(unsigned DiagID) const {
        assert(this && DiagID-DIAG_UPPER_LIMIT < DiagInfo.size() &&
               "Invalid diagnosic ID");
        return DiagInfo[DiagID-DIAG_UPPER_LIMIT].first;
      }
      
      unsigned getOrCreateDiagID(Diagnostic::Level L, const char *Message,
                                 Diagnostic &Diags) {
        DiagDesc D(L, Message);
        // Check to see if it already exists.
        std::map<DiagDesc, unsigned>::iterator I = DiagIDs.lower_bound(D);
        if (I != DiagIDs.end() && I->first == D)
          return I->second;
        
        // If not, assign a new ID.
        unsigned ID = DiagInfo.size()+DIAG_UPPER_LIMIT;
        DiagIDs.insert(std::make_pair(D, ID));
        DiagInfo.push_back(D);

        // If this is a warning, and all warnings are supposed to map to errors,
        // insert the mapping now.
        if (L == Diagnostic::Warning && Diags.getWarningsAsErrors())
          Diags.setDiagnosticMapping((diag::kind)ID, diag::MAP_ERROR);
        return ID;
      }
    };
    
  } // end diag namespace 
} // end clang namespace 


//===----------------------------------------------------------------------===//
// Common Diagnostic implementation
//===----------------------------------------------------------------------===//

static void DummyArgToStringFn(Diagnostic::ArgumentKind AK, intptr_t QT,
                               const char *Modifier, unsigned ML,
                               const char *Argument, unsigned ArgLen,
                               llvm::SmallVectorImpl<char> &Output) {
  const char *Str = "<can't format argument>";
  Output.append(Str, Str+strlen(Str));
}


Diagnostic::Diagnostic(DiagnosticClient *client) : Client(client) {
  IgnoreAllWarnings = false;
  WarningsAsErrors = false;
  WarnOnExtensions = false;
  ErrorOnExtensions = false;
  SuppressSystemWarnings = false;
  // Clear all mappings, setting them to MAP_DEFAULT.
  memset(DiagMappings, 0, sizeof(DiagMappings));
  
  ErrorOccurred = false;
  FatalErrorOccurred = false;
  NumDiagnostics = 0;
  NumErrors = 0;
  CustomDiagInfo = 0;
  CurDiagID = ~0U;
  LastDiagLevel = Fatal;
  
  ArgToStringFn = DummyArgToStringFn;
}

Diagnostic::~Diagnostic() {
  delete CustomDiagInfo;
}

/// getCustomDiagID - Return an ID for a diagnostic with the specified message
/// and level.  If this is the first request for this diagnosic, it is
/// registered and created, otherwise the existing ID is returned.
unsigned Diagnostic::getCustomDiagID(Level L, const char *Message) {
  if (CustomDiagInfo == 0) 
    CustomDiagInfo = new diag::CustomDiagInfo();
  return CustomDiagInfo->getOrCreateDiagID(L, Message, *this);
}


/// isBuiltinWarningOrExtension - Return true if the unmapped diagnostic
/// level of the specified diagnostic ID is a Warning or Extension.
/// This only works on builtin diagnostics, not custom ones, and is not legal to
/// call on NOTEs.
bool Diagnostic::isBuiltinWarningOrExtension(unsigned DiagID) {
  return DiagID < diag::DIAG_UPPER_LIMIT && getBuiltinDiagClass(DiagID) < ERROR;
}


/// getDescription - Given a diagnostic ID, return a description of the
/// issue.
const char *Diagnostic::getDescription(unsigned DiagID) const {
  if (DiagID < diag::DIAG_START_LEX)
    return DiagnosticTextCommon[DiagID];
  else if (DiagID < diag::DIAG_START_PARSE)
    return DiagnosticTextLex[DiagID - diag::DIAG_START_LEX - 1];
  else if (DiagID < diag::DIAG_START_AST)
    return DiagnosticTextParse[DiagID - diag::DIAG_START_PARSE - 1];
  else if (DiagID < diag::DIAG_START_SEMA)
    return DiagnosticTextAST[DiagID - diag::DIAG_START_AST - 1];
  else if (DiagID < diag::DIAG_START_ANALYSIS)
    return DiagnosticTextSema[DiagID - diag::DIAG_START_SEMA - 1];
  else if (DiagID < diag::DIAG_UPPER_LIMIT)
    return DiagnosticTextAnalysis[DiagID - diag::DIAG_START_ANALYSIS - 1];
  return CustomDiagInfo->getDescription(DiagID);
}

/// getDiagnosticLevel - Based on the way the client configured the Diagnostic
/// object, classify the specified diagnostic ID into a Level, consumable by
/// the DiagnosticClient.
Diagnostic::Level Diagnostic::getDiagnosticLevel(unsigned DiagID) const {
  // Handle custom diagnostics, which cannot be mapped.
  if (DiagID >= diag::DIAG_UPPER_LIMIT)
    return CustomDiagInfo->getLevel(DiagID);
  
  unsigned DiagClass = getBuiltinDiagClass(DiagID);
  assert(DiagClass != NOTE && "Cannot get the diagnostic level of a note!");
  return getDiagnosticLevel(DiagID, DiagClass);
}

/// getDiagnosticLevel - Based on the way the client configured the Diagnostic
/// object, classify the specified diagnostic ID into a Level, consumable by
/// the DiagnosticClient.
Diagnostic::Level
Diagnostic::getDiagnosticLevel(unsigned DiagID, unsigned DiagClass) const {
  // Specific non-error diagnostics may be mapped to various levels from ignored
  // to error.  Errors can only be mapped to fatal.
  switch (getDiagnosticMapping((diag::kind)DiagID)) {
  case diag::MAP_DEFAULT: break;
  case diag::MAP_IGNORE:  return Diagnostic::Ignored;
  case diag::MAP_WARNING: DiagClass = WARNING; break;
  case diag::MAP_ERROR:   DiagClass = ERROR; break;
  case diag::MAP_FATAL:   DiagClass = FATAL; break;
  }
  
  // Map diagnostic classes based on command line argument settings.
  if (DiagClass == EXTENSION) {
    if (ErrorOnExtensions)
      DiagClass = ERROR;
    else if (WarnOnExtensions)
      DiagClass = WARNING;
    else
      return Ignored;
  } else if (DiagClass == EXTWARN) {
    DiagClass = ErrorOnExtensions ? ERROR : WARNING;
  }
  
  // If warnings are globally mapped to ignore or error, do it.
  if (DiagClass == WARNING) {
    if (IgnoreAllWarnings)
      return Diagnostic::Ignored;
    if (WarningsAsErrors)
      DiagClass = ERROR;
  }
  
  switch (DiagClass) {
  default: assert(0 && "Unknown diagnostic class!");
  case WARNING:     return Diagnostic::Warning;
  case ERROR:       return Diagnostic::Error;
  case FATAL:       return Diagnostic::Fatal;
  }
}

/// ProcessDiag - This is the method used to report a diagnostic that is
/// finally fully formed.
void Diagnostic::ProcessDiag() {
  DiagnosticInfo Info(this);
  
  // If a fatal error has already been emitted, silence all subsequent
  // diagnostics.
  if (FatalErrorOccurred)
    return;
  
  // Figure out the diagnostic level of this message.
  Diagnostic::Level DiagLevel;
  unsigned DiagID = Info.getID();
  
  // ShouldEmitInSystemHeader - True if this diagnostic should be produced even
  // in a system header.
  bool ShouldEmitInSystemHeader;
  
  if (DiagID >= diag::DIAG_UPPER_LIMIT) {
    // Handle custom diagnostics, which cannot be mapped.
    DiagLevel = CustomDiagInfo->getLevel(DiagID);
    
    // Custom diagnostics always are emitted in system headers.
    ShouldEmitInSystemHeader = true;
  } else {
    // Get the class of the diagnostic.  If this is a NOTE, map it onto whatever
    // the diagnostic level was for the previous diagnostic so that it is
    // filtered the same as the previous diagnostic.
    unsigned DiagClass = getBuiltinDiagClass(DiagID);
    if (DiagClass == NOTE) {
      DiagLevel = Diagnostic::Note;
      ShouldEmitInSystemHeader = false;  // extra consideration is needed
    } else {
      // If this is not an error and we are in a system header, we ignore it. 
      // Check the original Diag ID here, because we also want to ignore
      // extensions and warnings in -Werror and -pedantic-errors modes, which
      // *map* warnings/extensions to errors.
      ShouldEmitInSystemHeader = DiagClass == ERROR;
      
      DiagLevel = getDiagnosticLevel(DiagID, DiagClass);
    }
  }

  if (DiagLevel != Diagnostic::Note)
    LastDiagLevel = DiagLevel;
  
  // If the client doesn't care about this message, don't issue it.  If this is
  // a note and the last real diagnostic was ignored, ignore it too.
  if (DiagLevel == Diagnostic::Ignored ||
      (DiagLevel == Diagnostic::Note && LastDiagLevel == Diagnostic::Ignored))
    return;

  // If this diagnostic is in a system header and is not a clang error, suppress
  // it.
  if (SuppressSystemWarnings && !ShouldEmitInSystemHeader &&
      Info.getLocation().isValid() &&
      Info.getLocation().getSpellingLoc().isInSystemHeader() &&
      (DiagLevel != Diagnostic::Note || LastDiagLevel == Diagnostic::Ignored))
    return;

  if (DiagLevel >= Diagnostic::Error) {
    ErrorOccurred = true;
    ++NumErrors;
    
    if (DiagLevel == Diagnostic::Fatal)
      FatalErrorOccurred = true;
  }
  
  // Finally, report it.
  Client->HandleDiagnostic(DiagLevel, Info);
  if (Client->IncludeInDiagnosticCounts()) ++NumDiagnostics;
}


DiagnosticClient::~DiagnosticClient() {}


/// ModifierIs - Return true if the specified modifier matches specified string.
template <std::size_t StrLen>
static bool ModifierIs(const char *Modifier, unsigned ModifierLen,
                       const char (&Str)[StrLen]) {
  return StrLen-1 == ModifierLen && !memcmp(Modifier, Str, StrLen-1);
}

/// HandleSelectModifier - Handle the integer 'select' modifier.  This is used
/// like this:  %select{foo|bar|baz}2.  This means that the integer argument
/// "%2" has a value from 0-2.  If the value is 0, the diagnostic prints 'foo'.
/// If the value is 1, it prints 'bar'.  If it has the value 2, it prints 'baz'.
/// This is very useful for certain classes of variant diagnostics.
static void HandleSelectModifier(unsigned ValNo,
                                 const char *Argument, unsigned ArgumentLen,
                                 llvm::SmallVectorImpl<char> &OutStr) {
  const char *ArgumentEnd = Argument+ArgumentLen;
  
  // Skip over 'ValNo' |'s.
  while (ValNo) {
    const char *NextVal = std::find(Argument, ArgumentEnd, '|');
    assert(NextVal != ArgumentEnd && "Value for integer select modifier was"
           " larger than the number of options in the diagnostic string!");
    Argument = NextVal+1;  // Skip this string.
    --ValNo;
  }
  
  // Get the end of the value.  This is either the } or the |.
  const char *EndPtr = std::find(Argument, ArgumentEnd, '|');
  // Add the value to the output string.
  OutStr.append(Argument, EndPtr);
}

/// HandleIntegerSModifier - Handle the integer 's' modifier.  This adds the
/// letter 's' to the string if the value is not 1.  This is used in cases like
/// this:  "you idiot, you have %4 parameter%s4!".
static void HandleIntegerSModifier(unsigned ValNo,
                                   llvm::SmallVectorImpl<char> &OutStr) {
  if (ValNo != 1)
    OutStr.push_back('s');
}


/// PluralNumber - Parse an unsigned integer and advance Start.
static unsigned PluralNumber(const char *&Start, const char *End)
{
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
static bool TestPluralRange(unsigned Val, const char *&Start, const char *End)
{
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
static bool EvalPluralExpr(unsigned ValNo, const char *Start, const char *End)
{
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
    if(Start == End)
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
static void HandlePluralModifier(unsigned ValNo,
                                 const char *Argument, unsigned ArgumentLen,
                                 llvm::SmallVectorImpl<char> &OutStr)
{
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
      ExprEnd = std::find(Argument, ArgumentEnd, '|');
      OutStr.append(Argument, ExprEnd);
      return;
    }
    Argument = std::find(Argument, ArgumentEnd - 1, '|') + 1;
  }
}


/// FormatDiagnostic - Format this diagnostic into a string, substituting the
/// formal arguments into the %0 slots.  The result is appended onto the Str
/// array.
void DiagnosticInfo::
FormatDiagnostic(llvm::SmallVectorImpl<char> &OutStr) const {
  const char *DiagStr = getDiags()->getDescription(getID());
  const char *DiagEnd = DiagStr+strlen(DiagStr);
  
  while (DiagStr != DiagEnd) {
    if (DiagStr[0] != '%') {
      // Append non-%0 substrings to Str if we have one.
      const char *StrEnd = std::find(DiagStr, DiagEnd, '%');
      OutStr.append(DiagStr, StrEnd);
      DiagStr = StrEnd;
      continue;
    } else if (DiagStr[1] == '%') {
      OutStr.push_back('%');  // %% -> %.
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
        
        for (; DiagStr[0] != '}'; ++DiagStr)
          assert(DiagStr[0] && "Mismatched {}'s in diagnostic string!");
        ArgumentLen = DiagStr-Argument;
        ++DiagStr;  // Skip }.
      }
    }
      
    assert(isdigit(*DiagStr) && "Invalid format for argument in diagnostic");
    unsigned ArgNo = *DiagStr++ - '0';

    switch (getArgKind(ArgNo)) {
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
      OutStr.append(S, S + strlen(S));
      break;
    }
    // ---- INTEGERS ----
    case Diagnostic::ak_sint: {
      int Val = getArgSInt(ArgNo);
      
      if (ModifierIs(Modifier, ModifierLen, "select")) {
        HandleSelectModifier((unsigned)Val, Argument, ArgumentLen, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "s")) {
        HandleIntegerSModifier(Val, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "plural")) {
        HandlePluralModifier((unsigned)Val, Argument, ArgumentLen, OutStr);
      } else {
        assert(ModifierLen == 0 && "Unknown integer modifier");
        // FIXME: Optimize
        std::string S = llvm::itostr(Val);
        OutStr.append(S.begin(), S.end());
      }
      break;
    }
    case Diagnostic::ak_uint: {
      unsigned Val = getArgUInt(ArgNo);
      
      if (ModifierIs(Modifier, ModifierLen, "select")) {
        HandleSelectModifier(Val, Argument, ArgumentLen, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "s")) {
        HandleIntegerSModifier(Val, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "plural")) {
        HandlePluralModifier((unsigned)Val, Argument, ArgumentLen, OutStr);
      } else {
        assert(ModifierLen == 0 && "Unknown integer modifier");
        
        // FIXME: Optimize
        std::string S = llvm::utostr_32(Val);
        OutStr.append(S.begin(), S.end());
      }
      break;
    }
    // ---- NAMES and TYPES ----
    case Diagnostic::ak_identifierinfo: {
      OutStr.push_back('\'');
      const IdentifierInfo *II = getArgIdentifier(ArgNo);
      assert(ModifierLen == 0 && "No modifiers for strings yet");
      OutStr.append(II->getName(), II->getName() + II->getLength());
      OutStr.push_back('\'');
      break;
    }
    case Diagnostic::ak_qualtype:
    case Diagnostic::ak_declarationname:
    case Diagnostic::ak_nameddecl:
      OutStr.push_back('\'');
      getDiags()->ConvertArgToString(getArgKind(ArgNo), getRawArg(ArgNo),
                                     Modifier, ModifierLen,
                                     Argument, ArgumentLen, OutStr);
      OutStr.push_back('\'');
      break;
    }
  }
}

/// IncludeInDiagnosticCounts - This method (whose default implementation
///  returns true) indicates whether the diagnostics handled by this
///  DiagnosticClient should be included in the number of diagnostics
///  reported by Diagnostic.
bool DiagnosticClient::IncludeInDiagnosticCounts() const { return true; }
