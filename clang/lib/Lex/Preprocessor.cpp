//===--- Preprocess.cpp - C Language Family Preprocessor Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Preprocessor interface.
//
//===----------------------------------------------------------------------===//
//
// Options to support:
//   -H       - Print the name of each header file used.
//   -d[DNI] - Dump various things.
//   -fworking-directory - #line's with preprocessor's working dir.
//   -fpreprocessed
//   -dependency-file,-M,-MM,-MF,-MG,-MP,-MT,-MQ,-MD,-MMD
//   -W*
//   -w
//
// Messages to emit:
//   "Multiple include guards may be useful for:\n"
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/ScratchBuffer.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Streams.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//

PreprocessorFactory::~PreprocessorFactory() {}

Preprocessor::Preprocessor(Diagnostic &diags, const LangOptions &opts,
                           TargetInfo &target, SourceManager &SM, 
                           HeaderSearch &Headers,
                           IdentifierInfoLookup* IILookup)
  : Diags(&diags), Features(opts), Target(target),FileMgr(Headers.getFileMgr()),
    SourceMgr(SM), HeaderInfo(Headers), Identifiers(opts, IILookup),
    CurPPLexer(0), CurDirLookup(0), Callbacks(0) {
  ScratchBuf = new ScratchBuffer(SourceMgr);

  // Clear stats.
  NumDirectives = NumDefined = NumUndefined = NumPragma = 0;
  NumIf = NumElse = NumEndif = 0;
  NumEnteredSourceFiles = 0;
  NumMacroExpanded = NumFnMacroExpanded = NumBuiltinMacroExpanded = 0;
  NumFastMacroExpanded = NumTokenPaste = NumFastTokenPaste = 0;
  MaxIncludeStackDepth = 0; 
  NumSkipped = 0;

  // Default to discarding comments.
  KeepComments = false;
  KeepMacroComments = false;
  
  // Macro expansion is enabled.
  DisableMacroExpansion = false;
  InMacroArgs = false;
  NumCachedTokenLexers = 0;

  CachedLexPos = 0;

  // "Poison" __VA_ARGS__, which can only appear in the expansion of a macro.
  // This gets unpoisoned where it is allowed.
  (Ident__VA_ARGS__ = getIdentifierInfo("__VA_ARGS__"))->setIsPoisoned();
  
  // Initialize the pragma handlers.
  PragmaHandlers = new PragmaNamespace(0);
  RegisterBuiltinPragmas();
  
  // Initialize builtin macros like __LINE__ and friends.
  RegisterBuiltinMacros();
}

Preprocessor::~Preprocessor() {
  assert(BacktrackPositions.empty() && "EnableBacktrack/Backtrack imbalance!");

  while (!IncludeMacroStack.empty()) {
    delete IncludeMacroStack.back().TheLexer;
    delete IncludeMacroStack.back().TheTokenLexer;
    IncludeMacroStack.pop_back();
  }

  // Free any macro definitions.
  for (llvm::DenseMap<IdentifierInfo*, MacroInfo*>::iterator I =
       Macros.begin(), E = Macros.end(); I != E; ++I) {
    // We don't need to free the MacroInfo objects directly.  These
    // will be released when the BumpPtrAllocator 'BP' object gets
    // destroyed. We still need to run the dstor, however, to free
    // memory alocated by MacroInfo.
    I->second->Destroy(BP);
    I->first->setHasMacroDefinition(false);
  }
  
  // Free any cached macro expanders.
  for (unsigned i = 0, e = NumCachedTokenLexers; i != e; ++i)
    delete TokenLexerCache[i];
  
  // Release pragma information.
  delete PragmaHandlers;

  // Delete the scratch buffer info.
  delete ScratchBuf;

  delete Callbacks;
}

void Preprocessor::setPTHManager(PTHManager* pm) {
  PTH.reset(pm);
  FileMgr.setStatCache(PTH->createStatCache());
}

void Preprocessor::DumpToken(const Token &Tok, bool DumpFlags) const {
  llvm::cerr << tok::getTokenName(Tok.getKind()) << " '"
             << getSpelling(Tok) << "'";
  
  if (!DumpFlags) return;
  
  llvm::cerr << "\t";
  if (Tok.isAtStartOfLine())
    llvm::cerr << " [StartOfLine]";
  if (Tok.hasLeadingSpace())
    llvm::cerr << " [LeadingSpace]";
  if (Tok.isExpandDisabled())
    llvm::cerr << " [ExpandDisabled]";
  if (Tok.needsCleaning()) {
    const char *Start = SourceMgr.getCharacterData(Tok.getLocation());
    llvm::cerr << " [UnClean='" << std::string(Start, Start+Tok.getLength())
               << "']";
  }
  
  llvm::cerr << "\tLoc=<";
  DumpLocation(Tok.getLocation());
  llvm::cerr << ">";
}

void Preprocessor::DumpLocation(SourceLocation Loc) const {
  Loc.dump(SourceMgr);
}

void Preprocessor::DumpMacro(const MacroInfo &MI) const {
  llvm::cerr << "MACRO: ";
  for (unsigned i = 0, e = MI.getNumTokens(); i != e; ++i) {
    DumpToken(MI.getReplacementToken(i));
    llvm::cerr << "  ";
  }
  llvm::cerr << "\n";
}

void Preprocessor::PrintStats() {
  llvm::cerr << "\n*** Preprocessor Stats:\n";
  llvm::cerr << NumDirectives << " directives found:\n";
  llvm::cerr << "  " << NumDefined << " #define.\n";
  llvm::cerr << "  " << NumUndefined << " #undef.\n";
  llvm::cerr << "  #include/#include_next/#import:\n";
  llvm::cerr << "    " << NumEnteredSourceFiles << " source files entered.\n";
  llvm::cerr << "    " << MaxIncludeStackDepth << " max include stack depth\n";
  llvm::cerr << "  " << NumIf << " #if/#ifndef/#ifdef.\n";
  llvm::cerr << "  " << NumElse << " #else/#elif.\n";
  llvm::cerr << "  " << NumEndif << " #endif.\n";
  llvm::cerr << "  " << NumPragma << " #pragma.\n";
  llvm::cerr << NumSkipped << " #if/#ifndef#ifdef regions skipped\n";

  llvm::cerr << NumMacroExpanded << "/" << NumFnMacroExpanded << "/"
             << NumBuiltinMacroExpanded << " obj/fn/builtin macros expanded, "
             << NumFastMacroExpanded << " on the fast path.\n";
  llvm::cerr << (NumFastTokenPaste+NumTokenPaste)
             << " token paste (##) operations performed, "
             << NumFastTokenPaste << " on the fast path.\n";
}

//===----------------------------------------------------------------------===//
// Token Spelling
//===----------------------------------------------------------------------===//


/// getSpelling() - Return the 'spelling' of this token.  The spelling of a
/// token are the characters used to represent the token in the source file
/// after trigraph expansion and escaped-newline folding.  In particular, this
/// wants to get the true, uncanonicalized, spelling of things like digraphs
/// UCNs, etc.
std::string Preprocessor::getSpelling(const Token &Tok) const {
  assert((int)Tok.getLength() >= 0 && "Token character range is bogus!");

  // If this token contains nothing interesting, return it directly.
  const char* TokStart = SourceMgr.getCharacterData(Tok.getLocation());
  if (!Tok.needsCleaning())
    return std::string(TokStart, TokStart+Tok.getLength());
  
  std::string Result;
  Result.reserve(Tok.getLength());
  
  // Otherwise, hard case, relex the characters into the string.
  for (const char *Ptr = TokStart, *End = TokStart+Tok.getLength();
       Ptr != End; ) {
    unsigned CharSize;
    Result.push_back(Lexer::getCharAndSizeNoWarn(Ptr, CharSize, Features));
    Ptr += CharSize;
  }
  assert(Result.size() != unsigned(Tok.getLength()) &&
         "NeedsCleaning flag set on something that didn't need cleaning!");
  return Result;
}

/// getSpelling - This method is used to get the spelling of a token into a
/// preallocated buffer, instead of as an std::string.  The caller is required
/// to allocate enough space for the token, which is guaranteed to be at least
/// Tok.getLength() bytes long.  The actual length of the token is returned.
///
/// Note that this method may do two possible things: it may either fill in
/// the buffer specified with characters, or it may *change the input pointer*
/// to point to a constant buffer with the data already in it (avoiding a
/// copy).  The caller is not allowed to modify the returned buffer pointer
/// if an internal buffer is returned.
unsigned Preprocessor::getSpelling(const Token &Tok,
                                   const char *&Buffer) const {
  assert((int)Tok.getLength() >= 0 && "Token character range is bogus!");
  
  // If this token is an identifier, just return the string from the identifier
  // table, which is very quick.
  if (const IdentifierInfo *II = Tok.getIdentifierInfo()) {
    Buffer = II->getName();
    return II->getLength();
  }

  // Otherwise, compute the start of the token in the input lexer buffer.
  const char *TokStart = 0;
  
  if (Tok.isLiteral())
    TokStart = Tok.getLiteralData();
  
  if (TokStart == 0)
    TokStart = SourceMgr.getCharacterData(Tok.getLocation());

  // If this token contains nothing interesting, return it directly.
  if (!Tok.needsCleaning()) {
    Buffer = TokStart;
    return Tok.getLength();
  }
  
  // Otherwise, hard case, relex the characters into the string.
  char *OutBuf = const_cast<char*>(Buffer);
  for (const char *Ptr = TokStart, *End = TokStart+Tok.getLength();
       Ptr != End; ) {
    unsigned CharSize;
    *OutBuf++ = Lexer::getCharAndSizeNoWarn(Ptr, CharSize, Features);
    Ptr += CharSize;
  }
  assert(unsigned(OutBuf-Buffer) != Tok.getLength() &&
         "NeedsCleaning flag set on something that didn't need cleaning!");
  
  return OutBuf-Buffer;
}

/// CreateString - Plop the specified string into a scratch buffer and return a
/// location for it.  If specified, the source location provides a source
/// location for the token.
void Preprocessor::CreateString(const char *Buf, unsigned Len, Token &Tok,
                                SourceLocation InstantiationLoc) {
  Tok.setLength(Len);
  
  const char *DestPtr;
  SourceLocation Loc = ScratchBuf->getToken(Buf, Len, DestPtr);
  
  if (InstantiationLoc.isValid())
    Loc = SourceMgr.createInstantiationLoc(Loc, InstantiationLoc,
                                           InstantiationLoc, Len);
  Tok.setLocation(Loc);
  
  // If this is a literal token, set the pointer data.
  if (Tok.isLiteral())
    Tok.setLiteralData(DestPtr);
}


/// AdvanceToTokenCharacter - Given a location that specifies the start of a
/// token, return a new location that specifies a character within the token.
SourceLocation Preprocessor::AdvanceToTokenCharacter(SourceLocation TokStart, 
                                                     unsigned CharNo) {
  // If they request the first char of the token, we're trivially done.
  if (CharNo == 0) return TokStart;
  
  // Figure out how many physical characters away the specified instantiation
  // character is.  This needs to take into consideration newlines and
  // trigraphs.
  const char *TokPtr = SourceMgr.getCharacterData(TokStart);
  unsigned PhysOffset = 0;
  
  // The usual case is that tokens don't contain anything interesting.  Skip
  // over the uninteresting characters.  If a token only consists of simple
  // chars, this method is extremely fast.
  while (CharNo && Lexer::isObviouslySimpleCharacter(*TokPtr))
    ++TokPtr, --CharNo, ++PhysOffset;
  
  // If we have a character that may be a trigraph or escaped newline, use a
  // lexer to parse it correctly.
  if (CharNo != 0) {
    // Skip over characters the remaining characters.
    for (; CharNo; --CharNo) {
      unsigned Size;
      Lexer::getCharAndSizeNoWarn(TokPtr, Size, Features);
      TokPtr += Size;
      PhysOffset += Size;
    }
  }
  
  return TokStart.getFileLocWithOffset(PhysOffset);
}

/// \brief Computes the source location just past the end of the
/// token at this source location.
///
/// This routine can be used to produce a source location that
/// points just past the end of the token referenced by \p Loc, and
/// is generally used when a diagnostic needs to point just after a
/// token where it expected something different that it received. If
/// the returned source location would not be meaningful (e.g., if
/// it points into a macro), this routine returns an invalid
/// source location.
SourceLocation Preprocessor::getLocForEndOfToken(SourceLocation Loc) {
  if (Loc.isInvalid() || !Loc.isFileID())
    return SourceLocation();

  unsigned Len = Lexer::MeasureTokenLength(Loc, getSourceManager());
  return AdvanceToTokenCharacter(Loc, Len);
}



//===----------------------------------------------------------------------===//
// Preprocessor Initialization Methods
//===----------------------------------------------------------------------===//

// Append a #define line to Buf for Macro.  Macro should be of the form XXX,
// in which case we emit "#define XXX 1" or "XXX=Y z W" in which case we emit
// "#define XXX Y z W".  To get a #define with no value, use "XXX=".
static void DefineBuiltinMacro(std::vector<char> &Buf, const char *Macro,
                               const char *Command = "#define ") {
  Buf.insert(Buf.end(), Command, Command+strlen(Command));
  if (const char *Equal = strchr(Macro, '=')) {
    // Turn the = into ' '.
    Buf.insert(Buf.end(), Macro, Equal);
    Buf.push_back(' ');
    Buf.insert(Buf.end(), Equal+1, Equal+strlen(Equal));
  } else {
    // Push "macroname 1".
    Buf.insert(Buf.end(), Macro, Macro+strlen(Macro));
    Buf.push_back(' ');
    Buf.push_back('1');
  }
  Buf.push_back('\n');
}

/// PickFP - This is used to pick a value based on the FP semantics of the
/// specified FP model.
template <typename T>
static T PickFP(const llvm::fltSemantics *Sem, T IEEESingleVal,
                T IEEEDoubleVal, T X87DoubleExtendedVal, T PPCDoubleDoubleVal) {
  if (Sem == &llvm::APFloat::IEEEsingle)
    return IEEESingleVal;
  if (Sem == &llvm::APFloat::IEEEdouble)
    return IEEEDoubleVal;
  if (Sem == &llvm::APFloat::x87DoubleExtended)
    return X87DoubleExtendedVal;
  assert(Sem == &llvm::APFloat::PPCDoubleDouble);
  return PPCDoubleDoubleVal;
}

static void DefineFloatMacros(std::vector<char> &Buf, const char *Prefix,
                              const llvm::fltSemantics *Sem) {
  const char *DenormMin, *Epsilon, *Max, *Min;
  DenormMin = PickFP(Sem, "1.40129846e-45F", "4.9406564584124654e-324", 
                     "3.64519953188247460253e-4951L",
                     "4.94065645841246544176568792868221e-324L");
  int Digits = PickFP(Sem, 6, 15, 18, 31);
  Epsilon = PickFP(Sem, "1.19209290e-7F", "2.2204460492503131e-16",
                   "1.08420217248550443401e-19L",
                   "4.94065645841246544176568792868221e-324L");
  int HasInifinity = 1, HasQuietNaN = 1;
  int MantissaDigits = PickFP(Sem, 24, 53, 64, 106);
  int Min10Exp = PickFP(Sem, -37, -307, -4931, -291);
  int Max10Exp = PickFP(Sem, 38, 308, 4932, 308);
  int MinExp = PickFP(Sem, -125, -1021, -16381, -968);
  int MaxExp = PickFP(Sem, 128, 1024, 16384, 1024);
  Min = PickFP(Sem, "1.17549435e-38F", "2.2250738585072014e-308",
               "3.36210314311209350626e-4932L",
               "2.00416836000897277799610805135016e-292L");
  Max = PickFP(Sem, "3.40282347e+38F", "1.7976931348623157e+308",
               "1.18973149535723176502e+4932L",
               "1.79769313486231580793728971405301e+308L");
  
  char MacroBuf[60];
  sprintf(MacroBuf, "__%s_DENORM_MIN__=%s", Prefix, DenormMin);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_DIG__=%d", Prefix, Digits);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_EPSILON__=%s", Prefix, Epsilon);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_HAS_INFINITY__=%d", Prefix, HasInifinity);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_HAS_QUIET_NAN__=%d", Prefix, HasQuietNaN);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MANT_DIG__=%d", Prefix, MantissaDigits);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MAX_10_EXP__=%d", Prefix, Max10Exp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MAX_EXP__=%d", Prefix, MaxExp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MAX__=%s", Prefix, Max);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MIN_10_EXP__=(%d)", Prefix, Min10Exp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MIN_EXP__=(%d)", Prefix, MinExp);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_MIN__=%s", Prefix, Min);
  DefineBuiltinMacro(Buf, MacroBuf);
  sprintf(MacroBuf, "__%s_HAS_DENORM__=1", Prefix);
  DefineBuiltinMacro(Buf, MacroBuf);
}


/// DefineTypeSize - Emit a macro to the predefines buffer that declares a macro
/// named MacroName with the max value for a type with width 'TypeWidth' a
/// signedness of 'isSigned' and with a value suffix of 'ValSuffix' (e.g. LL).
static void DefineTypeSize(const char *MacroName, unsigned TypeWidth,
                           const char *ValSuffix, bool isSigned,
                           std::vector<char> &Buf) {
  char MacroBuf[60];
  long long MaxVal;
  if (isSigned)
    MaxVal = (1LL << (TypeWidth - 1)) - 1;
  else
    MaxVal = ~0LL >> (64-TypeWidth);
  
  sprintf(MacroBuf, "%s=%llu%s", MacroName, MaxVal, ValSuffix);
  DefineBuiltinMacro(Buf, MacroBuf);
}

static void DefineType(const char *MacroName, TargetInfo::IntType Ty,
                       std::vector<char> &Buf) {
  char MacroBuf[60];
  sprintf(MacroBuf, "%s=%s", MacroName, TargetInfo::getTypeName(Ty));
  DefineBuiltinMacro(Buf, MacroBuf);
}


static void InitializePredefinedMacros(Preprocessor &PP, 
                                       std::vector<char> &Buf) {
  char MacroBuf[60];
  // Compiler version introspection macros.
  DefineBuiltinMacro(Buf, "__llvm__=1");   // LLVM Backend
  DefineBuiltinMacro(Buf, "__clang__=1");  // Clang Frontend
  
  // Currently claim to be compatible with GCC 4.2.1-5621.
  DefineBuiltinMacro(Buf, "__APPLE_CC__=5621");
  DefineBuiltinMacro(Buf, "__GNUC_MINOR__=2");
  DefineBuiltinMacro(Buf, "__GNUC_PATCHLEVEL__=1");
  DefineBuiltinMacro(Buf, "__GNUC__=4");
  DefineBuiltinMacro(Buf, "__GXX_ABI_VERSION=1002");
  DefineBuiltinMacro(Buf, "__VERSION__=\"4.2.1 Compatible Clang Compiler\"");
  
  
  // Initialize language-specific preprocessor defines.
  
  // These should all be defined in the preprocessor according to the
  // current language configuration.
  if (!PP.getLangOptions().Microsoft)
    DefineBuiltinMacro(Buf, "__STDC__=1");
  if (PP.getLangOptions().AsmPreprocessor)
    DefineBuiltinMacro(Buf, "__ASSEMBLER__=1");
  if (PP.getLangOptions().C99 && !PP.getLangOptions().CPlusPlus)
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199901L");
  else if (0) // STDC94 ?
    DefineBuiltinMacro(Buf, "__STDC_VERSION__=199409L");
  
  if (PP.getLangOptions().CPlusPlus0x)
    DefineBuiltinMacro(Buf, "__GXX_EXPERIMENTAL_CXX0X__");

  if (PP.getLangOptions().Freestanding)
    DefineBuiltinMacro(Buf, "__STDC_HOSTED__=0");
  else
    DefineBuiltinMacro(Buf, "__STDC_HOSTED__=1");
  
  if (PP.getLangOptions().ObjC1) {
    DefineBuiltinMacro(Buf, "__OBJC__=1");
    if (PP.getLangOptions().ObjCNonFragileABI)
      DefineBuiltinMacro(Buf, "__OBJC2__=1");

    if (PP.getLangOptions().getGCMode() != LangOptions::NonGC)
      DefineBuiltinMacro(Buf, "__OBJC_GC__=1");
    
    if (PP.getLangOptions().NeXTRuntime)
      DefineBuiltinMacro(Buf, "__NEXT_RUNTIME__=1");
  }
  
  // darwin_constant_cfstrings controls this. This is also dependent
  // on other things like the runtime I believe.  This is set even for C code.
  DefineBuiltinMacro(Buf, "__CONSTANT_CFSTRINGS__=1");
  
  if (PP.getLangOptions().ObjC2)
    DefineBuiltinMacro(Buf, "OBJC_NEW_PROPERTIES");

  if (PP.getLangOptions().PascalStrings)
    DefineBuiltinMacro(Buf, "__PASCAL_STRINGS__");

  if (PP.getLangOptions().Blocks) {
    DefineBuiltinMacro(Buf, "__block=__attribute__((__blocks__(byref)))");
    DefineBuiltinMacro(Buf, "__BLOCKS__=1");
  }
  
  if (PP.getLangOptions().CPlusPlus) {
    DefineBuiltinMacro(Buf, "__DEPRECATED=1");
    DefineBuiltinMacro(Buf, "__EXCEPTIONS=1");
    DefineBuiltinMacro(Buf, "__GNUG__=4");
    DefineBuiltinMacro(Buf, "__GXX_WEAK__=1");
    DefineBuiltinMacro(Buf, "__cplusplus=1");
    DefineBuiltinMacro(Buf, "__private_extern__=extern");
  }
  
  // Filter out some microsoft extensions when trying to parse in ms-compat
  // mode. 
  if (PP.getLangOptions().Microsoft) {
    DefineBuiltinMacro(Buf, "_cdecl=__cdecl");
    DefineBuiltinMacro(Buf, "__int8=__INT8_TYPE__");
    DefineBuiltinMacro(Buf, "__int16=__INT16_TYPE__");
    DefineBuiltinMacro(Buf, "__int32=__INT32_TYPE__");
    DefineBuiltinMacro(Buf, "__int64=__INT64_TYPE__");
  }
  
  if (PP.getLangOptions().Optimize)
    DefineBuiltinMacro(Buf, "__OPTIMIZE__=1");
  if (PP.getLangOptions().OptimizeSize)
    DefineBuiltinMacro(Buf, "__OPTIMIZE_SIZE__=1");
    
  // Initialize target-specific preprocessor defines.
  const TargetInfo &TI = PP.getTargetInfo();
  
  // Define type sizing macros based on the target properties.
  assert(TI.getCharWidth() == 8 && "Only support 8-bit char so far");
  DefineBuiltinMacro(Buf, "__CHAR_BIT__=8");

  unsigned IntMaxWidth;
  const char *IntMaxSuffix;
  if (TI.getIntMaxType() == TargetInfo::SignedLongLong) {
    IntMaxWidth = TI.getLongLongWidth();
    IntMaxSuffix = "LL";
  } else if (TI.getIntMaxType() == TargetInfo::SignedLong) {
    IntMaxWidth = TI.getLongWidth();
    IntMaxSuffix = "L";
  } else {
    assert(TI.getIntMaxType() == TargetInfo::SignedInt);
    IntMaxWidth = TI.getIntWidth();
    IntMaxSuffix = "";
  }
  
  DefineTypeSize("__SCHAR_MAX__", TI.getCharWidth(), "", true, Buf);
  DefineTypeSize("__SHRT_MAX__", TI.getShortWidth(), "", true, Buf);
  DefineTypeSize("__INT_MAX__", TI.getIntWidth(), "", true, Buf);
  DefineTypeSize("__LONG_MAX__", TI.getLongWidth(), "L", true, Buf);
  DefineTypeSize("__LONG_LONG_MAX__", TI.getLongLongWidth(), "LL", true, Buf);
  DefineTypeSize("__WCHAR_MAX__", TI.getWCharWidth(), "", true, Buf);
  DefineTypeSize("__INTMAX_MAX__", IntMaxWidth, IntMaxSuffix, true, Buf);

  DefineType("__INTMAX_TYPE__", TI.getIntMaxType(), Buf);
  DefineType("__UINTMAX_TYPE__", TI.getUIntMaxType(), Buf);
  DefineType("__PTRDIFF_TYPE__", TI.getPtrDiffType(0), Buf);
  DefineType("__INTPTR_TYPE__", TI.getIntPtrType(), Buf);
  DefineType("__SIZE_TYPE__", TI.getSizeType(), Buf);
  DefineType("__WCHAR_TYPE__", TI.getWCharType(), Buf);
  // FIXME: TargetInfo hookize __WINT_TYPE__.
  DefineBuiltinMacro(Buf, "__WINT_TYPE__=int");
  
  DefineFloatMacros(Buf, "FLT", &TI.getFloatFormat());
  DefineFloatMacros(Buf, "DBL", &TI.getDoubleFormat());
  DefineFloatMacros(Buf, "LDBL", &TI.getLongDoubleFormat());

  // Define a __POINTER_WIDTH__ macro for stdint.h.
  sprintf(MacroBuf, "__POINTER_WIDTH__=%d", (int)TI.getPointerWidth(0));
  DefineBuiltinMacro(Buf, MacroBuf);
  
  if (!TI.isCharSigned())
    DefineBuiltinMacro(Buf, "__CHAR_UNSIGNED__");  

  // Define fixed-sized integer types for stdint.h
  assert(TI.getCharWidth() == 8 && "unsupported target types");
  assert(TI.getShortWidth() == 16 && "unsupported target types");
  DefineBuiltinMacro(Buf, "__INT8_TYPE__=char");
  DefineBuiltinMacro(Buf, "__INT16_TYPE__=short");
  
  if (TI.getIntWidth() == 32)
    DefineBuiltinMacro(Buf, "__INT32_TYPE__=int");
  else {
    assert(TI.getLongLongWidth() == 32 && "unsupported target types");
    DefineBuiltinMacro(Buf, "__INT32_TYPE__=long long");
  }
  
  // 16-bit targets doesn't necessarily have a 64-bit type.
  if (TI.getLongLongWidth() == 64)
    DefineBuiltinMacro(Buf, "__INT64_TYPE__=long long");
  
  // Add __builtin_va_list typedef.
  {
    const char *VAList = TI.getVAListDeclaration();
    Buf.insert(Buf.end(), VAList, VAList+strlen(VAList));
    Buf.push_back('\n');
  }
  
  if (const char *Prefix = TI.getUserLabelPrefix()) {
    sprintf(MacroBuf, "__USER_LABEL_PREFIX__=%s", Prefix);
    DefineBuiltinMacro(Buf, MacroBuf);
  }
  
  // Build configuration options.  FIXME: these should be controlled by
  // command line options or something.
  DefineBuiltinMacro(Buf, "__DYNAMIC__=1");
  DefineBuiltinMacro(Buf, "__FINITE_MATH_ONLY__=0");
  DefineBuiltinMacro(Buf, "__NO_INLINE__=1");
  DefineBuiltinMacro(Buf, "__PIC__=1");

  // Macros to control C99 numerics and <float.h>
  DefineBuiltinMacro(Buf, "__FLT_EVAL_METHOD__=0");
  DefineBuiltinMacro(Buf, "__FLT_RADIX__=2");
  sprintf(MacroBuf, "__DECIMAL_DIG__=%d",
          PickFP(&TI.getLongDoubleFormat(), -1/*FIXME*/, 17, 21, 33));
  DefineBuiltinMacro(Buf, MacroBuf);
  
  // Get other target #defines.
  TI.getTargetDefines(PP.getLangOptions(), Buf);
}


/// EnterMainSourceFile - Enter the specified FileID as the main source file,
/// which implicitly adds the builtin defines etc.
void Preprocessor::EnterMainSourceFile() {
  // We do not allow the preprocessor to reenter the main file.  Doing so will
  // cause FileID's to accumulate information from both runs (e.g. #line
  // information) and predefined macros aren't guaranteed to be set properly.
  assert(NumEnteredSourceFiles == 0 && "Cannot reenter the main file!");
  FileID MainFileID = SourceMgr.getMainFileID();
  
  // Enter the main file source buffer.
  EnterSourceFile(MainFileID, 0);
  
  // Tell the header info that the main file was entered.  If the file is later
  // #imported, it won't be re-entered.
  if (const FileEntry *FE = SourceMgr.getFileEntryForID(MainFileID))
    HeaderInfo.IncrementIncludeCount(FE);
    
  std::vector<char> PrologFile;
  PrologFile.reserve(4080);
  
  // Install things like __POWERPC__, __GNUC__, etc into the macro table.
  InitializePredefinedMacros(*this, PrologFile);
  
  // Add on the predefines from the driver.  Wrap in a #line directive to report
  // that they come from the command line.
  const char *LineDirective = "# 1 \"<command line>\" 1\n";
  PrologFile.insert(PrologFile.end(),
                    LineDirective, LineDirective+strlen(LineDirective));
  
  PrologFile.insert(PrologFile.end(), Predefines.begin(), Predefines.end());
  
  LineDirective = "# 2 \"<built-in>\" 2\n";
  PrologFile.insert(PrologFile.end(),
                    LineDirective, LineDirective+strlen(LineDirective));
  
  // Memory buffer must end with a null byte!
  PrologFile.push_back(0);

  // Now that we have emitted the predefined macros, #includes, etc into
  // PrologFile, preprocess it to populate the initial preprocessor state.
  llvm::MemoryBuffer *SB = 
    llvm::MemoryBuffer::getMemBufferCopy(&PrologFile.front(),&PrologFile.back(),
                                         "<built-in>");
  assert(SB && "Cannot fail to create predefined source buffer");
  FileID FID = SourceMgr.createFileIDForMemBuffer(SB);
  assert(!FID.isInvalid() && "Could not create FileID for predefines?");
  
  // Start parsing the predefines.
  EnterSourceFile(FID, 0);
}


//===----------------------------------------------------------------------===//
// Lexer Event Handling.
//===----------------------------------------------------------------------===//

/// LookUpIdentifierInfo - Given a tok::identifier token, look up the
/// identifier information for the token and install it into the token.
IdentifierInfo *Preprocessor::LookUpIdentifierInfo(Token &Identifier,
                                                   const char *BufPtr) {
  assert(Identifier.is(tok::identifier) && "Not an identifier!");
  assert(Identifier.getIdentifierInfo() == 0 && "Identinfo already exists!");
  
  // Look up this token, see if it is a macro, or if it is a language keyword.
  IdentifierInfo *II;
  if (BufPtr && !Identifier.needsCleaning()) {
    // No cleaning needed, just use the characters from the lexed buffer.
    II = getIdentifierInfo(BufPtr, BufPtr+Identifier.getLength());
  } else {
    // Cleaning needed, alloca a buffer, clean into it, then use the buffer.
    llvm::SmallVector<char, 64> IdentifierBuffer;
    IdentifierBuffer.resize(Identifier.getLength());
    const char *TmpBuf = &IdentifierBuffer[0];
    unsigned Size = getSpelling(Identifier, TmpBuf);
    II = getIdentifierInfo(TmpBuf, TmpBuf+Size);
  }
  Identifier.setIdentifierInfo(II);
  return II;
}


/// HandleIdentifier - This callback is invoked when the lexer reads an
/// identifier.  This callback looks up the identifier in the map and/or
/// potentially macro expands it or turns it into a named token (like 'for').
///
/// Note that callers of this method are guarded by checking the
/// IdentifierInfo's 'isHandleIdentifierCase' bit.  If this method changes, the
/// IdentifierInfo methods that compute these properties will need to change to
/// match.
void Preprocessor::HandleIdentifier(Token &Identifier) {
  assert(Identifier.getIdentifierInfo() &&
         "Can't handle identifiers without identifier info!");
  
  IdentifierInfo &II = *Identifier.getIdentifierInfo();

  // If this identifier was poisoned, and if it was not produced from a macro
  // expansion, emit an error.
  if (II.isPoisoned() && CurPPLexer) {
    if (&II != Ident__VA_ARGS__)   // We warn about __VA_ARGS__ with poisoning.
      Diag(Identifier, diag::err_pp_used_poisoned_id);
    else
      Diag(Identifier, diag::ext_pp_bad_vaargs_use);
  }
  
  // If this is a macro to be expanded, do it.
  if (MacroInfo *MI = getMacroInfo(&II)) {
    if (!DisableMacroExpansion && !Identifier.isExpandDisabled()) {
      if (MI->isEnabled()) {
        if (!HandleMacroExpandedIdentifier(Identifier, MI))
          return;
      } else {
        // C99 6.10.3.4p2 says that a disabled macro may never again be
        // expanded, even if it's in a context where it could be expanded in the
        // future.
        Identifier.setFlag(Token::DisableExpand);
      }
    }
  }

  // C++ 2.11p2: If this is an alternative representation of a C++ operator,
  // then we act as if it is the actual operator and not the textual
  // representation of it.
  if (II.isCPlusPlusOperatorKeyword())
    Identifier.setIdentifierInfo(0);

  // If this is an extension token, diagnose its use.
  // We avoid diagnosing tokens that originate from macro definitions.
  if (II.isExtensionToken() && Features.C99 && !DisableMacroExpansion)
    Diag(Identifier, diag::ext_token_used);
}
