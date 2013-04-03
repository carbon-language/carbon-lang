//===--- MacroInfo.cpp - Information about #defined identifiers -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MacroInfo interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
using namespace clang;

MacroInfo::MacroInfo(SourceLocation DefLoc)
  : Location(DefLoc),
    ArgumentList(0),
    NumArguments(0),
    IsDefinitionLengthCached(false),
    IsFunctionLike(false),
    IsC99Varargs(false),
    IsGNUVarargs(false),
    IsBuiltinMacro(false),
    HasCommaPasting(false),
    IsDisabled(false),
    IsUsed(false),
    IsAllowRedefinitionsWithoutWarning(false),
    IsWarnIfUnused(false),
    FromASTFile(false) {
}

unsigned MacroInfo::getDefinitionLengthSlow(SourceManager &SM) const {
  assert(!IsDefinitionLengthCached);
  IsDefinitionLengthCached = true;

  if (ReplacementTokens.empty())
    return (DefinitionLength = 0);

  const Token &firstToken = ReplacementTokens.front();
  const Token &lastToken = ReplacementTokens.back();
  SourceLocation macroStart = firstToken.getLocation();
  SourceLocation macroEnd = lastToken.getLocation();
  assert(macroStart.isValid() && macroEnd.isValid());
  assert((macroStart.isFileID() || firstToken.is(tok::comment)) &&
         "Macro defined in macro?");
  assert((macroEnd.isFileID() || lastToken.is(tok::comment)) &&
         "Macro defined in macro?");
  std::pair<FileID, unsigned>
      startInfo = SM.getDecomposedExpansionLoc(macroStart);
  std::pair<FileID, unsigned>
      endInfo = SM.getDecomposedExpansionLoc(macroEnd);
  assert(startInfo.first == endInfo.first &&
         "Macro definition spanning multiple FileIDs ?");
  assert(startInfo.second <= endInfo.second);
  DefinitionLength = endInfo.second - startInfo.second;
  DefinitionLength += lastToken.getLength();

  return DefinitionLength;
}

/// \brief Return true if the specified macro definition is equal to
/// this macro in spelling, arguments, and whitespace.
///
/// \param Syntactically if true, the macro definitions can be identical even
/// if they use different identifiers for the function macro parameters.
/// Otherwise the comparison is lexical and this implements the rules in
/// C99 6.10.3.
bool MacroInfo::isIdenticalTo(const MacroInfo &Other, Preprocessor &PP,
                              bool Syntactically) const {
  bool Lexically = !Syntactically;

  // Check # tokens in replacement, number of args, and various flags all match.
  if (ReplacementTokens.size() != Other.ReplacementTokens.size() ||
      getNumArgs() != Other.getNumArgs() ||
      isFunctionLike() != Other.isFunctionLike() ||
      isC99Varargs() != Other.isC99Varargs() ||
      isGNUVarargs() != Other.isGNUVarargs())
    return false;

  if (Lexically) {
    // Check arguments.
    for (arg_iterator I = arg_begin(), OI = Other.arg_begin(), E = arg_end();
         I != E; ++I, ++OI)
      if (*I != *OI) return false;
  }

  // Check all the tokens.
  for (unsigned i = 0, e = ReplacementTokens.size(); i != e; ++i) {
    const Token &A = ReplacementTokens[i];
    const Token &B = Other.ReplacementTokens[i];
    if (A.getKind() != B.getKind())
      return false;

    // If this isn't the first first token, check that the whitespace and
    // start-of-line characteristics match.
    if (i != 0 &&
        (A.isAtStartOfLine() != B.isAtStartOfLine() ||
         A.hasLeadingSpace() != B.hasLeadingSpace()))
      return false;

    // If this is an identifier, it is easy.
    if (A.getIdentifierInfo() || B.getIdentifierInfo()) {
      if (A.getIdentifierInfo() == B.getIdentifierInfo())
        continue;
      if (Lexically)
        return false;
      // With syntactic equivalence the parameter names can be different as long
      // as they are used in the same place.
      int AArgNum = getArgumentNum(A.getIdentifierInfo());
      if (AArgNum == -1)
        return false;
      if (AArgNum != Other.getArgumentNum(B.getIdentifierInfo()))
        return false;
      continue;
    }

    // Otherwise, check the spelling.
    if (PP.getSpelling(A) != PP.getSpelling(B))
      return false;
  }

  return true;
}

MacroDirective::DefInfo MacroDirective::getDefinition(bool AllowHidden) {
  MacroDirective *MD = this;
  SourceLocation UndefLoc;
  Optional<bool> isPublic;
  for (; MD; MD = MD->getPrevious()) {
    if (!AllowHidden && MD->isHidden())
      continue;

    if (DefMacroDirective *DefMD = dyn_cast<DefMacroDirective>(MD))
      return DefInfo(DefMD, UndefLoc,
                     !isPublic.hasValue() || isPublic.getValue());

    if (UndefMacroDirective *UndefMD = dyn_cast<UndefMacroDirective>(MD)) {
      UndefLoc = UndefMD->getLocation();
      continue;
    }

    VisibilityMacroDirective *VisMD = cast<VisibilityMacroDirective>(MD);
    if (!isPublic.hasValue())
      isPublic = VisMD->isPublic();
  }

  return DefInfo();
}

const MacroDirective::DefInfo
MacroDirective::findDirectiveAtLoc(SourceLocation L, SourceManager &SM) const {
  assert(L.isValid() && "SourceLocation is invalid.");
  for (DefInfo Def = getDefinition(); Def; Def = Def.getPreviousDefinition()) {
    if (Def.getLocation().isInvalid() ||  // For macros defined on the command line.
        SM.isBeforeInTranslationUnit(Def.getLocation(), L))
      return (!Def.isUndefined() ||
              SM.isBeforeInTranslationUnit(L, Def.getUndefLocation()))
                  ? Def : DefInfo();
  }
  return DefInfo();
}
