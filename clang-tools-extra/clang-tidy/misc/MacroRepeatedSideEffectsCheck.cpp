//===--- MacroRepeatedSideEffectsCheck.cpp - clang-tidy--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MacroRepeatedSideEffectsCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroArgs.h"

namespace clang {
namespace tidy {
namespace misc {

namespace {
class MacroRepeatedPPCallbacks : public PPCallbacks {
public:
  MacroRepeatedPPCallbacks(ClangTidyCheck &Check, SourceManager &SM,
                           Preprocessor &PP)
      : Check(Check), SM(SM), PP(PP) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;

private:
  ClangTidyCheck &Check;
  SourceManager &SM;
  Preprocessor &PP;

  unsigned CountArgumentExpansions(const MacroInfo *MI,
                                   const IdentifierInfo *Arg) const;

  bool HasSideEffects(const Token *ResultArgToks) const;
};
} // namespace

void MacroRepeatedPPCallbacks::MacroExpands(const Token &MacroNameTok,
                                            const MacroDefinition &MD,
                                            SourceRange Range,
                                            const MacroArgs *Args) {
  // Ignore macro argument expansions.
  if (!Range.getBegin().isFileID())
    return;

  const MacroInfo *MI = MD.getMacroInfo();

  // Bail out if the contents of the macro are containing keywords that are
  // making the macro too complex.
  if (std::find_if(
          MI->tokens().begin(), MI->tokens().end(), [](const Token &T) {
            return T.isOneOf(tok::question, tok::kw_if, tok::kw_else,
                             tok::kw_switch, tok::kw_case, tok::kw_break,
                             tok::kw_while, tok::kw_do, tok::kw_for,
                             tok::kw_continue, tok::kw_goto, tok::kw_return);
          }) != MI->tokens().end())
    return;

  for (unsigned ArgNo = 0U; ArgNo < MI->getNumArgs(); ++ArgNo) {
    const IdentifierInfo *Arg = *(MI->arg_begin() + ArgNo);
    const Token *ResultArgToks = Args->getUnexpArgument(ArgNo);

    if (HasSideEffects(ResultArgToks) &&
        CountArgumentExpansions(MI, Arg) >= 2) {
      Check.diag(ResultArgToks->getLocation(),
                 "side effects in the %ordinal0 macro argument '%1' are "
                 "repeated in macro expansion")
          << (ArgNo + 1) << Arg->getName();
      Check.diag(MI->getDefinitionLoc(), "macro %0 defined here",
                 DiagnosticIDs::Note)
          << MacroNameTok.getIdentifierInfo();
    }
  }
}

unsigned MacroRepeatedPPCallbacks::CountArgumentExpansions(
    const MacroInfo *MI, const IdentifierInfo *Arg) const {
  unsigned CountInMacro = 0;
  bool SkipParen = false;
  int SkipParenCount = 0;
  for (const auto &T : MI->tokens()) {
    // If current token is a parenthesis, skip it.
    if (SkipParen) {
      if (T.is(tok::l_paren))
        SkipParenCount++;
      else if (T.is(tok::r_paren))
        SkipParenCount--;
      SkipParen = (SkipParenCount != 0);
      if (SkipParen)
        continue;
    }

    IdentifierInfo *TII = T.getIdentifierInfo();
    // If not existent, skip it.
    if (TII == nullptr)
      continue;

    // If a builtin is found within the macro definition, skip next
    // parenthesis.
    if (TII->getBuiltinID() != 0) {
      SkipParen = true;
      continue;
    }

    // If another macro is found within the macro definition, skip the macro
    // and the eventual arguments.
    if (TII->hasMacroDefinition()) {
      const MacroInfo *M = PP.getMacroDefinition(TII).getMacroInfo();
      if (M != nullptr && M->isFunctionLike())
        SkipParen = true;
      continue;
    }

    // Count argument.
    if (TII == Arg)
      CountInMacro++;
  }
  return CountInMacro;
}

bool MacroRepeatedPPCallbacks::HasSideEffects(
    const Token *ResultArgToks) const {
  for (; ResultArgToks->isNot(tok::eof); ++ResultArgToks) {
    if (ResultArgToks->isOneOf(tok::plusplus, tok::minusminus))
      return true;
  }
  return false;
}

void MacroRepeatedSideEffectsCheck::registerPPCallbacks(
    CompilerInstance &Compiler) {
  Compiler.getPreprocessor().addPPCallbacks(
      ::llvm::make_unique<MacroRepeatedPPCallbacks>(
          *this, Compiler.getSourceManager(), Compiler.getPreprocessor()));
}

} // namespace misc
} // namespace tidy
} // namespace clang
