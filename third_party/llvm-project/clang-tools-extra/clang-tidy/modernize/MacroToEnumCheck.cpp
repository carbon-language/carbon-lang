//===--- MacroToEnumCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MacroToEnumCheck.h"
#include "IntegralLiteralExpressionMatcher.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <string>

namespace clang {
namespace tidy {
namespace modernize {

static bool hasOnlyComments(SourceLocation Loc, const LangOptions &Options,
                            StringRef Text) {
  // Use a lexer to look for tokens; if we find something other than a single
  // hash, then there were intervening tokens between macro definitions.
  std::string Buffer{Text};
  Lexer Lex(Loc, Options, Buffer.c_str(), Buffer.c_str(),
            Buffer.c_str() + Buffer.size());
  Token Tok;
  bool SeenHash = false;
  while (!Lex.LexFromRawLexer(Tok)) {
    if (Tok.getKind() == tok::hash && !SeenHash) {
      SeenHash = true;
      continue;
    }
    return false;
  }

  // Everything in between was whitespace, so now just look for two blank lines,
  // consisting of two consecutive EOL sequences, either '\n', '\r' or '\r\n'.
  enum class WhiteSpace {
    Nothing,
    CR,
    LF,
    CRLF,
    CRLFCR,
  };

  WhiteSpace State = WhiteSpace::Nothing;
  for (char C : Text) {
    switch (C) {
    case '\r':
      if (State == WhiteSpace::CR)
        return false;

      State = State == WhiteSpace::CRLF ? WhiteSpace::CRLFCR : WhiteSpace::CR;
      break;

    case '\n':
      if (State == WhiteSpace::LF || State == WhiteSpace::CRLFCR)
        return false;

      State = State == WhiteSpace::CR ? WhiteSpace::CRLF : WhiteSpace::LF;
      break;

    default:
      State = WhiteSpace::Nothing;
      break;
    }
  }

  return true;
}

static StringRef getTokenName(const Token &Tok) {
  return Tok.is(tok::raw_identifier) ? Tok.getRawIdentifier()
                                     : Tok.getIdentifierInfo()->getName();
}

namespace {

struct EnumMacro {
  EnumMacro(Token Name, const MacroDirective *Directive)
      : Name(Name), Directive(Directive) {}

  Token Name;
  const MacroDirective *Directive;
};

using MacroList = SmallVector<EnumMacro>;

enum class IncludeGuard { None, FileChanged, IfGuard, DefineGuard };

struct FileState {
  FileState()
      : ConditionScopes(0), LastLine(0), GuardScanner(IncludeGuard::None) {}

  int ConditionScopes;
  unsigned int LastLine;
  IncludeGuard GuardScanner;
  SourceLocation LastMacroLocation;
};

} // namespace

class MacroToEnumCallbacks : public PPCallbacks {
public:
  MacroToEnumCallbacks(MacroToEnumCheck *Check, const LangOptions &LangOptions,
                       const SourceManager &SM)
      : Check(Check), LangOpts(LangOptions), SM(SM) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          Optional<FileEntryRef> File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    clearCurrentEnum(HashLoc);
  }

  // Keep track of macro definitions that look like enums.
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;

  // Undefining an enum-like macro results in the enum set being dropped.
  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override;

  // Conditional compilation clears any adjacent enum-like macros.
  // Macros used in conditional expressions clear any adjacent enum-like
  // macros.
  // Include guards are either
  //   #if !defined(GUARD)
  // or
  //   #ifndef GUARD
  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override {
    conditionStart(Loc);
    checkCondition(ConditionRange);
  }
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    conditionStart(Loc);
    checkName(MacroNameTok);
  }
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override {
    conditionStart(Loc);
    checkName(MacroNameTok);
  }
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override {
    checkCondition(ConditionRange);
  }
  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override {
    checkName(MacroNameTok);
  }
  void Elifdef(SourceLocation Loc, SourceRange ConditionRange,
      SourceLocation IfLoc) override {
    PPCallbacks::Elifdef(Loc, ConditionRange, IfLoc);
  }
  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override {
    checkName(MacroNameTok);
  }
  void Elifndef(SourceLocation Loc, SourceRange ConditionRange,
      SourceLocation IfLoc) override {
    PPCallbacks::Elifndef(Loc, ConditionRange, IfLoc);
  }
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override;
  void PragmaDirective(SourceLocation Loc,
                       PragmaIntroducerKind Introducer) override;

  // After we've seen everything, issue warnings and fix-its.
  void EndOfMainFile() override;

  void invalidateRange(SourceRange Range);

private:
  void newEnum() {
    if (Enums.empty() || !Enums.back().empty())
      Enums.emplace_back();
  }
  bool insideConditional() const {
    return (CurrentFile->GuardScanner == IncludeGuard::DefineGuard &&
            CurrentFile->ConditionScopes > 1) ||
           (CurrentFile->GuardScanner != IncludeGuard::DefineGuard &&
            CurrentFile->ConditionScopes > 0);
  }
  bool isConsecutiveMacro(const MacroDirective *MD) const;
  void rememberLastMacroLocation(const MacroDirective *MD) {
    CurrentFile->LastLine = SM.getSpellingLineNumber(MD->getLocation());
    CurrentFile->LastMacroLocation = Lexer::getLocForEndOfToken(
        MD->getMacroInfo()->getDefinitionEndLoc(), 0, SM, LangOpts);
  }
  void clearLastMacroLocation() {
    CurrentFile->LastLine = 0;
    CurrentFile->LastMacroLocation = SourceLocation{};
  }
  void clearCurrentEnum(SourceLocation Loc);
  void conditionStart(const SourceLocation &Loc);
  void checkCondition(SourceRange ConditionRange);
  void checkName(const Token &MacroNameTok);
  void rememberExpressionName(const Token &Tok);
  void rememberExpressionTokens(ArrayRef<Token> MacroTokens);
  void invalidateExpressionNames();
  void issueDiagnostics();
  void warnMacroEnum(const EnumMacro &Macro) const;
  void fixEnumMacro(const MacroList &MacroList) const;
  bool isInitializer(ArrayRef<Token> MacroTokens);

  MacroToEnumCheck *Check;
  const LangOptions &LangOpts;
  const SourceManager &SM;
  SmallVector<MacroList> Enums;
  SmallVector<FileState> Files;
  std::vector<std::string> ExpressionNames;
  FileState *CurrentFile = nullptr;
};

bool MacroToEnumCallbacks::isConsecutiveMacro(const MacroDirective *MD) const {
  if (CurrentFile->LastMacroLocation.isInvalid())
    return false;

  SourceLocation Loc = MD->getLocation();
  if (CurrentFile->LastLine + 1 == SM.getSpellingLineNumber(Loc))
    return true;

  SourceLocation Define =
      SM.translateLineCol(SM.getFileID(Loc), SM.getSpellingLineNumber(Loc), 1);
  CharSourceRange BetweenMacros{
      SourceRange{CurrentFile->LastMacroLocation, Define}, true};
  CharSourceRange CharRange =
      Lexer::makeFileCharRange(BetweenMacros, SM, LangOpts);
  StringRef BetweenText = Lexer::getSourceText(CharRange, SM, LangOpts);
  return hasOnlyComments(Define, LangOpts, BetweenText);
}

void MacroToEnumCallbacks::clearCurrentEnum(SourceLocation Loc) {
  // Only drop the most recent Enum set if the directive immediately follows.
  if (!Enums.empty() && !Enums.back().empty() &&
      SM.getSpellingLineNumber(Loc) == CurrentFile->LastLine + 1)
    Enums.pop_back();

  clearLastMacroLocation();
}

void MacroToEnumCallbacks::conditionStart(const SourceLocation &Loc) {
  ++CurrentFile->ConditionScopes;
  clearCurrentEnum(Loc);
  if (CurrentFile->GuardScanner == IncludeGuard::FileChanged)
    CurrentFile->GuardScanner = IncludeGuard::IfGuard;
}

void MacroToEnumCallbacks::checkCondition(SourceRange Range) {
  CharSourceRange CharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Range), SM, LangOpts);
  std::string Text = Lexer::getSourceText(CharRange, SM, LangOpts).str();
  Lexer Lex(CharRange.getBegin(), LangOpts, Text.data(), Text.data(),
            Text.data() + Text.size());
  Token Tok;
  bool End = false;
  while (!End) {
    End = Lex.LexFromRawLexer(Tok);
    if (Tok.is(tok::raw_identifier) &&
        Tok.getRawIdentifier().str() != "defined")
      checkName(Tok);
  }
}

void MacroToEnumCallbacks::checkName(const Token &MacroNameTok) {
  rememberExpressionName(MacroNameTok);

  StringRef Id = getTokenName(MacroNameTok);
  llvm::erase_if(Enums, [&Id](const MacroList &MacroList) {
    return llvm::any_of(MacroList, [&Id](const EnumMacro &Macro) {
      return getTokenName(Macro.Name) == Id;
    });
  });
}

void MacroToEnumCallbacks::rememberExpressionName(const Token &Tok) {
  std::string Id = getTokenName(Tok).str();
  auto Pos = llvm::lower_bound(ExpressionNames, Id);
  if (Pos == ExpressionNames.end() || *Pos != Id) {
    ExpressionNames.insert(Pos, Id);
  }
}

void MacroToEnumCallbacks::rememberExpressionTokens(
    ArrayRef<Token> MacroTokens) {
  for (Token Tok : MacroTokens) {
    if (Tok.isAnyIdentifier())
      rememberExpressionName(Tok);
  }
}

void MacroToEnumCallbacks::FileChanged(SourceLocation Loc,
                                       FileChangeReason Reason,
                                       SrcMgr::CharacteristicKind FileType,
                                       FileID PrevFID) {
  newEnum();
  if (Reason == EnterFile) {
    Files.emplace_back();
    if (!SM.isInMainFile(Loc))
      Files.back().GuardScanner = IncludeGuard::FileChanged;
  } else if (Reason == ExitFile) {
    assert(CurrentFile->ConditionScopes == 0);
    Files.pop_back();
  }
  CurrentFile = &Files.back();
}

bool MacroToEnumCallbacks::isInitializer(ArrayRef<Token> MacroTokens)
{
  IntegralLiteralExpressionMatcher Matcher(MacroTokens, LangOpts.C99 == 0);
  bool Matched = Matcher.match();
  bool isC = !LangOpts.CPlusPlus;
  if (isC && (Matcher.largestLiteralSize() != LiteralSize::Int &&
              Matcher.largestLiteralSize() != LiteralSize::UnsignedInt))
    return false;

  return Matched;
}


// Any defined but rejected macro is scanned for identifiers that
// are to be excluded as enums.
void MacroToEnumCallbacks::MacroDefined(const Token &MacroNameTok,
                                        const MacroDirective *MD) {
  // Include guards are never candidates for becoming an enum.
  if (CurrentFile->GuardScanner == IncludeGuard::IfGuard) {
    CurrentFile->GuardScanner = IncludeGuard::DefineGuard;
    return;
  }

  if (insideConditional())
    return;

  if (SM.getFilename(MD->getLocation()).empty())
    return;

  const MacroInfo *Info = MD->getMacroInfo();
  ArrayRef<Token> MacroTokens = Info->tokens();
  if (Info->isBuiltinMacro() || MacroTokens.empty())
    return;
  if (Info->isFunctionLike()) {
    rememberExpressionTokens(MacroTokens);
    return;
  }

  if (!isInitializer(MacroTokens))
    return;

  if (!isConsecutiveMacro(MD))
    newEnum();
  Enums.back().emplace_back(MacroNameTok, MD);
  rememberLastMacroLocation(MD);
}

// Any macro that is undefined removes all adjacent macros from consideration as
// an enum and starts a new enum scan.
void MacroToEnumCallbacks::MacroUndefined(const Token &MacroNameTok,
                                          const MacroDefinition &MD,
                                          const MacroDirective *Undef) {
  rememberExpressionName(MacroNameTok);

  auto MatchesToken = [&MacroNameTok](const EnumMacro &Macro) {
    return getTokenName(Macro.Name) == getTokenName(MacroNameTok);
  };

  auto It = llvm::find_if(Enums, [MatchesToken](const MacroList &MacroList) {
    return llvm::any_of(MacroList, MatchesToken);
  });
  if (It != Enums.end())
    Enums.erase(It);

  clearLastMacroLocation();
  CurrentFile->GuardScanner = IncludeGuard::None;
}

void MacroToEnumCallbacks::Endif(SourceLocation Loc, SourceLocation IfLoc) {
  // The if directive for the include guard isn't counted in the
  // ConditionScopes.
  if (CurrentFile->ConditionScopes == 0 &&
      CurrentFile->GuardScanner == IncludeGuard::DefineGuard)
    return;

  // We don't need to clear the current enum because the start of the
  // conditional block already took care of that.
  assert(CurrentFile->ConditionScopes > 0);
  --CurrentFile->ConditionScopes;
}

namespace {

template <size_t N>
bool textEquals(const char (&Needle)[N], const char *HayStack) {
  return StringRef{HayStack, N - 1} == Needle;
}

template <size_t N> size_t len(const char (&)[N]) { return N - 1; }

} // namespace

void MacroToEnumCallbacks::PragmaDirective(SourceLocation Loc,
                                           PragmaIntroducerKind Introducer) {
  if (CurrentFile->GuardScanner != IncludeGuard::FileChanged)
    return;

  bool Invalid = false;
  const char *Text = SM.getCharacterData(
      Lexer::getLocForEndOfToken(Loc, 0, SM, LangOpts), &Invalid);
  if (Invalid)
    return;

  while (*Text && std::isspace(*Text))
    ++Text;

  if (textEquals("pragma", Text))
    return;

  Text += len("pragma");
  while (*Text && std::isspace(*Text))
    ++Text;

  if (textEquals("once", Text))
    CurrentFile->GuardScanner = IncludeGuard::IfGuard;
}

void MacroToEnumCallbacks::invalidateExpressionNames() {
  for (const std::string &Id : ExpressionNames) {
    llvm::erase_if(Enums, [Id](const MacroList &MacroList) {
      return llvm::any_of(MacroList, [&Id](const EnumMacro &Macro) {
        return getTokenName(Macro.Name) == Id;
      });
    });
  }
}

void MacroToEnumCallbacks::EndOfMainFile() {
    invalidateExpressionNames();
    issueDiagnostics();
}

void MacroToEnumCallbacks::invalidateRange(SourceRange Range) {
  llvm::erase_if(Enums, [Range](const MacroList &MacroList) {
    return llvm::any_of(MacroList, [Range](const EnumMacro &Macro) {
      return Macro.Directive->getLocation() >= Range.getBegin() &&
             Macro.Directive->getLocation() <= Range.getEnd();
    });
  });
}

void MacroToEnumCallbacks::issueDiagnostics() {
  for (const MacroList &MacroList : Enums) {
    if (MacroList.empty())
      continue;

    for (const EnumMacro &Macro : MacroList)
      warnMacroEnum(Macro);

    fixEnumMacro(MacroList);
  }
}

void MacroToEnumCallbacks::warnMacroEnum(const EnumMacro &Macro) const {
  Check->diag(Macro.Directive->getLocation(),
              "macro '%0' defines an integral constant; prefer an enum instead")
      << getTokenName(Macro.Name);
}

void MacroToEnumCallbacks::fixEnumMacro(const MacroList &MacroList) const {
  SourceLocation Begin =
      MacroList.front().Directive->getMacroInfo()->getDefinitionLoc();
  Begin = SM.translateLineCol(SM.getFileID(Begin),
                              SM.getSpellingLineNumber(Begin), 1);
  DiagnosticBuilder Diagnostic =
      Check->diag(Begin, "replace macro with enum")
      << FixItHint::CreateInsertion(Begin, "enum {\n");

  for (size_t I = 0u; I < MacroList.size(); ++I) {
    const EnumMacro &Macro = MacroList[I];
    SourceLocation DefineEnd =
        Macro.Directive->getMacroInfo()->getDefinitionLoc();
    SourceLocation DefineBegin = SM.translateLineCol(
        SM.getFileID(DefineEnd), SM.getSpellingLineNumber(DefineEnd), 1);
    CharSourceRange DefineRange;
    DefineRange.setBegin(DefineBegin);
    DefineRange.setEnd(DefineEnd);
    Diagnostic << FixItHint::CreateRemoval(DefineRange);

    SourceLocation NameEnd = Lexer::getLocForEndOfToken(
        Macro.Directive->getMacroInfo()->getDefinitionLoc(), 0, SM, LangOpts);
    Diagnostic << FixItHint::CreateInsertion(NameEnd, " =");

    SourceLocation ValueEnd = Lexer::getLocForEndOfToken(
        Macro.Directive->getMacroInfo()->getDefinitionEndLoc(), 0, SM,
        LangOpts);
    if (I < MacroList.size() - 1)
      Diagnostic << FixItHint::CreateInsertion(ValueEnd, ",");
  }

  SourceLocation End = Lexer::getLocForEndOfToken(
      MacroList.back().Directive->getMacroInfo()->getDefinitionEndLoc(), 0, SM,
      LangOpts);
  End = SM.translateLineCol(SM.getFileID(End),
                            SM.getSpellingLineNumber(End) + 1, 1);
  Diagnostic << FixItHint::CreateInsertion(End, "};\n");
}

void MacroToEnumCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  auto Callback = std::make_unique<MacroToEnumCallbacks>(this, getLangOpts(), SM);
  PPCallback = Callback.get();
  PP->addPPCallbacks(std::move(Callback));
}

void MacroToEnumCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  using namespace ast_matchers;
  auto TopLevelDecl = hasParent(translationUnitDecl());
  Finder->addMatcher(decl(TopLevelDecl).bind("top"), this);
}

static bool isValid(SourceRange Range) {
  return Range.getBegin().isValid() && Range.getEnd().isValid();
}

static bool empty(SourceRange Range) {
  return Range.getBegin() == Range.getEnd();
}

void MacroToEnumCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto *TLDecl = Result.Nodes.getNodeAs<Decl>("top");
  if (TLDecl == nullptr)
      return;

  SourceRange Range = TLDecl->getSourceRange();
  if (auto *TemplateFn = Result.Nodes.getNodeAs<FunctionTemplateDecl>("top")) {
    if (TemplateFn->isThisDeclarationADefinition() && TemplateFn->hasBody())
      Range = SourceRange{TemplateFn->getBeginLoc(),
                          TemplateFn->getUnderlyingDecl()->getBodyRBrace()};
  }

  if (isValid(Range) && !empty(Range))
    PPCallback->invalidateRange(Range);
}

} // namespace modernize
} // namespace tidy
} // namespace clang
