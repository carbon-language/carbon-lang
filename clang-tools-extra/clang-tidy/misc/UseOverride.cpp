//===--- UseOverride.cpp - clang-tidy -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseOverride.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void UseOverride::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(methodDecl(isOverride()).bind("method"), this);
}

// Re-lex the tokens to get precise locations to insert 'override' and remove
// 'virtual'.
static SmallVector<Token, 16> ParseTokens(CharSourceRange Range,
                                          const SourceManager &Sources,
                                          LangOptions LangOpts) {
  std::pair<FileID, unsigned> LocInfo =
      Sources.getDecomposedLoc(Range.getBegin());
  StringRef File = Sources.getBufferData(LocInfo.first);
  const char *TokenBegin = File.data() + LocInfo.second;
  Lexer RawLexer(Sources.getLocForStartOfFile(LocInfo.first), LangOpts,
                 File.begin(), TokenBegin, File.end());
  SmallVector<Token, 16> Tokens;
  Token Tok;
  while (!RawLexer.LexFromRawLexer(Tok)) {
    if (Tok.is(tok::semi) || Tok.is(tok::l_brace))
      break;
    if (Sources.isBeforeInTranslationUnit(Range.getEnd(), Tok.getLocation()))
      break;
    Tokens.push_back(Tok);
  }
  return Tokens;
}

static StringRef GetText(const Token &Tok, const SourceManager &Sources) {
  return StringRef(Sources.getCharacterData(Tok.getLocation()),
                   Tok.getLength());
}

void UseOverride::check(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *Method = Result.Nodes.getStmtAs<FunctionDecl>("method");
  const SourceManager &Sources = *Result.SourceManager;

  assert(Method != nullptr);
  if (Method->getInstantiatedFromMemberFunction() != nullptr)
    Method = Method->getInstantiatedFromMemberFunction();

  if (Method->isImplicit() || Method->getLocation().isMacroID() ||
      Method->isOutOfLine())
    return;

  if (Method->getAttr<clang::OverrideAttr>() != nullptr &&
      !Method->isVirtualAsWritten())
    return; // Nothing to do.

  DiagnosticBuilder Diag = diag(Method->getLocation(),
                                "Prefer using 'override' instead of 'virtual'");

  CharSourceRange FileRange =
      Lexer::makeFileCharRange(CharSourceRange::getTokenRange(
                                   Method->getLocStart(), Method->getLocEnd()),
                               Sources, Result.Context->getLangOpts());

  if (!FileRange.isValid())
    return;

  // FIXME: Instead of re-lexing and looking for specific macros such as
  // 'ABSTRACT', properly store the location of 'virtual' and '= 0' in each
  // FunctionDecl.
  SmallVector<Token, 16> Tokens = ParseTokens(FileRange, Sources,
                                              Result.Context->getLangOpts());

  // Add 'override' on inline declarations that don't already have it.
  if (Method->getAttr<clang::OverrideAttr>() == nullptr) {
    SourceLocation InsertLoc;
    StringRef ReplacementText = "override ";

    if (Method->hasAttrs()) {
      for (const clang::Attr *A : Method->getAttrs()) {
        if (!A->isImplicit()) {
          InsertLoc = Sources.getExpansionLoc(A->getLocation());
          break;
        }
      }
    }

    if (InsertLoc.isInvalid() && Method->doesThisDeclarationHaveABody())
      InsertLoc = Method->getBody()->getLocStart();

    if (!InsertLoc.isValid()) {
      if (Tokens.size() > 2 && GetText(Tokens.back(), Sources) == "0" &&
          GetText(Tokens[Tokens.size() - 2], Sources) == "=") {
        InsertLoc = Tokens[Tokens.size() - 2].getLocation();
      } else if (GetText(Tokens.back(), Sources) == "ABSTRACT") {
        InsertLoc = Tokens.back().getLocation();
      }
    }

    if (!InsertLoc.isValid()) {
      InsertLoc = FileRange.getEnd();
      ReplacementText = " override";
    }
    Diag << FixItHint::CreateInsertion(InsertLoc, ReplacementText);
  }

  if (Method->isVirtualAsWritten()) {
    for (Token Tok : Tokens) {
      if (Tok.is(tok::raw_identifier) && GetText(Tok, Sources) == "virtual") {
        Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
            Tok.getLocation(), Tok.getLocation()));
        break;
      }
    }
  }
}

} // namespace tidy
} // namespace clang
