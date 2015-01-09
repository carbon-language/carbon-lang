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
static SmallVector<Token, 16>
ParseTokens(CharSourceRange Range, const MatchFinder::MatchResult &Result) {
  const SourceManager &Sources = *Result.SourceManager;
  std::pair<FileID, unsigned> LocInfo =
      Sources.getDecomposedLoc(Range.getBegin());
  StringRef File = Sources.getBufferData(LocInfo.first);
  const char *TokenBegin = File.data() + LocInfo.second;
  Lexer RawLexer(Sources.getLocForStartOfFile(LocInfo.first),
                 Result.Context->getLangOpts(), File.begin(), TokenBegin,
                 File.end());
  SmallVector<Token, 16> Tokens;
  Token Tok;
  while (!RawLexer.LexFromRawLexer(Tok)) {
    if (Tok.is(tok::semi) || Tok.is(tok::l_brace))
      break;
    if (Sources.isBeforeInTranslationUnit(Range.getEnd(), Tok.getLocation()))
      break;
    if (Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = Result.Context->Idents.get(StringRef(
          Sources.getCharacterData(Tok.getLocation()), Tok.getLength()));
      Tok.setIdentifierInfo(&Info);
      Tok.setKind(Info.getTokenID());
    }
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

  bool HasVirtual = Method->isVirtualAsWritten();
  bool HasOverride = Method->getAttr<OverrideAttr>();
  bool HasFinal = Method->getAttr<FinalAttr>();

  bool OnlyVirtualSpecified = HasVirtual && !HasOverride && !HasFinal;
  unsigned KeywordCount = HasVirtual + HasOverride + HasFinal;

  if (!OnlyVirtualSpecified && KeywordCount == 1)
    return; // Nothing to do.

  DiagnosticBuilder Diag = diag(
      Method->getLocation(),
      OnlyVirtualSpecified
          ? "Prefer using 'override' or (rarely) 'final' instead of 'virtual'"
          : "Annotate this function with 'override' or (rarely) 'final'");

  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Method->getSourceRange()), Sources,
      Result.Context->getLangOpts());

  if (!FileRange.isValid())
    return;

  // FIXME: Instead of re-lexing and looking for specific macros such as
  // 'ABSTRACT', properly store the location of 'virtual' and '= 0' in each
  // FunctionDecl.
  SmallVector<Token, 16> Tokens = ParseTokens(FileRange, Result);

  // Add 'override' on inline declarations that don't already have it.
  if (!HasFinal && !HasOverride) {
    SourceLocation InsertLoc;
    StringRef ReplacementText = "override ";

    for (Token T : Tokens) {
      if (T.is(tok::kw___attribute)) {
        InsertLoc = T.getLocation();
        break;
      }
    }

    if (Method->hasAttrs()) {
      for (const clang::Attr *A : Method->getAttrs()) {
        if (!A->isImplicit()) {
          SourceLocation Loc =
              Sources.getExpansionLoc(A->getRange().getBegin());
          if (!InsertLoc.isValid() ||
              Sources.isBeforeInTranslationUnit(Loc, InsertLoc))
            InsertLoc = Loc;
        }
      }
    }

    if (InsertLoc.isInvalid() && Method->doesThisDeclarationHaveABody() &&
        Method->getBody() && !Method->isDefaulted())
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

  if (HasFinal && HasOverride) {
    SourceLocation OverrideLoc = Method->getAttr<OverrideAttr>()->getLocation();
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getTokenRange(OverrideLoc, OverrideLoc));
  }

  if (Method->isVirtualAsWritten()) {
    for (Token Tok : Tokens) {
      if (Tok.is(tok::kw_virtual)) {
        Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
            Tok.getLocation(), Tok.getLocation()));
        break;
      }
    }
  }
}

} // namespace tidy
} // namespace clang
