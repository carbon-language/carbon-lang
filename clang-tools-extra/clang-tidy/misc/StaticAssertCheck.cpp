//===--- StaticAssertCheck.cpp - clang-tidy -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StaticAssertCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <string>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

StaticAssertCheck::StaticAssertCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void StaticAssertCheck::registerMatchers(MatchFinder *Finder) {
  auto IsAlwaysFalse = ignoringParenImpCasts(
      anyOf(boolLiteral(equals(false)).bind("isAlwaysFalse"),
            integerLiteral(equals(0)).bind("isAlwaysFalse")));
  auto AssertExprRoot = anyOf(
      binaryOperator(
          hasOperatorName("&&"),
          hasEitherOperand(ignoringImpCasts(stringLiteral().bind("assertMSG"))),
          anyOf(binaryOperator(hasEitherOperand(IsAlwaysFalse)), anything()))
          .bind("assertExprRoot"),
      IsAlwaysFalse);
  auto Condition = expr(anyOf(
      expr(ignoringParenCasts(anyOf(
          AssertExprRoot,
          unaryOperator(hasUnaryOperand(ignoringParenCasts(AssertExprRoot)))))),
      anything()));

  Finder->addMatcher(
      stmt(anyOf(conditionalOperator(hasCondition(Condition.bind("condition"))),
                 ifStmt(hasCondition(Condition.bind("condition")))),
           unless(isInTemplateInstantiation())).bind("condStmt"),
      this);
}

void StaticAssertCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext *ASTCtx = Result.Context;
  const LangOptions &Opts = ASTCtx->getLangOpts();
  const SourceManager &SM = ASTCtx->getSourceManager();
  const auto *CondStmt = Result.Nodes.getNodeAs<Stmt>("condStmt");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *IsAlwaysFalse = Result.Nodes.getNodeAs<Expr>("isAlwaysFalse");
  const auto *AssertMSG = Result.Nodes.getNodeAs<StringLiteral>("assertMSG");
  const auto *AssertExprRoot =
      Result.Nodes.getNodeAs<BinaryOperator>("assertExprRoot");
  SourceLocation AssertExpansionLoc = CondStmt->getLocStart();

  if (!Opts.CPlusPlus11 || !AssertExpansionLoc.isValid() ||
      !AssertExpansionLoc.isMacroID())
    return;

  StringRef MacroName =
      Lexer::getImmediateMacroName(AssertExpansionLoc, SM, Opts);

  if (MacroName != "assert" || Condition->isValueDependent() ||
      Condition->isTypeDependent() || Condition->isInstantiationDependent() ||
      !Condition->isEvaluatable(*ASTCtx))
    return;

  // False literal is not the result of macro expansion.
  if (IsAlwaysFalse) {
    SourceLocation FalseLiteralLoc =
        SM.getImmediateSpellingLoc(IsAlwaysFalse->getExprLoc());
    if (!FalseLiteralLoc.isMacroID())
      return;

    StringRef FalseMacroName =
        Lexer::getImmediateMacroName(FalseLiteralLoc, SM, Opts);
    if (FalseMacroName.compare_lower("false") == 0)
      return;
  }

  SourceLocation AssertLoc = SM.getImmediateMacroCallerLoc(AssertExpansionLoc);

  SmallVector<FixItHint, 4> FixItHints;
  SourceLocation LastParenLoc;
  if (AssertLoc.isValid() && !AssertLoc.isMacroID() &&
      (LastParenLoc = getLastParenLoc(ASTCtx, AssertLoc)).isValid()) {
    FixItHints.push_back(
        FixItHint::CreateReplacement(SourceRange(AssertLoc), "static_assert"));

    std::string StaticAssertMSG = ", \"\"";
    if (AssertExprRoot) {
      FixItHints.push_back(FixItHint::CreateRemoval(
          SourceRange(AssertExprRoot->getOperatorLoc())));
      FixItHints.push_back(FixItHint::CreateRemoval(
          SourceRange(AssertMSG->getLocStart(), AssertMSG->getLocEnd())));
      StaticAssertMSG = (Twine(", \"") + AssertMSG->getString() + "\"").str();
    }

    FixItHints.push_back(
        FixItHint::CreateInsertion(LastParenLoc, StaticAssertMSG));
  }

  diag(AssertLoc, "found assert() that could be replaced by static_assert()")
      << FixItHints;
}

SourceLocation StaticAssertCheck::getLastParenLoc(const ASTContext *ASTCtx,
                                                  SourceLocation AssertLoc) {
  const LangOptions &Opts = ASTCtx->getLangOpts();
  const SourceManager &SM = ASTCtx->getSourceManager();

  llvm::MemoryBuffer *Buffer = SM.getBuffer(SM.getFileID(AssertLoc));
  if (!Buffer)
    return SourceLocation();

  const char *BufferPos = SM.getCharacterData(AssertLoc);

  Token Token;
  Lexer Lexer(SM.getLocForStartOfFile(SM.getFileID(AssertLoc)), Opts,
              Buffer->getBufferStart(), BufferPos, Buffer->getBufferEnd());

  //        assert                          first left parenthesis
  if (Lexer.LexFromRawLexer(Token) || Lexer.LexFromRawLexer(Token) ||
      !Token.is(tok::l_paren))
    return SourceLocation();

  unsigned int ParenCount = 1;
  while (ParenCount && !Lexer.LexFromRawLexer(Token)) {
    if (Token.is(tok::l_paren))
      ++ParenCount;
    else if (Token.is(tok::r_paren))
      --ParenCount;
  }

  return Token.getLocation();
}

} // namespace tidy
} // namespace clang
