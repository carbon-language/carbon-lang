//===-- AddOverride/AddOverrideActions.cpp - add C++11 override -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the definition of the AddOverrideFixer class
/// which is used as an ASTMatcher callback.
///
//===----------------------------------------------------------------------===//

#include "AddOverrideActions.h"
#include "AddOverrideMatchers.h"
#include "Core/Transform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

namespace {

SourceLocation
backwardSkipWhitespacesAndComments(const SourceManager &SM,
                                   const clang::ASTContext &Context,
                                   SourceLocation Loc) {
  for (;;) {
    do {
      Loc = Loc.getLocWithOffset(-1);
    } while (isWhitespace(*FullSourceLoc(Loc, SM).getCharacterData()));

    Token Tok;
    SourceLocation Beginning =
        Lexer::GetBeginningOfToken(Loc, SM, Context.getLangOpts());
    const bool Invalid =
        Lexer::getRawToken(Beginning, Tok, SM, Context.getLangOpts());

    assert(!Invalid && "Expected a valid token.");
    if (Invalid || Tok.getKind() != tok::comment)
      return Loc.getLocWithOffset(1);
  }
}

} // end anonymous namespace

void AddOverrideFixer::run(const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;

  const CXXMethodDecl *M = Result.Nodes.getDeclAs<CXXMethodDecl>(MethodId);
  assert(M && "Bad Callback. No node provided");

  if (const FunctionDecl *TemplateMethod = M->getTemplateInstantiationPattern())
    M = cast<CXXMethodDecl>(TemplateMethod);

  if (!Owner.isFileModifiable(SM, M->getLocStart()))
    return;

  // First check that there isn't already an override attribute.
  if (M->hasAttr<OverrideAttr>())
    return;

  // FIXME: Pure methods are not supported yet as it is difficult to track down
  // the location of '= 0'.
  if (M->isPure())
    return;

  if (M->getParent()->hasAnyDependentBases())
    return;

  SourceLocation StartLoc;
  if (M->hasInlineBody()) {
    // Insert the override specifier before the function body.
    StartLoc = backwardSkipWhitespacesAndComments(SM, *Result.Context,
                                                  M->getBody()->getLocStart());
  } else {
    StartLoc = SM.getSpellingLoc(M->getLocEnd());
    StartLoc = Lexer::getLocForEndOfToken(StartLoc, 0, SM, LangOptions());
  }

  std::string ReplacementText = " override";
  if (DetectMacros) {
    assert(PP && "No access to Preprocessor object for macro detection");
    clang::TokenValue Tokens[] = { PP->getIdentifierInfo("override") };
    llvm::StringRef MacroName = PP->getLastMacroWithSpelling(StartLoc, Tokens);
    if (!MacroName.empty())
      ReplacementText = (" " + MacroName).str();
  }
  Owner.addReplacementForCurrentTU(
      tooling::Replacement(SM, StartLoc, 0, ReplacementText));
  ++AcceptedChanges;
}
