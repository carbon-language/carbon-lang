//===--- GoogleTidyModule.cpp - clang-tidy --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GoogleTidyModule.h"
#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void ExplicitConstructorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(constructorDecl().bind("ctor"), this);
}

// Looks for the token matching the predicate and returns the range of the found
// token including trailing whitespace.
SourceRange FindToken(const SourceManager &Sources, LangOptions LangOpts,
                      SourceLocation StartLoc, SourceLocation EndLoc,
                      bool (*Pred)(const Token &)) {
  FileID File = Sources.getFileID(Sources.getSpellingLoc(StartLoc));
  StringRef Buf = Sources.getBufferData(File);
  const char *StartChar = Sources.getCharacterData(StartLoc);
  Lexer Lex(StartLoc, LangOpts, StartChar, StartChar, Buf.end());
  Lex.SetCommentRetentionState(true);
  Token Tok;
  do {
    Lex.LexFromRawLexer(Tok);
    if (Pred(Tok)) {
      Token NextTok;
      Lex.LexFromRawLexer(NextTok);
      return SourceRange(Tok.getLocation(), NextTok.getLocation());
    }
  } while (Tok.isNot(tok::eof) && Tok.getLocation() < EndLoc);

  return SourceRange();
}

void ExplicitConstructorCheck::check(const MatchFinder::MatchResult &Result) {
  const CXXConstructorDecl *Ctor =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  // Do not be confused: isExplicit means 'explicit' keyword is present,
  // isImplicit means that it's a compiler-generated constructor.
  if (Ctor->isOutOfLine() || Ctor->isImplicit() || Ctor->isDeleted())
    return;

  if (Ctor->isExplicit() && Ctor->isCopyOrMoveConstructor()) {
    auto isKWExplicit = [](const Token &Tok) {
      return Tok.is(tok::raw_identifier) &&
             StringRef(Tok.getRawIdentifierData(), Tok.getLength()) ==
                 "explicit";
    };
    SourceRange ExplicitTokenRange =
        FindToken(*Result.SourceManager, Result.Context->getLangOpts(),
                  Ctor->getOuterLocStart(), Ctor->getLocEnd(), isKWExplicit);
    if (ExplicitTokenRange.isValid()) {
      DiagnosticBuilder Diag = diag(ExplicitTokenRange.getBegin(),
                                    "%0 constructor declared explicit.")
                               << (Ctor->isMoveConstructor() ? "Move" : "Copy");
      Diag << FixItHint::CreateRemoval(
          CharSourceRange::getCharRange(ExplicitTokenRange));
    }
  }

  if (Ctor->isExplicit() || Ctor->isCopyOrMoveConstructor() ||
      Ctor->getNumParams() == 0 || Ctor->getMinRequiredArguments() > 1)
    return;

  SourceLocation Loc = Ctor->getLocation();
  diag(Loc, "Single-argument constructors must be explicit")
      << FixItHint::CreateInsertion(Loc, "explicit ");
}

class GoogleModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.addCheckFactory(
        "google-explicit-constructor",
        new ClangTidyCheckFactory<ExplicitConstructorCheck>());
  }
};

// Register the GoogleTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<GoogleModule> X("google-module",
                                                    "Adds Google lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the GoogleModule.
volatile int GoogleModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
