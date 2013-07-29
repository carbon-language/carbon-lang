//===--- LLVMTidyModule.cpp - clang-tidy ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LLVMTidyModule.h"
#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void
NamespaceCommentCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(namespaceDecl().bind("namespace"), this);
}

void NamespaceCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const NamespaceDecl *ND = Result.Nodes.getNodeAs<NamespaceDecl>("namespace");
  Token Tok;
  SourceLocation Loc = ND->getRBraceLoc().getLocWithOffset(1);
  while (Lexer::getRawToken(Loc, Tok, *Result.SourceManager,
                            Result.Context->getLangOpts())) {
    Loc = Loc.getLocWithOffset(1);
  }
  // FIXME: Check that this namespace is "long".
  if (Tok.is(tok::comment)) {
    // FIXME: Check comment content.
    return;
  }
  std::string Fix = " // namespace";
  if (!ND->isAnonymousNamespace())
    Fix = Fix.append(" ").append(ND->getNameAsString());

  Context->Diag(ND->getLocation(),
                "namespace not terminated with a closing comment")
      << FixItHint::CreateInsertion(ND->getRBraceLoc().getLocWithOffset(1),
                                    Fix);
}

namespace {
class IncludeOrderPPCallbacks : public PPCallbacks {
public:
  explicit IncludeOrderPPCallbacks(ClangTidyContext &Context)
      : Context(Context) {}

  virtual void InclusionDirective(SourceLocation HashLoc,
                                  const Token &IncludeTok, StringRef FileName,
                                  bool IsAngled, CharSourceRange FilenameRange,
                                  const FileEntry *File, StringRef SearchPath,
                                  StringRef RelativePath,
                                  const Module *Imported) {
    // FIXME: This is a dummy implementation to show how to get at preprocessor
    // information. Implement a real include order check.
    Context.Diag(HashLoc, "This is an include");
  }

private:
  ClangTidyContext &Context;
};
} // namespace

void IncludeOrderCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  Compiler.getPreprocessor()
      .addPPCallbacks(new IncludeOrderPPCallbacks(*Context));
}

class LLVMModule : public ClangTidyModule {
public:
  virtual ~LLVMModule() {}

  virtual void addCheckFactories(ClangTidyCheckFactories &CheckFactories) {
    CheckFactories.addCheckFactory(
        "llvm-include-order", new ClangTidyCheckFactory<IncludeOrderCheck>());
    CheckFactories.addCheckFactory(
        "llvm-namespace-comment",
        new ClangTidyCheckFactory<NamespaceCommentCheck>());
  }
};

// Register the LLVMTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<LLVMModule> X("llvm-module",
                                                  "Adds LLVM lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the LLVMModule.
volatile int LLVMModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
