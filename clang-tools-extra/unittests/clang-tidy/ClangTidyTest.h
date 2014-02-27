//===--- ClangTidyTest.h - clang-tidy ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_TIDY_CLANG_TIDY_TEST_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_TIDY_CLANG_TIDY_TEST_H

#include "ClangTidy.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace tidy {
namespace test {

class TestPPAction : public PreprocessOnlyAction {
public:
  TestPPAction(ClangTidyCheck &Check, ClangTidyContext *Context)
      : Check(Check), Context(Context) {}

private:
  bool BeginSourceFileAction(CompilerInstance &Compiler,
                             llvm::StringRef file_name) LLVM_OVERRIDE {
    Context->setSourceManager(&Compiler.getSourceManager());
    Check.registerPPCallbacks(Compiler);
    return true;
  }

  ClangTidyCheck &Check;
  ClangTidyContext *Context;
};

template <typename T> std::string runCheckOnCode(StringRef Code) {
  T Check;
  SmallVector<ClangTidyError, 16> Errors;
  ClangTidyContext Context(&Errors);
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  Check.setContext(&Context);

  if (!tooling::runToolOnCode(new TestPPAction(Check, &Context), Code))
    return "";
  ast_matchers::MatchFinder Finder;
  Check.registerMatchers(&Finder);
  OwningPtr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));
  if (!tooling::runToolOnCode(Factory->create(), Code))
    return "";
  DiagConsumer.finish();
  tooling::Replacements Fixes;
  for (SmallVector<ClangTidyError, 16>::const_iterator I = Errors.begin(),
                                                       E = Errors.end();
       I != E; ++I)
    Fixes.insert(I->Fix.begin(), I->Fix.end());
  return tooling::applyAllReplacements(Code, Fixes);
}

} // namespace test
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_TIDY_CLANG_TIDY_TEST_H
