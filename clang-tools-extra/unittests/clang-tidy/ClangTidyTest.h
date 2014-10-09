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

class TestClangTidyAction : public ASTFrontendAction {
public:
  TestClangTidyAction(ClangTidyCheck &Check, ast_matchers::MatchFinder &Finder,
                      ClangTidyContext &Context)
      : Check(Check), Finder(Finder), Context(Context) {}

private:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    Context.setSourceManager(&Compiler.getSourceManager());
    Check.registerPPCallbacks(Compiler);
    return Finder.newASTConsumer();
  }

  ClangTidyCheck &Check;
  ast_matchers::MatchFinder &Finder;
  ClangTidyContext &Context;
};

template <typename T>
std::string runCheckOnCode(StringRef Code,
                           std::vector<ClangTidyError> *Errors = nullptr,
                           const Twine &Filename = "input.cc",
                           ArrayRef<std::string> ExtraArgs = None) {
  ClangTidyOptions Options;
  Options.Checks = "*";
  ClangTidyContext Context(llvm::make_unique<DefaultOptionsProvider>(
      ClangTidyGlobalOptions(), Options));
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  T Check("test-check", &Context);
  ast_matchers::MatchFinder Finder;
  Check.registerMatchers(&Finder);

  std::vector<std::string> ArgCXX11(1, "clang-tidy");
  ArgCXX11.push_back("-fsyntax-only");
  ArgCXX11.push_back("-std=c++11");
  ArgCXX11.insert(ArgCXX11.end(), ExtraArgs.begin(), ExtraArgs.end());
  ArgCXX11.push_back(Filename.str());
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions()));
  tooling::ToolInvocation Invocation(
      ArgCXX11, new TestClangTidyAction(Check, Finder, Context), Files.get());
  Invocation.mapVirtualFile(Filename.str(), Code);
  Invocation.setDiagnosticConsumer(&DiagConsumer);
  if (!Invocation.run())
    return "";

  DiagConsumer.finish();
  tooling::Replacements Fixes;
  for (const ClangTidyError &Error : Context.getErrors())
    Fixes.insert(Error.Fix.begin(), Error.Fix.end());
  if (Errors)
    *Errors = Context.getErrors();
  return tooling::applyAllReplacements(Code, Fixes);
}

} // namespace test
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_TIDY_CLANG_TIDY_TEST_H
