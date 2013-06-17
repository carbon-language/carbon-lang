//===-- AddOverride/AddOverride.cpp - add C++11 override -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the AddOverrideTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "AddOverride.h"
#include "AddOverrideActions.h"
#include "AddOverrideMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;

static llvm::cl::opt<bool> DetectMacros(
    "override-macros",
    llvm::cl::desc(
        "Detect and use macros that expand to the 'override' keyword."));

int AddOverrideTransform::apply(const FileOverrides &InputStates,
                                const CompilationDatabase &Database,
                                const std::vector<std::string> &SourcePaths,
                                FileOverrides &ResultStates) {
  RefactoringTool AddOverrideTool(Database, SourcePaths);

  unsigned AcceptedChanges = 0;

  MatchFinder Finder;
  AddOverrideFixer Fixer(AddOverrideTool.getReplacements(), AcceptedChanges,
                         DetectMacros);
  Finder.addMatcher(makeCandidateForOverrideAttrMatcher(), &Fixer);

  // Make Fixer available to handleBeginSource().
  this->Fixer = &Fixer;

  setOverrides(InputStates);

  if (int result = AddOverrideTool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  RewriterContainer Rewrite(AddOverrideTool.getFiles(), InputStates);

  // FIXME: Do something if some replacements didn't get applied?
  AddOverrideTool.applyAllReplacements(Rewrite.getRewriter());

  collectResults(Rewrite.getRewriter(), InputStates, ResultStates);

  setAcceptedChanges(AcceptedChanges);

  return 0;
}

bool AddOverrideTransform::handleBeginSource(clang::CompilerInstance &CI,
                                             llvm::StringRef Filename) {
  assert(Fixer != NULL && "Fixer must be set");
  Fixer->setPreprocessor(CI.getPreprocessor());
  return Transform::handleBeginSource(CI, Filename);
}
