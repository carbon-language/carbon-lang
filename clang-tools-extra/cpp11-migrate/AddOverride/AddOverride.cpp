//===-- AddOverride/AddOverride.cpp - add C++11 override ------------------===//
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

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;
namespace cl = llvm::cl;

static cl::opt<bool> DetectMacros(
    "override-macros",
    cl::desc("Detect and use macros that expand to the 'override' keyword."),
    cl::cat(TransformsOptionsCategory));

int AddOverrideTransform::apply(FileOverrides &InputStates,
                                const CompilationDatabase &Database,
                                const std::vector<std::string> &SourcePaths) {
  ClangTool AddOverrideTool(Database, SourcePaths);
  unsigned AcceptedChanges = 0;
  MatchFinder Finder;
  AddOverrideFixer Fixer(getReplacements(), AcceptedChanges, DetectMacros,
                         /*Owner=*/ *this);
  Finder.addMatcher(makeCandidateForOverrideAttrMatcher(), &Fixer);

  // Make Fixer available to handleBeginSource().
  this->Fixer = &Fixer;

  setOverrides(InputStates);

  if (int result = AddOverrideTool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  setAcceptedChanges(AcceptedChanges);
  return 0;
}

bool AddOverrideTransform::handleBeginSource(clang::CompilerInstance &CI,
                                             llvm::StringRef Filename) {
  assert(Fixer != NULL && "Fixer must be set");
  Fixer->setPreprocessor(CI.getPreprocessor());
  return Transform::handleBeginSource(CI, Filename);
}
