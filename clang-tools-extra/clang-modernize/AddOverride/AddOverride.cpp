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

int AddOverrideTransform::apply(const CompilationDatabase &Database,
                                const std::vector<std::string> &SourcePaths) {
  ClangTool AddOverrideTool(Database, SourcePaths);
  unsigned AcceptedChanges = 0;
  MatchFinder Finder;
  AddOverrideFixer Fixer(AcceptedChanges, DetectMacros,
                         /*Owner=*/ *this);
  Finder.addMatcher(makeCandidateForOverrideAttrMatcher(), &Fixer);

  // Make Fixer available to handleBeginSource().
  this->Fixer = &Fixer;

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

struct AddOverrideFactory : TransformFactory {
  AddOverrideFactory() {
    // if detecting macros is enabled, do not impose requirements on the
    // compiler. It is assumed that the macros use is "C++11-aware", meaning it
    // won't expand to override if the compiler doesn't support the specifier.
    if (!DetectMacros) {
      Since.Clang = Version(3, 0);
      Since.Gcc = Version(4, 7);
      Since.Icc = Version(14);
      Since.Msvc = Version(8);
    }
  }

  Transform *createTransform(const TransformOptions &Opts) override {
    return new AddOverrideTransform(Opts);
  }
};

// Register the factory using this statically initialized variable.
static TransformFactoryRegistry::Add<AddOverrideFactory>
X("add-override", "Make use of override specifier where possible");

// This anchor is used to force the linker to link in the generated object file
// and thus register the factory.
volatile int AddOverrideTransformAnchorSource = 0;
