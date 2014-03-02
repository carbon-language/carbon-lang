//===-- UseNullptr/UseNullptr.cpp - C++11 nullptr migration ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the UseNullptrTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "UseNullptr.h"
#include "NullptrActions.h"
#include "NullptrMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;
namespace cl = llvm::cl;

static cl::opt<std::string>
UserNullMacroNames("user-null-macros",
                   cl::desc("Comma-separated list of user-defined "
                            "macro names that behave like NULL"),
                   cl::cat(TransformsOptionsCategory), cl::init(""));

int UseNullptrTransform::apply(const CompilationDatabase &Database,
                               const std::vector<std::string> &SourcePaths) {
  ClangTool UseNullptrTool(Database, SourcePaths);

  unsigned AcceptedChanges = 0;

  llvm::SmallVector<llvm::StringRef, 1> MacroNames;
  if (!UserNullMacroNames.empty()) {
    llvm::StringRef S = UserNullMacroNames;
    S.split(MacroNames, ",");
  }
  MatchFinder Finder;
  NullptrFixer Fixer(AcceptedChanges, MacroNames, /*Owner=*/ *this);

  Finder.addMatcher(makeCastSequenceMatcher(), &Fixer);

  if (int result = UseNullptrTool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  setAcceptedChanges(AcceptedChanges);

  return 0;
}

struct UseNullptrFactory : TransformFactory {
  UseNullptrFactory() {
    Since.Clang = Version(3, 0);
    Since.Gcc = Version(4, 6);
    Since.Icc = Version(12, 1);
    Since.Msvc = Version(10);
  }

  Transform *createTransform(const TransformOptions &Opts) override {
    return new UseNullptrTransform(Opts);
  }
};

// Register the factory using this statically initialized variable.
static TransformFactoryRegistry::Add<UseNullptrFactory>
X("use-nullptr", "Make use of nullptr keyword where possible");

// This anchor is used to force the linker to link in the generated object file
// and thus register the factory.
volatile int UseNullptrTransformAnchorSource = 0;
