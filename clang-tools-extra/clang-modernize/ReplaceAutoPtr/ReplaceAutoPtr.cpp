//===-- ReplaceAutoPtr.cpp ---------- std::auto_ptr replacement -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the ReplaceAutoPtrTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "ReplaceAutoPtr.h"
#include "ReplaceAutoPtrActions.h"
#include "ReplaceAutoPtrMatchers.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

int
ReplaceAutoPtrTransform::apply(const CompilationDatabase &Database,
                               const std::vector<std::string> &SourcePaths) {
  ClangTool Tool(Database, SourcePaths);
  unsigned AcceptedChanges = 0;
  MatchFinder Finder;
  AutoPtrReplacer Replacer(AcceptedChanges, /*Owner=*/ *this);
  OwnershipTransferFixer Fixer(AcceptedChanges, /*Owner=*/ *this);

  Finder.addMatcher(makeAutoPtrTypeLocMatcher(), &Replacer);
  Finder.addMatcher(makeAutoPtrUsingDeclMatcher(), &Replacer);
  Finder.addMatcher(makeTransferOwnershipExprMatcher(), &Fixer);

  if (Tool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return 1;
  }

  setAcceptedChanges(AcceptedChanges);

  return 0;
}

struct ReplaceAutoPtrFactory : TransformFactory {
  ReplaceAutoPtrFactory() {
    Since.Clang = Version(3, 0);
    Since.Gcc = Version(4, 6);
    Since.Icc = Version(13);
    Since.Msvc = Version(11);
  }

  Transform *createTransform(const TransformOptions &Opts) override {
    return new ReplaceAutoPtrTransform(Opts);
  }
};

// Register the factory using this statically initialized variable.
static TransformFactoryRegistry::Add<ReplaceAutoPtrFactory>
X("replace-auto_ptr", "Replace std::auto_ptr (deprecated) by std::unique_ptr"
                      " (EXPERIMENTAL)");

// This anchor is used to force the linker to link in the generated object file
// and thus register the factory.
volatile int ReplaceAutoPtrTransformAnchorSource = 0;
