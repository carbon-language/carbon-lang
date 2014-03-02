//===-- UseAuto/UseAuto.cpp - Use auto type specifier ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the UseAutoTransform class.
///
//===----------------------------------------------------------------------===//

#include "UseAuto.h"
#include "UseAutoActions.h"
#include "UseAutoMatchers.h"

using clang::ast_matchers::MatchFinder;
using namespace clang;
using namespace clang::tooling;

int UseAutoTransform::apply(const clang::tooling::CompilationDatabase &Database,
                            const std::vector<std::string> &SourcePaths) {
  ClangTool UseAutoTool(Database, SourcePaths);

  unsigned AcceptedChanges = 0;

  MatchFinder Finder;
  ReplacementsVec Replaces;
  IteratorReplacer ReplaceIterators(AcceptedChanges, Options().MaxRiskLevel,
                                    /*Owner=*/ *this);
  NewReplacer ReplaceNew(AcceptedChanges, Options().MaxRiskLevel,
                         /*Owner=*/ *this);

  Finder.addMatcher(makeIteratorDeclMatcher(), &ReplaceIterators);
  Finder.addMatcher(makeDeclWithNewMatcher(), &ReplaceNew);

  if (int Result = UseAutoTool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return Result;
  }

  setAcceptedChanges(AcceptedChanges);

  return 0;
}

struct UseAutoFactory : TransformFactory {
  UseAutoFactory() {
    Since.Clang = Version(2, 9);
    Since.Gcc = Version(4, 4);
    Since.Icc = Version(12);
    Since.Msvc = Version(10);
  }

  Transform *createTransform(const TransformOptions &Opts) override {
    return new UseAutoTransform(Opts);
  }
};

// Register the factory using this statically initialized variable.
static TransformFactoryRegistry::Add<UseAutoFactory>
X("use-auto", "Use of 'auto' type specifier");

// This anchor is used to force the linker to link in the generated object file
// and thus register the factory.
volatile int UseAutoTransformAnchorSource = 0;
