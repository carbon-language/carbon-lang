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

int UseAutoTransform::apply(const FileOverrides &InputStates,
                            const clang::tooling::CompilationDatabase &Database,
                            const std::vector<std::string> &SourcePaths,
                            FileOverrides &ResultStates) {
  RefactoringTool UseAutoTool(Database, SourcePaths);

  unsigned AcceptedChanges = 0;

  MatchFinder Finder;
  IteratorReplacer ReplaceIterators(UseAutoTool.getReplacements(),
                                    AcceptedChanges, Options().MaxRiskLevel);
  NewReplacer ReplaceNew(UseAutoTool.getReplacements(), AcceptedChanges,
                         Options().MaxRiskLevel);

  Finder.addMatcher(makeIteratorDeclMatcher(), &ReplaceIterators);
  Finder.addMatcher(makeDeclWithNewMatcher(), &ReplaceNew);

  if (int Result = UseAutoTool.run(createActionFactory(Finder, InputStates))) {
    llvm::errs() << "Error encountered during translation.\n";
    return Result;
  }

  RewriterContainer Rewrite(UseAutoTool.getFiles(), InputStates);

  // FIXME: Do something if some replacements didn't get applied?
  UseAutoTool.applyAllReplacements(Rewrite.getRewriter());

  collectResults(Rewrite.getRewriter(), InputStates, ResultStates);

  setAcceptedChanges(AcceptedChanges);

  return 0;
}
