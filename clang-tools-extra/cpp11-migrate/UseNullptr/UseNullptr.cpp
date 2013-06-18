//===-- LoopConvert/LoopConvert.cpp - C++11 for-loop migration --*- C++ -*-===//
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
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;

int UseNullptrTransform::apply(FileOverrides &InputStates,
                               const CompilationDatabase &Database,
                               const std::vector<std::string> &SourcePaths) {
  ClangTool UseNullptrTool(Database, SourcePaths);

  unsigned AcceptedChanges = 0;

  MatchFinder Finder;
  NullptrFixer Fixer(getReplacements(), AcceptedChanges,
                     Options().MaxRiskLevel);
  Finder.addMatcher(makeCastSequenceMatcher(), &Fixer);

  setOverrides(InputStates);

  if (int result = UseNullptrTool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  setAcceptedChanges(AcceptedChanges);

  return 0;
}
