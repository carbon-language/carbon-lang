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
/// \brief This file provides the implementation of the LoopConvertTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "LoopConvert.h"
#include "LoopActions.h"
#include "LoopMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;

int LoopConvertTransform::apply(RiskLevel MaxRisk,
                                const CompilationDatabase &Database,
                                const std::vector<std::string> &SourcePaths) {
  RefactoringTool LoopTool(Database, SourcePaths);
  StmtAncestorASTVisitor ParentFinder;
  StmtGeneratedVarNameMap GeneratedDecls;
  ReplacedVarsMap ReplacedVars;
  unsigned AcceptedChanges = 0;
  unsigned DeferredChanges = 0;
  unsigned RejectedChanges = 0;

  MatchFinder Finder;
  LoopFixer ArrayLoopFixer(&ParentFinder, &LoopTool.getReplacements(),
                           &GeneratedDecls, &ReplacedVars, &AcceptedChanges,
                           &DeferredChanges, &RejectedChanges,
                           MaxRisk, LFK_Array);
  Finder.addMatcher(makeArrayLoopMatcher(), &ArrayLoopFixer);
  LoopFixer IteratorLoopFixer(&ParentFinder, &LoopTool.getReplacements(),
                              &GeneratedDecls, &ReplacedVars,
                              &AcceptedChanges, &DeferredChanges,
                              &RejectedChanges,
                              MaxRisk, LFK_Iterator);
  Finder.addMatcher(makeIteratorLoopMatcher(), &IteratorLoopFixer);
  LoopFixer PseudoarrrayLoopFixer(&ParentFinder, &LoopTool.getReplacements(),
                                  &GeneratedDecls, &ReplacedVars,
                                  &AcceptedChanges, &DeferredChanges,
                                  &RejectedChanges,
                                  MaxRisk, LFK_PseudoArray);
  Finder.addMatcher(makePseudoArrayLoopMatcher(), &PseudoarrrayLoopFixer);
  if (int result = LoopTool.run(newFrontendActionFactory(&Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  if (AcceptedChanges > 0) {
    setChangesMade();
  }

  if (RejectedChanges > 0 || DeferredChanges > 0) {
    setChangesNotMade();
  }

  return 0;
}
