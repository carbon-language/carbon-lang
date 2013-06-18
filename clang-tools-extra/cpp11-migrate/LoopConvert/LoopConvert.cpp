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
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

using clang::ast_matchers::MatchFinder;
using namespace clang::tooling;
using namespace clang;

int LoopConvertTransform::apply(FileOverrides &InputStates,
                                const CompilationDatabase &Database,
                                const std::vector<std::string> &SourcePaths) {
  ClangTool LoopTool(Database, SourcePaths);

  StmtAncestorASTVisitor ParentFinder;
  StmtGeneratedVarNameMap GeneratedDecls;
  ReplacedVarsMap ReplacedVars;
  unsigned AcceptedChanges = 0;
  unsigned DeferredChanges = 0;
  unsigned RejectedChanges = 0;

  MatchFinder Finder;
  LoopFixer ArrayLoopFixer(&ParentFinder, &getReplacements(), &GeneratedDecls,
                           &ReplacedVars, &AcceptedChanges, &DeferredChanges,
                           &RejectedChanges, Options().MaxRiskLevel, LFK_Array,
                           /*Owner=*/ *this);
  Finder.addMatcher(makeArrayLoopMatcher(), &ArrayLoopFixer);
  LoopFixer IteratorLoopFixer(
      &ParentFinder, &getReplacements(), &GeneratedDecls, &ReplacedVars,
      &AcceptedChanges, &DeferredChanges, &RejectedChanges,
      Options().MaxRiskLevel, LFK_Iterator, /*Owner=*/ *this);
  Finder.addMatcher(makeIteratorLoopMatcher(), &IteratorLoopFixer);
  LoopFixer PseudoarrrayLoopFixer(
      &ParentFinder, &getReplacements(), &GeneratedDecls, &ReplacedVars,
      &AcceptedChanges, &DeferredChanges, &RejectedChanges,
      Options().MaxRiskLevel, LFK_PseudoArray, /*Owner=*/ *this);
  Finder.addMatcher(makePseudoArrayLoopMatcher(), &PseudoarrrayLoopFixer);

  setOverrides(InputStates);

  if (int result = LoopTool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return result;
  }

  setAcceptedChanges(AcceptedChanges);
  setRejectedChanges(RejectedChanges);
  setDeferredChanges(DeferredChanges);

  return 0;
}
