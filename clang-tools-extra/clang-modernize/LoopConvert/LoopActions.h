//===-- LoopConvert/LoopActions.h - C++11 For loop migration ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares matchers and callbacks for use in migrating C++
/// for loops.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_LOOP_ACTIONS_H
#define CLANG_MODERNIZE_LOOP_ACTIONS_H

#include "Core/Transform.h"
#include "StmtAncestor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

struct Usage;
class Confidence;
// The main computational result of ForLoopIndexUseVisitor.
typedef llvm::SmallVector<Usage, 8> UsageResult;

enum LoopFixerKind {
  LFK_Array,
  LFK_Iterator,
  LFK_PseudoArray
};

struct TUTrackingInfo {

  /// \brief Reset and initialize per-TU tracking information.
  ///
  /// Must be called before using container accessors.
  void reset() {
    ParentFinder.reset(new StmtAncestorASTVisitor);
    GeneratedDecls.clear();
    ReplacedVars.clear();
  }

  /// \name Accessors
  /// \{
  StmtAncestorASTVisitor &getParentFinder() { return *ParentFinder; }
  StmtGeneratedVarNameMap &getGeneratedDecls() { return GeneratedDecls; }
  ReplacedVarsMap &getReplacedVars() { return ReplacedVars; }
  /// \}

private:
  std::unique_ptr<StmtAncestorASTVisitor> ParentFinder;
  StmtGeneratedVarNameMap GeneratedDecls;
  ReplacedVarsMap ReplacedVars;
};

/// \brief The callback to be used for loop migration matchers.
///
/// The callback does extra checking not possible in matchers, and attempts to
/// convert the for loop, if possible.
class LoopFixer : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  LoopFixer(TUTrackingInfo &TUInfo, unsigned *AcceptedChanges,
            unsigned *DeferredChanges, unsigned *RejectedChanges,
            RiskLevel MaxRisk, LoopFixerKind FixerKind, Transform &Owner)
      : TUInfo(TUInfo), AcceptedChanges(AcceptedChanges),
        DeferredChanges(DeferredChanges), RejectedChanges(RejectedChanges),
        MaxRisk(MaxRisk), FixerKind(FixerKind), Owner(Owner) {}

  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result);

private:
  TUTrackingInfo &TUInfo;
  unsigned *AcceptedChanges;
  unsigned *DeferredChanges;
  unsigned *RejectedChanges;
  RiskLevel MaxRisk;
  LoopFixerKind FixerKind;
  Transform &Owner;

  /// \brief Computes the changes needed to convert a given for loop, and
  /// applies it.
  void doConversion(clang::ASTContext *Context, const clang::VarDecl *IndexVar,
                    const clang::VarDecl *MaybeContainer,
                    llvm::StringRef ContainerString, const UsageResult &Usages,
                    const clang::DeclStmt *AliasDecl, bool AliasUseRequired,
                    bool AliasFromForInit, const clang::ForStmt *TheLoop,
                    bool ContainerNeedsDereference, bool DerefByValue,
                    bool DerefByConstRef);

  /// \brief Given a loop header that would be convertible, discover all usages
  /// of the index variable and convert the loop if possible.
  void findAndVerifyUsages(clang::ASTContext *Context,
                           const clang::VarDecl *LoopVar,
                           const clang::VarDecl *EndVar,
                           const clang::Expr *ContainerExpr,
                           const clang::Expr *BoundExpr,
                           bool ContainerNeedsDereference, bool DerefByValue,
                           bool DerefByConstRef, const clang::ForStmt *TheLoop,
                           Confidence ConfidenceLevel);

  /// \brief Determine if the change should be deferred or rejected, returning
  /// text which refers to the container iterated over if the change should
  /// proceed.
  llvm::StringRef checkDeferralsAndRejections(clang::ASTContext *Context,
                                              const clang::Expr *ContainerExpr,
                                              Confidence ConfidenceLevel,
                                              const clang::ForStmt *TheLoop);
};

#endif // CLANG_MODERNIZE_LOOP_ACTIONS_H
