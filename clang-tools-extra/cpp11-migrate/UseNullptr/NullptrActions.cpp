//===-- nullptr-convert/NullptrActions.cpp - Matcher callback -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the definition of the NullptrFixer class which is
///  used as an ASTMatcher callback. Also within this file is a helper AST
///  visitor class used to identify sequences of explicit casts.
///
//===----------------------------------------------------------------------===//

#include "NullptrActions.h"
#include "NullptrMatchers.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

namespace {

/// \brief Replaces the provided range with the text "nullptr", but only if 
/// the start and end location are both in main file.
/// Returns true if and only if a replacement was made.
bool ReplaceWithNullptr(tooling::Replacements &Replace, SourceManager &SM,
                        SourceLocation StartLoc, SourceLocation EndLoc) {
  if (SM.isFromSameFile(StartLoc, EndLoc) && SM.isFromMainFile(StartLoc)) {
    CharSourceRange Range(SourceRange(StartLoc, EndLoc), true);
    Replace.insert(tooling::Replacement(SM, Range, "nullptr"));
    return true;
  } else
    return false;
}

}

/// \brief Looks for a sequences of 0 or more explicit casts with an implicit
/// null-to-pointer cast within.
///
/// The matcher this visitor is used with will find a top-most explicit cast
/// (i.e. it has no explicit casts as an ancestor) where an implicit cast is
/// nested within. However, there is no guarantee that only explicit casts
/// exist between the found top-most explicit cast and the possibly more than
/// one nested implicit cast. This visitor finds all cast sequences with an
/// implicit cast to null within and creates a replacement leaving the
/// outermost explicit cast unchanged to avoid introducing ambiguities.
class CastSequenceVisitor : public RecursiveASTVisitor<CastSequenceVisitor> {
public:
  CastSequenceVisitor(tooling::Replacements &R, SourceManager &SM,
                      unsigned &AcceptedChanges)
      : Replace(R), SM(SM), AcceptedChanges(AcceptedChanges), FirstSubExpr(0) {}

  // Only VisitStmt is overridden as we shouldn't find other base AST types
  // within a cast expression.
  bool VisitStmt(Stmt *S) {
    CastExpr *C = dyn_cast<CastExpr>(S);

    if (!C) {
      ResetFirstSubExpr();
      return true;
    } else if (!FirstSubExpr) {
      // Get the subexpression of the outermost explicit cast
      FirstSubExpr = C->getSubExpr();
    }

    if (C->getCastKind() == CK_NullToPointer ||
        C->getCastKind() == CK_NullToMemberPointer) {

      SourceLocation StartLoc = FirstSubExpr->getLocStart();
      SourceLocation EndLoc = FirstSubExpr->getLocEnd();

      // If the start/end location is a macro, get the expansion location.
      StartLoc = SM.getFileLoc(StartLoc);
      EndLoc = SM.getFileLoc(EndLoc);

      AcceptedChanges +=
          ReplaceWithNullptr(Replace, SM, StartLoc, EndLoc) ? 1 : 0;

      ResetFirstSubExpr();
    }

    return true;
  }

private:
  void ResetFirstSubExpr() { FirstSubExpr = 0; }

private:
  tooling::Replacements &Replace;
  SourceManager &SM;
  unsigned &AcceptedChanges;
  Expr *FirstSubExpr;
};

void NullptrFixer::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;

  const CastExpr *NullCast = Result.Nodes.getNodeAs<CastExpr>(CastSequence);
  if (NullCast) {
    // Given an explicit cast with an implicit null-to-pointer cast within
    // use CastSequenceVisitor to identify sequences of explicit casts that can
    // be converted into 'nullptr'.
    CastSequenceVisitor Visitor(Replace, SM, AcceptedChanges);
    Visitor.TraverseStmt(const_cast<CastExpr *>(NullCast));
  }

  const CastExpr *Cast = Result.Nodes.getNodeAs<CastExpr>(ImplicitCastNode);
  if (Cast) {
    const Expr *E = Cast->IgnoreParenImpCasts();

    SourceLocation StartLoc = E->getLocStart();
    SourceLocation EndLoc = E->getLocEnd();

    // If the start/end location is a macro, get the expansion location.
    StartLoc = SM.getFileLoc(StartLoc);
    EndLoc = SM.getFileLoc(EndLoc);

    AcceptedChanges +=
        ReplaceWithNullptr(Replace, SM, StartLoc, EndLoc) ? 1 : 0;
  }
}
