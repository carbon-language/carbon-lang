//=- ReachableCodePathInsensitive.cpp ---------------------------*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a flow-sensitive, path-insensitive analysis of
// determining reachable blocks within a CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Analysis/Analyses/ReachableCode.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;

static SourceLocation GetUnreachableLoc(const CFGBlock &b, SourceRange &R1,
                                        SourceRange &R2) {
  const Stmt *S = 0;
  unsigned sn = 0;
  R1 = R2 = SourceRange();

top:
  if (sn < b.size()) {
    CFGStmt CS = b[sn].getAs<CFGStmt>();
    if (!CS)
      goto top;
    
    S = CS.getStmt(); 
  } else if (b.getTerminator())
    S = b.getTerminator();
  else
    return SourceLocation();

  switch (S->getStmtClass()) {
    case Expr::BinaryOperatorClass: {
      const BinaryOperator *BO = cast<BinaryOperator>(S);
      if (BO->getOpcode() == BO_Comma) {
        if (sn+1 < b.size())
          return b[sn+1].getAs<CFGStmt>().getStmt()->getLocStart();
        const CFGBlock *n = &b;
        while (1) {
          if (n->getTerminator())
            return n->getTerminator()->getLocStart();
          if (n->succ_size() != 1)
            return SourceLocation();
          n = n[0].succ_begin()[0];
          if (n->pred_size() != 1)
            return SourceLocation();
          if (!n->empty())
            return n[0][0].getAs<CFGStmt>().getStmt()->getLocStart();
        }
      }
      R1 = BO->getLHS()->getSourceRange();
      R2 = BO->getRHS()->getSourceRange();
      return BO->getOperatorLoc();
    }
    case Expr::UnaryOperatorClass: {
      const UnaryOperator *UO = cast<UnaryOperator>(S);
      R1 = UO->getSubExpr()->getSourceRange();
      return UO->getOperatorLoc();
    }
    case Expr::CompoundAssignOperatorClass: {
      const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(S);
      R1 = CAO->getLHS()->getSourceRange();
      R2 = CAO->getRHS()->getSourceRange();
      return CAO->getOperatorLoc();
    }
    case Expr::ConditionalOperatorClass: {
      const ConditionalOperator *CO = cast<ConditionalOperator>(S);
      return CO->getQuestionLoc();
    }
    case Expr::MemberExprClass: {
      const MemberExpr *ME = cast<MemberExpr>(S);
      R1 = ME->getSourceRange();
      return ME->getMemberLoc();
    }
    case Expr::ArraySubscriptExprClass: {
      const ArraySubscriptExpr *ASE = cast<ArraySubscriptExpr>(S);
      R1 = ASE->getLHS()->getSourceRange();
      R2 = ASE->getRHS()->getSourceRange();
      return ASE->getRBracketLoc();
    }
    case Expr::CStyleCastExprClass: {
      const CStyleCastExpr *CSC = cast<CStyleCastExpr>(S);
      R1 = CSC->getSubExpr()->getSourceRange();
      return CSC->getLParenLoc();
    }
    case Expr::CXXFunctionalCastExprClass: {
      const CXXFunctionalCastExpr *CE = cast <CXXFunctionalCastExpr>(S);
      R1 = CE->getSubExpr()->getSourceRange();
      return CE->getTypeBeginLoc();
    }
    case Expr::ImplicitCastExprClass:
      ++sn;
      goto top;
    case Stmt::CXXTryStmtClass: {
      return cast<CXXTryStmt>(S)->getHandler(0)->getCatchLoc();
    }
    default: ;
  }
  R1 = S->getSourceRange();
  return S->getLocStart();
}

static SourceLocation MarkLiveTop(const CFGBlock *Start,
                                  llvm::BitVector &reachable,
                                  SourceManager &SM) {

  // Prep work worklist.
  llvm::SmallVector<const CFGBlock*, 32> WL;
  WL.push_back(Start);

  SourceRange R1, R2;
  SourceLocation top = GetUnreachableLoc(*Start, R1, R2);

  bool FromMainFile = false;
  bool FromSystemHeader = false;
  bool TopValid = false;

  if (top.isValid()) {
    FromMainFile = SM.isFromMainFile(top);
    FromSystemHeader = SM.isInSystemHeader(top);
    TopValid = true;
  }

  // Solve
  CFGBlock::FilterOptions FO;
  FO.IgnoreDefaultsWithCoveredEnums = 1;

  while (!WL.empty()) {
    const CFGBlock *item = WL.back();
    WL.pop_back();

    SourceLocation c = GetUnreachableLoc(*item, R1, R2);
    if (c.isValid()
        && (!TopValid
            || (SM.isFromMainFile(c) && !FromMainFile)
            || (FromSystemHeader && !SM.isInSystemHeader(c))
            || SM.isBeforeInTranslationUnit(c, top))) {
          top = c;
          FromMainFile = SM.isFromMainFile(top);
          FromSystemHeader = SM.isInSystemHeader(top);
        }

    reachable.set(item->getBlockID());
    for (CFGBlock::filtered_succ_iterator I =
	   item->filtered_succ_start_end(FO); I.hasMore(); ++I)
      if (const CFGBlock *B = *I) {
        unsigned blockID = B->getBlockID();
        if (!reachable[blockID]) {
          reachable.set(blockID);
          WL.push_back(B);
        }
      }
  }

  return top;
}

static int LineCmp(const void *p1, const void *p2) {
  SourceLocation *Line1 = (SourceLocation *)p1;
  SourceLocation *Line2 = (SourceLocation *)p2;
  return !(*Line1 < *Line2);
}

namespace {
struct ErrLoc {
  SourceLocation Loc;
  SourceRange R1;
  SourceRange R2;
  ErrLoc(SourceLocation l, SourceRange r1, SourceRange r2)
  : Loc(l), R1(r1), R2(r2) { }
};
}
namespace clang { namespace reachable_code {

/// ScanReachableFromBlock - Mark all blocks reachable from Start.
/// Returns the total number of blocks that were marked reachable.
unsigned ScanReachableFromBlock(const CFGBlock &Start,
                                llvm::BitVector &Reachable) {
  unsigned count = 0;
  llvm::SmallVector<const CFGBlock*, 32> WL;

    // Prep work queue
  Reachable.set(Start.getBlockID());
  ++count;
  WL.push_back(&Start);

  // Find the reachable blocks from 'Start'.
  CFGBlock::FilterOptions FO;
  FO.IgnoreDefaultsWithCoveredEnums = 1;

  while (!WL.empty()) {
    const CFGBlock *item = WL.back();
    WL.pop_back();

      // Look at the successors and mark then reachable.
    for (CFGBlock::filtered_succ_iterator I= item->filtered_succ_start_end(FO);
         I.hasMore(); ++I)
      if (const CFGBlock *B = *I) {
        unsigned blockID = B->getBlockID();
        if (!Reachable[blockID]) {
          Reachable.set(blockID);
          ++count;
          WL.push_back(B);
        }
      }
  }
  return count;
}

void FindUnreachableCode(AnalysisContext &AC, Callback &CB) {
  CFG *cfg = AC.getCFG();
  if (!cfg)
    return;

  // Scan for reachable blocks.
  llvm::BitVector reachable(cfg->getNumBlockIDs());
  unsigned numReachable = ScanReachableFromBlock(cfg->getEntry(), reachable);

    // If there are no unreachable blocks, we're done.
  if (numReachable == cfg->getNumBlockIDs())
    return;

  SourceRange R1, R2;

  llvm::SmallVector<ErrLoc, 24> lines;
  bool AddEHEdges = AC.getAddEHEdges();

  // First, give warnings for blocks with no predecessors, as they
  // can't be part of a loop.
  for (CFG::iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
    CFGBlock &b = **I;
    if (!reachable[b.getBlockID()]) {
      if (b.pred_empty()) {
        if (!AddEHEdges
        && dyn_cast_or_null<CXXTryStmt>(b.getTerminator().getStmt())) {
            // When not adding EH edges from calls, catch clauses
            // can otherwise seem dead.  Avoid noting them as dead.
          numReachable += ScanReachableFromBlock(b, reachable);
          continue;
        }
        SourceLocation c = GetUnreachableLoc(b, R1, R2);
        if (!c.isValid()) {
            // Blocks without a location can't produce a warning, so don't mark
            // reachable blocks from here as live.
          reachable.set(b.getBlockID());
          ++numReachable;
          continue;
        }
        lines.push_back(ErrLoc(c, R1, R2));
          // Avoid excessive errors by marking everything reachable from here
        numReachable += ScanReachableFromBlock(b, reachable);
      }
    }
  }

  if (numReachable < cfg->getNumBlockIDs()) {
      // And then give warnings for the tops of loops.
    for (CFG::iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
      CFGBlock &b = **I;
      if (!reachable[b.getBlockID()])
          // Avoid excessive errors by marking everything reachable from here
        lines.push_back(ErrLoc(MarkLiveTop(&b, reachable,
                                         AC.getASTContext().getSourceManager()),
                               SourceRange(), SourceRange()));
    }
  }

  llvm::array_pod_sort(lines.begin(), lines.end(), LineCmp);

  for (llvm::SmallVectorImpl<ErrLoc>::iterator I=lines.begin(), E=lines.end();
       I != E; ++I)
    if (I->Loc.isValid())
      CB.HandleUnreachable(I->Loc, I->R1, I->R2);
}

}} // end namespace clang::reachable_code
