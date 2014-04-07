//===- ThreadSafetyCommon.h ------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Parts of thread safety analysis that are not specific to thread safety
// itself have been factored into classes here, where they can be potentially
// used by other analyses.  Currently these include:
//
// * Generalize clang CFG visitors.
// * Conversion of the clang CFG to SSA form.
// * Translation of clang Exprs to TIL SExprs
//
// UNDER CONSTRUCTION.  USE AT YOUR OWN RISK.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_THREAD_SAFETY_COMMON_H
#define LLVM_CLANG_THREAD_SAFETY_COMMON_H

#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <vector>


namespace clang {
namespace threadSafety {


// Simple Visitor class for traversing a clang CFG.
class CFGVisitor {
public:
  // Enter the CFG for Decl D, and perform any initial setup operations.
  void enterCFG(CFG *Cfg, const NamedDecl *D, const CFGBlock *First) {}

  // Enter a CFGBlock.
  void enterCFGBlock(const CFGBlock *B) {}

  // Process an ordinary statement.
  void handleStatement(const Stmt *S) {}

  // Process a destructor call
  void handleDestructorCall(const VarDecl *VD, const CXXDestructorDecl *DD) {}

  // Process a successor edge.
  void handleSuccessor(const CFGBlock *Succ) {}

  // Process a successor back edge to a previously visited block.
  void handleSuccessorBackEdge(const CFGBlock *Succ) {}

  // Leave a CFGBlock.
  void exitCFGBlock(const CFGBlock *B) {}

  // Leave the CFG, and perform any final cleanup operations.
  void exitCFG(const CFGBlock *Last) {}
};


// Walks the clang CFG, and invokes methods on a given CFGVisitor.
class CFGWalker {
public:
  CFGWalker() : CFGraph(0), FDecl(0), ACtx(0), SortedGraph(0) {}

  ~CFGWalker() { }

  // Initialize the CFGWalker.  This setup only needs to be done once, even
  // if there are multiple passes over the CFG.
  bool init(AnalysisDeclContext &AC) {
    ACtx = &AC;
    CFGraph = AC.getCFG();
    if (!CFGraph)
      return false;

    FDecl = dyn_cast_or_null<NamedDecl>(AC.getDecl());
    if (!FDecl) // ignore anonymous functions
      return false;

    SortedGraph = AC.getAnalysis<PostOrderCFGView>();
    if (!SortedGraph)
      return false;

    return true;
  }

  // Traverse the CFG, calling methods on V as appropriate.
  template <class Visitor>
  void walk(Visitor &V) {
    PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

    V.enterCFG(CFGraph, FDecl, &CFGraph->getEntry());

    for (const CFGBlock* CurrBlock : *SortedGraph) {
      VisitedBlocks.insert(CurrBlock);

      V.enterCFGBlock(CurrBlock);

      // Process statements
      for (CFGBlock::const_iterator BI = CurrBlock->begin(),
                                    BE = CurrBlock->end();
           BI != BE; ++BI) {
        switch (BI->getKind()) {
        case CFGElement::Statement: {
          V.handleStatement(BI->castAs<CFGStmt>().getStmt());
          break;
        }
        case CFGElement::AutomaticObjectDtor: {
          CFGAutomaticObjDtor AD = BI->castAs<CFGAutomaticObjDtor>();
          CXXDestructorDecl *DD = const_cast<CXXDestructorDecl*>(
              AD.getDestructorDecl(ACtx->getASTContext()));
          VarDecl *VD = const_cast<VarDecl*>(AD.getVarDecl());
          V.handleDestructorCall(VD, DD);
          break;
        }
        default:
          break;
        }
      }

      // Process successors
      for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
                                         SE = CurrBlock->succ_end();
           SI != SE; ++SI) {
        if (*SI == 0)
          continue;

        if (VisitedBlocks.alreadySet(*SI)) {
          V.handleSuccessorBackEdge(*SI);
          continue;
        }
        V.handleSuccessor(*SI);
      }

      V.exitCFGBlock(CurrBlock);
    }
    V.exitCFG(&CFGraph->getExit());
  }

public:
  CFG *CFGraph;
  const NamedDecl *FDecl;
  AnalysisDeclContext *ACtx;
  PostOrderCFGView *SortedGraph;
};


// Translate clang::Expr to til::SExpr.
class SExprBuilder {
public:
  typedef llvm::DenseMap<const Stmt*, til::Variable*> StatementMap;

  /// \brief Encapsulates the lexical context of a function call.  The lexical
  /// context includes the arguments to the call, including the implicit object
  /// argument.  When an attribute containing a mutex expression is attached to
  /// a method, the expression may refer to formal parameters of the method.
  /// Actual arguments must be substituted for formal parameters to derive
  /// the appropriate mutex expression in the lexical context where the function
  /// is called.  PrevCtx holds the context in which the arguments themselves
  /// should be evaluated; multiple calling contexts can be chained together
  /// by the lock_returned attribute.
  struct CallingContext {
    const NamedDecl *AttrDecl;  // The decl to which the attr is attached.
    const Expr *SelfArg;        // Implicit object argument -- e.g. 'this'
    unsigned NumArgs;           // Number of funArgs
    const Expr *const *FunArgs; // Function arguments
    CallingContext *Prev;       // The previous context; or 0 if none.
    bool SelfArrow;             // is Self referred to with -> or .?

    CallingContext(const NamedDecl *D = 0, const Expr *S = 0, unsigned N = 0,
                   const Expr *const *A = 0, CallingContext *P = 0)
        : AttrDecl(D), SelfArg(S), NumArgs(N), FunArgs(A), Prev(P),
          SelfArrow(false)
    {}
  };

  til::SExpr *lookupStmt(const Stmt *S);
  void insertStmt(const Stmt *S, til::Variable *V);

  // Translate a clang statement or expression to a TIL expression.
  // Also performs substitution of variables; Ctx provides the context.
  // Dispatches on the type of S.
  til::SExpr *translate(const Stmt *S, CallingContext *Ctx);


  til::SExpr *translateDeclRefExpr(const DeclRefExpr *DRE,
                                   CallingContext *Ctx) ;
  til::SExpr *translateCXXThisExpr(const CXXThisExpr *TE, CallingContext *Ctx);
  til::SExpr *translateMemberExpr(const MemberExpr *ME, CallingContext *Ctx);
  til::SExpr *translateCallExpr(const CallExpr *CE, CallingContext *Ctx);
  til::SExpr *translateCXXMemberCallExpr(const CXXMemberCallExpr *ME,
                                         CallingContext *Ctx);
  til::SExpr *translateCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE,
                                           CallingContext *Ctx);
  til::SExpr *translateUnaryOperator(const UnaryOperator *UO,
                                     CallingContext *Ctx);
  til::SExpr *translateBinaryOperator(const BinaryOperator *BO,
                                      CallingContext *Ctx);
  til::SExpr *translateCastExpr(const CastExpr *CE, CallingContext *Ctx);
  til::SExpr *translateArraySubscriptExpr(const ArraySubscriptExpr *E,
                                          CallingContext *Ctx);
  til::SExpr *translateConditionalOperator(const ConditionalOperator *C,
                                           CallingContext *Ctx);
  til::SExpr *translateBinaryConditionalOperator(
      const BinaryConditionalOperator *C, CallingContext *Ctx);


  SExprBuilder(til::MemRegionRef A, StatementMap *SM = 0)
      : Arena(A), SMap(SM), SelfVar(0) {
    // FIXME: we don't always have a self-variable.
    SelfVar = new (Arena) til::Variable(til::Variable::VK_SFun);
  }

protected:
  til::MemRegionRef Arena;
  StatementMap *SMap;       // Map from Stmt to TIL Variables
  til::Variable *SelfVar;   // Variable to use for 'this'
};


// Dump an SCFG to llvm::errs().
void printSCFG(CFGWalker &walker);


} // end namespace threadSafety

} // end namespace clang

#endif  // LLVM_CLANG_THREAD_SAFETY_COMMON_H
