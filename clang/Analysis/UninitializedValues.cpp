//==- UninitializedValues.cpp - Find Unintialized Values --------*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Uninitialized Values analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/UninitializedValues.h"
#include "clang/Analysis/CFGStmtVisitor.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "DataflowSolver.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Dataflow initialization logic.
//===----------------------------------------------------------------------===//      

namespace {

class RegisterDeclsAndExprs : public CFGStmtVisitor<RegisterDeclsAndExprs> {
  UninitializedValues::AnalysisDataTy& AD;
public:
  RegisterDeclsAndExprs(UninitializedValues::AnalysisDataTy& ad) :  AD(ad) {}
  
  void VisitBlockVarDecl(BlockVarDecl* VD) {
    if (AD.VMap.find(VD) == AD.VMap.end())
      AD.VMap[VD] = AD.NumDecls++;
  }
      
  void VisitDeclChain(ScopedDecl* D) {
    for (; D != NULL; D = D->getNextDeclarator())
      if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(D))
        VisitBlockVarDecl(VD);
  }
  
  void BlockStmt_VisitExpr(Expr* E) {
    if (AD.EMap.find(E) == AD.EMap.end())
      AD.EMap[E] = AD.NumBlockExprs++;
      
    Visit(E);
  }
  
  void VisitDeclRefExpr(DeclRefExpr* DR) {
    VisitDeclChain(DR->getDecl());
  }
  
  void VisitDeclStmt(DeclStmt* S) {
    VisitDeclChain(S->getDecl());
  }
  
  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }
  
};
  
} // end anonymous namespace

void UninitializedValues::InitializeValues(const CFG& cfg) {
  RegisterDeclsAndExprs R(this->getAnalysisData());
  
  for (CFG::const_iterator I=cfg.begin(), E=cfg.end(); I!=E; ++I)
    for (CFGBlock::const_iterator BI=I->begin(), BE=I->end(); BI!=BE; ++BI)
      R.BlockStmt_Visit(*BI);
  
  // Initialize the values of the last block.
//  UninitializedValues::ValTy& V = getBlockDataMap()[&cfg.getEntry()];
//  V.resetValues(getAnalysisData());
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//      

namespace {

class TransferFuncs : public CFGStmtVisitor<TransferFuncs,bool> {
  UninitializedValues::ValTy V;
  UninitializedValues::AnalysisDataTy& AD;
  bool InitWithAssigns;
public:
  TransferFuncs(UninitializedValues::AnalysisDataTy& ad, 
                bool init_with_assigns=true) : 
    AD(ad), InitWithAssigns(init_with_assigns) {
    V.resetValues(AD);
  }
  
  UninitializedValues::ValTy& getVal() { return V; }
  
  bool VisitDeclRefExpr(DeclRefExpr* DR);
  bool VisitBinaryOperator(BinaryOperator* B);
  bool VisitUnaryOperator(UnaryOperator* U);
  bool VisitStmt(Stmt* S);
  bool VisitCallExpr(CallExpr* C);
  bool BlockStmt_VisitExpr(Expr* E);
  bool VisitDeclStmt(DeclStmt* D);
  
  static inline bool Initialized() { return true; }
  static inline bool Uninitialized() { return false; }
};


bool TransferFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {
  if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(DR->getDecl())) {
    assert ( AD.VMap.find(VD) != AD.VMap.end() && "Unknown VarDecl.");
    if (AD.Observer)
      AD.Observer->ObserveDeclRefExpr(V,AD,DR,VD);
      
    return V.DeclBV[ AD.VMap[VD] ];    
  }
  else
    return Initialized();
}

bool TransferFuncs::VisitBinaryOperator(BinaryOperator* B) {
  if (CFG::hasImplicitControlFlow(B)) {
    assert ( AD.EMap.find(B) != AD.EMap.end() && "Unknown block-level expr.");
    return V.ExprBV[ AD.EMap[B] ];
  }
  
  if (B->isAssignmentOp()) {
    // Get the Decl for the LHS, if any
    for (Stmt* S  = B->getLHS() ;; ) {
      if (ParenExpr* P = dyn_cast<ParenExpr>(S))
        S = P->getSubExpr();
      else if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S))
        if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(DR->getDecl())) {
          assert ( AD.VMap.find(VD) != AD.VMap.end() && "Unknown VarDecl.");
          
          if(InitWithAssigns) {
            // Pseudo-hack to prevent cascade of warnings.  If the RHS uses
            // an uninitialized value, then we are already going to flag a warning
            // related to the "cause".  Thus, propogating uninitialized doesn't
            // make sense, since we are just adding extra messages that don't
            // contribute to diagnosing the bug.  In InitWithAssigns mode
            // we unconditionally set the assigned variable to Initialized to
            // prevent Uninitialized propogation.
            return V.DeclBV[AD.VMap[VD]] = Initialized();
          }
          else 
            return V.DeclBV[ AD.VMap[VD] ] = Visit(B->getRHS());
        }

      break;
    }
  }
  
  return VisitStmt(B);
}

bool TransferFuncs::VisitDeclStmt(DeclStmt* S) {
  bool x = Initialized();
  
  for (ScopedDecl* D = S->getDecl(); D != NULL; D = D->getNextDeclarator())
    if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(D))
      if (Stmt* I = VD->getInit()) {
        assert ( AD.EMap.find(cast<Expr>(I)) != 
                 AD.EMap.end() && "Unknown Expr.");
                 
        assert ( AD.VMap.find(VD) != AD.VMap.end() && "Unknown VarDecl.");
        x = V.ExprBV[ AD.EMap[cast<Expr>(I)] ];
        V.DeclBV[ AD.VMap[VD] ] = x;
      }
      
  return x;
}

bool TransferFuncs::VisitCallExpr(CallExpr* C) {
  VisitStmt(C);
  return Initialized();
}

bool TransferFuncs::VisitUnaryOperator(UnaryOperator* U) {
  switch (U->getOpcode()) {
    case UnaryOperator::AddrOf: {
      // Blast through parentheses and find the decl (if any).  Treat it
      // as initialized from this point forward.
      for (Stmt* S = U->getSubExpr() ;; )
        if (ParenExpr* P = dyn_cast<ParenExpr>(S))
          S = P->getSubExpr();
        else if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S)) {
          if (BlockVarDecl* VD = dyn_cast<BlockVarDecl>(DR->getDecl())) {
            assert ( AD.VMap.find(VD) != AD.VMap.end() && "Unknown VarDecl.");
            V.DeclBV[ AD.VMap[VD] ] = Initialized();
          }
          break;
        }
        else {
          // Evaluate the transfer function for subexpressions, even
          // if we cannot reason more deeply about the &-expression.
          return Visit(U->getSubExpr());
        }

      return Initialized();
    }

    default:
      return Visit(U->getSubExpr());
  }      
}

bool TransferFuncs::VisitStmt(Stmt* S) {
  bool x = Initialized();

  // We don't stop at the first subexpression that is Uninitialized because
  // evaluating some subexpressions may result in propogating "Uninitialized"
  // or "Initialized" to variables referenced in the other subexpressions.
  for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (Visit(*I) == Uninitialized())
      x = Uninitialized();
  
  return x;
}

bool TransferFuncs::BlockStmt_VisitExpr(Expr* E) {
  assert ( AD.EMap.find(E) != AD.EMap.end() );
  return V.ExprBV[ AD.EMap[E] ] = Visit(E);
}
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Merge operator.
//
//  In our transfer functions we take the approach that any
//  combination of unintialized values, e.g. Unitialized + ___ = Unitialized.
//
//  Merges take the opposite approach.
//
//  In the merge of dataflow values (for Decls) we prefer unsoundness, and
//  prefer false negatives to false positives.  At merges, if a value for a
//  tracked Decl is EVER initialized in any of the predecessors we treat it as
//  initialized at the confluence point.
//
//  For tracked CFGBlock-level expressions (such as the result of
//  short-circuit), we do the opposite merge: if a value is EVER uninitialized
//  in a predecessor we treat it as uninitalized at the confluence point.
//  The reason we do this is because dataflow values for tracked Exprs are
//  not as control-dependent as dataflow values for tracked Decls.
//===----------------------------------------------------------------------===//      

namespace {
struct Merge {
  void operator()(UninitializedValues::ValTy& Dst,
                  UninitializedValues::ValTy& Src) {
    assert (Dst.DeclBV.size() == Src.DeclBV.size() 
            && "Bitvector sizes do not match.");
            
    Dst.DeclBV |= Src.DeclBV;
    
    assert (Dst.ExprBV.size() == Src.ExprBV.size()
            && "Bitvector sizes do not match.");

    Dst.ExprBV &= Src.ExprBV;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Unitialized values checker.   Scan an AST and flag variable uses
//===----------------------------------------------------------------------===//      

UninitializedValues_ValueTypes::ObserverTy::~ObserverTy() {}

namespace {

class UninitializedValuesChecker : public UninitializedValues::ObserverTy {
  ASTContext &Ctx;
  Diagnostic &Diags;
  llvm::SmallPtrSet<BlockVarDecl*,10> AlreadyWarned;
  
public:
  UninitializedValuesChecker(ASTContext &ctx, Diagnostic &diags)
    : Ctx(ctx), Diags(diags) {}
    
  virtual void ObserveDeclRefExpr(UninitializedValues::ValTy& V,
                                  UninitializedValues::AnalysisDataTy& AD,
                                  DeclRefExpr* DR, BlockVarDecl* VD) {

    assert ( AD.VMap.find(VD) != AD.VMap.end() && "Unknown VarDecl.");
    if (V.DeclBV[ AD.VMap[VD] ] == TransferFuncs::Uninitialized())
      if (AlreadyWarned.insert(VD))
        Diags.Report(DR->getSourceRange().Begin(), diag::warn_uninit_val);
  }
};

} // end anonymous namespace

namespace clang {

void CheckUninitializedValues(CFG& cfg, ASTContext &Ctx, Diagnostic &Diags) {

  typedef DataflowSolver<UninitializedValues,TransferFuncs,Merge> Solver;
  
  // Compute the unitialized values information.
  UninitializedValues U;
  Solver S(U);
  S.runOnCFG(cfg);
  
  // Scan for DeclRefExprs that use uninitialized values.
  UninitializedValuesChecker Observer(Ctx,Diags);
  U.getAnalysisData().Observer = &Observer;

  for (CFG::iterator I=cfg.begin(), E=cfg.end(); I!=E; ++I)
    S.runOnBlock(&*I);
}

}
