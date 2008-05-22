//==- DeadStores.cpp - Check for stores to dead variables --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a DeadStores, a flow-sensitive checker that looks for
//  stores to variables that are no longer live.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Visitors/CFGRecStmtVisitor.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

namespace {
  
class VISIBILITY_HIDDEN DeadStoreObs : public LiveVariables::ObserverTy {
  ASTContext &Ctx;
  Diagnostic &Diags;
  DiagnosticClient &Client;
public:
  DeadStoreObs(ASTContext &ctx, Diagnostic &diags, DiagnosticClient &client)
    : Ctx(ctx), Diags(diags), Client(client) {}    
  
  virtual ~DeadStoreObs() {}
  
  unsigned GetDiag(VarDecl* VD) {      
    std::string msg = "value stored to '" + std::string(VD->getName()) +
                      "' is never used";
    
    return Diags.getCustomDiagID(Diagnostic::Warning, msg.c_str());
                               
  }
  
  void CheckDeclRef(DeclRefExpr* DR, Expr* Val,
                    const LiveVariables::AnalysisDataTy& AD,
                    const LiveVariables::ValTy& Live) {
    
    if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl()))
      if (VD->hasLocalStorage() && !Live(VD, AD)) {
        SourceRange R = Val->getSourceRange();        
        Diags.Report(&Client,
                     Ctx.getFullLoc(DR->getSourceRange().getBegin()),
                     GetDiag(VD), 0, 0, &R, 1);
      }
  }
  
  virtual void ObserveStmt(Stmt* S,
                           const LiveVariables::AnalysisDataTy& AD,
                           const LiveVariables::ValTy& Live) {
    
    // Skip statements in macros.
    if (S->getLocStart().isMacroID())
      return;
    
    if (BinaryOperator* B = dyn_cast<BinaryOperator>(S)) {    
      if (!B->isAssignmentOp()) return; // Skip non-assignments.
      
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(B->getLHS()))
        CheckDeclRef(DR, B->getRHS(), AD, Live);
    }
    else if (UnaryOperator* U = dyn_cast<UnaryOperator>(S)) {
      if (!U->isIncrementOp())
        return;
      
      Expr *Ex = U->getSubExpr()->IgnoreParenCasts();
      
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(Ex))
        CheckDeclRef(DR, U, AD, Live);
    }    
    else if (DeclStmt* DS = dyn_cast<DeclStmt>(S))
      // Iterate through the decls.  Warn if any initializers are complex
      // expressions that are not live (never used).
      for (ScopedDecl* SD = DS->getDecl(); SD; SD = SD->getNextDeclarator()) {        
        
        VarDecl* V = dyn_cast<VarDecl>(SD);
        if (!V) continue;
        
        if (V->hasLocalStorage())
          if (Expr* E = V->getInit()) {
            if (!Live(V, AD)) {
              // Special case: check for initializations with constants.
              //
              //  e.g. : int x = 0;
              //
              // If x is EVER assigned a new value later, don't issue
              // a warning.  This is because such initialization can be
              // due to defensive programming.
              if (!E->isConstantExpr(Ctx,NULL)) {
                // Flag a warning.
                SourceRange R = E->getSourceRange();
                Diags.Report(&Client,
                             Ctx.getFullLoc(V->getLocation()),
                             GetDiag(V), 0, 0, &R, 1);
              }
            }
          }
      }
  }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver function to invoke the Dead-Stores checker on a CFG.
//===----------------------------------------------------------------------===//

void clang::CheckDeadStores(CFG& cfg, ASTContext &Ctx, Diagnostic &Diags) {  
  LiveVariables L(cfg);
  L.runOnCFG(cfg);
  DeadStoreObs A(Ctx, Diags, Diags.getClient());
  L.runOnAllBlocks(cfg, &A);
}

//===----------------------------------------------------------------------===//
// BugReporter-based invocation of the Dead-Stores checker.
//===----------------------------------------------------------------------===//
  
namespace {

class VISIBILITY_HIDDEN DiagBugReport : public RangedBugReport {
  std::list<std::string> Strs;
  FullSourceLoc L;
public:
  DiagBugReport(BugType& D, FullSourceLoc l) :
    RangedBugReport(D, NULL), L(l) {}
  
  virtual ~DiagBugReport() {}
  virtual FullSourceLoc getLocation(SourceManager&) { return L; }
  
  void addString(const std::string& s) { Strs.push_back(s); }  
  
  typedef std::list<std::string>::const_iterator str_iterator;
  str_iterator str_begin() const { return Strs.begin(); }
  str_iterator str_end() const { return Strs.end(); }
};
  
class VISIBILITY_HIDDEN DiagCollector : public DiagnosticClient {
  std::list<DiagBugReport> Reports;
  BugType& D;
public:
  DiagCollector(BugType& d) : D(d) {}
  
  virtual ~DiagCollector() {}
  
  virtual void HandleDiagnostic(Diagnostic &Diags, 
                                Diagnostic::Level DiagLevel,
                                FullSourceLoc Pos,
                                diag::kind ID,
                                const std::string *Strs,
                                unsigned NumStrs,
                                const SourceRange *Ranges, 
                                unsigned NumRanges) {
    
    // FIXME: Use a map from diag::kind to BugType, instead of having just
    //  one BugType.
    
    Reports.push_back(DiagBugReport(D, Pos));
    DiagBugReport& R = Reports.back();
    
    for ( ; NumRanges ; --NumRanges, ++Ranges)
      R.addRange(*Ranges);
    
    for ( ; NumStrs ; --NumStrs, ++Strs)
      R.addString(*Strs);    
  }
  
  // Iterators.
  
  typedef std::list<DiagBugReport>::iterator iterator;
  iterator begin() { return Reports.begin(); }
  iterator end() { return Reports.end(); }
};
  
class VISIBILITY_HIDDEN DeadStoresChecker : public BugTypeCacheLocation {
public:
  virtual const char* getName() const {
    return "dead store";
  }
  
  virtual const char* getDescription() const {
    return "Value stored to variable is never subsequently read.";
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    
    // Run the dead store checker and collect the diagnostics.
    DiagCollector C(*this);    
    DeadStoreObs A(BR.getContext(), BR.getDiagnostic(), C);
    GRExprEngine& Eng = BR.getEngine();
    Eng.getLiveness().runOnAllBlocks(Eng.getCFG(), &A);
    
    // Emit the bug reports.
    
    for (DiagCollector::iterator I = C.begin(), E = C.end(); I != E; ++I)
      BR.EmitWarning(*I);    
  }
};
} // end anonymous namespace

BugType* clang::MakeDeadStoresChecker() {
  return new DeadStoresChecker();
}
