//=- AnalysisBasedWarnings.cpp - Sema warnings based on libAnalysis -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines analysis_warnings::[Policy,Executor].
// Together they are used by Sema to issue warnings based on inexpensive
// static analysis algorithms in libAnalysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Analyses/ReachableCode.h"
#include "clang/Analysis/Analyses/UninitializedValuesV2.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Casting.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Unreachable code analysis.
//===----------------------------------------------------------------------===//

namespace {
  class UnreachableCodeHandler : public reachable_code::Callback {
    Sema &S;
  public:
    UnreachableCodeHandler(Sema &s) : S(s) {}

    void HandleUnreachable(SourceLocation L, SourceRange R1, SourceRange R2) {
      S.Diag(L, diag::warn_unreachable) << R1 << R2;
    }
  };
}

/// CheckUnreachable - Check for unreachable code.
static void CheckUnreachable(Sema &S, AnalysisContext &AC) {
  UnreachableCodeHandler UC(S);
  reachable_code::FindUnreachableCode(AC, UC);
}

//===----------------------------------------------------------------------===//
// Check for missing return value.
//===----------------------------------------------------------------------===//

enum ControlFlowKind {
  UnknownFallThrough,
  NeverFallThrough,
  MaybeFallThrough,
  AlwaysFallThrough,
  NeverFallThroughOrReturn
};

/// CheckFallThrough - Check that we don't fall off the end of a
/// Statement that should return a value.
///
/// \returns AlwaysFallThrough iff we always fall off the end of the statement,
/// MaybeFallThrough iff we might or might not fall off the end,
/// NeverFallThroughOrReturn iff we never fall off the end of the statement or
/// return.  We assume NeverFallThrough iff we never fall off the end of the
/// statement but we may return.  We assume that functions not marked noreturn
/// will return.
static ControlFlowKind CheckFallThrough(AnalysisContext &AC) {
  CFG *cfg = AC.getCFG();
  if (cfg == 0) return UnknownFallThrough;

  // The CFG leaves in dead things, and we don't want the dead code paths to
  // confuse us, so we mark all live things first.
  llvm::BitVector live(cfg->getNumBlockIDs());
  unsigned count = reachable_code::ScanReachableFromBlock(cfg->getEntry(),
                                                          live);

  bool AddEHEdges = AC.getAddEHEdges();
  if (!AddEHEdges && count != cfg->getNumBlockIDs())
    // When there are things remaining dead, and we didn't add EH edges
    // from CallExprs to the catch clauses, we have to go back and
    // mark them as live.
    for (CFG::iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
      CFGBlock &b = **I;
      if (!live[b.getBlockID()]) {
        if (b.pred_begin() == b.pred_end()) {
          if (b.getTerminator() && isa<CXXTryStmt>(b.getTerminator()))
            // When not adding EH edges from calls, catch clauses
            // can otherwise seem dead.  Avoid noting them as dead.
            count += reachable_code::ScanReachableFromBlock(b, live);
          continue;
        }
      }
    }

  // Now we know what is live, we check the live precessors of the exit block
  // and look for fall through paths, being careful to ignore normal returns,
  // and exceptional paths.
  bool HasLiveReturn = false;
  bool HasFakeEdge = false;
  bool HasPlainEdge = false;
  bool HasAbnormalEdge = false;

  // Ignore default cases that aren't likely to be reachable because all
  // enums in a switch(X) have explicit case statements.
  CFGBlock::FilterOptions FO;
  FO.IgnoreDefaultsWithCoveredEnums = 1;

  for (CFGBlock::filtered_pred_iterator
	 I = cfg->getExit().filtered_pred_start_end(FO); I.hasMore(); ++I) {
    const CFGBlock& B = **I;
    if (!live[B.getBlockID()])
      continue;
    if (B.size() == 0) {
      if (B.getTerminator() && isa<CXXTryStmt>(B.getTerminator())) {
        HasAbnormalEdge = true;
        continue;
      }

      // A labeled empty statement, or the entry block...
      HasPlainEdge = true;
      continue;
    }
    CFGElement CE = B[B.size()-1];
    if (CFGInitializer CI = CE.getAs<CFGInitializer>()) {
      // A base or member initializer.
      HasPlainEdge = true;
      continue;
    }
    if (CFGMemberDtor MD = CE.getAs<CFGMemberDtor>()) {
      // A member destructor.
      HasPlainEdge = true;
      continue;
    }
    if (CFGBaseDtor BD = CE.getAs<CFGBaseDtor>()) {
      // A base destructor.
      HasPlainEdge = true;
      continue;
    }
    CFGStmt CS = CE.getAs<CFGStmt>();
    if (!CS.isValid())
      continue;
    Stmt *S = CS.getStmt();
    if (isa<ReturnStmt>(S)) {
      HasLiveReturn = true;
      continue;
    }
    if (isa<ObjCAtThrowStmt>(S)) {
      HasFakeEdge = true;
      continue;
    }
    if (isa<CXXThrowExpr>(S)) {
      HasFakeEdge = true;
      continue;
    }
    if (const AsmStmt *AS = dyn_cast<AsmStmt>(S)) {
      if (AS->isMSAsm()) {
        HasFakeEdge = true;
        HasLiveReturn = true;
        continue;
      }
    }
    if (isa<CXXTryStmt>(S)) {
      HasAbnormalEdge = true;
      continue;
    }

    bool NoReturnEdge = false;
    if (CallExpr *C = dyn_cast<CallExpr>(S)) {
      if (std::find(B.succ_begin(), B.succ_end(), &cfg->getExit())
            == B.succ_end()) {
        HasAbnormalEdge = true;
        continue;
      }
      Expr *CEE = C->getCallee()->IgnoreParenCasts();
      if (getFunctionExtInfo(CEE->getType()).getNoReturn()) {
        NoReturnEdge = true;
        HasFakeEdge = true;
      } else if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE)) {
        ValueDecl *VD = DRE->getDecl();
        if (VD->hasAttr<NoReturnAttr>()) {
          NoReturnEdge = true;
          HasFakeEdge = true;
        }
      }
    }
    // FIXME: Add noreturn message sends.
    if (NoReturnEdge == false)
      HasPlainEdge = true;
  }
  if (!HasPlainEdge) {
    if (HasLiveReturn)
      return NeverFallThrough;
    return NeverFallThroughOrReturn;
  }
  if (HasAbnormalEdge || HasFakeEdge || HasLiveReturn)
    return MaybeFallThrough;
  // This says AlwaysFallThrough for calls to functions that are not marked
  // noreturn, that don't return.  If people would like this warning to be more
  // accurate, such functions should be marked as noreturn.
  return AlwaysFallThrough;
}

namespace {

struct CheckFallThroughDiagnostics {
  unsigned diag_MaybeFallThrough_HasNoReturn;
  unsigned diag_MaybeFallThrough_ReturnsNonVoid;
  unsigned diag_AlwaysFallThrough_HasNoReturn;
  unsigned diag_AlwaysFallThrough_ReturnsNonVoid;
  unsigned diag_NeverFallThroughOrReturn;
  bool funMode;
  SourceLocation FuncLoc;

  static CheckFallThroughDiagnostics MakeForFunction(const Decl *Func) {
    CheckFallThroughDiagnostics D;
    D.FuncLoc = Func->getLocation();
    D.diag_MaybeFallThrough_HasNoReturn =
      diag::warn_falloff_noreturn_function;
    D.diag_MaybeFallThrough_ReturnsNonVoid =
      diag::warn_maybe_falloff_nonvoid_function;
    D.diag_AlwaysFallThrough_HasNoReturn =
      diag::warn_falloff_noreturn_function;
    D.diag_AlwaysFallThrough_ReturnsNonVoid =
      diag::warn_falloff_nonvoid_function;

    // Don't suggest that virtual functions be marked "noreturn", since they
    // might be overridden by non-noreturn functions.
    bool isVirtualMethod = false;
    if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Func))
      isVirtualMethod = Method->isVirtual();
    
    if (!isVirtualMethod)
      D.diag_NeverFallThroughOrReturn =
        diag::warn_suggest_noreturn_function;
    else
      D.diag_NeverFallThroughOrReturn = 0;
    
    D.funMode = true;
    return D;
  }

  static CheckFallThroughDiagnostics MakeForBlock() {
    CheckFallThroughDiagnostics D;
    D.diag_MaybeFallThrough_HasNoReturn =
      diag::err_noreturn_block_has_return_expr;
    D.diag_MaybeFallThrough_ReturnsNonVoid =
      diag::err_maybe_falloff_nonvoid_block;
    D.diag_AlwaysFallThrough_HasNoReturn =
      diag::err_noreturn_block_has_return_expr;
    D.diag_AlwaysFallThrough_ReturnsNonVoid =
      diag::err_falloff_nonvoid_block;
    D.diag_NeverFallThroughOrReturn =
      diag::warn_suggest_noreturn_block;
    D.funMode = false;
    return D;
  }

  bool checkDiagnostics(Diagnostic &D, bool ReturnsVoid,
                        bool HasNoReturn) const {
    if (funMode) {
      return (ReturnsVoid ||
              D.getDiagnosticLevel(diag::warn_maybe_falloff_nonvoid_function,
                                   FuncLoc) == Diagnostic::Ignored)
        && (!HasNoReturn ||
            D.getDiagnosticLevel(diag::warn_noreturn_function_has_return_expr,
                                 FuncLoc) == Diagnostic::Ignored)
        && (!ReturnsVoid ||
            D.getDiagnosticLevel(diag::warn_suggest_noreturn_block, FuncLoc)
              == Diagnostic::Ignored);
    }

    // For blocks.
    return  ReturnsVoid && !HasNoReturn
            && (!ReturnsVoid ||
                D.getDiagnosticLevel(diag::warn_suggest_noreturn_block, FuncLoc)
                  == Diagnostic::Ignored);
  }
};

}

/// CheckFallThroughForFunctionDef - Check that we don't fall off the end of a
/// function that should return a value.  Check that we don't fall off the end
/// of a noreturn function.  We assume that functions and blocks not marked
/// noreturn will return.
static void CheckFallThroughForBody(Sema &S, const Decl *D, const Stmt *Body,
                                    QualType BlockTy,
                                    const CheckFallThroughDiagnostics& CD,
                                    AnalysisContext &AC) {

  bool ReturnsVoid = false;
  bool HasNoReturn = false;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    ReturnsVoid = FD->getResultType()->isVoidType();
    HasNoReturn = FD->hasAttr<NoReturnAttr>() ||
       FD->getType()->getAs<FunctionType>()->getNoReturnAttr();
  }
  else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    ReturnsVoid = MD->getResultType()->isVoidType();
    HasNoReturn = MD->hasAttr<NoReturnAttr>();
  }
  else if (isa<BlockDecl>(D)) {
    if (const FunctionType *FT =
          BlockTy->getPointeeType()->getAs<FunctionType>()) {
      if (FT->getResultType()->isVoidType())
        ReturnsVoid = true;
      if (FT->getNoReturnAttr())
        HasNoReturn = true;
    }
  }

  Diagnostic &Diags = S.getDiagnostics();

  // Short circuit for compilation speed.
  if (CD.checkDiagnostics(Diags, ReturnsVoid, HasNoReturn))
      return;

  // FIXME: Function try block
  if (const CompoundStmt *Compound = dyn_cast<CompoundStmt>(Body)) {
    switch (CheckFallThrough(AC)) {
      case UnknownFallThrough:
        break;

      case MaybeFallThrough:
        if (HasNoReturn)
          S.Diag(Compound->getRBracLoc(),
                 CD.diag_MaybeFallThrough_HasNoReturn);
        else if (!ReturnsVoid)
          S.Diag(Compound->getRBracLoc(),
                 CD.diag_MaybeFallThrough_ReturnsNonVoid);
        break;
      case AlwaysFallThrough:
        if (HasNoReturn)
          S.Diag(Compound->getRBracLoc(),
                 CD.diag_AlwaysFallThrough_HasNoReturn);
        else if (!ReturnsVoid)
          S.Diag(Compound->getRBracLoc(),
                 CD.diag_AlwaysFallThrough_ReturnsNonVoid);
        break;
      case NeverFallThroughOrReturn:
        if (ReturnsVoid && !HasNoReturn && CD.diag_NeverFallThroughOrReturn)
          S.Diag(Compound->getLBracLoc(),
                 CD.diag_NeverFallThroughOrReturn);
        break;
      case NeverFallThrough:
        break;
    }
  }
}

//===----------------------------------------------------------------------===//
// -Wuninitialized
//===----------------------------------------------------------------------===//

namespace {
class UninitValsDiagReporter : public UninitVariablesHandler {
  Sema &S;
public:
  UninitValsDiagReporter(Sema &S) : S(S) {}
  
  void handleUseOfUninitVariable(const DeclRefExpr *dr, const VarDecl *vd) {
    S.Diag(dr->getLocStart(), diag::warn_var_is_uninit)
      << vd->getDeclName() << dr->getSourceRange();
  }
};
}

//===----------------------------------------------------------------------===//
// AnalysisBasedWarnings - Worker object used by Sema to execute analysis-based
//  warnings on a function, method, or block.
//===----------------------------------------------------------------------===//

clang::sema::AnalysisBasedWarnings::Policy::Policy() {
  enableCheckFallThrough = 1;
  enableCheckUnreachable = 0;
}

clang::sema::AnalysisBasedWarnings::AnalysisBasedWarnings(Sema &s) : S(s) {
  Diagnostic &D = S.getDiagnostics();
  DefaultPolicy.enableCheckUnreachable = (unsigned)
    (D.getDiagnosticLevel(diag::warn_unreachable, SourceLocation()) !=
        Diagnostic::Ignored);
}

void clang::sema::
AnalysisBasedWarnings::IssueWarnings(sema::AnalysisBasedWarnings::Policy P,
                                     const Decl *D, QualType BlockTy) {

  assert(BlockTy.isNull() || isa<BlockDecl>(D));

  // We avoid doing analysis-based warnings when there are errors for
  // two reasons:
  // (1) The CFGs often can't be constructed (if the body is invalid), so
  //     don't bother trying.
  // (2) The code already has problems; running the analysis just takes more
  //     time.
  Diagnostic &Diags = S.getDiagnostics();

  if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred())
    return;

  // Do not do any analysis for declarations in system headers if we are
  // going to just ignore them.
  if (Diags.getSuppressSystemWarnings() &&
      S.SourceMgr.isInSystemHeader(D->getLocation()))
    return;

  // For code in dependent contexts, we'll do this at instantiation time.
  if (cast<DeclContext>(D)->isDependentContext())
    return;

  const Stmt *Body = D->getBody();
  assert(Body);

  // Don't generate EH edges for CallExprs as we'd like to avoid the n^2
  // explosion for destrutors that can result and the compile time hit.
  AnalysisContext AC(D, 0, /*useUnoptimizedCFG=*/false, /*addehedges=*/false,
                     /*addImplicitDtors=*/true, /*addInitializers=*/true);

  // Warning: check missing 'return'
  if (P.enableCheckFallThrough) {
    const CheckFallThroughDiagnostics &CD =
      (isa<BlockDecl>(D) ? CheckFallThroughDiagnostics::MakeForBlock()
                         : CheckFallThroughDiagnostics::MakeForFunction(D));
    CheckFallThroughForBody(S, D, Body, BlockTy, CD, AC);
  }

  // Warning: check for unreachable code
  if (P.enableCheckUnreachable)
    CheckUnreachable(S, AC);
  
  if (Diags.getDiagnosticLevel(diag::warn_var_is_uninit, D->getLocStart())
      != Diagnostic::Ignored) {
    if (!S.getLangOptions().CPlusPlus) {
      CFG *cfg = AC.getCFG();
      if (cfg) {
        UninitValsDiagReporter reporter(S);
        runUninitializedVariablesAnalysis(*cast<DeclContext>(D), *cfg,
                                          reporter);
      }
    }
  }
}

void clang::sema::
AnalysisBasedWarnings::IssueWarnings(sema::AnalysisBasedWarnings::Policy P,
                                     const BlockExpr *E) {
  return IssueWarnings(P, E->getBlockDecl(), E->getType());
}

void clang::sema::
AnalysisBasedWarnings::IssueWarnings(sema::AnalysisBasedWarnings::Policy P,
                                     const ObjCMethodDecl *D) {
  return IssueWarnings(P, D, QualType());
}

void clang::sema::
AnalysisBasedWarnings::IssueWarnings(sema::AnalysisBasedWarnings::Policy P,
                                     const FunctionDecl *D) {
  return IssueWarnings(P, D, QualType());
}
