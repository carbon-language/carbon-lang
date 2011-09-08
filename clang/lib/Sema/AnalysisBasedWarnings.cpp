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
#include "clang/Sema/ScopeInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Analyses/ReachableCode.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Analysis/Analyses/UninitializedValues.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <vector>

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
  unsigned count = reachable_code::ScanReachableFromBlock(&cfg->getEntry(),
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
            count += reachable_code::ScanReachableFromBlock(&b, live);
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

    // Destructors can appear after the 'return' in the CFG.  This is
    // normal.  We need to look pass the destructors for the return
    // statement (if it exists).
    CFGBlock::const_reverse_iterator ri = B.rbegin(), re = B.rend();
    bool hasNoReturnDtor = false;
    
    for ( ; ri != re ; ++ri) {
      CFGElement CE = *ri;

      // FIXME: The right solution is to just sever the edges in the
      // CFG itself.
      if (const CFGImplicitDtor *iDtor = ri->getAs<CFGImplicitDtor>())
        if (iDtor->isNoReturn(AC.getASTContext())) {
          hasNoReturnDtor = true;
          HasFakeEdge = true;
          break;
        }
      
      if (isa<CFGStmt>(CE))
        break;
    }
    
    if (hasNoReturnDtor)
      continue;
    
    // No more CFGElements in the block?
    if (ri == re) {
      if (B.getTerminator() && isa<CXXTryStmt>(B.getTerminator())) {
        HasAbnormalEdge = true;
        continue;
      }
      // A labeled empty statement, or the entry block...
      HasPlainEdge = true;
      continue;
    }

    CFGStmt CS = cast<CFGStmt>(*ri);
    const Stmt *S = CS.getStmt();
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
    if (const CallExpr *C = dyn_cast<CallExpr>(S)) {
      if (std::find(B.succ_begin(), B.succ_end(), &cfg->getExit())
            == B.succ_end()) {
        HasAbnormalEdge = true;
        continue;
      }
      const Expr *CEE = C->getCallee()->IgnoreParenCasts();
      QualType calleeType = CEE->getType();
      if (calleeType == AC.getASTContext().BoundMemberTy) {
        calleeType = Expr::findBoundMemberType(CEE);
        assert(!calleeType.isNull() && "analyzing unresolved call?");
      }
      if (getFunctionExtInfo(calleeType).getNoReturn()) {
        NoReturnEdge = true;
        HasFakeEdge = true;
      } else if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE)) {
        const ValueDecl *VD = DRE->getDecl();
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
                                    const BlockExpr *blkExpr,
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
    QualType BlockTy = blkExpr->getType();
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
        if (ReturnsVoid && !HasNoReturn && CD.diag_NeverFallThroughOrReturn) {
          if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
            S.Diag(Compound->getLBracLoc(), CD.diag_NeverFallThroughOrReturn)
              << FD;
          } else {
            S.Diag(Compound->getLBracLoc(), CD.diag_NeverFallThroughOrReturn);
          }
        }
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
/// ContainsReference - A visitor class to search for references to
/// a particular declaration (the needle) within any evaluated component of an
/// expression (recursively).
class ContainsReference : public EvaluatedExprVisitor<ContainsReference> {
  bool FoundReference;
  const DeclRefExpr *Needle;

public:
  ContainsReference(ASTContext &Context, const DeclRefExpr *Needle)
    : EvaluatedExprVisitor<ContainsReference>(Context),
      FoundReference(false), Needle(Needle) {}

  void VisitExpr(Expr *E) {
    // Stop evaluating if we already have a reference.
    if (FoundReference)
      return;

    EvaluatedExprVisitor<ContainsReference>::VisitExpr(E);
  }

  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E == Needle)
      FoundReference = true;
    else
      EvaluatedExprVisitor<ContainsReference>::VisitDeclRefExpr(E);
  }

  bool doesContainReference() const { return FoundReference; }
};
}

/// DiagnoseUninitializedUse -- Helper function for diagnosing uses of an
/// uninitialized variable. This manages the different forms of diagnostic
/// emitted for particular types of uses. Returns true if the use was diagnosed
/// as a warning. If a pariticular use is one we omit warnings for, returns
/// false.
static bool DiagnoseUninitializedUse(Sema &S, const VarDecl *VD,
                                     const Expr *E, bool isAlwaysUninit) {
  bool isSelfInit = false;

  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (isAlwaysUninit) {
      // Inspect the initializer of the variable declaration which is
      // being referenced prior to its initialization. We emit
      // specialized diagnostics for self-initialization, and we
      // specifically avoid warning about self references which take the
      // form of:
      //
      //   int x = x;
      //
      // This is used to indicate to GCC that 'x' is intentionally left
      // uninitialized. Proven code paths which access 'x' in
      // an uninitialized state after this will still warn.
      //
      // TODO: Should we suppress maybe-uninitialized warnings for
      // variables initialized in this way?
      if (const Expr *Initializer = VD->getInit()) {
        if (DRE == Initializer->IgnoreParenImpCasts())
          return false;

        ContainsReference CR(S.Context, DRE);
        CR.Visit(const_cast<Expr*>(Initializer));
        isSelfInit = CR.doesContainReference();
      }
      if (isSelfInit) {
        S.Diag(DRE->getLocStart(),
               diag::warn_uninit_self_reference_in_init)
        << VD->getDeclName() << VD->getLocation() << DRE->getSourceRange();
      } else {
        S.Diag(DRE->getLocStart(), diag::warn_uninit_var)
          << VD->getDeclName() << DRE->getSourceRange();
      }
    } else {
      S.Diag(DRE->getLocStart(), diag::warn_maybe_uninit_var)
        << VD->getDeclName() << DRE->getSourceRange();
    }
  } else {
    const BlockExpr *BE = cast<BlockExpr>(E);
    S.Diag(BE->getLocStart(),
           isAlwaysUninit ? diag::warn_uninit_var_captured_by_block
                          : diag::warn_maybe_uninit_var_captured_by_block)
      << VD->getDeclName();
  }

  // Report where the variable was declared when the use wasn't within
  // the initializer of that declaration.
  if (!isSelfInit)
    S.Diag(VD->getLocStart(), diag::note_uninit_var_def)
      << VD->getDeclName();

  return true;
}

static void SuggestInitializationFixit(Sema &S, const VarDecl *VD) {
  // Don't issue a fixit if there is already an initializer.
  if (VD->getInit())
    return;

  // Suggest possible initialization (if any).
  const char *initialization = 0;
  QualType VariableTy = VD->getType().getCanonicalType();

  if (VariableTy->isObjCObjectPointerType() ||
      VariableTy->isBlockPointerType()) {
    // Check if 'nil' is defined.
    if (S.PP.getMacroInfo(&S.getASTContext().Idents.get("nil")))
      initialization = " = nil";
    else
      initialization = " = 0";
  }
  else if (VariableTy->isRealFloatingType())
    initialization = " = 0.0";
  else if (VariableTy->isBooleanType() && S.Context.getLangOptions().CPlusPlus)
    initialization = " = false";
  else if (VariableTy->isEnumeralType())
    return;
  else if (VariableTy->isPointerType() || VariableTy->isMemberPointerType()) {
    if (S.Context.getLangOptions().CPlusPlus0x)
      initialization = " = nullptr";
    // Check if 'NULL' is defined.
    else if (S.PP.getMacroInfo(&S.getASTContext().Idents.get("NULL")))
      initialization = " = NULL";
    else
      initialization = " = 0";
  }
  else if (VariableTy->isScalarType())
    initialization = " = 0";

  if (initialization) {
    SourceLocation loc = S.PP.getLocForEndOfToken(VD->getLocEnd());
    S.Diag(loc, diag::note_var_fixit_add_initialization)
      << FixItHint::CreateInsertion(loc, initialization);
  }
}

typedef std::pair<const Expr*, bool> UninitUse;

namespace {
struct SLocSort {
  bool operator()(const UninitUse &a, const UninitUse &b) {
    SourceLocation aLoc = a.first->getLocStart();
    SourceLocation bLoc = b.first->getLocStart();
    return aLoc.getRawEncoding() < bLoc.getRawEncoding();
  }
};

class UninitValsDiagReporter : public UninitVariablesHandler {
  Sema &S;
  typedef SmallVector<UninitUse, 2> UsesVec;
  typedef llvm::DenseMap<const VarDecl *, UsesVec*> UsesMap;
  UsesMap *uses;
  
public:
  UninitValsDiagReporter(Sema &S) : S(S), uses(0) {}
  ~UninitValsDiagReporter() { 
    flushDiagnostics();
  }
  
  void handleUseOfUninitVariable(const Expr *ex, const VarDecl *vd,
                                 bool isAlwaysUninit) {
    if (!uses)
      uses = new UsesMap();
    
    UsesVec *&vec = (*uses)[vd];
    if (!vec)
      vec = new UsesVec();
    
    vec->push_back(std::make_pair(ex, isAlwaysUninit));
  }
  
  void flushDiagnostics() {
    if (!uses)
      return;
    
    for (UsesMap::iterator i = uses->begin(), e = uses->end(); i != e; ++i) {
      const VarDecl *vd = i->first;
      UsesVec *vec = i->second;

      // Sort the uses by their SourceLocations.  While not strictly
      // guaranteed to produce them in line/column order, this will provide
      // a stable ordering.
      std::sort(vec->begin(), vec->end(), SLocSort());
      
      for (UsesVec::iterator vi = vec->begin(), ve = vec->end(); vi != ve;
           ++vi) {
        if (!DiagnoseUninitializedUse(S, vd, vi->first,
                                      /*isAlwaysUninit=*/vi->second))
          continue;

        SuggestInitializationFixit(S, vd);

        // Skip further diagnostics for this variable. We try to warn only on
        // the first point at which a variable is used uninitialized.
        break;
      }

      delete vec;
    }
    delete uses;
  }
};
}


//===----------------------------------------------------------------------===//
// -Wthread-safety
//===----------------------------------------------------------------------===//

namespace {
/// \brief Implements a set of CFGBlocks using a BitVector.
///
/// This class contains a minimal interface, primarily dictated by the SetType
/// template parameter of the llvm::po_iterator template, as used with external
/// storage. We also use this set to keep track of which CFGBlocks we visit
/// during the analysis.
class CFGBlockSet {
  llvm::BitVector VisitedBlockIDs;

public:
  // po_iterator requires this iterator, but the only interface needed is the
  // value_type typedef.
  struct iterator {
    typedef const CFGBlock *value_type;
  };

  CFGBlockSet() {}
  CFGBlockSet(const CFG *G) : VisitedBlockIDs(G->getNumBlockIDs(), false) {}

  /// \brief Set the bit associated with a particular CFGBlock.
  /// This is the important method for the SetType template parameter.
  bool insert(const CFGBlock *Block) {
    // Note that insert() is called by po_iterator, which doesn't check to make
    // sure that Block is non-null.  Moreover, the CFGBlock iterator will
    // occasionally hand out null pointers for pruned edges, so we catch those
    // here.
    if (Block == 0)
      return false;  // if an edge is trivially false.
    if (VisitedBlockIDs.test(Block->getBlockID()))
      return false;
    VisitedBlockIDs.set(Block->getBlockID());
    return true;
  }

  /// \brief Check if the bit for a CFGBlock has been already set.
  /// This method is for tracking visited blocks in the main threadsafety loop.
  /// Block must not be null.
  bool alreadySet(const CFGBlock *Block) {
    return VisitedBlockIDs.test(Block->getBlockID());
  }
};

/// \brief We create a helper class which we use to iterate through CFGBlocks in
/// the topological order.
class TopologicallySortedCFG {
  typedef llvm::po_iterator<const CFG*, CFGBlockSet, true>  po_iterator;

  std::vector<const CFGBlock*> Blocks;

public:
  typedef std::vector<const CFGBlock*>::reverse_iterator iterator;

  TopologicallySortedCFG(const CFG *CFGraph) {
    Blocks.reserve(CFGraph->getNumBlockIDs());
    CFGBlockSet BSet(CFGraph);

    for (po_iterator I = po_iterator::begin(CFGraph, BSet),
         E = po_iterator::end(CFGraph, BSet); I != E; ++I) {
      Blocks.push_back(*I);
    }
  }

  iterator begin() {
    return Blocks.rbegin();
  }

  iterator end() {
    return Blocks.rend();
  }
};

/// \brief A LockID object uniquely identifies a particular lock acquired, and
/// is built from an Expr* (i.e. calling a lock function).
///
/// Thread-safety analysis works by comparing lock expressions.  Within the
/// body of a function, an expression such as "x->foo->bar.mu" will resolve to
/// a particular lock object at run-time.  Subsequent occurrences of the same
/// expression (where "same" means syntactic equality) will refer to the same
/// run-time object if three conditions hold:
/// (1) Local variables in the expression, such as "x" have not changed.
/// (2) Values on the heap that affect the expression have not changed.
/// (3) The expression involves only pure function calls.
/// The current implementation assumes, but does not verify, that multiple uses
/// of the same lock expression satisfies these criteria.
///
/// Clang introduces an additional wrinkle, which is that it is difficult to
/// derive canonical expressions, or compare expressions directly for equality.
/// Thus, we identify a lock not by an Expr, but by the set of named
/// declarations that are referenced by the Expr.  In other words,
/// x->foo->bar.mu will be a four element vector with the Decls for
/// mu, bar, and foo, and x.  The vector will uniquely identify the expression
/// for all practical purposes.
///
/// Note we will need to perform substitution on "this" and function parameter
/// names when constructing a lock expression.
///
/// For example:
/// class C { Mutex Mu;  void lock() EXCLUSIVE_LOCK_FUNCTION(this->Mu); };
/// void myFunc(C *X) { ... X->lock() ... }
/// The original expression for the lock acquired by myFunc is "this->Mu", but
/// "X" is substituted for "this" so we get X->Mu();
///
/// For another example:
/// foo(MyList *L) EXCLUSIVE_LOCKS_REQUIRED(L->Mu) { ... }
/// MyList *MyL;
/// foo(MyL);  // requires lock MyL->Mu to be held
///
/// FIXME: In C++0x Mutexes are the objects that control access to shared
/// variables, while Locks are the objects that acquire and release Mutexes. We
/// may want to switch to this new terminology soon, in which case we should
/// rename this class "Mutex" and rename "LockId" to "MutexId", as well as
/// making sure that the terms Lock and Mutex throughout this code are
/// consistent with C++0x
///
/// FIXME: We should also pick one and canonicalize all usage of lock vs acquire
/// and unlock vs release as verbs.
class LockID {
  SmallVector<NamedDecl*, 2> DeclSeq;
 
  /// Build a Decl sequence representing the lock from the given expression.
  /// Recursive function that bottoms out when the final DeclRefExpr is reached.
  void buildLock(Expr *Exp) {
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Exp)) {
      NamedDecl *ND = cast<NamedDecl>(DRE->getDecl()->getCanonicalDecl());
      DeclSeq.push_back(ND);
    } else if (MemberExpr *ME = dyn_cast<MemberExpr>(Exp)) {
      NamedDecl *ND = ME->getMemberDecl();
      DeclSeq.push_back(ND);
      buildLock(ME->getBase());
    } else if (isa<CXXThisExpr>(Exp)) {
      return;
    } else {
      // FIXME: add diagnostic
      llvm::report_fatal_error("Expected lock expression!");
    }
  }

public:
  LockID(Expr *LExpr) {
    buildLock(LExpr);
    assert(!DeclSeq.empty());
  }

  bool operator==(const LockID &other) const {
    return DeclSeq == other.DeclSeq;
  }

  bool operator!=(const LockID &other) const {
    return !(*this == other);
  }

  // SmallVector overloads Operator< to do lexicographic ordering. Note that
  // we use pointer equality (and <) to compare NamedDecls. This means the order
  // of LockIDs in a lockset is nondeterministic. In order to output
  // diagnostics in a deterministic ordering, we must order all diagnostics to
  // output by SourceLocation when iterating through this lockset.
  bool operator<(const LockID &other) const {
    return DeclSeq < other.DeclSeq;
  }

  /// \brief Returns the name of the first Decl in the list for a given LockID;
  /// e.g. the lock expression foo.bar() has name "bar".
  /// The caret will point unambiguously to the lock expression, so using this
  /// name in diagnostics is a way to get simple, and consistent, lock names.
  /// We do not want to output the entire expression text for security reasons.
  StringRef getName() const {
    return DeclSeq.front()->getName();
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    for (SmallVectorImpl<NamedDecl*>::const_iterator I = DeclSeq.begin(),
         E = DeclSeq.end(); I != E; ++I) {
      ID.AddPointer(*I);
    }
  }
};

/// \brief This is a helper class that stores info about the most recent
/// accquire of a Lock.
///
/// The main body of the analysis maps LockIDs to LockDatas.
struct LockData {
  SourceLocation AcquireLoc;

  LockData(SourceLocation Loc) : AcquireLoc(Loc) {}

  bool operator==(const LockData &other) const {
    return AcquireLoc == other.AcquireLoc;
  }

  bool operator!=(const LockData &other) const {
    return !(*this == other);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(AcquireLoc.getRawEncoding());
  }
};

/// A Lockset maps each LockID (defined above) to information about how it has
/// been locked.
typedef llvm::ImmutableMap<LockID, LockData> Lockset;

/// \brief We use this class to visit different types of expressions in
/// CFGBlocks, and build up the lockset.
/// An expression may cause us to add or remove locks from the lockset, or else
/// output error messages related to missing locks.
/// FIXME: In future, we may be able to not inherit from a visitor.
class BuildLockset : public StmtVisitor<BuildLockset> {
  Sema &S;
  Lockset LSet;
  Lockset::Factory &LocksetFactory;

  // Helper functions
  void removeLock(SourceLocation UnlockLoc, Expr *LockExp);
  void addLock(SourceLocation LockLoc, Expr *LockExp);
  const ValueDecl *getValueDecl(Expr *Exp);
  void checkAccess(Expr *Exp);
  void checkDereference(Expr *Exp);

public:
  BuildLockset(Sema &S, Lockset LS, Lockset::Factory &F)
    : StmtVisitor<BuildLockset>(), S(S), LSet(LS),
      LocksetFactory(F) {}

  Lockset getLockset() {
    return LSet;
  }

  void VisitUnaryOperator(UnaryOperator *UO);
  void VisitBinaryOperator(BinaryOperator *BO);
  void VisitCastExpr(CastExpr *CE);
  void VisitCXXMemberCallExpr(CXXMemberCallExpr *Exp);
};

/// \brief Add a new lock to the lockset, warning if the lock is already there.
/// \param LockLoc The source location of the acquire
/// \param LockExp The lock expression corresponding to the lock to be added
void BuildLockset::addLock(SourceLocation LockLoc, Expr *LockExp) {
  LockID Lock(LockExp);
  LockData NewLockData(LockLoc);

  if (LSet.contains(Lock))
    S.Diag(LockLoc, diag::warn_double_lock) << Lock.getName();

  LSet = LocksetFactory.add(LSet, Lock, NewLockData);
}

/// \brief Remove a lock from the lockset, warning if the lock is not there.
/// \param LockExp The lock expression corresponding to the lock to be removed
/// \param UnlockLoc The source location of the unlock (only used in error msg)
void BuildLockset::removeLock(SourceLocation UnlockLoc, Expr *LockExp) {
  LockID Lock(LockExp);

  Lockset NewLSet = LocksetFactory.remove(LSet, Lock);
  if(NewLSet == LSet)
    S.Diag(UnlockLoc, diag::warn_unlock_but_no_acquire) << Lock.getName();

  LSet = NewLSet;
}

/// \brief Gets the value decl pointer from DeclRefExprs or MemberExprs
const ValueDecl *BuildLockset::getValueDecl(Expr *Exp) {
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Exp))
    return DR->getDecl();

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(Exp))
    return ME->getMemberDecl();

  return 0;
}

/// \brief This method identifies variable dereferences and checks pt_guarded_by
/// and pt_guarded_var annotations. Note that we only check these annotations
/// at the time a pointer is dereferenced.
/// FIXME: We need to check for other types of pointer dereferences
/// (e.g. [], ->) and deal with them here.
/// \param Exp An expression that has been read or written.
void BuildLockset::checkDereference(Expr *Exp) {
  UnaryOperator *UO = dyn_cast<UnaryOperator>(Exp);
  if (!UO || UO->getOpcode() != clang::UO_Deref)
    return;
  Exp = UO->getSubExpr()->IgnoreParenCasts();

  const ValueDecl *D = getValueDecl(Exp);
  if(!D || !D->hasAttrs())
    return;

  if (D->getAttr<PtGuardedVarAttr>() && LSet.isEmpty())
    S.Diag(Exp->getExprLoc(), diag::warn_var_deref_requires_any_lock)
      << D->getName();

  const AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0, Size = ArgAttrs.size(); i < Size; ++i) {
    if (ArgAttrs[i]->getKind() != attr::PtGuardedBy)
      continue;
    PtGuardedByAttr *PGBAttr = cast<PtGuardedByAttr>(ArgAttrs[i]);
    LockID Lock(PGBAttr->getArg());
    if (!LSet.contains(Lock))
      S.Diag(Exp->getExprLoc(), diag::warn_var_deref_requires_lock)
        << D->getName() << Lock.getName();
  }
}

/// \brief Checks guarded_by and guarded_var attributes.
/// Whenever we identify an access (read or write) of a DeclRefExpr or
/// MemberExpr, we need to check whether there are any guarded_by or
/// guarded_var attributes, and make sure we hold the appropriate locks.
void BuildLockset::checkAccess(Expr *Exp) {
  const ValueDecl *D = getValueDecl(Exp);
  if(!D || !D->hasAttrs())
    return;

  if (D->getAttr<GuardedVarAttr>() && LSet.isEmpty())
    S.Diag(Exp->getExprLoc(), diag::warn_variable_requires_any_lock)
      << D->getName();

  const AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0, Size = ArgAttrs.size(); i < Size; ++i) {
    if (ArgAttrs[i]->getKind() != attr::GuardedBy)
      continue;
    GuardedByAttr *GBAttr = cast<GuardedByAttr>(ArgAttrs[i]);
    LockID Lock(GBAttr->getArg());
    if (!LSet.contains(Lock))
      S.Diag(Exp->getExprLoc(), diag::warn_variable_requires_lock)
        << D->getName() << Lock.getName();
  }
}

/// \brief For unary operations which read and write a variable, we need to
/// check whether we hold any required locks. Reads are checked in
/// VisitCastExpr.
void BuildLockset::VisitUnaryOperator(UnaryOperator *UO) {
  switch (UO->getOpcode()) {
    case clang::UO_PostDec:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PreInc: {
      Expr *SubExp = UO->getSubExpr()->IgnoreParenCasts();
      checkAccess(SubExp);
      checkDereference(SubExp);
      break;
    }
    default:
      break;
  }
}

/// For binary operations which assign to a variable (writes), we need to check
/// whether we hold any required locks.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitBinaryOperator(BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;
  Expr *LHSExp = BO->getLHS()->IgnoreParenCasts();
  checkAccess(LHSExp);
  checkDereference(LHSExp);
}

/// Whenever we do an LValue to Rvalue cast, we are reading a variable and
/// need to ensure we hold any required locks. 
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitCastExpr(CastExpr *CE) {
  if (CE->getCastKind() != CK_LValueToRValue)
    return;
  Expr *SubExp = CE->getSubExpr()->IgnoreParenCasts();
  checkAccess(SubExp);
  checkDereference(SubExp);
}


/// \brief When visiting CXXMemberCallExprs we need to examine the attributes on
/// the method that is being called and add, remove or check locks in the
/// lockset accordingly.
/// 
/// FIXME: For classes annotated with one of the guarded annotations, we need
/// to treat const method calls as reads and non-const method calls as writes,
/// and check that the appropriate locks are held. Non-const method calls with 
/// the same signature as const method calls can be also treated as reads.
void BuildLockset::VisitCXXMemberCallExpr(CXXMemberCallExpr *Exp) {
  NamedDecl *D = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());

  SourceLocation ExpLocation = Exp->getExprLoc();
  Expr *Parent = Exp->getImplicitObjectArgument();

  if(!D || !D->hasAttrs())
    return;

  AttrVec &ArgAttrs = D->getAttrs();
  for(unsigned i = 0; i < ArgAttrs.size(); ++i) {
    Attr *Attr = ArgAttrs[i];
    switch (Attr->getKind()) {
      // When we encounter an exclusive lock function, we need to add the lock
      // to our lockset.
      case attr::ExclusiveLockFunction: {
        ExclusiveLockFunctionAttr *ELFAttr =
          cast<ExclusiveLockFunctionAttr>(Attr);

        if (ELFAttr->args_size() == 0) {// The lock held is the "this" object.
          addLock(ExpLocation, Parent);
          break;
        }

        for (ExclusiveLockFunctionAttr::args_iterator I = ELFAttr->args_begin(),
             E = ELFAttr->args_end(); I != E; ++I)
          addLock(ExpLocation, *I);
        // FIXME: acquired_after/acquired_before annotations
        break;
      }

      // When we encounter an unlock function, we need to remove unlocked locks
      // from the lockset, and flag a warning if they are not there.
      case attr::UnlockFunction: {
        UnlockFunctionAttr *UFAttr = cast<UnlockFunctionAttr>(Attr);

        if (UFAttr->args_size() == 0) { // The lock held is the "this" object.
          removeLock(ExpLocation, Parent);
          break;
        }

        for (UnlockFunctionAttr::args_iterator I = UFAttr->args_begin(),
            E = UFAttr->args_end(); I != E; ++I)
          removeLock(ExpLocation, *I);
        break;
      }

      // Ignore other (non thread-safety) attributes
      default:
        break;
    }
  }
}

typedef std::pair<SourceLocation, PartialDiagnostic> DelayedDiag;
typedef llvm::SmallVector<DelayedDiag, 4> DiagList;

struct SortDiagBySourceLocation {
  Sema &S;

  SortDiagBySourceLocation(Sema &S) : S(S) {}

  bool operator()(const DelayedDiag &left, const DelayedDiag &right) {
    // Although this call will be slow, this is only called when outputting
    // multiple warnings.
    return S.getSourceManager().isBeforeInTranslationUnit(left.first,
                                                          right.first);
  }
};
} // end anonymous namespace

/// \brief Emit all buffered diagnostics in order of sourcelocation.
/// We need to output diagnostics produced while iterating through
/// the lockset in deterministic order, so this function orders diagnostics
/// and outputs them.
static void EmitDiagnostics(Sema &S, DiagList &D) {
  SortDiagBySourceLocation SortDiagBySL(S);
  sort(D.begin(), D.end(), SortDiagBySL);
  for (DiagList::iterator I = D.begin(), E = D.end(); I != E; ++I)
    S.Diag(I->first, I->second);
}

/// \brief Compute the intersection of two locksets and issue warnings for any
/// locks in the symmetric difference.
///
/// This function is used at a merge point in the CFG when comparing the lockset
/// of each branch being merged. For example, given the following sequence:
/// A; if () then B; else C; D; we need to check that the lockset after B and C
/// are the same. In the event of a difference, we use the intersection of these
/// two locksets at the start of D.
static Lockset intersectAndWarn(Sema &S, Lockset LSet1, Lockset LSet2,
                                Lockset::Factory &Fact) {
  Lockset Intersection = LSet1;
  DiagList Warnings;

  for (Lockset::iterator I = LSet2.begin(), E = LSet2.end(); I != E; ++I) {
    if (!LSet1.contains(I.getKey())) {
      const LockID &MissingLock = I.getKey();
      const LockData &MissingLockData = I.getData();
      PartialDiagnostic Warning =
        S.PDiag(diag::warn_lock_not_released_in_scope) << MissingLock.getName();
      Warnings.push_back(DelayedDiag(MissingLockData.AcquireLoc, Warning));
    }
  }

  for (Lockset::iterator I = LSet1.begin(), E = LSet1.end(); I != E; ++I) {
    if (!LSet2.contains(I.getKey())) {
      const LockID &MissingLock = I.getKey();
      const LockData &MissingLockData = I.getData();
      PartialDiagnostic Warning =
        S.PDiag(diag::warn_lock_not_released_in_scope) << MissingLock.getName();
      Warnings.push_back(DelayedDiag(MissingLockData.AcquireLoc, Warning));
      Intersection = Fact.remove(Intersection, MissingLock);
    }
  }

  EmitDiagnostics(S, Warnings);
  return Intersection;
}

/// \brief Returns the location of the first Stmt in a Block.
static SourceLocation getFirstStmtLocation(CFGBlock *Block) {
  SourceLocation Loc;
  for (CFGBlock::const_iterator BI = Block->begin(), BE = Block->end();
       BI != BE; ++BI) {
    if (const CFGStmt *CfgStmt = dyn_cast<CFGStmt>(&(*BI))) {
      Loc = CfgStmt->getStmt()->getLocStart();
      if (Loc.isValid()) return Loc;
    }
  }
  if (Stmt *S = Block->getTerminator().getStmt()) {
    Loc = S->getLocStart();
    if (Loc.isValid()) return Loc;
  }
  return Loc;
}

/// \brief Warn about different locksets along backedges of loops.
/// This function is called when we encounter a back edge. At that point,
/// we need to verify that the lockset before taking the backedge is the
/// same as the lockset before entering the loop.
///
/// \param LoopEntrySet Locks held before starting the loop
/// \param LoopReentrySet Locks held in the last CFG block of the loop
static void warnBackEdgeUnequalLocksets(Sema &S, const Lockset LoopReentrySet,
                                        const Lockset LoopEntrySet,
                                        SourceLocation FirstLocInLoop) {
  assert(FirstLocInLoop.isValid());
  DiagList Warnings;

  // Warn for locks held at the start of the loop, but not the end.
  for (Lockset::iterator I = LoopEntrySet.begin(), E = LoopEntrySet.end();
       I != E; ++I) {
    if (!LoopReentrySet.contains(I.getKey())) {
      const LockID &MissingLock = I.getKey();
      // We report this error at the location of the first statement in a loop
      PartialDiagnostic Warning =
        S.PDiag(diag::warn_expecting_lock_held_on_loop)
          << MissingLock.getName();
      Warnings.push_back(DelayedDiag(FirstLocInLoop, Warning));
    }
  }

  // Warn for locks held at the end of the loop, but not at the start.
  for (Lockset::iterator I = LoopReentrySet.begin(), E = LoopReentrySet.end();
       I != E; ++I) {
    if (!LoopEntrySet.contains(I.getKey())) {
      const LockID &MissingLock = I.getKey();
      const LockData &MissingLockData = I.getData();
      PartialDiagnostic Warning =
        S.PDiag(diag::warn_lock_not_released_in_scope) << MissingLock.getName();
      Warnings.push_back(DelayedDiag(MissingLockData.AcquireLoc, Warning));
    }
  }

  EmitDiagnostics(S, Warnings);
}

/// \brief Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of locks that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
static void checkThreadSafety(Sema &S, AnalysisContext &AC) {
  CFG *CFGraph = AC.getCFG();
  if (!CFGraph) return;

  Lockset::Factory LocksetFactory;

  // FIXME: Swith to SmallVector? Otherwise improve performance impact?
  std::vector<Lockset> EntryLocksets(CFGraph->getNumBlockIDs(),
                                     LocksetFactory.getEmptyMap());
  std::vector<Lockset> ExitLocksets(CFGraph->getNumBlockIDs(),
                                    LocksetFactory.getEmptyMap());

  // We need to explore the CFG via a "topological" ordering.
  // That way, we will be guaranteed to have information about required
  // predecessor locksets when exploring a new block.
  TopologicallySortedCFG SortedGraph(CFGraph);
  CFGBlockSet VisitedBlocks(CFGraph);

  for (TopologicallySortedCFG::iterator I = SortedGraph.begin(),
       E = SortedGraph.end(); I!= E; ++I) {
    const CFGBlock *CurrBlock = *I;
    int CurrBlockID = CurrBlock->getBlockID();

    VisitedBlocks.insert(CurrBlock);

    // Use the default initial lockset in case there are no predecessors.
    Lockset &Entryset = EntryLocksets[CurrBlockID];
    Lockset &Exitset = ExitLocksets[CurrBlockID];

    // Iterate through the predecessor blocks and warn if the lockset for all
    // predecessors is not the same. We take the entry lockset of the current
    // block to be the intersection of all previous locksets.
    // FIXME: By keeping the intersection, we may output more errors in future
    // for a lock which is not in the intersection, but was in the union. We
    // may want to also keep the union in future. As an example, let's say
    // the intersection contains Lock L, and the union contains L and M.
    // Later we unlock M. At this point, we would output an error because we
    // never locked M; although the real error is probably that we forgot to
    // lock M on all code paths. Conversely, let's say that later we lock M.
    // In this case, we should compare against the intersection instead of the
    // union because the real error is probably that we forgot to unlock M on
    // all code paths.
    bool LocksetInitialized = false;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {

      // if *PI -> CurrBlock is a back edge
      if (*PI == 0 || !VisitedBlocks.alreadySet(*PI))
        continue;

      int PrevBlockID = (*PI)->getBlockID();
      if (!LocksetInitialized) {
        Entryset = ExitLocksets[PrevBlockID];
        LocksetInitialized = true;
      } else {
        Entryset = intersectAndWarn(S, Entryset, ExitLocksets[PrevBlockID],
                                LocksetFactory);
      }
    }

    BuildLockset LocksetBuilder(S, Entryset, LocksetFactory);
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      if (const CFGStmt *CfgStmt = dyn_cast<CFGStmt>(&*BI))
        LocksetBuilder.Visit(const_cast<Stmt*>(CfgStmt->getStmt()));
    }
    Exitset = LocksetBuilder.getLockset();

    // For every back edge from CurrBlock (the end of the loop) to another block
    // (FirstLoopBlock) we need to check that the Lockset of Block is equal to
    // the one held at the beginning of FirstLoopBlock. We can look up the
    // Lockset held at the beginning of FirstLoopBlock in the EntryLockSets map.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {

      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == 0 || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      SourceLocation FirstLoopLocation = getFirstStmtLocation(FirstLoopBlock);

      assert(FirstLoopLocation.isValid());
      // Fail gracefully in release code.
      if (!FirstLoopLocation.isValid())
        continue;

      Lockset PreLoop = EntryLocksets[FirstLoopBlock->getBlockID()];
      Lockset LoopEnd = ExitLocksets[CurrBlockID];
      warnBackEdgeUnequalLocksets(S, LoopEnd, PreLoop, FirstLoopLocation);
    }
  }

  Lockset FinalLockset = ExitLocksets[CFGraph->getExit().getBlockID()];
  if (!FinalLockset.isEmpty()) {
    DiagList Warnings;
    for (Lockset::iterator I=FinalLockset.begin(), E=FinalLockset.end();
         I != E; ++I) {
      const LockID &MissingLock = I.getKey();
      const LockData &MissingLockData = I.getData();

      std::string FunName = "<unknown>";
      if (const NamedDecl *ContextDecl = dyn_cast<NamedDecl>(AC.getDecl())) {
        FunName = ContextDecl->getDeclName().getAsString();
      }

      PartialDiagnostic Warning =
        S.PDiag(diag::warn_locks_not_released)
          << MissingLock.getName() << FunName;
      Warnings.push_back(DelayedDiag(MissingLockData.AcquireLoc, Warning));
    }
    EmitDiagnostics(S, Warnings);
  }
}


//===----------------------------------------------------------------------===//
// AnalysisBasedWarnings - Worker object used by Sema to execute analysis-based
//  warnings on a function, method, or block.
//===----------------------------------------------------------------------===//

clang::sema::AnalysisBasedWarnings::Policy::Policy() {
  enableCheckFallThrough = 1;
  enableCheckUnreachable = 0;
  enableThreadSafetyAnalysis = 0;
}

clang::sema::AnalysisBasedWarnings::AnalysisBasedWarnings(Sema &s)
  : S(s),
    NumFunctionsAnalyzed(0),
    NumFunctionsWithBadCFGs(0),
    NumCFGBlocks(0),
    MaxCFGBlocksPerFunction(0),
    NumUninitAnalysisFunctions(0),
    NumUninitAnalysisVariables(0),
    MaxUninitAnalysisVariablesPerFunction(0),
    NumUninitAnalysisBlockVisits(0),
    MaxUninitAnalysisBlockVisitsPerFunction(0) {
  Diagnostic &D = S.getDiagnostics();
  DefaultPolicy.enableCheckUnreachable = (unsigned)
    (D.getDiagnosticLevel(diag::warn_unreachable, SourceLocation()) !=
        Diagnostic::Ignored);
  DefaultPolicy.enableThreadSafetyAnalysis = (unsigned)
    (D.getDiagnosticLevel(diag::warn_double_lock, SourceLocation()) !=
     Diagnostic::Ignored);

}

static void flushDiagnostics(Sema &S, sema::FunctionScopeInfo *fscope) {
  for (SmallVectorImpl<sema::PossiblyUnreachableDiag>::iterator
       i = fscope->PossiblyUnreachableDiags.begin(),
       e = fscope->PossiblyUnreachableDiags.end();
       i != e; ++i) {
    const sema::PossiblyUnreachableDiag &D = *i;
    S.Diag(D.Loc, D.PD);
  }
}

void clang::sema::
AnalysisBasedWarnings::IssueWarnings(sema::AnalysisBasedWarnings::Policy P,
                                     sema::FunctionScopeInfo *fscope,
                                     const Decl *D, const BlockExpr *blkExpr) {

  // We avoid doing analysis-based warnings when there are errors for
  // two reasons:
  // (1) The CFGs often can't be constructed (if the body is invalid), so
  //     don't bother trying.
  // (2) The code already has problems; running the analysis just takes more
  //     time.
  Diagnostic &Diags = S.getDiagnostics();

  // Do not do any analysis for declarations in system headers if we are
  // going to just ignore them.
  if (Diags.getSuppressSystemWarnings() &&
      S.SourceMgr.isInSystemHeader(D->getLocation()))
    return;

  // For code in dependent contexts, we'll do this at instantiation time.
  if (cast<DeclContext>(D)->isDependentContext())
    return;

  if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred()) {
    // Flush out any possibly unreachable diagnostics.
    flushDiagnostics(S, fscope);
    return;
  }
  
  const Stmt *Body = D->getBody();
  assert(Body);

  AnalysisContext AC(D, 0);

  // Don't generate EH edges for CallExprs as we'd like to avoid the n^2
  // explosion for destrutors that can result and the compile time hit.
  AC.getCFGBuildOptions().PruneTriviallyFalseEdges = true;
  AC.getCFGBuildOptions().AddEHEdges = false;
  AC.getCFGBuildOptions().AddInitializers = true;
  AC.getCFGBuildOptions().AddImplicitDtors = true;
  
  // Force that certain expressions appear as CFGElements in the CFG.  This
  // is used to speed up various analyses.
  // FIXME: This isn't the right factoring.  This is here for initial
  // prototyping, but we need a way for analyses to say what expressions they
  // expect to always be CFGElements and then fill in the BuildOptions
  // appropriately.  This is essentially a layering violation.
  if (P.enableCheckUnreachable) {
    // Unreachable code analysis requires a linearized CFG.
    AC.getCFGBuildOptions().setAllAlwaysAdd();
  }
  else {
    AC.getCFGBuildOptions()
      .setAlwaysAdd(Stmt::BinaryOperatorClass)
      .setAlwaysAdd(Stmt::BlockExprClass)
      .setAlwaysAdd(Stmt::CStyleCastExprClass)
      .setAlwaysAdd(Stmt::DeclRefExprClass)
      .setAlwaysAdd(Stmt::ImplicitCastExprClass)
      .setAlwaysAdd(Stmt::UnaryOperatorClass);
  }

  // Construct the analysis context with the specified CFG build options.
  
  // Emit delayed diagnostics.
  if (!fscope->PossiblyUnreachableDiags.empty()) {
    bool analyzed = false;

    // Register the expressions with the CFGBuilder.
    for (SmallVectorImpl<sema::PossiblyUnreachableDiag>::iterator
         i = fscope->PossiblyUnreachableDiags.begin(),
         e = fscope->PossiblyUnreachableDiags.end();
         i != e; ++i) {
      if (const Stmt *stmt = i->stmt)
        AC.registerForcedBlockExpression(stmt);
    }

    if (AC.getCFG()) {
      analyzed = true;
      for (SmallVectorImpl<sema::PossiblyUnreachableDiag>::iterator
            i = fscope->PossiblyUnreachableDiags.begin(),
            e = fscope->PossiblyUnreachableDiags.end();
            i != e; ++i)
      {
        const sema::PossiblyUnreachableDiag &D = *i;
        bool processed = false;
        if (const Stmt *stmt = i->stmt) {
          const CFGBlock *block = AC.getBlockForRegisteredExpression(stmt);
          assert(block);
          if (CFGReverseBlockReachabilityAnalysis *cra = AC.getCFGReachablityAnalysis()) {
            // Can this block be reached from the entrance?
            if (cra->isReachable(&AC.getCFG()->getEntry(), block))
              S.Diag(D.Loc, D.PD);
            processed = true;
          }
        }
        if (!processed) {
          // Emit the warning anyway if we cannot map to a basic block.
          S.Diag(D.Loc, D.PD);
        }
      }
    }

    if (!analyzed)
      flushDiagnostics(S, fscope);
  }
  
  
  // Warning: check missing 'return'
  if (P.enableCheckFallThrough) {
    const CheckFallThroughDiagnostics &CD =
      (isa<BlockDecl>(D) ? CheckFallThroughDiagnostics::MakeForBlock()
                         : CheckFallThroughDiagnostics::MakeForFunction(D));
    CheckFallThroughForBody(S, D, Body, blkExpr, CD, AC);
  }

  // Warning: check for unreachable code
  if (P.enableCheckUnreachable)
    CheckUnreachable(S, AC);
  
  // Check for thread safety violations
  if (P.enableThreadSafetyAnalysis)
    checkThreadSafety(S, AC);

  if (Diags.getDiagnosticLevel(diag::warn_uninit_var, D->getLocStart())
      != Diagnostic::Ignored ||
      Diags.getDiagnosticLevel(diag::warn_maybe_uninit_var, D->getLocStart())
      != Diagnostic::Ignored) {
    if (CFG *cfg = AC.getCFG()) {
      UninitValsDiagReporter reporter(S);
      UninitVariablesAnalysisStats stats;
      std::memset(&stats, 0, sizeof(UninitVariablesAnalysisStats));
      runUninitializedVariablesAnalysis(*cast<DeclContext>(D), *cfg, AC,
                                        reporter, stats);

      if (S.CollectStats && stats.NumVariablesAnalyzed > 0) {
        ++NumUninitAnalysisFunctions;
        NumUninitAnalysisVariables += stats.NumVariablesAnalyzed;
        NumUninitAnalysisBlockVisits += stats.NumBlockVisits;
        MaxUninitAnalysisVariablesPerFunction =
            std::max(MaxUninitAnalysisVariablesPerFunction,
                     stats.NumVariablesAnalyzed);
        MaxUninitAnalysisBlockVisitsPerFunction =
            std::max(MaxUninitAnalysisBlockVisitsPerFunction,
                     stats.NumBlockVisits);
      }
    }
  }

  // Collect statistics about the CFG if it was built.
  if (S.CollectStats && AC.isCFGBuilt()) {
    ++NumFunctionsAnalyzed;
    if (CFG *cfg = AC.getCFG()) {
      // If we successfully built a CFG for this context, record some more
      // detail information about it.
      NumCFGBlocks += cfg->getNumBlockIDs();
      MaxCFGBlocksPerFunction = std::max(MaxCFGBlocksPerFunction,
                                         cfg->getNumBlockIDs());
    } else {
      ++NumFunctionsWithBadCFGs;
    }
  }
}

void clang::sema::AnalysisBasedWarnings::PrintStats() const {
  llvm::errs() << "\n*** Analysis Based Warnings Stats:\n";

  unsigned NumCFGsBuilt = NumFunctionsAnalyzed - NumFunctionsWithBadCFGs;
  unsigned AvgCFGBlocksPerFunction =
      !NumCFGsBuilt ? 0 : NumCFGBlocks/NumCFGsBuilt;
  llvm::errs() << NumFunctionsAnalyzed << " functions analyzed ("
               << NumFunctionsWithBadCFGs << " w/o CFGs).\n"
               << "  " << NumCFGBlocks << " CFG blocks built.\n"
               << "  " << AvgCFGBlocksPerFunction
               << " average CFG blocks per function.\n"
               << "  " << MaxCFGBlocksPerFunction
               << " max CFG blocks per function.\n";

  unsigned AvgUninitVariablesPerFunction = !NumUninitAnalysisFunctions ? 0
      : NumUninitAnalysisVariables/NumUninitAnalysisFunctions;
  unsigned AvgUninitBlockVisitsPerFunction = !NumUninitAnalysisFunctions ? 0
      : NumUninitAnalysisBlockVisits/NumUninitAnalysisFunctions;
  llvm::errs() << NumUninitAnalysisFunctions
               << " functions analyzed for uninitialiazed variables\n"
               << "  " << NumUninitAnalysisVariables << " variables analyzed.\n"
               << "  " << AvgUninitVariablesPerFunction
               << " average variables per function.\n"
               << "  " << MaxUninitAnalysisVariablesPerFunction
               << " max variables per function.\n"
               << "  " << NumUninitAnalysisBlockVisits << " block visits.\n"
               << "  " << AvgUninitBlockVisitsPerFunction
               << " average block visits per function.\n"
               << "  " << MaxUninitAnalysisBlockVisitsPerFunction
               << " max block visits per function.\n";
}
