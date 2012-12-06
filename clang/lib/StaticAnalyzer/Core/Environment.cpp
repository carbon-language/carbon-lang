//== Environment.cpp - Map from Stmt* to Locations/Values -------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the Environment and EnvironmentManager classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

static const Expr *ignoreTransparentExprs(const Expr *E) {
  E = E->IgnoreParens();

  switch (E->getStmtClass()) {
  case Stmt::OpaqueValueExprClass:
    E = cast<OpaqueValueExpr>(E)->getSourceExpr();
    break;
  case Stmt::ExprWithCleanupsClass:
    E = cast<ExprWithCleanups>(E)->getSubExpr();
    break;
  case Stmt::CXXBindTemporaryExprClass:
    E = cast<CXXBindTemporaryExpr>(E)->getSubExpr();
    break;
  case Stmt::SubstNonTypeTemplateParmExprClass:
    E = cast<SubstNonTypeTemplateParmExpr>(E)->getReplacement();
    break;
  case Stmt::CXXDefaultArgExprClass:
    E = cast<CXXDefaultArgExpr>(E)->getExpr();
    break;
  default:
    // This is the base case: we can't look through more than we already have.
    return E;
  }

  return ignoreTransparentExprs(E);
}

static const Stmt *ignoreTransparentExprs(const Stmt *S) {
  if (const Expr *E = dyn_cast<Expr>(S))
    return ignoreTransparentExprs(E);
  return S;
}

EnvironmentEntry::EnvironmentEntry(const Stmt *S, const LocationContext *L)
  : std::pair<const Stmt *,
              const StackFrameContext *>(ignoreTransparentExprs(S),
                                         L ? L->getCurrentStackFrame() : 0) {}

SVal Environment::lookupExpr(const EnvironmentEntry &E) const {
  const SVal* X = ExprBindings.lookup(E);
  if (X) {
    SVal V = *X;
    return V;
  }
  return UnknownVal();
}

SVal Environment::getSVal(const EnvironmentEntry &Entry,
                          SValBuilder& svalBuilder) const {
  const Stmt *S = Entry.getStmt();
  const LocationContext *LCtx = Entry.getLocationContext();

  switch (S->getStmtClass()) {
  case Stmt::CXXBindTemporaryExprClass:
  case Stmt::CXXDefaultArgExprClass:
  case Stmt::ExprWithCleanupsClass:
  case Stmt::GenericSelectionExprClass:
  case Stmt::OpaqueValueExprClass:
  case Stmt::ParenExprClass:
  case Stmt::SubstNonTypeTemplateParmExprClass:
    llvm_unreachable("Should have been handled by ignoreTransparentExprs");

  case Stmt::AddrLabelExprClass:
    return svalBuilder.makeLoc(cast<AddrLabelExpr>(S));

  case Stmt::CharacterLiteralClass: {
    const CharacterLiteral *C = cast<CharacterLiteral>(S);
    return svalBuilder.makeIntVal(C->getValue(), C->getType());
  }

  case Stmt::CXXBoolLiteralExprClass:
    return svalBuilder.makeBoolVal(cast<CXXBoolLiteralExpr>(S));

  case Stmt::CXXScalarValueInitExprClass:
  case Stmt::ImplicitValueInitExprClass: {
    QualType Ty = cast<Expr>(S)->getType();
    return svalBuilder.makeZeroVal(Ty);
  }

  case Stmt::IntegerLiteralClass:
    return svalBuilder.makeIntVal(cast<IntegerLiteral>(S));

  case Stmt::ObjCBoolLiteralExprClass:
    return svalBuilder.makeBoolVal(cast<ObjCBoolLiteralExpr>(S));

  // For special C0xx nullptr case, make a null pointer SVal.
  case Stmt::CXXNullPtrLiteralExprClass:
    return svalBuilder.makeNull();

  case Stmt::ObjCStringLiteralClass: {
    MemRegionManager &MRMgr = svalBuilder.getRegionManager();
    const ObjCStringLiteral *SL = cast<ObjCStringLiteral>(S);
    return svalBuilder.makeLoc(MRMgr.getObjCStringRegion(SL));
  }

  case Stmt::StringLiteralClass: {
    MemRegionManager &MRMgr = svalBuilder.getRegionManager();
    const StringLiteral *SL = cast<StringLiteral>(S);
    return svalBuilder.makeLoc(MRMgr.getStringRegion(SL));
  }

  case Stmt::ReturnStmtClass: {
    const ReturnStmt *RS = cast<ReturnStmt>(S);
    if (const Expr *RE = RS->getRetValue())
      return getSVal(EnvironmentEntry(RE, LCtx), svalBuilder);
    return UndefinedVal();        
  }
    
  // Handle all other Stmt* using a lookup.
  default:
    break;
  }
  
  return lookupExpr(EnvironmentEntry(S, LCtx));
}

Environment EnvironmentManager::bindExpr(Environment Env,
                                         const EnvironmentEntry &E,
                                         SVal V,
                                         bool Invalidate) {
  if (V.isUnknown()) {
    if (Invalidate)
      return Environment(F.remove(Env.ExprBindings, E));
    else
      return Env;
  }
  return Environment(F.add(Env.ExprBindings, E, V));
}

namespace {
class MarkLiveCallback : public SymbolVisitor {
  SymbolReaper &SymReaper;
public:
  MarkLiveCallback(SymbolReaper &symreaper) : SymReaper(symreaper) {}
  bool VisitSymbol(SymbolRef sym) {
    SymReaper.markLive(sym);
    return true;
  }
  bool VisitMemRegion(const MemRegion *R) {
    SymReaper.markLive(R);
    return true;
  }
};
} // end anonymous namespace

// removeDeadBindings:
//  - Remove subexpression bindings.
//  - Remove dead block expression bindings.
//  - Keep live block expression bindings:
//   - Mark their reachable symbols live in SymbolReaper,
//     see ScanReachableSymbols.
//   - Mark the region in DRoots if the binding is a loc::MemRegionVal.
Environment
EnvironmentManager::removeDeadBindings(Environment Env,
                                       SymbolReaper &SymReaper,
                                       ProgramStateRef ST) {

  // We construct a new Environment object entirely, as this is cheaper than
  // individually removing all the subexpression bindings (which will greatly
  // outnumber block-level expression bindings).
  Environment NewEnv = getInitialEnvironment();

  MarkLiveCallback CB(SymReaper);
  ScanReachableSymbols RSScaner(ST, CB);

  llvm::ImmutableMapRef<EnvironmentEntry,SVal>
    EBMapRef(NewEnv.ExprBindings.getRootWithoutRetain(),
             F.getTreeFactory());

  // Iterate over the block-expr bindings.
  for (Environment::iterator I = Env.begin(), E = Env.end();
       I != E; ++I) {

    const EnvironmentEntry &BlkExpr = I.getKey();
    const SVal &X = I.getData();

    if (SymReaper.isLive(BlkExpr.getStmt(), BlkExpr.getLocationContext())) {
      // Copy the binding to the new map.
      EBMapRef = EBMapRef.add(BlkExpr, X);

      // If the block expr's value is a memory region, then mark that region.
      if (isa<loc::MemRegionVal>(X)) {
        const MemRegion *R = cast<loc::MemRegionVal>(X).getRegion();
        SymReaper.markLive(R);
      }

      // Mark all symbols in the block expr's value live.
      RSScaner.scan(X);
      continue;
    } else {
      SymExpr::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end();
      for (; SI != SE; ++SI)
        SymReaper.maybeDead(*SI);
    }
  }

  NewEnv.ExprBindings = EBMapRef.asImmutableMap();
  return NewEnv;
}

void Environment::print(raw_ostream &Out, const char *NL,
                        const char *Sep) const {
  bool isFirst = true;

  for (Environment::iterator I = begin(), E = end(); I != E; ++I) {
    const EnvironmentEntry &En = I.getKey();
    
    if (isFirst) {
      Out << NL << NL
          << "Expressions:"
          << NL;      
      isFirst = false;
    } else {
      Out << NL;
    }
    
    const Stmt *S = En.getStmt();
    
    Out << " (" << (const void*) En.getLocationContext() << ','
      << (const void*) S << ") ";
    LangOptions LO; // FIXME.
    S->printPretty(Out, 0, PrintingPolicy(LO));
    Out << " : " << I.getData();
  }
}
