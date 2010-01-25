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
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;

SVal Environment::GetSVal(const Stmt *E, ValueManager& ValMgr) const {

  for (;;) {

    switch (E->getStmtClass()) {

      case Stmt::AddrLabelExprClass:
        return ValMgr.makeLoc(cast<AddrLabelExpr>(E));

        // ParenExprs are no-ops.

      case Stmt::ParenExprClass:
        E = cast<ParenExpr>(E)->getSubExpr();
        continue;

      case Stmt::CharacterLiteralClass: {
        const CharacterLiteral* C = cast<CharacterLiteral>(E);
        return ValMgr.makeIntVal(C->getValue(), C->getType());
      }

      case Stmt::IntegerLiteralClass: {
        // In C++, this expression may have been bound to a temporary object.
        SVal const *X = ExprBindings.lookup(E);
        if (X)
          return *X;
        else
          return ValMgr.makeIntVal(cast<IntegerLiteral>(E));
      }

      // Casts where the source and target type are the same
      // are no-ops.  We blast through these to get the descendant
      // subexpression that has a value.

      case Stmt::ImplicitCastExprClass:
      case Stmt::CStyleCastExprClass: {
        const CastExpr* C = cast<CastExpr>(E);
        QualType CT = C->getType();

        if (CT->isVoidType())
          return UnknownVal();

        break;
      }

        // Handle all other Stmt* using a lookup.

      default:
        break;
    };

    break;
  }

  return LookupExpr(E);
}

Environment EnvironmentManager::BindExpr(Environment Env, const Stmt *S,
                                         SVal V, bool Invalidate) {
  assert(S);

  if (V.isUnknown()) {
    if (Invalidate)
      return Environment(F.Remove(Env.ExprBindings, S), Env.ACtx);
    else
      return Env;
  }

  return Environment(F.Add(Env.ExprBindings, S, V), Env.ACtx);
}

namespace {
class MarkLiveCallback : public SymbolVisitor {
  SymbolReaper &SymReaper;
public:
  MarkLiveCallback(SymbolReaper &symreaper) : SymReaper(symreaper) {}
  bool VisitSymbol(SymbolRef sym) { SymReaper.markLive(sym); return true; }
};
} // end anonymous namespace

// RemoveDeadBindings:
//  - Remove subexpression bindings.
//  - Remove dead block expression bindings.
//  - Keep live block expression bindings:
//   - Mark their reachable symbols live in SymbolReaper,
//     see ScanReachableSymbols.
//   - Mark the region in DRoots if the binding is a loc::MemRegionVal.

Environment
EnvironmentManager::RemoveDeadBindings(Environment Env, const Stmt *S,
                                       SymbolReaper &SymReaper,
                                       const GRState *ST,
                              llvm::SmallVectorImpl<const MemRegion*> &DRoots) {

  CFG &C = *Env.getAnalysisContext().getCFG();

  // We construct a new Environment object entirely, as this is cheaper than
  // individually removing all the subexpression bindings (which will greatly
  // outnumber block-level expression bindings).
  Environment NewEnv = getInitialEnvironment(&Env.getAnalysisContext());

  // Iterate over the block-expr bindings.
  for (Environment::iterator I = Env.begin(), E = Env.end();
       I != E; ++I) {

    const Stmt *BlkExpr = I.getKey();

    // Not a block-level expression?
    if (!C.isBlkExpr(BlkExpr))
      continue;

    const SVal &X = I.getData();

    if (SymReaper.isLive(S, BlkExpr)) {
      // Copy the binding to the new map.
      NewEnv.ExprBindings = F.Add(NewEnv.ExprBindings, BlkExpr, X);

      // If the block expr's value is a memory region, then mark that region.
      if (isa<loc::MemRegionVal>(X)) {
        const MemRegion* R = cast<loc::MemRegionVal>(X).getRegion();
        DRoots.push_back(R);
        // Mark the super region of the RX as live.
        // e.g.: int x; char *y = (char*) &x; if (*y) ...
        // 'y' => element region. 'x' is its super region.
        // We only add one level super region for now.

        // FIXME: maybe multiple level of super regions should be added.
        if (const SubRegion *SR = dyn_cast<SubRegion>(R))
          DRoots.push_back(SR->getSuperRegion());
      }

      // Mark all symbols in the block expr's value live.
      MarkLiveCallback cb(SymReaper);
      ST->scanReachableSymbols(X, cb);
      continue;
    }

    // Otherwise the expression is dead with a couple exceptions.
    // Do not misclean LogicalExpr or ConditionalOperator.  It is dead at the
    // beginning of itself, but we need its UndefinedVal to determine its
    // SVal.
    if (X.isUndef() && cast<UndefinedVal>(X).getData())
      NewEnv.ExprBindings = F.Add(NewEnv.ExprBindings, BlkExpr, X);
  }

  return NewEnv;
}
