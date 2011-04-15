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

#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRState.h"

using namespace clang;
using namespace ento;

SVal Environment::lookupExpr(const Stmt* E) const {
  const SVal* X = ExprBindings.lookup(E);
  if (X) {
    SVal V = *X;
    return V;
  }
  return UnknownVal();
}

SVal Environment::getSVal(const Stmt *E, SValBuilder& svalBuilder) const {
  for (;;) {
    switch (E->getStmtClass()) {
      case Stmt::AddrLabelExprClass:
        return svalBuilder.makeLoc(cast<AddrLabelExpr>(E));
      case Stmt::OpaqueValueExprClass: {
        const OpaqueValueExpr *ope = cast<OpaqueValueExpr>(E);
        E = ope->getSourceExpr();
        continue;        
      }        
      case Stmt::ParenExprClass:
        // ParenExprs are no-ops.
        E = cast<ParenExpr>(E)->getSubExpr();
        continue;
      case Stmt::GenericSelectionExprClass:
        // GenericSelectionExprs are no-ops.
        E = cast<GenericSelectionExpr>(E)->getResultExpr();
        continue;
      case Stmt::CharacterLiteralClass: {
        const CharacterLiteral* C = cast<CharacterLiteral>(E);
        return svalBuilder.makeIntVal(C->getValue(), C->getType());
      }
      case Stmt::CXXBoolLiteralExprClass: {
        const SVal *X = ExprBindings.lookup(E);
        if (X) 
          return *X;
        else 
          return svalBuilder.makeBoolVal(cast<CXXBoolLiteralExpr>(E));
      }
      case Stmt::IntegerLiteralClass: {
        // In C++, this expression may have been bound to a temporary object.
        SVal const *X = ExprBindings.lookup(E);
        if (X)
          return *X;
        else
          return svalBuilder.makeIntVal(cast<IntegerLiteral>(E));
      }
      case Stmt::ImplicitCastExprClass:
      case Stmt::CXXFunctionalCastExprClass:
      case Stmt::CStyleCastExprClass: {
        // We blast through no-op casts to get the descendant
        // subexpression that has a value.
        const CastExpr* C = cast<CastExpr>(E);
        QualType CT = C->getType();
        if (CT->isVoidType())
          return UnknownVal();
        if (C->getCastKind() == CK_NoOp) {
          E = C->getSubExpr();
          continue;
        }
        break;
      }
      case Stmt::ExprWithCleanupsClass:
        E = cast<ExprWithCleanups>(E)->getSubExpr();
        continue;
      case Stmt::CXXBindTemporaryExprClass:
        E = cast<CXXBindTemporaryExpr>(E)->getSubExpr();
        continue;
      // Handle all other Stmt* using a lookup.
      default:
        break;
    };
    break;
  }
  return lookupExpr(E);
}

Environment EnvironmentManager::bindExpr(Environment Env, const Stmt *S,
                                         SVal V, bool Invalidate) {
  assert(S);

  if (V.isUnknown()) {
    if (Invalidate)
      return Environment(F.remove(Env.ExprBindings, S));
    else
      return Env;
  }

  return Environment(F.add(Env.ExprBindings, S, V));
}

static inline const Stmt *MakeLocation(const Stmt *S) {
  return (const Stmt*) (((uintptr_t) S) | 0x1);
}

Environment EnvironmentManager::bindExprAndLocation(Environment Env,
                                                    const Stmt *S,
                                                    SVal location, SVal V) {
  return Environment(F.add(F.add(Env.ExprBindings, MakeLocation(S), location),
                           S, V));
}

namespace {
class MarkLiveCallback : public SymbolVisitor {
  SymbolReaper &SymReaper;
public:
  MarkLiveCallback(SymbolReaper &symreaper) : SymReaper(symreaper) {}
  bool VisitSymbol(SymbolRef sym) { SymReaper.markLive(sym); return true; }
};
} // end anonymous namespace

static bool isBlockExprInCallers(const Stmt *E, const LocationContext *LC) {
  const LocationContext *ParentLC = LC->getParent();
  while (ParentLC) {
    CFG &C = *ParentLC->getCFG();
    if (C.isBlkExpr(E))
      return true;
    ParentLC = ParentLC->getParent();
  }

  return false;
}

// In addition to mapping from Stmt * - > SVals in the Environment, we also
// maintain a mapping from Stmt * -> SVals (locations) that were used during
// a load and store.
static inline bool IsLocation(const Stmt *S) {
  return (bool) (((uintptr_t) S) & 0x1);
}

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
                                       const GRState *ST,
                              llvm::SmallVectorImpl<const MemRegion*> &DRoots) {

  CFG &C = *SymReaper.getLocationContext()->getCFG();

  // We construct a new Environment object entirely, as this is cheaper than
  // individually removing all the subexpression bindings (which will greatly
  // outnumber block-level expression bindings).
  Environment NewEnv = getInitialEnvironment();
  
  llvm::SmallVector<std::pair<const Stmt*, SVal>, 10> deferredLocations;

  // Iterate over the block-expr bindings.
  for (Environment::iterator I = Env.begin(), E = Env.end();
       I != E; ++I) {

    const Stmt *BlkExpr = I.getKey();
    
    // For recorded locations (used when evaluating loads and stores), we
    // consider them live only when their associated normal expression is
    // also live.
    // NOTE: This assumes that loads/stores that evaluated to UnknownVal
    // still have an entry in the map.
    if (IsLocation(BlkExpr)) {
      deferredLocations.push_back(std::make_pair(BlkExpr, I.getData()));
      continue;
    }
    
    const SVal &X = I.getData();

    // Block-level expressions in callers are assumed always live.
    if (isBlockExprInCallers(BlkExpr, SymReaper.getLocationContext())) {
      NewEnv.ExprBindings = F.add(NewEnv.ExprBindings, BlkExpr, X);

      if (isa<loc::MemRegionVal>(X)) {
        const MemRegion* R = cast<loc::MemRegionVal>(X).getRegion();
        DRoots.push_back(R);
      }

      // Mark all symbols in the block expr's value live.
      MarkLiveCallback cb(SymReaper);
      ST->scanReachableSymbols(X, cb);
      continue;
    }

    // Not a block-level expression?
    if (!C.isBlkExpr(BlkExpr))
      continue;

    if (SymReaper.isLive(BlkExpr)) {
      // Copy the binding to the new map.
      NewEnv.ExprBindings = F.add(NewEnv.ExprBindings, BlkExpr, X);

      // If the block expr's value is a memory region, then mark that region.
      if (isa<loc::MemRegionVal>(X)) {
        const MemRegion* R = cast<loc::MemRegionVal>(X).getRegion();
        DRoots.push_back(R);
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
      NewEnv.ExprBindings = F.add(NewEnv.ExprBindings, BlkExpr, X);
  }
  
  // Go through he deferred locations and add them to the new environment if
  // the correspond Stmt* is in the map as well.
  for (llvm::SmallVectorImpl<std::pair<const Stmt*, SVal> >::iterator
      I = deferredLocations.begin(), E = deferredLocations.end(); I != E; ++I) {
    const Stmt *S = (Stmt*) (((uintptr_t) I->first) & (uintptr_t) ~0x1);
    if (NewEnv.ExprBindings.lookup(S))
      NewEnv.ExprBindings = F.add(NewEnv.ExprBindings, I->first, I->second);
  }

  return NewEnv;
}
