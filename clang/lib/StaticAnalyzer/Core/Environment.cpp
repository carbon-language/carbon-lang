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

#include "clang/AST/ExprObjC.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

using namespace clang;
using namespace ento;

SVal Environment::lookupExpr(const Stmt *E) const {
  const SVal* X = ExprBindings.lookup(E);
  if (X) {
    SVal V = *X;
    return V;
  }
  return UnknownVal();
}

SVal Environment::getSVal(const Stmt *E, SValBuilder& svalBuilder,
			  bool useOnlyDirectBindings) const {

  if (useOnlyDirectBindings) {
    // This branch is rarely taken, but can be exercised by
    // checkers that explicitly bind values to arbitrary
    // expressions.  It is crucial that we do not ignore any
    // expression here, and do a direct lookup.
    return lookupExpr(E);
  }

  for (;;) {
    if (const Expr *Ex = dyn_cast<Expr>(E))
      E = Ex->IgnoreParens();

    switch (E->getStmtClass()) {
      case Stmt::AddrLabelExprClass:
        return svalBuilder.makeLoc(cast<AddrLabelExpr>(E));
      case Stmt::OpaqueValueExprClass: {
        const OpaqueValueExpr *ope = cast<OpaqueValueExpr>(E);
        E = ope->getSourceExpr();
        continue;        
      }        
      case Stmt::ParenExprClass:
      case Stmt::GenericSelectionExprClass:
        llvm_unreachable("ParenExprs and GenericSelectionExprs should "
                         "have been handled by IgnoreParens()");
        return UnknownVal();
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
      // For special C0xx nullptr case, make a null pointer SVal.
      case Stmt::CXXNullPtrLiteralExprClass:
        return svalBuilder.makeNull();
      case Stmt::ExprWithCleanupsClass:
        E = cast<ExprWithCleanups>(E)->getSubExpr();
        continue;
      case Stmt::CXXBindTemporaryExprClass:
        E = cast<CXXBindTemporaryExpr>(E)->getSubExpr();
        continue;
      case Stmt::ObjCPropertyRefExprClass:
        return loc::ObjCPropRef(cast<ObjCPropertyRefExpr>(E));
        
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
                                       const ProgramState *ST) {

  // We construct a new Environment object entirely, as this is cheaper than
  // individually removing all the subexpression bindings (which will greatly
  // outnumber block-level expression bindings).
  Environment NewEnv = getInitialEnvironment();
  
  SmallVector<std::pair<const Stmt*, SVal>, 10> deferredLocations;

  MarkLiveCallback CB(SymReaper);
  ScanReachableSymbols RSScaner(ST, CB);

  llvm::ImmutableMapRef<const Stmt*,SVal>
    EBMapRef(NewEnv.ExprBindings.getRootWithoutRetain(),
             F.getTreeFactory());

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

    if (SymReaper.isLive(BlkExpr)) {
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
    }

    // Otherwise the expression is dead with a couple exceptions.
    // Do not misclean LogicalExpr or ConditionalOperator.  It is dead at the
    // beginning of itself, but we need its UndefinedVal to determine its
    // SVal.
    if (X.isUndef() && cast<UndefinedVal>(X).getData())
      EBMapRef = EBMapRef.add(BlkExpr, X);
  }
  
  // Go through he deferred locations and add them to the new environment if
  // the correspond Stmt* is in the map as well.
  for (SmallVectorImpl<std::pair<const Stmt*, SVal> >::iterator
      I = deferredLocations.begin(), E = deferredLocations.end(); I != E; ++I) {
    const Stmt *S = (Stmt*) (((uintptr_t) I->first) & (uintptr_t) ~0x1);
    if (EBMapRef.lookup(S))
      EBMapRef = EBMapRef.add(I->first, I->second);
  }

  NewEnv.ExprBindings = EBMapRef.asImmutableMap();
  return NewEnv;
}
