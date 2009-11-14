//== NullDerefChecker.cpp - Null dereference checker ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NullDerefChecker, a builtin check in GRExprEngine that performs
// checks for null pointers at loads and stores.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checkers/DereferenceChecker.h"
#include "clang/Analysis/PathSensitive/Checker.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "GRExprEngineInternalChecks.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN DereferenceChecker : public Checker {
  BuiltinBug *BT_null;
  BuiltinBug *BT_undef;
  llvm::SmallVector<ExplodedNode*, 2> ImplicitNullDerefNodes;
public:
  DereferenceChecker() : BT_null(0), BT_undef(0) {}
  static void *getTag() { static int tag = 0; return &tag; }
  void VisitLocation(CheckerContext &C, const Stmt *S, SVal location);
  
  std::pair<ExplodedNode * const*, ExplodedNode * const*>
  getImplicitNodes() const {    
    return std::make_pair(ImplicitNullDerefNodes.data(),
                          ImplicitNullDerefNodes.data() +
                          ImplicitNullDerefNodes.size());
  }
};
} // end anonymous namespace

void clang::RegisterDereferenceChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new DereferenceChecker());
}

std::pair<ExplodedNode * const *, ExplodedNode * const *>
clang::GetImplicitNullDereferences(GRExprEngine &Eng) {
  DereferenceChecker *checker = Eng.getChecker<DereferenceChecker>();
  if (!checker)
    return std::make_pair((ExplodedNode * const *) 0,
                          (ExplodedNode * const *) 0);
  return checker->getImplicitNodes();
}

void DereferenceChecker::VisitLocation(CheckerContext &C, const Stmt *S,
                                       SVal l) {
  // Check for dereference of an undefined value.
  if (l.isUndef()) {
    ExplodedNode *N = C.GenerateNode(S, true);
    if (N) {
      if (!BT_undef)
        BT_undef = new BuiltinBug("Dereference of undefined pointer value");
      
      EnhancedBugReport *report =
        new EnhancedBugReport(*BT_undef, BT_undef->getDescription(), N);
      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                                bugreporter::GetDerefExpr(N));
      C.EmitReport(report);
    }
    return;
  }
  
  DefinedOrUnknownSVal location = cast<DefinedOrUnknownSVal>(l);
  
  // Check for null dereferences.  
  if (!isa<Loc>(location))
    return;
  
  const GRState *state = C.getState();
  const GRState *notNullState, *nullState;
  llvm::tie(notNullState, nullState) = state->Assume(location);
  
  // The explicit NULL case.
  if (nullState) {
    // Generate an error node.
    ExplodedNode *N = C.GenerateNode(S, nullState, true);    
    if (N) {      
      if (!notNullState) {
        // We know that 'location' cannot be non-null.  This is what
        // we call an "explicit" null dereference.        
        if (!BT_null)
          BT_null = new BuiltinBug("Null pointer dereference",
                                   "Dereference of null pointer");

        EnhancedBugReport *report =
          new EnhancedBugReport(*BT_null, BT_null->getDescription(), N);
        report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                                  bugreporter::GetDerefExpr(N));
        
        C.EmitReport(report);
        return;
      }

      // Otherwise, we have the case where the location could either be
      // null or not-null.  Record the error node as an "implicit" null
      // dereference.
      ImplicitNullDerefNodes.push_back(N);
    }
  }
  
  // From this point forward, we know that the location is not null.
  assert(notNullState);
  C.addTransition(state != nullState ? C.GenerateNode(S, notNullState) :
                                       C.getPredecessor());
}
