//===--- NonNullParamChecker.cpp - Undefined arguments checker -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NonNullParamChecker, which checks for arguments expected not to
// be null due to:
//   - the corresponding parameters being declared to have nonnull attribute
//   - the corresponding parameters being references; since the call would form
//     a reference to a null pointer
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/Attr.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class NonNullParamChecker
  : public Checker< check::PreCall > {
  mutable OwningPtr<BugType> BTAttrNonNull;
  mutable OwningPtr<BugType> BTNullRefArg;
public:

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

  BugReport *genReportNullAttrNonNull(const ExplodedNode *ErrorN,
                                      const Expr *ArgE) const;
  BugReport *genReportReferenceToNullPointer(const ExplodedNode *ErrorN,
                                             const Expr *ArgE) const;
};
} // end anonymous namespace

void NonNullParamChecker::checkPreCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  const Decl *FD = Call.getDecl();
  if (!FD)
    return;

  const NonNullAttr *Att = FD->getAttr<NonNullAttr>();

  ProgramStateRef state = C.getState();

  CallEvent::param_type_iterator TyI = Call.param_type_begin(),
                                 TyE = Call.param_type_end();

  for (unsigned idx = 0, count = Call.getNumArgs(); idx != count; ++idx){

    // Check if the parameter is a reference. We want to report when reference
    // to a null pointer is passed as a paramter.
    bool haveRefTypeParam = false;
    if (TyI != TyE) {
      haveRefTypeParam = (*TyI)->isReferenceType();
      TyI++;
    }

    bool haveAttrNonNull = Att && Att->isNonNull(idx);
    if (!haveAttrNonNull) {
      // Check if the parameter is also marked 'nonnull'.
      ArrayRef<ParmVarDecl*> parms = Call.parameters();
      if (idx < parms.size())
        haveAttrNonNull = parms[idx]->hasAttr<NonNullAttr>();
    }

    if (!haveRefTypeParam && !haveAttrNonNull)
      continue;

    // If the value is unknown or undefined, we can't perform this check.
    const Expr *ArgE = Call.getArgExpr(idx);
    SVal V = Call.getArgSVal(idx);
    Optional<DefinedSVal> DV = V.getAs<DefinedSVal>();
    if (!DV)
      continue;

    // Process the case when the argument is not a location.
    assert(!haveRefTypeParam || DV->getAs<Loc>());

    if (haveAttrNonNull && !DV->getAs<Loc>()) {
      // If the argument is a union type, we want to handle a potential
      // transparent_union GCC extension.
      if (!ArgE)
        continue;

      QualType T = ArgE->getType();
      const RecordType *UT = T->getAsUnionType();
      if (!UT || !UT->getDecl()->hasAttr<TransparentUnionAttr>())
        continue;

      if (Optional<nonloc::CompoundVal> CSV =
              DV->getAs<nonloc::CompoundVal>()) {
        nonloc::CompoundVal::iterator CSV_I = CSV->begin();
        assert(CSV_I != CSV->end());
        V = *CSV_I;
        DV = V.getAs<DefinedSVal>();
        assert(++CSV_I == CSV->end());
        // FIXME: Handle (some_union){ some_other_union_val }, which turns into
        // a LazyCompoundVal inside a CompoundVal.
        if (!V.getAs<Loc>())
          continue;
        // Retrieve the corresponding expression.
        if (const CompoundLiteralExpr *CE = dyn_cast<CompoundLiteralExpr>(ArgE))
          if (const InitListExpr *IE =
                dyn_cast<InitListExpr>(CE->getInitializer()))
             ArgE = dyn_cast<Expr>(*(IE->begin()));

      } else {
        // FIXME: Handle LazyCompoundVals?
        continue;
      }
    }

    ConstraintManager &CM = C.getConstraintManager();
    ProgramStateRef stateNotNull, stateNull;
    llvm::tie(stateNotNull, stateNull) = CM.assumeDual(state, *DV);

    if (stateNull && !stateNotNull) {
      // Generate an error node.  Check for a null node in case
      // we cache out.
      if (ExplodedNode *errorNode = C.generateSink(stateNull)) {

        BugReport *R = 0;
        if (haveAttrNonNull)
          R = genReportNullAttrNonNull(errorNode, ArgE);
        else if (haveRefTypeParam)
          R = genReportReferenceToNullPointer(errorNode, ArgE);

        // Highlight the range of the argument that was null.
        R->addRange(Call.getArgSourceRange(idx));

        // Emit the bug report.
        C.emitReport(R);
      }

      // Always return.  Either we cached out or we just emitted an error.
      return;
    }

    // If a pointer value passed the check we should assume that it is
    // indeed not null from this point forward.
    assert(stateNotNull);
    state = stateNotNull;
  }

  // If we reach here all of the arguments passed the nonnull check.
  // If 'state' has been updated generated a new node.
  C.addTransition(state);
}

BugReport *NonNullParamChecker::genReportNullAttrNonNull(
  const ExplodedNode *ErrorNode, const Expr *ArgE) const {
  // Lazily allocate the BugType object if it hasn't already been
  // created. Ownership is transferred to the BugReporter object once
  // the BugReport is passed to 'EmitWarning'.
  if (!BTAttrNonNull)
    BTAttrNonNull.reset(new BugType(
        this, "Argument with 'nonnull' attribute passed null", "API"));

  BugReport *R = new BugReport(*BTAttrNonNull,
                  "Null pointer passed as an argument to a 'nonnull' parameter",
                  ErrorNode);
  if (ArgE)
    bugreporter::trackNullOrUndefValue(ErrorNode, ArgE, *R);

  return R;
}

BugReport *NonNullParamChecker::genReportReferenceToNullPointer(
  const ExplodedNode *ErrorNode, const Expr *ArgE) const {
  if (!BTNullRefArg)
    BTNullRefArg.reset(new BuiltinBug(this, "Dereference of null pointer"));

  BugReport *R = new BugReport(*BTNullRefArg,
                               "Forming reference to null pointer",
                               ErrorNode);
  if (ArgE) {
    const Expr *ArgEDeref = bugreporter::getDerefExpr(ArgE);
    if (ArgEDeref == 0)
      ArgEDeref = ArgE;
    bugreporter::trackNullOrUndefValue(ErrorNode,
                                       ArgEDeref,
                                       *R);
  }
  return R;

}

void ento::registerNonNullParamChecker(CheckerManager &mgr) {
  mgr.registerChecker<NonNullParamChecker>();
}
