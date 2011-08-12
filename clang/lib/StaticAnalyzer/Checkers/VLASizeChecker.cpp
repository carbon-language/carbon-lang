//=== VLASizeChecker.cpp - Undefined dereference checker --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines VLASizeChecker, a builtin check in ExprEngine that 
// performs checks for declaration of VLA of undefined or zero size.
// In addition, VLASizeChecker is responsible for defining the extent
// of the MemRegion that represents a VLA.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/AST/CharUnits.h"

using namespace clang;
using namespace ento;

namespace {
class VLASizeChecker : public Checker< check::PreStmt<DeclStmt> > {
  mutable llvm::OwningPtr<BugType> BT_zero;
  mutable llvm::OwningPtr<BugType> BT_undef;
  
public:
  void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const;
};
} // end anonymous namespace

void VLASizeChecker::checkPreStmt(const DeclStmt *DS, CheckerContext &C) const {
  if (!DS->isSingleDecl())
    return;
  
  const VarDecl *VD = dyn_cast<VarDecl>(DS->getSingleDecl());
  if (!VD)
    return;

  ASTContext &Ctx = C.getASTContext();
  const VariableArrayType *VLA = Ctx.getAsVariableArrayType(VD->getType());
  if (!VLA)
    return;

  // FIXME: Handle multi-dimensional VLAs.
  const Expr *SE = VLA->getSizeExpr();
  const GRState *state = C.getState();
  SVal sizeV = state->getSVal(SE);

  if (sizeV.isUndef()) {
    // Generate an error node.
    ExplodedNode *N = C.generateSink();
    if (!N)
      return;
    
    if (!BT_undef)
      BT_undef.reset(new BuiltinBug("Declared variable-length array (VLA) "
                                    "uses a garbage value as its size"));

    EnhancedBugReport *report =
      new EnhancedBugReport(*BT_undef, BT_undef->getName(), N);
    report->addRange(SE->getSourceRange());
    report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, SE);
    C.EmitReport(report);
    return;
  }

  // See if the size value is known. It can't be undefined because we would have
  // warned about that already.
  if (sizeV.isUnknown())
    return;
  
  // Check if the size is zero.
  DefinedSVal sizeD = cast<DefinedSVal>(sizeV);

  const GRState *stateNotZero, *stateZero;
  llvm::tie(stateNotZero, stateZero) = state->assume(sizeD);

  if (stateZero && !stateNotZero) {
    ExplodedNode *N = C.generateSink(stateZero);
    if (!BT_zero)
      BT_zero.reset(new BuiltinBug("Declared variable-length array (VLA) has "
                                   "zero size"));

    EnhancedBugReport *report =
      new EnhancedBugReport(*BT_zero, BT_zero->getName(), N);
    report->addRange(SE->getSourceRange());
    report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, SE);
    C.EmitReport(report);
    return;
  }
 
  // From this point on, assume that the size is not zero.
  state = stateNotZero;

  // VLASizeChecker is responsible for defining the extent of the array being
  // declared. We do this by multiplying the array length by the element size,
  // then matching that with the array region's extent symbol.

  // Convert the array length to size_t.
  SValBuilder &svalBuilder = C.getSValBuilder();
  QualType SizeTy = Ctx.getSizeType();
  NonLoc ArrayLength = cast<NonLoc>(svalBuilder.evalCast(sizeD, SizeTy, 
                                                         SE->getType()));

  // Get the element size.
  CharUnits EleSize = Ctx.getTypeSizeInChars(VLA->getElementType());
  SVal EleSizeVal = svalBuilder.makeIntVal(EleSize.getQuantity(), SizeTy);

  // Multiply the array length by the element size.
  SVal ArraySizeVal = svalBuilder.evalBinOpNN(state, BO_Mul, ArrayLength,
                                              cast<NonLoc>(EleSizeVal), SizeTy);

  // Finally, assume that the array's extent matches the given size.
  const LocationContext *LC = C.getPredecessor()->getLocationContext();
  DefinedOrUnknownSVal Extent =
    state->getRegion(VD, LC)->getExtent(svalBuilder);
  DefinedOrUnknownSVal ArraySize = cast<DefinedOrUnknownSVal>(ArraySizeVal);
  DefinedOrUnknownSVal sizeIsKnown =
    svalBuilder.evalEQ(state, Extent, ArraySize);
  state = state->assume(sizeIsKnown, true);

  // Assume should not fail at this point.
  assert(state);

  // Remember our assumptions!
  C.addTransition(state);
}

void ento::registerVLASizeChecker(CheckerManager &mgr) {
  mgr.registerChecker<VLASizeChecker>();
}
