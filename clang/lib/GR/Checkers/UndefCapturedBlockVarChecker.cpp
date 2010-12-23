// UndefCapturedBlockVarChecker.cpp - Uninitialized captured vars -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker detects blocks that capture uninitialized values.
//
//===----------------------------------------------------------------------===//

#include "ExprEngineInternalChecks.h"
#include "clang/GR/PathSensitive/CheckerVisitor.h"
#include "clang/GR/PathSensitive/ExprEngine.h"
#include "clang/GR/BugReporter/BugType.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class UndefCapturedBlockVarChecker
  : public CheckerVisitor<UndefCapturedBlockVarChecker> {
 BugType *BT;

public:
  UndefCapturedBlockVarChecker() : BT(0) {}
  static void *getTag() { static int tag = 0; return &tag; }
  void PostVisitBlockExpr(CheckerContext &C, const BlockExpr *BE);
};
} // end anonymous namespace

void ento::RegisterUndefCapturedBlockVarChecker(ExprEngine &Eng) {
  Eng.registerCheck(new UndefCapturedBlockVarChecker());
}

static const BlockDeclRefExpr *FindBlockDeclRefExpr(const Stmt *S,
                                                    const VarDecl *VD){
  if (const BlockDeclRefExpr *BR = dyn_cast<BlockDeclRefExpr>(S))
    if (BR->getDecl() == VD)
      return BR;

  for (Stmt::const_child_iterator I = S->child_begin(), E = S->child_end();
       I!=E; ++I)
    if (const Stmt *child = *I) {
      const BlockDeclRefExpr *BR = FindBlockDeclRefExpr(child, VD);
      if (BR)
        return BR;
    }

  return NULL;
}

void
UndefCapturedBlockVarChecker::PostVisitBlockExpr(CheckerContext &C,
                                                 const BlockExpr *BE) {
  if (!BE->hasBlockDeclRefExprs())
    return;

  const GRState *state = C.getState();
  const BlockDataRegion *R =
    cast<BlockDataRegion>(state->getSVal(BE).getAsRegion());

  BlockDataRegion::referenced_vars_iterator I = R->referenced_vars_begin(),
                                            E = R->referenced_vars_end();

  for (; I != E; ++I) {
    // This VarRegion is the region associated with the block; we need
    // the one associated with the encompassing context.
    const VarRegion *VR = *I;
    const VarDecl *VD = VR->getDecl();

    if (VD->getAttr<BlocksAttr>() || !VD->hasLocalStorage())
      continue;

    // Get the VarRegion associated with VD in the local stack frame.
    const LocationContext *LC = C.getPredecessor()->getLocationContext();
    VR = C.getSValBuilder().getRegionManager().getVarRegion(VD, LC);

    if (state->getSVal(VR).isUndef())
      if (ExplodedNode *N = C.generateSink()) {
        if (!BT)
          BT = new BuiltinBug("Captured block variable is uninitialized");

        // Generate a bug report.
        llvm::SmallString<128> buf;
        llvm::raw_svector_ostream os(buf);

        os << "Variable '" << VD->getName() << "' is captured by block with "
              "a garbage value";

        EnhancedBugReport *R = new EnhancedBugReport(*BT, os.str(), N);
        if (const Expr *Ex = FindBlockDeclRefExpr(BE->getBody(), VD))
          R->addRange(Ex->getSourceRange());
        R->addVisitorCreator(bugreporter::registerFindLastStore, VR);
        // need location of block
        C.EmitReport(R);
      }
  }
}
