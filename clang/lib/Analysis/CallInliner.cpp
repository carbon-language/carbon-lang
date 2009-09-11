//===--- CallInliner.cpp - Transfer function that inlines callee ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the callee inlining transfer function.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

using namespace clang;

namespace {
  
class VISIBILITY_HIDDEN CallInliner : public GRTransferFuncs {
  ASTContext &Ctx;
public:
  CallInliner(ASTContext &ctx) : Ctx(ctx) {}

  void EvalCall(ExplodedNodeSet& Dst, GRExprEngine& Engine,
                GRStmtNodeBuilder& Builder, CallExpr* CE, SVal L,
                ExplodedNode* Pred);
  
};

}

void CallInliner::EvalCall(ExplodedNodeSet& Dst, GRExprEngine& Engine,
                           GRStmtNodeBuilder& Builder, CallExpr* CE, SVal L,
                           ExplodedNode* Pred) {
  assert(0 && "TO BE IMPLEMENTED");
}
  
GRTransferFuncs *clang::CreateCallInliner(ASTContext &ctx) {
  return new CallInliner(ctx);
}
