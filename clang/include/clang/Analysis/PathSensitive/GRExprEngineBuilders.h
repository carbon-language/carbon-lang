//===-- GRExprEngineBuilders.h - "Builder" classes for GRExprEngine -*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines smart builder "references" which are used to marshal
//  builders between GRExprEngine objects and their related components.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GREXPRENGINE_BUILDERS
#define LLVM_CLANG_ANALYSIS_GREXPRENGINE_BUILDERS
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/Support/SaveAndRestore.h"

namespace clang {

class GRStmtNodeBuilderRef {
  ExplodedNodeSet &Dst;
  GRStmtNodeBuilder &B;
  GRExprEngine& Eng;
  ExplodedNode* Pred;
  const GRState* state;
  const Stmt* stmt;
  const unsigned OldSize;
  const bool AutoCreateNode;
  SaveAndRestore<bool> OldSink;
  SaveAndRestore<const void*> OldTag;
  SaveOr OldHasGen;

private:
  friend class GRExprEngine;

  GRStmtNodeBuilderRef(); // do not implement
  void operator=(const GRStmtNodeBuilderRef&); // do not implement

  GRStmtNodeBuilderRef(ExplodedNodeSet &dst,
                       GRStmtNodeBuilder &builder,
                       GRExprEngine& eng,
                       ExplodedNode* pred,
                       const GRState *st,
                       const Stmt* s, bool auto_create_node)
  : Dst(dst), B(builder), Eng(eng), Pred(pred),
    state(st), stmt(s), OldSize(Dst.size()), AutoCreateNode(auto_create_node),
    OldSink(B.BuildSinks), OldTag(B.Tag), OldHasGen(B.HasGeneratedNode) {}

public:

  ~GRStmtNodeBuilderRef() {
    // Handle the case where no nodes where generated.  Auto-generate that
    // contains the updated state if we aren't generating sinks.
    if (!B.BuildSinks && Dst.size() == OldSize && !B.HasGeneratedNode) {
      if (AutoCreateNode)
        B.MakeNode(Dst, const_cast<Stmt*>(stmt), Pred, state);
      else
        Dst.Add(Pred);
    }
  }

  const GRState *getState() { return state; }

  GRStateManager& getStateManager() {
    return Eng.getStateManager();
  }

  ExplodedNode* MakeNode(const GRState* state) {
    return B.MakeNode(Dst, const_cast<Stmt*>(stmt), Pred, state);
  }
};

} // end clang namespace
#endif
