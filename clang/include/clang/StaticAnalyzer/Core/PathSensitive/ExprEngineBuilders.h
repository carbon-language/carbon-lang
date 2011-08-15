//===-- ExprEngineBuilders.h - "Builder" classes for ExprEngine ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines smart builder "references" which are used to marshal
//  builders between ExprEngine objects and their related components.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_EXPRENGINE_BUILDERS
#define LLVM_CLANG_GR_EXPRENGINE_BUILDERS
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/Analysis/Support/SaveAndRestore.h"

namespace clang {

namespace ento {

class StmtNodeBuilderRef {
  ExplodedNodeSet &Dst;
  StmtNodeBuilder &B;
  ExprEngine& Eng;
  ExplodedNode *Pred;
  const ProgramState *state;
  const Stmt *stmt;
  const unsigned OldSize;
  const bool AutoCreateNode;
  SaveAndRestore<bool> OldSink;
  SaveOr OldHasGen;

private:
  friend class ExprEngine;

  StmtNodeBuilderRef(); // do not implement
  void operator=(const StmtNodeBuilderRef&); // do not implement

  StmtNodeBuilderRef(ExplodedNodeSet &dst,
                       StmtNodeBuilder &builder,
                       ExprEngine& eng,
                       ExplodedNode *pred,
                       const ProgramState *st,
                       const Stmt *s, bool auto_create_node)
  : Dst(dst), B(builder), Eng(eng), Pred(pred),
    state(st), stmt(s), OldSize(Dst.size()), AutoCreateNode(auto_create_node),
    OldSink(B.BuildSinks), OldHasGen(B.hasGeneratedNode) {}

public:

  ~StmtNodeBuilderRef() {
    // Handle the case where no nodes where generated.  Auto-generate that
    // contains the updated state if we aren't generating sinks.
    if (!B.BuildSinks && Dst.size() == OldSize && !B.hasGeneratedNode) {
      if (AutoCreateNode)
        B.MakeNode(Dst, const_cast<Stmt*>(stmt), Pred, state);
      else
        Dst.Add(Pred);
    }
  }

  const ProgramState *getState() { return state; }

  ProgramStateManager& getStateManager() {
    return Eng.getStateManager();
  }

  ExplodedNode *MakeNode(const ProgramState *state) {
    return B.MakeNode(Dst, const_cast<Stmt*>(stmt), Pred, state);
  }
};

} // end GR namespace

} // end clang namespace

#endif
