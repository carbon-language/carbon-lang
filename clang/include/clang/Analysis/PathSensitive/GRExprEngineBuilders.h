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

namespace clang {
  

// SaveAndRestore - A utility class that uses RAII to save and restore
//  the value of a variable.
template<typename T>
struct SaveAndRestore {
  SaveAndRestore(T& x) : X(x), old_value(x) {}
  ~SaveAndRestore() { X = old_value; }
  T get() { return old_value; }
private:  
  T& X;
  T old_value;
};

// SaveOr - Similar to SaveAndRestore.  Operates only on bools; the old
//  value of a variable is saved, and during the dstor the old value is
//  or'ed with the new value.
struct SaveOr {
  SaveOr(bool& x) : X(x), old_value(x) { x = false; }
  ~SaveOr() { X |= old_value; }
private:
  bool& X;
  const bool old_value;
};

class GRStmtNodeBuilderRef {
  GRExprEngine::NodeSet &Dst;
  GRExprEngine::StmtNodeBuilder &B;
  GRExprEngine& Eng;
  GRExprEngine::NodeTy* Pred;
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
  
  GRStmtNodeBuilderRef(GRExprEngine::NodeSet &dst,
                       GRExprEngine::StmtNodeBuilder &builder,
                       GRExprEngine& eng,
                       GRExprEngine::NodeTy* pred,
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
  
  GRStateRef getState() {
    return GRStateRef(state, Eng.getStateManager());
  }

  GRStateManager& getStateManager() {
    return Eng.getStateManager();
  }
  
  GRExprEngine::NodeTy* MakeNode(const GRState* state) {
    return B.MakeNode(Dst, const_cast<Stmt*>(stmt), Pred, state);    
  }    
};

} // end clang namespace
#endif
