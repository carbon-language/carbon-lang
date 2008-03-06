// CFRefCount.h - Transfer functions for the CF Ref. Count checker -*- C++ -*---
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CFRefCount, which defines the transfer functions
//  to implement the Core Foundation reference count checker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRREFCOUNT
#define LLVM_CLANG_ANALYSIS_GRREFCOUNT

#include "GRSimpleVals.h"

namespace clang {
  
class CFRefCount : public GRSimpleVals {
public:
  CFRefCount() {}
  virtual ~CFRefCount() {}
    
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        ValueStateManager& StateMgr,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        ValueManager& ValMgr,
                        CallExpr* CE, LVal L,
                        ExplodedNode<ValueState>* Pred);  
};
  
} // end clang namespace

#endif
