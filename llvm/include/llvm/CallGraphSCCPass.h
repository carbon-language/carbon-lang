//===- CallGraphSCCPass.h - Pass that operates BU on call graph -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the CallGraphSCCPass class, which is used for passes which
// are implemented as bottom-up traversals on the call graph.  Because there may
// be cycles in the call graph, passes of this type operate on the call-graph in
// SCC order: that is, they process function bottom-up, except for recursive
// functions, which they process all at once.
//
// These passes are inherently interprocedural, and are required to keep the
// call graph up-to-date if they do anything which could modify it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CALL_GRAPH_SCC_PASS_H
#define LLVM_CALL_GRAPH_SCC_PASS_H

#include "llvm/Pass.h"

class CallGraphNode;

struct CallGraphSCCPass : public Pass {

  /// runOnSCC - This method should be implemented by the subclass to perform
  /// whatever action is necessary for the specified SCC.  Note that
  /// non-recursive (or only self-recursive) functions will have an SCC size of
  /// 1, where recursive portions of the call graph will have SCC size > 1.
  ///
  virtual bool runOnSCC(const std::vector<CallGraphNode *> &SCC) = 0;

  /// run - Run this pass, returning true if a modification was made to the
  /// module argument.  This is implemented in terms of the runOnSCC method.
  ///
  virtual bool run(Module &M);


  /// getAnalysisUsage - For this class, we declare that we require and preserve
  /// the call graph.  If the derived class implements this method, it should
  /// always explicitly call the implementation here.
  virtual void getAnalysisUsage(AnalysisUsage &Info) const;
};

#endif
