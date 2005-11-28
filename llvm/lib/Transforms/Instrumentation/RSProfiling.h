//===- RSProfiling.cpp - Various profiling using random sampling ----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// See notes in RSProfiling.cpp
//
//===----------------------------------------------------------------------===//

namespace llvm {
  // By default, we provide some convienence stuff to clients, so they 
  // can just store the instructions they create to do profiling.
  // also, handle all chaining issues.
  // a client is free to overwrite these, as long as it implements the
  // chaining itself.
  struct RSProfilers : public ModulePass {
    std::set<Value*> profcode;
    virtual bool isProfiling(Value* v);
    virtual ~RSProfilers() {}
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    void IncrementCounterInBlock(BasicBlock *BB, unsigned CounterNum,
                                 GlobalValue *CounterArray);
  };
};
