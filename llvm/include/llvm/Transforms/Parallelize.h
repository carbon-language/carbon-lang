//===- Parallelize.h - Auto parallelization using DS Graphs -----*- C++ -*-===//
//
// Externally visible routines related to the IPO pass Parallelize in
// lib/Transforms/IPO/Parallelize.cpp.  That pass automatically parallelizes
// a program using the Cilk multi-threaded runtime system to execute
// parallel code.  The routines here are used only to identify functions
// marked as Cilk operations.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_PARALLELIZE_H
#define LLVM_TRANSFORMS_PARALLELIZE_H

class Function;
class CallInst;

//---------------------------------------------------------------------------- 
// Routines to identify Cilk functions, calls to Cilk functions, and syncs.
//---------------------------------------------------------------------------- 

extern bool isCilk(const Function& F);
extern bool isCilk(const CallInst& CI);
extern bool isSync(const CallInst& CI);

//===----------------------------------------------------------------------===//

#endif
