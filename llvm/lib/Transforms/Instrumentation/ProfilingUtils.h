//===- ProfilingUtils.h - Helper functions shared by profilers --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This files defines a few helper functions which are used by profile
// instrumentation code to instrument the code.  This allows the profiler pass
// to worry about *what* to insert, and these functions take care of *how* to do
// it.
//
//===----------------------------------------------------------------------===//

#ifndef PROFILINGUTILS_H
#define PROFILINGUTILS_H

namespace llvm {
  class Function;
  class GlobalValue;
  class ConstantPointerRef;
  class BasicBlock;

  void InsertProfilingInitCall(Function *MainFn, const char *FnName,
                               GlobalValue *Arr);
  void IncrementCounterInBlock(BasicBlock *BB, unsigned CounterNum,
                               ConstantPointerRef *CounterArray);
}

#endif
