//===-- MCJIT.h - Class definition for the MCJIT ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_MCJIT_H
#define LLVM_LIB_EXECUTIONENGINE_MCJIT_H

#include "llvm/PassManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

// FIXME: This makes all kinds of horrible assumptions for the time being,
// like only having one module, not needing to worry about multi-threading,
// blah blah. Purely in get-it-up-and-limping mode for now.

class MCJIT : public ExecutionEngine {
  MCJIT(Module *M, TargetMachine *tm, TargetJITInfo &tji,
        RTDyldMemoryManager *MemMgr, bool AllocateGVsWithCode);

  TargetMachine *TM;
  MCContext *Ctx;
  RTDyldMemoryManager *MemMgr;

  // FIXME: These may need moved to a separate 'jitstate' member like the
  // non-MC JIT does for multithreading and such. Just keep them here for now.
  PassManager PM;
  Module *M;
  // FIXME: This really doesn't belong here.
  SmallVector<char, 4096> Buffer; // Working buffer into which we JIT.
  raw_svector_ostream OS;

  RuntimeDyld Dyld;

public:
  ~MCJIT();

  /// @name ExecutionEngine interface implementation
  /// @{

  virtual void *getPointerToBasicBlock(BasicBlock *BB);

  virtual void *getPointerToFunction(Function *F);

  virtual void *recompileAndRelinkFunction(Function *F);

  virtual void freeMachineCodeForFunction(Function *F);

  virtual GenericValue runFunction(Function *F,
                                   const std::vector<GenericValue> &ArgValues);

  /// getPointerToNamedFunction - This method returns the address of the
  /// specified function by using the dlsym function call.  As such it is only
  /// useful for resolving library symbols, not code generated symbols.
  ///
  /// If AbortOnFailure is false and no function with the given name is
  /// found, this function silently returns a null pointer. Otherwise,
  /// it prints a message to stderr and aborts.
  ///
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);
  /// @}
  /// @name (Private) Registration Interfaces
  /// @{

  static void Register() {
    MCJITCtor = createJIT;
  }

  static ExecutionEngine *createJIT(Module *M,
                                    std::string *ErrorStr,
                                    JITMemoryManager *JMM,
                                    bool GVsWithCode,
                                    TargetMachine *TM);

  // @}
};

} // End llvm namespace

#endif
