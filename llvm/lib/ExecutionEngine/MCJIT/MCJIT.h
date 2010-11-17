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

#include "llvm/ExecutionEngine/ExecutionEngine.h"

namespace llvm {

class MCJIT : public ExecutionEngine {
  MCJIT(Module *M, TargetMachine &tm, TargetJITInfo &tji,
        JITMemoryManager *JMM, CodeGenOpt::Level OptLevel,
        bool AllocateGVsWithCode);
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

  /// @}
  /// @name (Private) Registration Interfaces
  /// @{

  static void Register() {
    MCJITCtor = createJIT;
  }

  // FIXME: This routine is scheduled for termination. Do not use it.
  static TargetMachine *selectTarget(Module *M,
                                     StringRef MArch,
                                     StringRef MCPU,
                                     const SmallVectorImpl<std::string>& MAttrs,
                                     std::string *Err);

  static ExecutionEngine *createJIT(Module *M,
                                    std::string *ErrorStr,
                                    JITMemoryManager *JMM,
                                    CodeGenOpt::Level OptLevel,
                                    bool GVsWithCode,
                                    CodeModel::Model CMM,
                                    StringRef MArch,
                                    StringRef MCPU,
                                    const SmallVectorImpl<std::string>& MAttrs);

  // @}
};

} // End llvm namespace

#endif
