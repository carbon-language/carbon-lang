//===-- X86TargetMachine.h - Define TargetMachine for the X86 ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETMACHINE_H
#define X86TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "X86InstrInfo.h"

namespace llvm {

class X86TargetMachine : public TargetMachine {
  X86InstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
public:
  X86TargetMachine(const Module &M);

  virtual const X86InstrInfo     &getInstrInfo() const { return InstrInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return FrameInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const TargetSchedInfo &getSchedInfo()  const { abort(); }
  virtual const TargetRegInfo   &getRegInfo()    const { abort(); }
  virtual const TargetCacheInfo  &getCacheInfo() const { abort(); }

  /// addPassesToJITCompile - Add passes to the specified pass manager to
  /// implement a fast dynamic compiler for this target.  Return true if this is
  /// not supported for this target.
  ///
  virtual bool addPassesToJITCompile(FunctionPassManager &PM);

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method should returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);
  
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);

  virtual void replaceMachineCodeForFunction (void *Old, void *New);

  /// getJITStubForFunction - Create or return a stub for the specified
  /// function.  This stub acts just like the specified function, except that it
  /// allows the "address" of the function to be taken without having to
  /// generate code for it.
  virtual void *getJITStubForFunction(Function *F, MachineCodeEmitter &MCE);
};

} // End llvm namespace

#endif
