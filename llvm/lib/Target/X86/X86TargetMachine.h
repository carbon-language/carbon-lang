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
#include "X86JITInfo.h"

namespace llvm {
class IntrinsicLowering;

class X86TargetMachine : public TargetMachine {
  X86InstrInfo    InstrInfo;
  TargetFrameInfo FrameInfo;
  X86JITInfo      JITInfo;
public:
  X86TargetMachine(const Module &M, IntrinsicLowering *IL);

  virtual const X86InstrInfo     &getInstrInfo() const { return InstrInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return FrameInfo; }
  virtual       TargetJITInfo    *getJITInfo()         { return &JITInfo; }
  virtual const MRegisterInfo    *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  // deprecated interfaces
  virtual const TargetSchedInfo  &getSchedInfo() const { abort(); }
  virtual const TargetRegInfo    &getRegInfo()   const { abort(); }

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method should returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);
  
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
};

  // this is implemented in X86CodeEmitter.cpp
  namespace X86 {
    void emitInstruction(MachineCodeEmitter& mce,
                         const X86InstrInfo& ii,
                         const MachineInstr& MI);
  }

} // End llvm namespace

#endif
