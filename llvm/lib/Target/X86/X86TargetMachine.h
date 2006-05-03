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
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86JITInfo.h"
#include "X86Subtarget.h"
#include "X86ISelLowering.h"

namespace llvm {

class X86TargetMachine : public TargetMachine {
  const TargetData DataLayout;       // Calculates type size & alignment
  X86InstrInfo      InstrInfo;
  X86Subtarget      Subtarget;
  TargetFrameInfo   FrameInfo;
  X86JITInfo        JITInfo;
  X86TargetLowering TLInfo;
public:
  X86TargetMachine(const Module &M, const std::string &FS);

  virtual const X86InstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual       TargetJITInfo    *getJITInfo()         { return &JITInfo; }
  virtual const TargetSubtarget  *getSubtargetImpl() const{ return &Subtarget; }
  virtual       X86TargetLowering *getTargetLowering() { return &TLInfo; }
  virtual const MRegisterInfo    *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }

  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);

  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType, bool Fast);

  static unsigned getModuleMatchQuality(const Module &M);
  static unsigned getJITMatchQuality();
};
} // End llvm namespace

#endif
