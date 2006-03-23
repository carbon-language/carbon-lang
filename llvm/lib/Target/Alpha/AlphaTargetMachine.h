//===-- AlphaTargetMachine.h - Define TargetMachine for Alpha ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Alpha-specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHA_TARGETMACHINE_H
#define ALPHA_TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "AlphaInstrInfo.h"
#include "AlphaJITInfo.h"
#include "AlphaSubtarget.h"

namespace llvm {

class GlobalValue;

class AlphaTargetMachine : public TargetMachine {
  AlphaInstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
  AlphaJITInfo JITInfo;
  AlphaSubtarget Subtarget;

public:
  AlphaTargetMachine(const Module &M, const std::string &FS);

  virtual const AlphaInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const TargetSubtarget  *getSubtargetImpl() const{ return &Subtarget; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual TargetJITInfo* getJITInfo() {
    return &JITInfo;
  }

  static unsigned getJITMatchQuality();

  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);

  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType, bool Fast);

  static unsigned getModuleMatchQuality(const Module &M);
};

} // end namespace llvm

#endif
