//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_TARGETMACHINE_H
#define PPC_TARGETMACHINE_H

#include "PPCFrameInfo.h"
#include "PPCSubtarget.h"
#include "PPCJITInfo.h"
#include "PPCInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class PassManager;
class IntrinsicLowering;
class GlobalValue;
class IntrinsicLowering;

class PPCTargetMachine : public TargetMachine {
  PPCInstrInfo    InstrInfo;
  PPCSubtarget    Subtarget;
  PPCFrameInfo    FrameInfo;
  PPCJITInfo      JITInfo;
public:
  PPCTargetMachine(const Module &M, IntrinsicLowering *IL,
                   const std::string &FS);

  virtual const PPCInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual       TargetJITInfo    *getJITInfo()         { return &JITInfo; }
  virtual const TargetSubtarget  *getSubtargetImpl() const{ return &Subtarget; }
  virtual const MRegisterInfo    *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  static unsigned getJITMatchQuality();

  static unsigned getModuleMatchQuality(const Module &M);
  
  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType);
  
  bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                  MachineCodeEmitter &MCE);
};
  
} // end namespace llvm

#endif
