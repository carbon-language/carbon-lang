//==- SystemZTargetMachine.h - Define TargetMachine for SystemZ ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SystemZ specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef SYSTEMZTARGETMACHINE_H
#define SYSTEMZTARGETMACHINE_H

#include "SystemZFrameLowering.h"
#include "SystemZISelLowering.h"
#include "SystemZInstrInfo.h"
#include "SystemZRegisterInfo.h"
#include "SystemZSelectionDAGInfo.h"
#include "SystemZSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class SystemZTargetMachine : public LLVMTargetMachine {
  SystemZSubtarget        Subtarget;
  const DataLayout        DL;
  SystemZInstrInfo        InstrInfo;
  SystemZTargetLowering   TLInfo;
  SystemZSelectionDAGInfo TSInfo;
  SystemZFrameLowering    FrameLowering;

public:
  SystemZTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  // Override TargetMachine.
  virtual const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  virtual const SystemZInstrInfo *getInstrInfo() const override {
    return &InstrInfo;
  }
  virtual const SystemZSubtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  virtual const DataLayout *getDataLayout() const override {
    return &DL;
  }
  virtual const SystemZRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const SystemZTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  virtual const TargetSelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  // Override LLVMTargetMachine
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

} // end namespace llvm

#endif
