//===-- MSP430TargetMachine.h - Define TargetMachine for MSP430 -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MSP430 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TARGET_MSP430_TARGETMACHINE_H
#define LLVM_TARGET_MSP430_TARGETMACHINE_H

#include "MSP430Subtarget.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// MSP430TargetMachine
///
class MSP430TargetMachine : public LLVMTargetMachine {
  MSP430Subtarget        Subtarget;

public:
  MSP430TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);

  const TargetFrameLowering *getFrameLowering() const override {
    return getSubtargetImpl()->getFrameLowering();
  }
  const MSP430InstrInfo *getInstrInfo() const override {
    return getSubtargetImpl()->getInstrInfo();
  }
  const DataLayout *getDataLayout() const override {
    return getSubtargetImpl()->getDataLayout();
  }
  const MSP430Subtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  const TargetRegisterInfo *getRegisterInfo() const override {
    return getSubtargetImpl()->getRegisterInfo();
  }
  const MSP430TargetLowering *getTargetLowering() const override {
    return getSubtargetImpl()->getTargetLowering();
  }
  const MSP430SelectionDAGInfo *getSelectionDAGInfo() const override {
    return getSubtargetImpl()->getSelectionDAGInfo();
  }
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
}; // MSP430TargetMachine.

} // end namespace llvm

#endif // LLVM_TARGET_MSP430_TARGETMACHINE_H
