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

#include "MSP430FrameLowering.h"
#include "MSP430ISelLowering.h"
#include "MSP430InstrInfo.h"
#include "MSP430RegisterInfo.h"
#include "MSP430SelectionDAGInfo.h"
#include "MSP430Subtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// MSP430TargetMachine
///
class MSP430TargetMachine : public LLVMTargetMachine {
  MSP430Subtarget        Subtarget;
  const DataLayout       DL;       // Calculates type size & alignment
  MSP430InstrInfo        InstrInfo;
  MSP430TargetLowering   TLInfo;
  MSP430SelectionDAGInfo TSInfo;
  MSP430FrameLowering    FrameLowering;

public:
  MSP430TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);

  const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const MSP430InstrInfo *getInstrInfo() const override  { return &InstrInfo; }
  const DataLayout *getDataLayout() const override     { return &DL;}
  const MSP430Subtarget *getSubtargetImpl() const override { return &Subtarget; }

  const TargetRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  const MSP430TargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const MSP430SelectionDAGInfo* getSelectionDAGInfo() const override {
    return &TSInfo;
  }
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
}; // MSP430TargetMachine.

} // end namespace llvm

#endif // LLVM_TARGET_MSP430_TARGETMACHINE_H
