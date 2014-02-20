//=== AArch64TargetMachine.h - Define TargetMachine for AArch64 -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the AArch64 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64TARGETMACHINE_H
#define LLVM_AARCH64TARGETMACHINE_H

#include "AArch64FrameLowering.h"
#include "AArch64ISelLowering.h"
#include "AArch64InstrInfo.h"
#include "AArch64SelectionDAGInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AArch64TargetMachine : public LLVMTargetMachine {
  AArch64Subtarget          Subtarget;
  AArch64InstrInfo          InstrInfo;
  const DataLayout          DL;
  AArch64TargetLowering     TLInfo;
  AArch64SelectionDAGInfo   TSInfo;
  AArch64FrameLowering      FrameLowering;

public:
  AArch64TargetMachine(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  const AArch64InstrInfo *getInstrInfo() const {
    return &InstrInfo;
  }

  const AArch64FrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }

  const AArch64TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  const AArch64SelectionDAGInfo *getSelectionDAGInfo() const {
    return &TSInfo;
  }

  const AArch64Subtarget *getSubtargetImpl() const { return &Subtarget; }

  const DataLayout *getDataLayout() const { return &DL; }

  const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  TargetPassConfig *createPassConfig(PassManagerBase &PM);

  virtual void addAnalysisPasses(PassManagerBase &PM);
};

}

#endif
