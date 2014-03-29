//===-- ARM64TargetMachine.h - Define TargetMachine for ARM64 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ARM64 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef ARM64TARGETMACHINE_H
#define ARM64TARGETMACHINE_H

#include "ARM64InstrInfo.h"
#include "ARM64ISelLowering.h"
#include "ARM64Subtarget.h"
#include "ARM64FrameLowering.h"
#include "ARM64SelectionDAGInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

class ARM64TargetMachine : public LLVMTargetMachine {
protected:
  ARM64Subtarget Subtarget;

private:
  const DataLayout DL;
  ARM64InstrInfo InstrInfo;
  ARM64TargetLowering TLInfo;
  ARM64FrameLowering FrameLowering;
  ARM64SelectionDAGInfo TSInfo;

public:
  ARM64TargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS,
                     const TargetOptions &Options, Reloc::Model RM,
                     CodeModel::Model CM, CodeGenOpt::Level OL);

  virtual const ARM64Subtarget *getSubtargetImpl() const { return &Subtarget; }
  virtual const ARM64TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }
  virtual const DataLayout *getDataLayout() const { return &DL; }
  virtual const ARM64FrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const ARM64InstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const ARM64RegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const ARM64SelectionDAGInfo *getSelectionDAGInfo() const {
    return &TSInfo;
  }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);

  /// \brief Register ARM64 analysis passes with a pass manager.
  virtual void addAnalysisPasses(PassManagerBase &PM);
};

} // end namespace llvm

#endif
