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
                     CodeModel::Model CM, CodeGenOpt::Level OL,
                     bool IsLittleEndian);

  const ARM64Subtarget *getSubtargetImpl() const override { return &Subtarget; }
  const ARM64TargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const DataLayout *getDataLayout() const override { return &DL; }
  const ARM64FrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const ARM64InstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const ARM64RegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const ARM64SelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  /// \brief Register ARM64 analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;
};

// ARM64leTargetMachine - ARM64 little endian target machine.
//
class ARM64leTargetMachine : public ARM64TargetMachine {
  virtual void anchor();
public:
  ARM64leTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

// ARM64beTargetMachine - ARM64 big endian target machine.
//
class ARM64beTargetMachine : public ARM64TargetMachine {
  virtual void anchor();
public:
  ARM64beTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
