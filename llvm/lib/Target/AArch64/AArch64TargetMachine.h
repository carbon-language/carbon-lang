//==-- AArch64TargetMachine.h - Define TargetMachine for AArch64 -*- C++ -*-==//
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

#ifndef AArch64TARGETMACHINE_H
#define AArch64TARGETMACHINE_H

#include "AArch64InstrInfo.h"
#include "AArch64ISelLowering.h"
#include "AArch64Subtarget.h"
#include "AArch64FrameLowering.h"
#include "AArch64SelectionDAGInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

class AArch64TargetMachine : public LLVMTargetMachine {
protected:
  AArch64Subtarget Subtarget;

private:
  const DataLayout DL;
  AArch64InstrInfo InstrInfo;
  AArch64TargetLowering TLInfo;
  AArch64FrameLowering FrameLowering;
  AArch64SelectionDAGInfo TSInfo;

public:
  AArch64TargetMachine(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL, bool IsLittleEndian);

  const AArch64Subtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  const AArch64TargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const DataLayout *getDataLayout() const override { return &DL; }
  const AArch64FrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const AArch64InstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const AArch64RegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const AArch64SelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  /// \brief Register AArch64 analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;
};

// AArch64leTargetMachine - AArch64 little endian target machine.
//
class AArch64leTargetMachine : public AArch64TargetMachine {
  virtual void anchor();
public:
  AArch64leTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                         StringRef FS, const TargetOptions &Options,
                         Reloc::Model RM, CodeModel::Model CM,
                         CodeGenOpt::Level OL);
};

// AArch64beTargetMachine - AArch64 big endian target machine.
//
class AArch64beTargetMachine : public AArch64TargetMachine {
  virtual void anchor();
public:
  AArch64beTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                         StringRef FS, const TargetOptions &Options,
                         Reloc::Model RM, CodeModel::Model CM,
                         CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
