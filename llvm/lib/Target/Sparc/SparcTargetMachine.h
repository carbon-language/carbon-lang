//===-- SparcTargetMachine.h - Define TargetMachine for Sparc ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Sparc specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCTARGETMACHINE_H
#define SPARCTARGETMACHINE_H

#include "SparcInstrInfo.h"
#include "SparcSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class SparcTargetMachine : public LLVMTargetMachine {
  SparcSubtarget Subtarget;
public:
  SparcTargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL, bool is64bit);

  const SparcInstrInfo *getInstrInfo() const override {
    return getSubtargetImpl()->getInstrInfo();
  }
  const TargetFrameLowering *getFrameLowering() const override {
    return getSubtargetImpl()->getFrameLowering();
  }
  const SparcSubtarget *getSubtargetImpl() const override { return &Subtarget; }
  const SparcRegisterInfo *getRegisterInfo() const override {
    return getSubtargetImpl()->getRegisterInfo();
  }
  const SparcTargetLowering *getTargetLowering() const override {
    return getSubtargetImpl()->getTargetLowering();
  }
  const SparcSelectionDAGInfo *getSelectionDAGInfo() const override {
    return getSubtargetImpl()->getSelectionDAGInfo();
  }
  SparcJITInfo *getJITInfo() override { return Subtarget.getJITInfo(); }
  const DataLayout *getDataLayout() const override {
    return getSubtargetImpl()->getDataLayout();
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  bool addCodeEmitter(PassManagerBase &PM, JITCodeEmitter &JCE) override;
};

/// SparcV8TargetMachine - Sparc 32-bit target machine
///
class SparcV8TargetMachine : public SparcTargetMachine {
  virtual void anchor();
public:
  SparcV8TargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS,
                       const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

/// SparcV9TargetMachine - Sparc 64-bit target machine
///
class SparcV9TargetMachine : public SparcTargetMachine {
  virtual void anchor();
public:
  SparcV9TargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS,
                       const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
