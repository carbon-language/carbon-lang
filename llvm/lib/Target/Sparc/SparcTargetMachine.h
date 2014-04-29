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

#include "SparcFrameLowering.h"
#include "SparcISelLowering.h"
#include "SparcInstrInfo.h"
#include "SparcJITInfo.h"
#include "SparcSelectionDAGInfo.h"
#include "SparcSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class SparcTargetMachine : public LLVMTargetMachine {
  SparcSubtarget Subtarget;
  const DataLayout DL;       // Calculates type size & alignment
  SparcInstrInfo InstrInfo;
  SparcTargetLowering TLInfo;
  SparcSelectionDAGInfo TSInfo;
  SparcFrameLowering FrameLowering;
  SparcJITInfo JITInfo;
public:
  SparcTargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL, bool is64bit);

  const SparcInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const TargetFrameLowering  *getFrameLowering() const override {
    return &FrameLowering;
  }
  const SparcSubtarget   *getSubtargetImpl() const override{ return &Subtarget; }
  const SparcRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const SparcTargetLowering* getTargetLowering() const override {
    return &TLInfo;
  }
  const SparcSelectionDAGInfo* getSelectionDAGInfo() const override {
    return &TSInfo;
  }
  SparcJITInfo *getJITInfo() override {
    return &JITInfo;
  }
  const DataLayout       *getDataLayout() const override { return &DL; }

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
