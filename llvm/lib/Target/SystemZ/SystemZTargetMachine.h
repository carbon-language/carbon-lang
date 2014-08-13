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


#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETMACHINE_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETMACHINE_H

#include "SystemZSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class TargetFrameLowering;

class SystemZTargetMachine : public LLVMTargetMachine {
  SystemZSubtarget        Subtarget;

public:
  SystemZTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  // Override TargetMachine.
  const SystemZSubtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  // Override LLVMTargetMachine
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

} // end namespace llvm

#endif
