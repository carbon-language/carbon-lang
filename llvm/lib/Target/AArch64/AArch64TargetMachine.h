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

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64TARGETMACHINE_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64TARGETMACHINE_H

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AArch64TargetMachine : public LLVMTargetMachine {
protected:
  AArch64Subtarget Subtarget;

public:
  AArch64TargetMachine(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL, bool IsLittleEndian);

  const AArch64Subtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  /// \brief Register AArch64 analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;

  /// \brief Query if the PBQP register allocator is being used
  bool isPBQPUsed() const { return usingPBQP; }

private:
  bool usingPBQP;
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
