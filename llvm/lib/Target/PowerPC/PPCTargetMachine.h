//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCTARGETMACHINE_H
#define LLVM_LIB_TARGET_POWERPC_PPCTARGETMACHINE_H

#include "PPCInstrInfo.h"
#include "PPCSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// PPCTargetMachine - Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
  PPCSubtarget Subtarget;

  mutable StringMap<std::unique_ptr<PPCSubtarget>> SubtargetMap;

public:
  PPCTargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL);

  const PPCSubtarget *getSubtargetImpl() const override { return &Subtarget; }
  const PPCSubtarget *getSubtargetImpl(const Function &F) const override;

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  /// \brief Register PPC analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;
};

/// PPC32TargetMachine - PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC32TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

/// PPC64TargetMachine - PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC64TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
