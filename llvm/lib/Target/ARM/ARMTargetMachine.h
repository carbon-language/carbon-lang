//===-- ARMTargetMachine.h - Define TargetMachine for ARM -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ARM specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMTARGETMACHINE_H
#define LLVM_LIB_TARGET_ARM_ARMTARGETMACHINE_H

#include "ARMInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class ARMBaseTargetMachine : public LLVMTargetMachine {
protected:
  ARMSubtarget        Subtarget;
  bool isLittle;
  mutable StringMap<std::unique_ptr<ARMSubtarget>> SubtargetMap;

public:
  ARMBaseTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS,
                       const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL,
                       bool isLittle);

  const ARMSubtarget *getSubtargetImpl() const override { return &Subtarget; }
  const ARMSubtarget *getSubtargetImpl(const Function &F) const override;

  /// \brief Register ARM analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

/// ARMTargetMachine - ARM target machine.
///
class ARMTargetMachine : public ARMBaseTargetMachine {
  virtual void anchor();
 public:
   ARMTargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS,
                    const TargetOptions &Options, Reloc::Model RM,
                    CodeModel::Model CM, CodeGenOpt::Level OL, bool isLittle);
};

/// ARMLETargetMachine - ARM little endian target machine.
///
class ARMLETargetMachine : public ARMTargetMachine {
  void anchor() override;
public:
  ARMLETargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

/// ARMBETargetMachine - ARM big endian target machine.
///
class ARMBETargetMachine : public ARMTargetMachine {
  void anchor() override;
public:
  ARMBETargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS,
                     const TargetOptions &Options, Reloc::Model RM,
                     CodeModel::Model CM, CodeGenOpt::Level OL);
};

/// ThumbTargetMachine - Thumb target machine.
/// Due to the way architectures are handled, this represents both
///   Thumb-1 and Thumb-2.
///
class ThumbTargetMachine : public ARMBaseTargetMachine {
  virtual void anchor();
public:
  ThumbTargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS,
                     const TargetOptions &Options, Reloc::Model RM,
                     CodeModel::Model CM, CodeGenOpt::Level OL, bool isLittle);
};

/// ThumbLETargetMachine - Thumb little endian target machine.
///
class ThumbLETargetMachine : public ThumbTargetMachine {
  void anchor() override;
public:
  ThumbLETargetMachine(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

/// ThumbBETargetMachine - Thumb big endian target machine.
///
class ThumbBETargetMachine : public ThumbTargetMachine {
  void anchor() override;
public:
  ThumbBETargetMachine(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
