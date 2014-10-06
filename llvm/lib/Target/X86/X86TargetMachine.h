//===-- X86TargetMachine.h - Define TargetMachine for the X86 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86TARGETMACHINE_H
#define LLVM_LIB_TARGET_X86_X86TARGETMACHINE_H
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class StringRef;

class X86TargetMachine final : public LLVMTargetMachine {
  virtual void anchor();
  X86Subtarget       Subtarget;

  mutable StringMap<std::unique_ptr<X86Subtarget>> SubtargetMap;

public:
  X86TargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL);
  const X86Subtarget *getSubtargetImpl() const override { return &Subtarget; }
  const X86Subtarget *getSubtargetImpl(const Function &F) const override;

  /// \brief Register X86 analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;

  // Set up the pass pipeline.
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

} // End llvm namespace

#endif
