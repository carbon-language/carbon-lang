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

#ifndef X86TARGETMACHINE_H
#define X86TARGETMACHINE_H

#include "X86.h"
#include "X86FrameLowering.h"
#include "X86ISelLowering.h"
#include "X86InstrInfo.h"
#include "X86JITInfo.h"
#include "X86SelectionDAGInfo.h"
#include "X86Subtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class StringRef;

class X86TargetMachine : public LLVMTargetMachine {
  virtual void anchor();
  X86Subtarget       Subtarget;
  X86FrameLowering   FrameLowering;
  InstrItineraryData InstrItins;
  const DataLayout   DL; // Calculates type size & alignment
  X86InstrInfo       InstrInfo;
  X86TargetLowering  TLInfo;
  X86SelectionDAGInfo TSInfo;
  X86JITInfo         JITInfo;

public:
  X86TargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL);

  const DataLayout *getDataLayout() const override { return &DL; }
  const X86InstrInfo *getInstrInfo() const override {
    return &InstrInfo;
  }
  const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  X86JITInfo *getJITInfo() override {
    return &JITInfo;
  }
  const X86Subtarget *getSubtargetImpl() const override { return &Subtarget; }
  const X86TargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const X86SelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }
  const X86RegisterInfo  *getRegisterInfo() const override {
    return &getInstrInfo()->getRegisterInfo();
  }
  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }

  /// \brief Register X86 analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;

  // Set up the pass pipeline.
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  bool addCodeEmitter(PassManagerBase &PM, JITCodeEmitter &JCE) override;
};

} // End llvm namespace

#endif
