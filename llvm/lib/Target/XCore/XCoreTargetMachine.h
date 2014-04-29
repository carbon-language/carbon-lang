//===-- XCoreTargetMachine.h - Define TargetMachine for XCore ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the XCore specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETMACHINE_H
#define XCORETARGETMACHINE_H

#include "XCoreFrameLowering.h"
#include "XCoreISelLowering.h"
#include "XCoreInstrInfo.h"
#include "XCoreSelectionDAGInfo.h"
#include "XCoreSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class XCoreTargetMachine : public LLVMTargetMachine {
  XCoreSubtarget Subtarget;
  const DataLayout DL;       // Calculates type size & alignment
  XCoreInstrInfo InstrInfo;
  XCoreFrameLowering FrameLowering;
  XCoreTargetLowering TLInfo;
  XCoreSelectionDAGInfo TSInfo;
public:
  XCoreTargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);

  const XCoreInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const XCoreFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const XCoreSubtarget *getSubtargetImpl() const override { return &Subtarget; }
  const XCoreTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const XCoreSelectionDAGInfo* getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  const TargetRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const DataLayout       *getDataLayout() const override { return &DL; }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  void addAnalysisPasses(PassManagerBase &PM) override;
};

} // end namespace llvm

#endif
