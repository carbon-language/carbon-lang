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

  virtual const DataLayout *getDataLayout() const { return &DL; }
  virtual const X86InstrInfo     *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual const TargetFrameLowering  *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual       X86JITInfo       *getJITInfo()         {
    return &JITInfo;
  }
  virtual const X86Subtarget     *getSubtargetImpl() const{ return &Subtarget; }
  virtual const X86TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }
  virtual const X86SelectionDAGInfo *getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const X86RegisterInfo  *getRegisterInfo() const {
    return &getInstrInfo()->getRegisterInfo();
  }
  virtual const InstrItineraryData *getInstrItineraryData() const {
    return &InstrItins;
  }

  /// \brief Register X86 analysis passes with a pass manager.
  virtual void addAnalysisPasses(PassManagerBase &PM);

  // Set up the pass pipeline.
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);

  virtual bool addCodeEmitter(PassManagerBase &PM,
                              JITCodeEmitter &JCE);
};

} // End llvm namespace

#endif
