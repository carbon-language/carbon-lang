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

#ifndef ARMTARGETMACHINE_H
#define ARMTARGETMACHINE_H

#include "ARMFrameLowering.h"
#include "ARMISelLowering.h"
#include "ARMInstrInfo.h"
#include "ARMJITInfo.h"
#include "ARMSelectionDAGInfo.h"
#include "ARMSubtarget.h"
#include "Thumb1FrameLowering.h"
#include "Thumb1InstrInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class ARMBaseTargetMachine : public LLVMTargetMachine {
protected:
  ARMSubtarget        Subtarget;
private:
  ARMJITInfo          JITInfo;
  InstrItineraryData  InstrItins;

public:
  ARMBaseTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS,
                       const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  virtual       ARMJITInfo       *getJITInfo()         { return &JITInfo; }
  virtual const ARMSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual const ARMTargetLowering *getTargetLowering() const {
    // Implemented by derived classes
    llvm_unreachable("getTargetLowering not implemented");
  }
  virtual const InstrItineraryData *getInstrItineraryData() const {
    return &InstrItins;
  }

  /// \brief Register ARM analysis passes with a pass manager.
  virtual void addAnalysisPasses(PassManagerBase &PM);

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);

  virtual bool addCodeEmitter(PassManagerBase &PM, JITCodeEmitter &MCE);
};

/// ARMTargetMachine - ARM target machine.
///
class ARMTargetMachine : public ARMBaseTargetMachine {
  virtual void anchor();
  ARMInstrInfo        InstrInfo;
  const DataLayout    DL;       // Calculates type size & alignment
  ARMTargetLowering   TLInfo;
  ARMSelectionDAGInfo TSInfo;
  ARMFrameLowering    FrameLowering;
 public:
  ARMTargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS,
                   const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL);

  virtual const ARMRegisterInfo  *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const ARMTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const ARMSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const ARMFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const ARMInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const DataLayout       *getDataLayout() const { return &DL; }
};

/// ThumbTargetMachine - Thumb target machine.
/// Due to the way architectures are handled, this represents both
///   Thumb-1 and Thumb-2.
///
class ThumbTargetMachine : public ARMBaseTargetMachine {
  virtual void anchor();
  // Either Thumb1InstrInfo or Thumb2InstrInfo.
  OwningPtr<ARMBaseInstrInfo> InstrInfo;
  const DataLayout    DL;   // Calculates type size & alignment
  ARMTargetLowering   TLInfo;
  ARMSelectionDAGInfo TSInfo;
  // Either Thumb1FrameLowering or ARMFrameLowering.
  OwningPtr<ARMFrameLowering> FrameLowering;
public:
  ThumbTargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS,
                     const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);

  /// returns either Thumb1RegisterInfo or Thumb2RegisterInfo
  virtual const ARMBaseRegisterInfo *getRegisterInfo() const {
    return &InstrInfo->getRegisterInfo();
  }

  virtual const ARMTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const ARMSelectionDAGInfo *getSelectionDAGInfo() const {
    return &TSInfo;
  }

  /// returns either Thumb1InstrInfo or Thumb2InstrInfo
  virtual const ARMBaseInstrInfo *getInstrInfo() const {
    return InstrInfo.get();
  }
  /// returns either Thumb1FrameLowering or ARMFrameLowering
  virtual const ARMFrameLowering *getFrameLowering() const {
    return FrameLowering.get();
  }
  virtual const DataLayout       *getDataLayout() const { return &DL; }
};

} // end namespace llvm

#endif
