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

#include "ARMInstrInfo.h"
#include "ARMELFWriterInfo.h"
#include "ARMFrameLowering.h"
#include "ARMJITInfo.h"
#include "ARMSubtarget.h"
#include "ARMISelLowering.h"
#include "ARMSelectionDAGInfo.h"
#include "Thumb1InstrInfo.h"
#include "Thumb1FrameLowering.h"
#include "Thumb2InstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/OwningPtr.h"

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
  virtual const InstrItineraryData *getInstrItineraryData() const {
    return &InstrItins;
  }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);

  virtual bool addCodeEmitter(PassManagerBase &PM, JITCodeEmitter &MCE);
};

/// ARMTargetMachine - ARM target machine.
///
class ARMTargetMachine : public ARMBaseTargetMachine {
  virtual void anchor();
  ARMInstrInfo        InstrInfo;
  const TargetData    DataLayout;       // Calculates type size & alignment
  ARMELFWriterInfo    ELFWriterInfo;
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
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
  virtual const ARMELFWriterInfo *getELFWriterInfo() const {
    return Subtarget.isTargetELF() ? &ELFWriterInfo : 0;
  }
};

/// ThumbTargetMachine - Thumb target machine.
/// Due to the way architectures are handled, this represents both
///   Thumb-1 and Thumb-2.
///
class ThumbTargetMachine : public ARMBaseTargetMachine {
  virtual void anchor();
  // Either Thumb1InstrInfo or Thumb2InstrInfo.
  OwningPtr<ARMBaseInstrInfo> InstrInfo;
  const TargetData    DataLayout;   // Calculates type size & alignment
  ARMELFWriterInfo    ELFWriterInfo;
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
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
  virtual const ARMELFWriterInfo *getELFWriterInfo() const {
    return Subtarget.isTargetELF() ? &ELFWriterInfo : 0;
  }
};

} // end namespace llvm

#endif
