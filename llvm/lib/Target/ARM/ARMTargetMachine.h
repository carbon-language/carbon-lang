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

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "ARMInstrInfo.h"
#include "ARMFrameInfo.h"
#include "ARMJITInfo.h"
#include "ARMSubtarget.h"
#include "ARMISelLowering.h"
#include "ARMSelectionDAGInfo.h"
#include "Thumb1InstrInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/ADT/OwningPtr.h"

namespace llvm {

class ARMBaseTargetMachine : public LLVMTargetMachine {
protected:
  ARMSubtarget        Subtarget;

private:
  ARMFrameInfo        FrameInfo;
  ARMJITInfo          JITInfo;
  InstrItineraryData  InstrItins;
  Reloc::Model        DefRelocModel;    // Reloc model before it's overridden.

public:
  ARMBaseTargetMachine(const Target &T, const std::string &TT,
                       const std::string &FS, bool isThumb);

  virtual const ARMFrameInfo     *getFrameInfo() const { return &FrameInfo; }
  virtual       ARMJITInfo       *getJITInfo()         { return &JITInfo; }
  virtual const ARMSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual const InstrItineraryData getInstrItineraryData() const {
    return InstrItins;
  }

  // Pass Pipeline Configuration
  virtual bool addPreISel(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreSched2(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              JITCodeEmitter &MCE);
};

/// ARMTargetMachine - ARM target machine.
///
class ARMTargetMachine : public ARMBaseTargetMachine {
  ARMInstrInfo        InstrInfo;
  const TargetData    DataLayout;       // Calculates type size & alignment
  ARMTargetLowering   TLInfo;
  ARMSelectionDAGInfo TSInfo;
public:
  ARMTargetMachine(const Target &T, const std::string &TT,
                   const std::string &FS);

  virtual const ARMRegisterInfo  *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const ARMTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const ARMSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual const ARMInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
};

/// ThumbTargetMachine - Thumb target machine.
/// Due to the way architectures are handled, this represents both
///   Thumb-1 and Thumb-2.
///
class ThumbTargetMachine : public ARMBaseTargetMachine {
  // Either Thumb1InstrInfo or Thumb2InstrInfo.
  OwningPtr<ARMBaseInstrInfo> InstrInfo;
  const TargetData    DataLayout;   // Calculates type size & alignment
  ARMTargetLowering   TLInfo;
  ARMSelectionDAGInfo TSInfo;
public:
  ThumbTargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);

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
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
};

} // end namespace llvm

#endif
