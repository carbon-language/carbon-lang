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
#include "Thumb1InstrInfo.h"
#include "Thumb2InstrInfo.h"

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
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreRegAlloc(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              MachineCodeEmitter &MCE);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              JITCodeEmitter &MCE);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              ObjectCodeEmitter &OCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    MachineCodeEmitter &MCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    JITCodeEmitter &MCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    ObjectCodeEmitter &OCE);
};

/// ARMTargetMachine - ARM target machine.
///
class ARMTargetMachine : public ARMBaseTargetMachine {
  ARMInstrInfo        InstrInfo;
  const TargetData    DataLayout;       // Calculates type size & alignment
  ARMTargetLowering   TLInfo;
public:
  ARMTargetMachine(const Target &T, const std::string &TT,
                   const std::string &FS);

  virtual const ARMRegisterInfo  *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual       ARMTargetLowering *getTargetLowering() const {
    return const_cast<ARMTargetLowering*>(&TLInfo);
  }

  virtual const ARMInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
};

/// ThumbTargetMachine - Thumb target machine.
/// Due to the way architectures are handled, this represents both
///   Thumb-1 and Thumb-2.
///
class ThumbTargetMachine : public ARMBaseTargetMachine {
  ARMBaseInstrInfo    *InstrInfo;   // either Thumb1InstrInfo or Thumb2InstrInfo
  const TargetData    DataLayout;   // Calculates type size & alignment
  ARMTargetLowering   TLInfo;
public:
  ThumbTargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);

  /// returns either Thumb1RegisterInfo of Thumb2RegisterInfo
  virtual const ARMBaseRegisterInfo *getRegisterInfo() const {
    return &InstrInfo->getRegisterInfo();
  }

  virtual ARMTargetLowering *getTargetLowering() const {
    return const_cast<ARMTargetLowering*>(&TLInfo);
  }

  /// returns either Thumb1InstrInfo or Thumb2InstrInfo
  virtual const ARMBaseInstrInfo *getInstrInfo() const { return InstrInfo; }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
};

} // end namespace llvm

#endif
