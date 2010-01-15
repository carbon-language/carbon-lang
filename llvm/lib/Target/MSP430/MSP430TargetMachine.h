//==-- MSP430TargetMachine.h - Define TargetMachine for MSP430 ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MSP430 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TARGET_MSP430_TARGETMACHINE_H
#define LLVM_TARGET_MSP430_TARGETMACHINE_H

#include "MSP430InstrInfo.h"
#include "MSP430ISelLowering.h"
#include "MSP430RegisterInfo.h"
#include "MSP430Subtarget.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// MSP430TargetMachine
///
class MSP430TargetMachine : public LLVMTargetMachine {
  MSP430Subtarget        Subtarget;
  const TargetData       DataLayout;       // Calculates type size & alignment
  MSP430InstrInfo        InstrInfo;
  MSP430TargetLowering   TLInfo;

  // MSP430 does not have any call stack frame, therefore not having
  // any MSP430 specific FrameInfo class.
  TargetFrameInfo       FrameInfo;

public:
  MSP430TargetMachine(const Target &T, const std::string &TT,
                      const std::string &FS);

  virtual const TargetFrameInfo *getFrameInfo() const { return &FrameInfo; }
  virtual const MSP430InstrInfo *getInstrInfo() const  { return &InstrInfo; }
  virtual const TargetData *getTargetData() const     { return &DataLayout;}
  virtual const MSP430Subtarget *getSubtargetImpl() const { return &Subtarget; }

  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual MSP430TargetLowering *getTargetLowering() const {
    return const_cast<MSP430TargetLowering*>(&TLInfo);
  }

  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
}; // MSP430TargetMachine.

} // end namespace llvm

#endif // LLVM_TARGET_MSP430_TARGETMACHINE_H
