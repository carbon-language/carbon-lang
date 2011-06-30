//==- SystemZTargetMachine.h - Define TargetMachine for SystemZ ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SystemZ specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TARGET_SYSTEMZ_TARGETMACHINE_H
#define LLVM_TARGET_SYSTEMZ_TARGETMACHINE_H

#include "SystemZInstrInfo.h"
#include "SystemZISelLowering.h"
#include "SystemZFrameLowering.h"
#include "SystemZSelectionDAGInfo.h"
#include "SystemZRegisterInfo.h"
#include "SystemZSubtarget.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// SystemZTargetMachine
///
class SystemZTargetMachine : public LLVMTargetMachine {
  SystemZSubtarget        Subtarget;
  const TargetData        DataLayout;       // Calculates type size & alignment
  SystemZInstrInfo        InstrInfo;
  SystemZTargetLowering   TLInfo;
  SystemZSelectionDAGInfo TSInfo;
  SystemZFrameLowering    FrameLowering;
public:
  SystemZTargetMachine(const Target &T, const std::string &TT,
                       const std::string &CPU, const std::string &FS);

  virtual const TargetFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const SystemZInstrInfo *getInstrInfo() const  { return &InstrInfo; }
  virtual const TargetData *getTargetData() const     { return &DataLayout;}
  virtual const SystemZSubtarget *getSubtargetImpl() const { return &Subtarget; }

  virtual const SystemZRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const SystemZTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const SystemZSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
}; // SystemZTargetMachine.

} // end namespace llvm

#endif // LLVM_TARGET_SystemZ_TARGETMACHINE_H
