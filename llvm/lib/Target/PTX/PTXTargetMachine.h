//===-- PTXTargetMachine.h - Define TargetMachine for PTX -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PTX specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_TARGET_MACHINE_H
#define PTX_TARGET_MACHINE_H

#include "PTXISelLowering.h"
#include "PTXInstrInfo.h"
#include "PTXFrameInfo.h"
#include "PTXSubtarget.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class PTXTargetMachine : public LLVMTargetMachine {
  private:
    const TargetData DataLayout;
    PTXFrameInfo FrameInfo;
    PTXInstrInfo InstrInfo;
    PTXTargetLowering TLInfo;
    PTXSubtarget Subtarget;

  public:
    PTXTargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);

    virtual const TargetData *getTargetData() const { return &DataLayout; }

    virtual const TargetFrameInfo *getFrameInfo() const { return &FrameInfo; }

    virtual const PTXInstrInfo *getInstrInfo() const { return &InstrInfo; }
    virtual const TargetRegisterInfo *getRegisterInfo() const {
      return &InstrInfo.getRegisterInfo(); }

    virtual const PTXTargetLowering *getTargetLowering() const {
      return &TLInfo; }

    virtual const PTXSubtarget *getSubtargetImpl() const { return &Subtarget; }

    virtual bool addInstSelector(PassManagerBase &PM,
                                 CodeGenOpt::Level OptLevel);
}; // class PTXTargetMachine
} // namespace llvm

#endif // PTX_TARGET_MACHINE_H
