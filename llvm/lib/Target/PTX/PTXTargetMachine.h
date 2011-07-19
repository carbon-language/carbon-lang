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
#include "PTXFrameLowering.h"
#include "PTXSubtarget.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class PTXTargetMachine : public LLVMTargetMachine {
  private:
    const TargetData  DataLayout;
    PTXSubtarget      Subtarget; // has to be initialized before FrameLowering
    PTXFrameLowering  FrameLowering;
    PTXInstrInfo      InstrInfo;
    PTXTargetLowering TLInfo;

  public:
    PTXTargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, Reloc::Model RM,
                     bool is64Bit);

    virtual const TargetData *getTargetData() const { return &DataLayout; }

    virtual const TargetFrameLowering *getFrameLowering() const {
      return &FrameLowering;
    }

    virtual const PTXInstrInfo *getInstrInfo() const { return &InstrInfo; }
    virtual const TargetRegisterInfo *getRegisterInfo() const {
      return &InstrInfo.getRegisterInfo(); }

    virtual const PTXTargetLowering *getTargetLowering() const {
      return &TLInfo; }

    virtual const PTXSubtarget *getSubtargetImpl() const { return &Subtarget; }

    virtual bool addInstSelector(PassManagerBase &PM,
                                 CodeGenOpt::Level OptLevel);
    virtual bool addPostRegAlloc(PassManagerBase &PM,
                                 CodeGenOpt::Level OptLevel);
}; // class PTXTargetMachine


class PTX32TargetMachine : public PTXTargetMachine {
public:

  PTX32TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, Reloc::Model RM);
}; // class PTX32TargetMachine

class PTX64TargetMachine : public PTXTargetMachine {
public:

  PTX64TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, Reloc::Model RM);
}; // class PTX32TargetMachine

} // namespace llvm

#endif // PTX_TARGET_MACHINE_H
