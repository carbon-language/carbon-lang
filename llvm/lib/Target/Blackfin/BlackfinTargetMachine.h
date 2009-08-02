//===-- BlackfinTargetMachine.h - TargetMachine for Blackfin ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Blackfin specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFINTARGETMACHINE_H
#define BLACKFINTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "BlackfinInstrInfo.h"
#include "BlackfinSubtarget.h"
#include "BlackfinISelLowering.h"

namespace llvm {

  class Module;

  class BlackfinTargetMachine : public LLVMTargetMachine {
    const TargetData DataLayout;
    BlackfinSubtarget Subtarget;
    BlackfinTargetLowering TLInfo;
    BlackfinInstrInfo InstrInfo;
    TargetFrameInfo FrameInfo;

  protected:
    virtual const TargetAsmInfo *createTargetAsmInfo() const;

  public:
    BlackfinTargetMachine(const Target &T, const Module &M,
                          const std::string &FS);

    virtual const BlackfinInstrInfo *getInstrInfo() const { return &InstrInfo; }
    virtual const TargetFrameInfo *getFrameInfo() const { return &FrameInfo; }
    virtual const BlackfinSubtarget *getSubtargetImpl() const {
      return &Subtarget;
    }
    virtual const BlackfinRegisterInfo *getRegisterInfo() const {
      return &InstrInfo.getRegisterInfo();
    }
    virtual BlackfinTargetLowering* getTargetLowering() const {
      return const_cast<BlackfinTargetLowering*>(&TLInfo);
    }
    virtual const TargetData *getTargetData() const { return &DataLayout; }
    static unsigned getModuleMatchQuality(const Module &M);
    virtual bool addInstSelector(PassManagerBase &PM,
                                 CodeGenOpt::Level OptLevel);
  };

} // end namespace llvm

#endif
