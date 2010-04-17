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
#include "BlackfinIntrinsicInfo.h"

namespace llvm {

  class BlackfinTargetMachine : public LLVMTargetMachine {
    const TargetData DataLayout;
    BlackfinSubtarget Subtarget;
    BlackfinTargetLowering TLInfo;
    BlackfinInstrInfo InstrInfo;
    TargetFrameInfo FrameInfo;
    BlackfinIntrinsicInfo IntrinsicInfo;
  public:
    BlackfinTargetMachine(const Target &T, const std::string &TT,
                          const std::string &FS);

    virtual const BlackfinInstrInfo *getInstrInfo() const { return &InstrInfo; }
    virtual const TargetFrameInfo *getFrameInfo() const { return &FrameInfo; }
    virtual const BlackfinSubtarget *getSubtargetImpl() const {
      return &Subtarget;
    }
    virtual const BlackfinRegisterInfo *getRegisterInfo() const {
      return &InstrInfo.getRegisterInfo();
    }
    virtual const BlackfinTargetLowering* getTargetLowering() const {
      return &TLInfo;
    }
    virtual const TargetData *getTargetData() const { return &DataLayout; }
    virtual bool addInstSelector(PassManagerBase &PM,
                                 CodeGenOpt::Level OptLevel);
    const TargetIntrinsicInfo *getIntrinsicInfo() const {
      return &IntrinsicInfo;
    }
  };

} // end namespace llvm

#endif
