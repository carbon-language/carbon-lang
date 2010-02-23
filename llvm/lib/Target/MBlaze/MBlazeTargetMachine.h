//===-- MBlazeTargetMachine.h - Define TargetMachine for MBlaze --- C++ ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MBlaze specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZE_TARGETMACHINE_H
#define MBLAZE_TARGETMACHINE_H

#include "MBlazeSubtarget.h"
#include "MBlazeInstrInfo.h"
#include "MBlazeISelLowering.h"
#include "MBlazeIntrinsicInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class formatted_raw_ostream;

  class MBlazeTargetMachine : public LLVMTargetMachine {
    MBlazeSubtarget       Subtarget;
    const TargetData    DataLayout; // Calculates type size & alignment
    MBlazeInstrInfo       InstrInfo;
    TargetFrameInfo     FrameInfo;
    MBlazeTargetLowering  TLInfo;
    MBlazeIntrinsicInfo IntrinsicInfo;
  public:
    MBlazeTargetMachine(const Target &T, const std::string &TT,
                      const std::string &FS);

    virtual const MBlazeInstrInfo *getInstrInfo() const
    { return &InstrInfo; }

    virtual const TargetFrameInfo *getFrameInfo() const
    { return &FrameInfo; }

    virtual const MBlazeSubtarget *getSubtargetImpl() const
    { return &Subtarget; }

    virtual const TargetData *getTargetData() const
    { return &DataLayout;}

    virtual const MBlazeRegisterInfo *getRegisterInfo() const
    { return &InstrInfo.getRegisterInfo(); }

    virtual MBlazeTargetLowering   *getTargetLowering() const
    { return const_cast<MBlazeTargetLowering*>(&TLInfo); }

    const TargetIntrinsicInfo *getIntrinsicInfo() const
    { return &IntrinsicInfo; }

    // Pass Pipeline Configuration
    virtual bool addInstSelector(PassManagerBase &PM,
                                 CodeGenOpt::Level OptLevel);

    virtual bool addPreEmitPass(PassManagerBase &PM,
                                CodeGenOpt::Level OptLevel);
  };
} // End llvm namespace

#endif
