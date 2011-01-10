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
#include "MBlazeSelectionDAGInfo.h"
#include "MBlazeIntrinsicInfo.h"
#include "MBlazeFrameLowering.h"
#include "MBlazeELFWriterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class formatted_raw_ostream;

  class MBlazeTargetMachine : public LLVMTargetMachine {
    MBlazeSubtarget        Subtarget;
    const TargetData       DataLayout; // Calculates type size & alignment
    MBlazeInstrInfo        InstrInfo;
    MBlazeFrameLowering    FrameLowering;
    MBlazeTargetLowering   TLInfo;
    MBlazeSelectionDAGInfo TSInfo;
    MBlazeIntrinsicInfo    IntrinsicInfo;
    MBlazeELFWriterInfo    ELFWriterInfo;
  public:
    MBlazeTargetMachine(const Target &T, const std::string &TT,
                      const std::string &FS);

    virtual const MBlazeInstrInfo *getInstrInfo() const
    { return &InstrInfo; }

    virtual const TargetFrameLowering *getFrameLowering() const
    { return &FrameLowering; }

    virtual const MBlazeSubtarget *getSubtargetImpl() const
    { return &Subtarget; }

    virtual const TargetData *getTargetData() const
    { return &DataLayout;}

    virtual const MBlazeRegisterInfo *getRegisterInfo() const
    { return &InstrInfo.getRegisterInfo(); }

    virtual const MBlazeTargetLowering *getTargetLowering() const
    { return &TLInfo; }

    virtual const MBlazeSelectionDAGInfo* getSelectionDAGInfo() const
    { return &TSInfo; }

    const TargetIntrinsicInfo *getIntrinsicInfo() const
    { return &IntrinsicInfo; }

    virtual const MBlazeELFWriterInfo *getELFWriterInfo() const {
      return &ELFWriterInfo;
    }

    // Pass Pipeline Configuration
    virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level Opt);
    virtual bool addPreEmitPass(PassManagerBase &PM,CodeGenOpt::Level Opt);
  };
} // End llvm namespace

#endif
