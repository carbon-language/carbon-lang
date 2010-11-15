//===-- MipsTargetMachine.h - Define TargetMachine for Mips -00--*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Mips specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETMACHINE_H
#define MIPSTARGETMACHINE_H

#include "MipsSubtarget.h"
#include "MipsInstrInfo.h"
#include "MipsISelLowering.h"
#include "MipsFrameInfo.h"
#include "MipsSelectionDAGInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class formatted_raw_ostream;

  class MipsTargetMachine : public LLVMTargetMachine {
    MipsSubtarget       Subtarget;
    const TargetData    DataLayout; // Calculates type size & alignment
    MipsInstrInfo       InstrInfo;
    MipsFrameInfo       FrameInfo;
    MipsTargetLowering  TLInfo;
    MipsSelectionDAGInfo TSInfo;
  public:
    MipsTargetMachine(const Target &T, const std::string &TT,
                      const std::string &FS, bool isLittle);

    virtual const MipsInstrInfo   *getInstrInfo()     const
    { return &InstrInfo; }
    virtual const TargetFrameInfo *getFrameInfo()     const
    { return &FrameInfo; }
    virtual const MipsSubtarget   *getSubtargetImpl() const
    { return &Subtarget; }
    virtual const TargetData      *getTargetData()    const
    { return &DataLayout;}

    virtual const MipsRegisterInfo *getRegisterInfo()  const {
      return &InstrInfo.getRegisterInfo();
    }

    virtual const MipsTargetLowering *getTargetLowering() const {
      return &TLInfo;
    }

    virtual const MipsSelectionDAGInfo* getSelectionDAGInfo() const {
      return &TSInfo;
    }

    // Pass Pipeline Configuration
    virtual bool addInstSelector(PassManagerBase &PM,
                                 CodeGenOpt::Level OptLevel);
    virtual bool addPreEmitPass(PassManagerBase &PM,
                                CodeGenOpt::Level OptLevel);
  };

/// MipselTargetMachine - Mipsel target machine.
///
class MipselTargetMachine : public MipsTargetMachine {
public:
  MipselTargetMachine(const Target &T, const std::string &TT,
                      const std::string &FS);
};

} // End llvm namespace

#endif
