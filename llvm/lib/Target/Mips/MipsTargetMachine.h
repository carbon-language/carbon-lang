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
#include "MipsFrameLowering.h"
#include "MipsSelectionDAGInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class formatted_raw_ostream;

  class MipsTargetMachine : public LLVMTargetMachine {
    MipsSubtarget       Subtarget;
    const TargetData    DataLayout; // Calculates type size & alignment
    MipsInstrInfo       InstrInfo;
    MipsFrameLowering   FrameLowering;
    MipsTargetLowering  TLInfo;
    MipsSelectionDAGInfo TSInfo;
  public:
    MipsTargetMachine(const Target &T, const std::string &TT,
                      const std::string &CPU, const std::string &FS,
                      bool isLittle);

    virtual const MipsInstrInfo   *getInstrInfo()     const
    { return &InstrInfo; }
    virtual const TargetFrameLowering *getFrameLowering()     const
    { return &FrameLowering; }
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
    virtual bool addPreRegAlloc(PassManagerBase &PM,
                                CodeGenOpt::Level OptLevel);
    virtual bool addPostRegAlloc(PassManagerBase &, CodeGenOpt::Level);
  };

/// MipselTargetMachine - Mipsel target machine.
///
class MipselTargetMachine : public MipsTargetMachine {
public:
  MipselTargetMachine(const Target &T, const std::string &TT,
                      const std::string &CPU, const std::string &FS);
};

} // End llvm namespace

#endif
