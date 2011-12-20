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
#include "MipsJITInfo.h"

namespace llvm {
  class formatted_raw_ostream;

  class MipsTargetMachine : public LLVMTargetMachine {
    MipsSubtarget       Subtarget;
    const TargetData    DataLayout; // Calculates type size & alignment
    MipsInstrInfo       InstrInfo;
    MipsFrameLowering   FrameLowering;
    MipsTargetLowering  TLInfo;
    MipsSelectionDAGInfo TSInfo;
    MipsJITInfo JITInfo;

  public:
    MipsTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL,
                      bool isLittle);

    virtual const MipsInstrInfo   *getInstrInfo()     const
    { return &InstrInfo; }
    virtual const TargetFrameLowering *getFrameLowering()     const
    { return &FrameLowering; }
    virtual const MipsSubtarget   *getSubtargetImpl() const
    { return &Subtarget; }
    virtual const TargetData      *getTargetData()    const
    { return &DataLayout;}
    virtual MipsJITInfo *getJITInfo()
    { return &JITInfo; }


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
    virtual bool addInstSelector(PassManagerBase &PM);
    virtual bool addPreEmitPass(PassManagerBase &PM);
    virtual bool addPreRegAlloc(PassManagerBase &PM);
    virtual bool addPostRegAlloc(PassManagerBase &);
    virtual bool addCodeEmitter(PassManagerBase &PM,
				 JITCodeEmitter &JCE);

  };

/// MipsebTargetMachine - Mips32 big endian target machine.
///
class MipsebTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  MipsebTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
};

/// MipselTargetMachine - Mips32 little endian target machine.
///
class MipselTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  MipselTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
};

/// Mips64ebTargetMachine - Mips64 big endian target machine.
///
class Mips64ebTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  Mips64ebTargetMachine(const Target &T, StringRef TT,
                        StringRef CPU, StringRef FS,
                        const TargetOptions &Options,
                        Reloc::Model RM, CodeModel::Model CM,
                        CodeGenOpt::Level OL);
};

/// Mips64elTargetMachine - Mips64 little endian target machine.
///
class Mips64elTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  Mips64elTargetMachine(const Target &T, StringRef TT,
                        StringRef CPU, StringRef FS,
                        const TargetOptions &Options,
                        Reloc::Model RM, CodeModel::Model CM,
                        CodeGenOpt::Level OL);
};
} // End llvm namespace

#endif
