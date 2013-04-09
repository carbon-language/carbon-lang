//===-- MipsTargetMachine.h - Define TargetMachine for Mips -----*- C++ -*-===//
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

#include "MipsFrameLowering.h"
#include "MipsISelLowering.h"
#include "MipsInstrInfo.h"
#include "MipsJITInfo.h"
#include "MipsSelectionDAGInfo.h"
#include "MipsSubtarget.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class formatted_raw_ostream;
class MipsRegisterInfo;

class MipsTargetMachine : public LLVMTargetMachine {
  MipsSubtarget       Subtarget;
  const DataLayout    DL; // Calculates type size & alignment
  OwningPtr<const MipsInstrInfo> InstrInfo;
  OwningPtr<const MipsFrameLowering> FrameLowering;
  OwningPtr<const MipsTargetLowering> TLInfo;
  OwningPtr<const MipsInstrInfo> InstrInfo16;
  OwningPtr<const MipsFrameLowering> FrameLowering16;
  OwningPtr<const MipsTargetLowering> TLInfo16;
  OwningPtr<const MipsInstrInfo> InstrInfoSE;
  OwningPtr<const MipsFrameLowering> FrameLoweringSE;
  OwningPtr<const MipsTargetLowering> TLInfoSE;
  MipsSelectionDAGInfo TSInfo;
  MipsJITInfo JITInfo;

public:
  MipsTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL,
                    bool isLittle);

  virtual ~MipsTargetMachine() {}

  virtual void addAnalysisPasses(PassManagerBase &PM);

  virtual const MipsInstrInfo *getInstrInfo() const
  { return InstrInfo.get(); }
  virtual const TargetFrameLowering *getFrameLowering() const
  { return FrameLowering.get(); }
  virtual const MipsSubtarget *getSubtargetImpl() const
  { return &Subtarget; }
  virtual const DataLayout *getDataLayout()    const
  { return &DL;}
  virtual MipsJITInfo *getJITInfo()
  { return &JITInfo; }

  virtual const MipsRegisterInfo *getRegisterInfo()  const {
    return &InstrInfo->getRegisterInfo();
  }

  virtual const MipsTargetLowering *getTargetLowering() const {
    return TLInfo.get();
  }

  virtual const MipsSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
  virtual bool addCodeEmitter(PassManagerBase &PM, JITCodeEmitter &JCE);

  // Set helper classes
  void setHelperClassesMips16();

  void setHelperClassesMipsSE();


};

/// MipsebTargetMachine - Mips32/64 big endian target machine.
///
class MipsebTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  MipsebTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
};

/// MipselTargetMachine - Mips32/64 little endian target machine.
///
class MipselTargetMachine : public MipsTargetMachine {
  virtual void anchor();
public:
  MipselTargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);
};

} // End llvm namespace

#endif
