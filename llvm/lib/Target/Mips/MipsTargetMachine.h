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
  std::unique_ptr<const MipsInstrInfo> InstrInfo;
  std::unique_ptr<const MipsFrameLowering> FrameLowering;
  std::unique_ptr<const MipsTargetLowering> TLInfo;
  std::unique_ptr<const MipsInstrInfo> InstrInfo16;
  std::unique_ptr<const MipsFrameLowering> FrameLowering16;
  std::unique_ptr<const MipsTargetLowering> TLInfo16;
  std::unique_ptr<const MipsInstrInfo> InstrInfoSE;
  std::unique_ptr<const MipsFrameLowering> FrameLoweringSE;
  std::unique_ptr<const MipsTargetLowering> TLInfoSE;
  MipsSelectionDAGInfo TSInfo;
  const InstrItineraryData &InstrItins;
  MipsJITInfo JITInfo;

public:
  MipsTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL,
                    bool isLittle);

  virtual ~MipsTargetMachine() {}

  void addAnalysisPasses(PassManagerBase &PM) override;

  const MipsInstrInfo *getInstrInfo() const override
  { return InstrInfo.get(); }
  const TargetFrameLowering *getFrameLowering() const override
  { return FrameLowering.get(); }
  const MipsSubtarget *getSubtargetImpl() const override
  { return &Subtarget; }
  const DataLayout *getDataLayout()    const override
  { return &DL;}

  const InstrItineraryData *getInstrItineraryData() const override {
    return Subtarget.inMips16Mode() ? nullptr : &InstrItins;
  }

  MipsJITInfo *getJITInfo() override { return &JITInfo; }

  const MipsRegisterInfo *getRegisterInfo()  const override {
    return &InstrInfo->getRegisterInfo();
  }

  const MipsTargetLowering *getTargetLowering() const override {
    return TLInfo.get();
  }

  const MipsSelectionDAGInfo* getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  bool addCodeEmitter(PassManagerBase &PM, JITCodeEmitter &JCE) override;

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
