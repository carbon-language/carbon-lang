//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_TARGETMACHINE_H
#define PPC_TARGETMACHINE_H

#include "PPCFrameLowering.h"
#include "PPCISelLowering.h"
#include "PPCInstrInfo.h"
#include "PPCJITInfo.h"
#include "PPCSelectionDAGInfo.h"
#include "PPCSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// PPCTargetMachine - Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
  PPCSubtarget        Subtarget;
  const DataLayout    DL;       // Calculates type size & alignment
  PPCInstrInfo        InstrInfo;
  PPCFrameLowering    FrameLowering;
  PPCJITInfo          JITInfo;
  PPCTargetLowering   TLInfo;
  PPCSelectionDAGInfo TSInfo;
  InstrItineraryData  InstrItins;

public:
  PPCTargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL, bool is64Bit);

  const PPCInstrInfo      *getInstrInfo() const override { return &InstrInfo; }
  const PPCFrameLowering  *getFrameLowering() const override {
    return &FrameLowering;
  }
        PPCJITInfo        *getJITInfo() override         { return &JITInfo; }
  const PPCTargetLowering *getTargetLowering() const override {
   return &TLInfo;
  }
  const PPCSelectionDAGInfo* getSelectionDAGInfo() const override {
    return &TSInfo;
  }
  const PPCRegisterInfo   *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  const DataLayout    *getDataLayout() const override    { return &DL; }
  const PPCSubtarget  *getSubtargetImpl() const override { return &Subtarget; }
  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  bool addCodeEmitter(PassManagerBase &PM,
                      JITCodeEmitter &JCE) override;

  /// \brief Register PPC analysis passes with a pass manager.
  void addAnalysisPasses(PassManagerBase &PM) override;
};

/// PPC32TargetMachine - PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC32TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

/// PPC64TargetMachine - PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC64TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
