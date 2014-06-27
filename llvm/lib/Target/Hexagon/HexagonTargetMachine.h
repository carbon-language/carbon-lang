//=-- HexagonTargetMachine.h - Define TargetMachine for Hexagon ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Hexagon specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonTARGETMACHINE_H
#define HexagonTARGETMACHINE_H

#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class Module;

class HexagonTargetMachine : public LLVMTargetMachine {
  HexagonSubtarget Subtarget;

public:
  HexagonTargetMachine(const Target &T, StringRef TT,StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  const HexagonInstrInfo *getInstrInfo() const override {
    return getSubtargetImpl()->getInstrInfo();
  }
  const HexagonSubtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  const HexagonRegisterInfo *getRegisterInfo() const override {
    return getSubtargetImpl()->getRegisterInfo();
  }
  const InstrItineraryData* getInstrItineraryData() const override {
    return &getSubtargetImpl()->getInstrItineraryData();
  }
  const HexagonTargetLowering* getTargetLowering() const override {
    return getSubtargetImpl()->getTargetLowering();
  }
  const HexagonFrameLowering* getFrameLowering() const override {
    return getSubtargetImpl()->getFrameLowering();
  }
  const HexagonSelectionDAGInfo* getSelectionDAGInfo() const override {
    return getSubtargetImpl()->getSelectionDAGInfo();
  }
  const DataLayout *getDataLayout() const override {
    return getSubtargetImpl()->getDataLayout();
  }
  static unsigned getModuleMatchQuality(const Module &M);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

extern bool flag_aligned_memcpy;

} // end namespace llvm

#endif
