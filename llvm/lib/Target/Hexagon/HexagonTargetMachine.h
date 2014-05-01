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

#include "HexagonFrameLowering.h"
#include "HexagonISelLowering.h"
#include "HexagonInstrInfo.h"
#include "HexagonSelectionDAGInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class Module;

class HexagonTargetMachine : public LLVMTargetMachine {
  const DataLayout DL;       // Calculates type size & alignment.
  HexagonSubtarget Subtarget;
  HexagonInstrInfo InstrInfo;
  HexagonTargetLowering TLInfo;
  HexagonSelectionDAGInfo TSInfo;
  HexagonFrameLowering FrameLowering;
  const InstrItineraryData* InstrItins;

public:
  HexagonTargetMachine(const Target &T, StringRef TT,StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  const HexagonInstrInfo *getInstrInfo() const override {
    return &InstrInfo;
  }
  const HexagonSubtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  const HexagonRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  const InstrItineraryData* getInstrItineraryData() const override {
    return InstrItins;
  }


  const HexagonTargetLowering* getTargetLowering() const override {
    return &TLInfo;
  }

  const HexagonFrameLowering* getFrameLowering() const override {
    return &FrameLowering;
  }

  const HexagonSelectionDAGInfo* getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  const DataLayout       *getDataLayout() const override { return &DL; }
  static unsigned getModuleMatchQuality(const Module &M);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

extern bool flag_aligned_memcpy;

} // end namespace llvm

#endif
