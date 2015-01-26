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

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONTARGETMACHINE_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONTARGETMACHINE_H

#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class Module;

class HexagonTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  const DataLayout DL; // Calculates type size & alignment.
  HexagonSubtarget Subtarget;

public:
  HexagonTargetMachine(const Target &T, StringRef TT,StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
  ~HexagonTargetMachine() override;
  const DataLayout *getDataLayout() const override { return &DL; }
  const HexagonSubtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  static unsigned getModuleMatchQuality(const Module &M);

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};

extern bool flag_aligned_memcpy;

} // end namespace llvm

#endif
