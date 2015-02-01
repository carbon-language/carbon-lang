//===-- AMDGPUTargetMachine.h - AMDGPU TargetMachine Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief The AMDGPU TargetMachine interface definition for hw codgen targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_AMDGPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_R600_AMDGPUTARGETMACHINE_H

#include "AMDGPUFrameLowering.h"
#include "AMDGPUInstrInfo.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "R600ISelLowering.h"
#include "llvm/IR/DataLayout.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// AMDGPU Target Machine (R600+)
//===----------------------------------------------------------------------===//

class AMDGPUTargetMachine : public LLVMTargetMachine {
private:
  const DataLayout DL;

protected:
  TargetLoweringObjectFile *TLOF;
  AMDGPUSubtarget Subtarget;
  AMDGPUIntrinsicInfo IntrinsicInfo;

public:
  AMDGPUTargetMachine(const Target &T, StringRef TT, StringRef FS,
                      StringRef CPU, TargetOptions Options, Reloc::Model RM,
                      CodeModel::Model CM, CodeGenOpt::Level OL);
  ~AMDGPUTargetMachine();
  // FIXME: This is currently broken, the DataLayout needs to move to
  // the target machine.
  const DataLayout *getDataLayout() const override {
    return &DL;
  }
  const AMDGPUSubtarget *getSubtargetImpl() const override {
    return &Subtarget;
  }
  const AMDGPUIntrinsicInfo *getIntrinsicInfo() const override {
    return &IntrinsicInfo;
  }
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetIRAnalysis getTargetIRAnalysis() override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF;
  }
};

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

class GCNTargetMachine : public AMDGPUTargetMachine {

public:
  GCNTargetMachine(const Target &T, StringRef TT, StringRef FS,
                    StringRef CPU, TargetOptions Options, Reloc::Model RM,
                    CodeModel::Model CM, CodeGenOpt::Level OL);
};

} // End namespace llvm

#endif
