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

#ifndef AMDGPU_TARGET_MACHINE_H
#define AMDGPU_TARGET_MACHINE_H

#include "AMDGPUInstrInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDILFrameLowering.h"
#include "AMDILIntrinsicInfo.h"
#include "R600ISelLowering.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/DataLayout.h"

namespace llvm {

MCAsmInfo* createMCAsmInfo(const Target &T, StringRef TT);

class AMDGPUTargetMachine : public LLVMTargetMachine {

  AMDGPUSubtarget Subtarget;
  const DataLayout Layout;
  AMDGPUFrameLowering FrameLowering;
  AMDGPUIntrinsicInfo IntrinsicInfo;
  const AMDGPUInstrInfo * InstrInfo;
  AMDGPUTargetLowering * TLInfo;
  const InstrItineraryData* InstrItins;

public:
   AMDGPUTargetMachine(const Target &T, StringRef TT, StringRef FS,
                       StringRef CPU,
                       TargetOptions Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
   ~AMDGPUTargetMachine();
   virtual const AMDGPUFrameLowering* getFrameLowering() const {
     return &FrameLowering;
   }
   virtual const AMDGPUIntrinsicInfo* getIntrinsicInfo() const {
     return &IntrinsicInfo;
   }
   virtual const AMDGPUInstrInfo *getInstrInfo() const {return InstrInfo;}
   virtual const AMDGPUSubtarget *getSubtargetImpl() const {return &Subtarget; }
   virtual const AMDGPURegisterInfo *getRegisterInfo() const {
      return &InstrInfo->getRegisterInfo();
   }
   virtual AMDGPUTargetLowering * getTargetLowering() const {
      return TLInfo;
   }
   virtual const InstrItineraryData* getInstrItineraryData() const {
      return InstrItins;
   }
   virtual const DataLayout* getDataLayout() const { return &Layout; }
   virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
};

} // End namespace llvm

#endif // AMDGPU_TARGET_MACHINE_H
