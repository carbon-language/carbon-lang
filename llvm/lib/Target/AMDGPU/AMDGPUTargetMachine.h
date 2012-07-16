//===-- AMDGPUTargetMachine.h - AMDGPU TargetMachine Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  The AMDGPU TargetMachine interface definition for hw codgen targets.
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
#include "llvm/Target/TargetData.h"

namespace llvm {

MCAsmInfo* createMCAsmInfo(const Target &T, StringRef TT);

class AMDGPUTargetMachine : public LLVMTargetMachine {

  AMDGPUSubtarget Subtarget;
  const TargetData DataLayout;
  AMDILFrameLowering FrameLowering;
  AMDILIntrinsicInfo IntrinsicInfo;
  const AMDGPUInstrInfo * InstrInfo;
  AMDGPUTargetLowering * TLInfo;
  const InstrItineraryData* InstrItins;
  bool mDump;

public:
   AMDGPUTargetMachine(const Target &T, StringRef TT, StringRef FS,
                       StringRef CPU,
                       TargetOptions Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
   ~AMDGPUTargetMachine();
   virtual const AMDILFrameLowering* getFrameLowering() const {
     return &FrameLowering;
   }
   virtual const AMDILIntrinsicInfo* getIntrinsicInfo() const {
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
   virtual const TargetData* getTargetData() const { return &DataLayout; }
   virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
   virtual bool addPassesToEmitFile(PassManagerBase &PM,
                                              formatted_raw_ostream &Out,
                                              CodeGenFileType FileType,
                                              bool DisableVerify,
                                              AnalysisID StartAfter = 0,
                                              AnalysisID StopAfter = 0);
};

} // End namespace llvm

#endif // AMDGPU_TARGET_MACHINE_H
