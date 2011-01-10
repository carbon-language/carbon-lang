//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC -----*- C++ -*-=//
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
#include "PPCSubtarget.h"
#include "PPCJITInfo.h"
#include "PPCInstrInfo.h"
#include "PPCISelLowering.h"
#include "PPCSelectionDAGInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {
class PassManager;
class GlobalValue;

/// PPCTargetMachine - Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
  PPCSubtarget        Subtarget;
  const TargetData    DataLayout;       // Calculates type size & alignment
  PPCInstrInfo        InstrInfo;
  PPCFrameLowering    FrameLowering;
  PPCJITInfo          JITInfo;
  PPCTargetLowering   TLInfo;
  PPCSelectionDAGInfo TSInfo;
  InstrItineraryData  InstrItins;

public:
  PPCTargetMachine(const Target &T, const std::string &TT,
                   const std::string &FS, bool is64Bit);

  virtual const PPCInstrInfo      *getInstrInfo() const { return &InstrInfo; }
  virtual const PPCFrameLowering  *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual       PPCJITInfo        *getJITInfo()         { return &JITInfo; }
  virtual const PPCTargetLowering *getTargetLowering() const { 
   return &TLInfo;
  }
  virtual const PPCSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const PPCRegisterInfo   *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  
  virtual const TargetData    *getTargetData() const    { return &DataLayout; }
  virtual const PPCSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual const InstrItineraryData *getInstrItineraryData() const {  
    return &InstrItins;
  }

  // Pass Pipeline Configuration
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              JITCodeEmitter &JCE);
  virtual bool getEnableTailMergeDefault() const;
};

/// PPC32TargetMachine - PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
public:
  PPC32TargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);
};

/// PPC64TargetMachine - PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
public:
  PPC64TargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);
};

} // end namespace llvm

#endif
