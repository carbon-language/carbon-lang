//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_TARGETMACHINE_H
#define PPC_TARGETMACHINE_H

#include "PPCFrameInfo.h"
#include "PPCSubtarget.h"
#include "PPCJITInfo.h"
#include "PPCInstrInfo.h"
#include "PPCISelLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {
class PassManager;
class GlobalValue;

class PPCTargetMachine : public TargetMachine {
  const TargetData DataLayout;       // Calculates type size & alignment
  PPCInstrInfo           InstrInfo;
  PPCSubtarget           Subtarget;
  PPCFrameInfo           FrameInfo;
  PPCJITInfo             JITInfo;
  PPCTargetLowering      TLInfo;
  InstrItineraryData     InstrItins;
public:
  PPCTargetMachine(const Module &M, const std::string &FS);

  virtual const PPCInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual       TargetJITInfo    *getJITInfo()         { return &JITInfo; }
  virtual const TargetSubtarget  *getSubtargetImpl() const{ return &Subtarget; }
  virtual       PPCTargetLowering *getTargetLowering() { return &TLInfo; }
  virtual const MRegisterInfo    *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItins;
  }
  

  static unsigned getJITMatchQuality();

  static unsigned getModuleMatchQuality(const Module &M);
  
  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType, bool Fast);
  
  bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                  MachineCodeEmitter &MCE);
};
  
} // end namespace llvm

#endif
