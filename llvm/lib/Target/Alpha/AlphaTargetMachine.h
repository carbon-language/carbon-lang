//===-- AlphaTargetMachine.h - Define TargetMachine for PowerPC -*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the Alpha-specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHA_TARGETMACHINE_H
#define ALPHA_TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "AlphaInstrInfo.h"
//#include "AlphaJITInfo.h"

namespace llvm {

class GlobalValue;
class IntrinsicLowering;

class AlphaTargetMachine : public TargetMachine {
  AlphaInstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
  //  AlphaJITInfo JITInfo;

public:
  AlphaTargetMachine(const Module &M, IntrinsicLowering *IL);
  
  virtual const AlphaInstrInfo *getInstrInfo() const { return &InstrInfo; }    
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  //  virtual TargetJITInfo *getJITInfo() {
  //    return &JITInfo;
  //  }
 
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
					  MachineCodeEmitter &MCE);
  
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
  
};

} // end namespace llvm

#endif
