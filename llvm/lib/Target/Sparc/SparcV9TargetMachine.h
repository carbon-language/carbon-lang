//===-- SparcTargetMachine.h - Define TargetMachine for Sparc ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the top-level UltraSPARC target machine.
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_TARGETMACHINE_H
#define SPARC_TARGETMACHINE_H

#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "SparcInstrInfo.h"
#include "SparcInternals.h"
#include "SparcRegInfo.h"
#include "SparcFrameInfo.h"
#include "SparcJITInfo.h"

namespace llvm {
  class PassManager;

class SparcTargetMachine : public TargetMachine {
  SparcInstrInfo instrInfo;
  SparcSchedInfo schedInfo;
  SparcRegInfo   regInfo;
  SparcFrameInfo frameInfo;
  SparcCacheInfo cacheInfo;
  SparcJITInfo   jitInfo;
public:
  SparcTargetMachine(IntrinsicLowering *IL);
  
  virtual const TargetInstrInfo  &getInstrInfo() const { return instrInfo; }
  virtual const TargetSchedInfo  &getSchedInfo() const { return schedInfo; }
  virtual const TargetRegInfo    &getRegInfo()   const { return regInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return frameInfo; }
  virtual const TargetCacheInfo  &getCacheInfo() const { return cacheInfo; }
  virtual       TargetJITInfo    *getJITInfo()         { return &jitInfo; }

  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);
};

} // End llvm namespace

#endif
