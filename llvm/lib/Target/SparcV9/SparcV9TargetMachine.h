//===-- SparcV9TargetMachine.h - Define TargetMachine for SparcV9 ---*- C++ -*-===//
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
#include "SparcV9InstrInfo.h"
#include "SparcV9Internals.h"
#include "SparcV9RegInfo.h"
#include "SparcV9FrameInfo.h"
#include "SparcV9JITInfo.h"

namespace llvm {
  class PassManager;

class SparcV9TargetMachine : public TargetMachine {
  SparcV9InstrInfo instrInfo;
  SparcV9SchedInfo schedInfo;
  SparcV9RegInfo   regInfo;
  SparcV9FrameInfo frameInfo;
  SparcV9JITInfo   jitInfo;
public:
  SparcV9TargetMachine(IntrinsicLowering *IL);
  
  virtual const TargetInstrInfo  &getInstrInfo() const { return instrInfo; }
  virtual const TargetSchedInfo  &getSchedInfo() const { return schedInfo; }
  virtual const TargetRegInfo    &getRegInfo()   const { return regInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return frameInfo; }
  virtual       TargetJITInfo    *getJITInfo()         { return &jitInfo; }

  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);
};

} // End llvm namespace

#endif
