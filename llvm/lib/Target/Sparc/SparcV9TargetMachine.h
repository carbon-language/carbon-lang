//===-- SparcTargetMachine.h - Define TargetMachine for Sparc ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the primary interface to machine description for the
// UltraSPARC.
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_TARGETMACHINE_H
#define SPARC_TARGETMACHINE_H

#include "llvm/PassManager.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "SparcInstrInfo.h"
#include "SparcInternals.h"
#include "SparcRegInfo.h"
#include "SparcFrameInfo.h"

namespace llvm {

class SparcTargetMachine : public TargetMachine {
  SparcInstrInfo instrInfo;
  SparcSchedInfo schedInfo;
  SparcRegInfo   regInfo;
  SparcFrameInfo frameInfo;
  SparcCacheInfo cacheInfo;
public:
  SparcTargetMachine();

  virtual const TargetInstrInfo  &getInstrInfo() const { return instrInfo; }
  virtual const TargetSchedInfo  &getSchedInfo() const { return schedInfo; }
  virtual const TargetRegInfo    &getRegInfo()   const { return regInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return frameInfo; }
  virtual const TargetCacheInfo  &getCacheInfo() const { return cacheInfo; }

  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
  virtual bool addPassesToJITCompile(FunctionPassManager &PM);
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);
  virtual void replaceMachineCodeForFunction(void *Old, void *New);

  /// getJITStubForFunction - Create or return a stub for the specified
  /// function.  This stub acts just like the specified function, except that it
  /// allows the "address" of the function to be taken without having to
  /// generate code for it.
  ///
  ///virtual void *getJITStubForFunction(Function *F, MachineCodeEmitter &MCE);
};

} // End llvm namespace

#endif
