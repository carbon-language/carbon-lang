//===-- SPUTargetMachine.cpp - Define TargetMachine for Cell SPU ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the Cell SPU target.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPUTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

extern "C" void LLVMInitializeCellSPUTarget() { 
  // Register the target.
  RegisterTargetMachine<SPUTargetMachine> X(TheCellSPUTarget);
}

const std::pair<unsigned, int> *
SPUFrameLowering::getCalleeSaveSpillSlots(unsigned &NumEntries) const {
  NumEntries = 1;
  return &LR[0];
}

SPUTargetMachine::SPUTargetMachine(const Target &T, StringRef TT,
                                   StringRef CPU,StringRef FS, Reloc::Model RM)
  : LLVMTargetMachine(T, TT, CPU, FS, RM),
    Subtarget(TT, CPU, FS),
    DataLayout(Subtarget.getTargetDataString()),
    InstrInfo(*this),
    FrameLowering(Subtarget),
    TLInfo(*this),
    TSInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool SPUTargetMachine::addInstSelector(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createSPUISelDag(*this));
  return false;
}

// passes to run just before printing the assembly
bool SPUTargetMachine::
addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel) 
{
  //align instructions with nops/lnops for dual issue
  PM.add(createSPUNopFillerPass(*this));
  return true;
}
