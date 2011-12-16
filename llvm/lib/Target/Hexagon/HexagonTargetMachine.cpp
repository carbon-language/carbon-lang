//===-- HexagonTargetMachine.cpp - Define TargetMachine for Hexagon -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "Hexagon.h"
#include "HexagonISelLowering.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/TargetRegistry.h"
#include <iostream>

using namespace llvm;

static cl::
opt<bool> DisableHardwareLoops(
                        "disable-hexagon-hwloops", cl::Hidden,
                        cl::desc("Disable Hardware Loops for Hexagon target"));

/// HexagonTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int HexagonTargetMachineModule;
int HexagonTargetMachineModule = 0;

extern "C" void LLVMInitializeHexagonTarget() {
  // Register the target.
  RegisterTargetMachine<HexagonTargetMachine> X(TheHexagonTarget);
}


/// HexagonTargetMachine ctor - Create an ILP32 architecture model.
///

/// Hexagon_TODO: Do I need an aggregate alignment?
///
HexagonTargetMachine::HexagonTargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           TargetOptions Options,
                                           Reloc::Model RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    DataLayout("e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-a0:0") ,
    Subtarget(TT, CPU, FS), InstrInfo(Subtarget), TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget),
    InstrItins(&Subtarget.getInstrItineraryData()) {
  setMCUseCFI(false);
}

// addPassesForOptimizations - Allow the backend (target) to add Target
// Independent Optimization passes to the Pass Manager.
bool HexagonTargetMachine::addPassesForOptimizations(PassManagerBase &PM) {

  PM.add(createConstantPropagationPass());
  PM.add(createLoopSimplifyPass());
  PM.add(createDeadCodeEliminationPass());
  PM.add(createConstantPropagationPass());
  PM.add(createLoopUnrollPass());
  PM.add(createLoopStrengthReducePass(getTargetLowering()));
  return true;
}

bool HexagonTargetMachine::addInstSelector(PassManagerBase &PM) {
  PM.add(createHexagonRemoveExtendOps(*this));
  PM.add(createHexagonISelDag(*this));
  return false;
}


bool HexagonTargetMachine::addPreRegAlloc(PassManagerBase &PM) {
  if (!DisableHardwareLoops) {
    PM.add(createHexagonHardwareLoops());
  }

  return false;
}

bool HexagonTargetMachine::addPostRegAlloc(PassManagerBase &PM) {
  PM.add(createHexagonCFGOptimizer(*this));
  return true;
}


bool HexagonTargetMachine::addPreSched2(PassManagerBase &PM) {
  PM.add(createIfConverterPass());
  return true;
}

bool HexagonTargetMachine::addPreEmitPass(PassManagerBase &PM) {

  if (!DisableHardwareLoops) {
    PM.add(createHexagonFixupHwLoops());
  }

  // Expand Spill code for predicate registers.
  PM.add(createHexagonExpandPredSpillCode(*this));

  // Split up TFRcondsets into conditional transfers.
  PM.add(createHexagonSplitTFRCondSets(*this));

  return false;
}
