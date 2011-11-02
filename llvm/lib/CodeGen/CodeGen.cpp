//===-- CodeGen.cpp -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the common initialization routines for the
// CodeGen library.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm-c/Initialization.h"

using namespace llvm;

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void llvm::initializeCodeGen(PassRegistry &Registry) {
  initializeCalculateSpillWeightsPass(Registry);
  initializeDeadMachineInstructionElimPass(Registry);
  initializeGCModuleInfoPass(Registry);
  initializeIfConverterPass(Registry);
  initializeLiveDebugVariablesPass(Registry);
  initializeLiveIntervalsPass(Registry);
  initializeLiveStacksPass(Registry);
  initializeLiveVariablesPass(Registry);
  initializeMachineBlockFrequencyInfoPass(Registry);
  initializeMachineBlockPlacementPass(Registry);
  initializeMachineBlockPlacementStatsPass(Registry);
  initializeMachineCSEPass(Registry);
  initializeMachineDominatorTreePass(Registry);
  initializeMachineLICMPass(Registry);
  initializeMachineLoopInfoPass(Registry);
  initializeMachineModuleInfoPass(Registry);
  initializeMachineSinkingPass(Registry);
  initializeMachineVerifierPassPass(Registry);
  initializeOptimizePHIsPass(Registry);
  initializePHIEliminationPass(Registry);
  initializePeepholeOptimizerPass(Registry);
  initializeProcessImplicitDefsPass(Registry);
  initializePEIPass(Registry);
  initializeRALinScanPass(Registry);
  initializeRegisterCoalescerPass(Registry);
  initializeRenderMachineFunctionPass(Registry);
  initializeSlotIndexesPass(Registry);
  initializeLoopSplitterPass(Registry);
  initializeStackProtectorPass(Registry);
  initializeStackSlotColoringPass(Registry);
  initializeStrongPHIEliminationPass(Registry);
  initializeTwoAddressInstructionPassPass(Registry);
  initializeUnreachableBlockElimPass(Registry);
  initializeUnreachableMachineBlockElimPass(Registry);
  initializeVirtRegMapPass(Registry);
  initializeLowerIntrinsicsPass(Registry);
}

void LLVMInitializeCodeGen(LLVMPassRegistryRef R) {
  initializeCodeGen(*unwrap(R));
}
