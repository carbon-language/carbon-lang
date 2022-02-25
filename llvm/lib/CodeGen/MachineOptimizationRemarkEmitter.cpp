///===- MachineOptimizationRemarkEmitter.cpp - Opt Diagnostic -*- C++ -*---===//
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
/// See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===---------------------------------------------------------------------===//
/// \file
/// Optimization diagnostic interfaces for machine passes.  It's packaged as an
/// analysis pass so that by using this service passes become dependent on MBFI
/// as well.  MBFI is used to compute the "hotness" of the diagnostic message.
///
///===---------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/LazyMachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

DiagnosticInfoMIROptimization::MachineArgument::MachineArgument(
    StringRef MKey, const MachineInstr &MI) {
  Key = std::string(MKey);

  raw_string_ostream OS(Val);
  MI.print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
           /*SkipDebugLoc=*/true);
}

Optional<uint64_t>
MachineOptimizationRemarkEmitter::computeHotness(const MachineBasicBlock &MBB) {
  if (!MBFI)
    return None;

  return MBFI->getBlockProfileCount(&MBB);
}

void MachineOptimizationRemarkEmitter::computeHotness(
    DiagnosticInfoMIROptimization &Remark) {
  const MachineBasicBlock *MBB = Remark.getBlock();
  if (MBB)
    Remark.setHotness(computeHotness(*MBB));
}

void MachineOptimizationRemarkEmitter::emit(
    DiagnosticInfoOptimizationBase &OptDiagCommon) {
  auto &OptDiag = cast<DiagnosticInfoMIROptimization>(OptDiagCommon);
  computeHotness(OptDiag);

  LLVMContext &Ctx = MF.getFunction().getContext();

  // Only emit it if its hotness meets the threshold.
  if (OptDiag.getHotness().getValueOr(0) <
      Ctx.getDiagnosticsHotnessThreshold()) {
    return;
  }

  Ctx.diagnose(OptDiag);
}

MachineOptimizationRemarkEmitterPass::MachineOptimizationRemarkEmitterPass()
    : MachineFunctionPass(ID) {
  initializeMachineOptimizationRemarkEmitterPassPass(
      *PassRegistry::getPassRegistry());
}

bool MachineOptimizationRemarkEmitterPass::runOnMachineFunction(
    MachineFunction &MF) {
  MachineBlockFrequencyInfo *MBFI;

  if (MF.getFunction().getContext().getDiagnosticsHotnessRequested())
    MBFI = &getAnalysis<LazyMachineBlockFrequencyInfoPass>().getBFI();
  else
    MBFI = nullptr;

  ORE = std::make_unique<MachineOptimizationRemarkEmitter>(MF, MBFI);
  return false;
}

void MachineOptimizationRemarkEmitterPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<LazyMachineBlockFrequencyInfoPass>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

char MachineOptimizationRemarkEmitterPass::ID = 0;
static const char ore_name[] = "Machine Optimization Remark Emitter";
#define ORE_NAME "machine-opt-remark-emitter"

INITIALIZE_PASS_BEGIN(MachineOptimizationRemarkEmitterPass, ORE_NAME, ore_name,
                      true, true)
INITIALIZE_PASS_DEPENDENCY(LazyMachineBlockFrequencyInfoPass)
INITIALIZE_PASS_END(MachineOptimizationRemarkEmitterPass, ORE_NAME, ore_name,
                    true, true)
