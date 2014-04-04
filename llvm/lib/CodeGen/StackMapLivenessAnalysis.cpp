//===-- StackMapLivenessAnalysis.cpp - StackMap live Out Analysis ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StackMap Liveness analysis pass. The pass calculates
// the liveness for each basic block in a function and attaches the register
// live-out information to a stackmap or patchpoint intrinsic if present.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "stackmaps"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/StackMapLivenessAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"


using namespace llvm;

namespace llvm {
cl::opt<bool> EnableStackMapLiveness("enable-stackmap-liveness",
  cl::Hidden, cl::desc("Enable StackMap Liveness Analysis Pass"));
cl::opt<bool> EnablePatchPointLiveness("enable-patchpoint-liveness",
  cl::Hidden, cl::desc("Enable PatchPoint Liveness Analysis Pass"));
}

STATISTIC(NumStackMapFuncVisited, "Number of functions visited");
STATISTIC(NumStackMapFuncSkipped, "Number of functions skipped");
STATISTIC(NumBBsVisited,          "Number of basic blocks visited");
STATISTIC(NumBBsHaveNoStackmap,   "Number of basic blocks with no stackmap");
STATISTIC(NumStackMaps,           "Number of StackMaps visited");

char StackMapLiveness::ID = 0;
char &llvm::StackMapLivenessID = StackMapLiveness::ID;
INITIALIZE_PASS(StackMapLiveness, "stackmap-liveness",
                "StackMap Liveness Analysis", false, false)

/// Default construct and initialize the pass.
StackMapLiveness::StackMapLiveness() : MachineFunctionPass(ID) {
  initializeStackMapLivenessPass(*PassRegistry::getPassRegistry());
}

/// Tell the pass manager which passes we depend on and what information we
/// preserve.
void StackMapLiveness::getAnalysisUsage(AnalysisUsage &AU) const {
  // We preserve all information.
  AU.setPreservesAll();
  AU.setPreservesCFG();
  // Default dependencie for all MachineFunction passes.
  AU.addRequired<MachineFunctionAnalysis>();
}

/// Calculate the liveness information for the given machine function.
bool StackMapLiveness::runOnMachineFunction(MachineFunction &_MF) {
  DEBUG(dbgs() << "********** COMPUTING STACKMAP LIVENESS: "
               << _MF.getName() << " **********\n");
  MF = &_MF;
  TRI = MF->getTarget().getRegisterInfo();
  ++NumStackMapFuncVisited;

  // Skip this function if there are no stackmaps or patchpoints to process.
  if (!((MF->getFrameInfo()->hasStackMap() && EnableStackMapLiveness) ||
        (MF->getFrameInfo()->hasPatchPoint() && EnablePatchPointLiveness))) {
    ++NumStackMapFuncSkipped;
    return false;
  }
  return calculateLiveness();
}

/// Performs the actual liveness calculation for the function.
bool StackMapLiveness::calculateLiveness() {
  bool HasChanged = false;
  // For all basic blocks in the function.
  for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
       MBBI != MBBE; ++MBBI) {
    DEBUG(dbgs() << "****** BB " << MBBI->getName() << " ******\n");
    LiveRegs.init(TRI);
    LiveRegs.addLiveOuts(MBBI);
    bool HasStackMap = false;
    // Reverse iterate over all instructions and add the current live register
    // set to an instruction if we encounter a stackmap or patchpoint
    // instruction.
    for (MachineBasicBlock::reverse_iterator I = MBBI->rbegin(),
         E = MBBI->rend(); I != E; ++I) {
      int Opc = I->getOpcode();
      if ((EnableStackMapLiveness && (Opc == TargetOpcode::STACKMAP)) ||
          (EnablePatchPointLiveness && (Opc == TargetOpcode::PATCHPOINT))) {
        addLiveOutSetToMI(*I);
        HasChanged = true;
        HasStackMap = true;
        ++NumStackMaps;
      }
      DEBUG(dbgs() << "   " << LiveRegs << "   " << *I);
      LiveRegs.stepBackward(*I);
    }
    ++NumBBsVisited;
    if (!HasStackMap)
      ++NumBBsHaveNoStackmap;
  }
  return HasChanged;
}

/// Add the current register live set to the instruction.
void StackMapLiveness::addLiveOutSetToMI(MachineInstr &MI) {
  uint32_t *Mask = createRegisterMask();
  MachineOperand MO = MachineOperand::CreateRegLiveOut(Mask);
  MI.addOperand(*MF, MO);
}

/// Create a register mask and initialize it with the registers from the
/// register live set.
uint32_t *StackMapLiveness::createRegisterMask() const {
  // The mask is owned and cleaned up by the Machine Function.
  uint32_t *Mask = MF->allocateRegisterMask(TRI->getNumRegs());
  for (LivePhysRegs::const_iterator RI = LiveRegs.begin(), RE = LiveRegs.end();
       RI != RE; ++RI)
    Mask[*RI / 32] |= 1U << (*RI % 32);
  return Mask;
}
