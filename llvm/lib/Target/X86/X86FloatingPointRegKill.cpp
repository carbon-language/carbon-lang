//===-- X86FloatingPoint.cpp - FP_REG_KILL inserter -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which inserts FP_REG_KILL instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-codegen"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Instructions.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumFPKill, "Number of FP_REG_KILL instructions added");

namespace {
  struct FPRegKiller : public MachineFunctionPass {
    static char ID;
    FPRegKiller() : MachineFunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "X86 FP_REG_KILL inserter";
    }
  };
  char FPRegKiller::ID = 0;
}

FunctionPass *llvm::createX87FPRegKillInserterPass() {
  return new FPRegKiller();
}

/// isFPStackVReg - Return true if the specified vreg is from a fp stack
/// register class.
static bool isFPStackVReg(unsigned RegNo, const MachineRegisterInfo &MRI) {
  if (!TargetRegisterInfo::isVirtualRegister(RegNo))
    return false;
  
  switch (MRI.getRegClass(RegNo)->getID()) {
  default: return false;
  case X86::RFP32RegClassID:
  case X86::RFP64RegClassID:
  case X86::RFP80RegClassID:
  return true;
  }
}


/// ContainsFPStackCode - Return true if the specific MBB has floating point
/// stack code, and thus needs an FP_REG_KILL.
static bool ContainsFPStackCode(MachineBasicBlock *MBB,
                                const MachineRegisterInfo &MRI) {
  // Scan the block, looking for instructions that define fp stack vregs.
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
       I != E; ++I) {
    if (I->getNumOperands() == 0 || !I->getOperand(0).isReg())
      continue;
    
    for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
      if (!I->getOperand(op).isReg() || !I->getOperand(op).isDef())
        continue;
      
      if (isFPStackVReg(I->getOperand(op).getReg(), MRI))
        return true;
    }
  }
  
  // Check PHI nodes in successor blocks.  These PHI's will be lowered to have
  // a copy of the input value in this block, which is a definition of the
  // value.
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
       E = MBB->succ_end(); SI != E; ++ SI) {
    MachineBasicBlock *SuccBB = *SI;
    for (MachineBasicBlock::iterator I = SuccBB->begin(), E = SuccBB->end();
         I != E; ++I) {
      // All PHI nodes are at the top of the block.
      if (!I->isPHI()) break;
      
      if (isFPStackVReg(I->getOperand(0).getReg(), MRI))
        return true;
    }
  }
  
  return false;
}                                 

bool FPRegKiller::runOnMachineFunction(MachineFunction &MF) {
  // If we are emitting FP stack code, scan the basic block to determine if this
  // block defines any FP values.  If so, put an FP_REG_KILL instruction before
  // the terminator of the block.

  // Note that FP stack instructions are used in all modes for long double,
  // so we always need to do this check.
  // Also note that it's possible for an FP stack register to be live across
  // an instruction that produces multiple basic blocks (SSE CMOV) so we
  // must check all the generated basic blocks.

  // Scan all of the machine instructions in these MBBs, checking for FP
  // stores.  (RFP32 and RFP64 will not exist in SSE mode, but RFP80 might.)

  // Fast-path: If nothing is using the x87 registers, we don't need to do
  // any scanning.
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  if (MRI.getRegClassVirtRegs(X86::RFP80RegisterClass).empty() &&
      MRI.getRegClassVirtRegs(X86::RFP64RegisterClass).empty() &&
      MRI.getRegClassVirtRegs(X86::RFP32RegisterClass).empty())
    return false;

  bool Changed = false;
  MachineFunction::iterator MBBI = MF.begin();
  MachineFunction::iterator EndMBB = MF.end();
  for (; MBBI != EndMBB; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    
    // If this block returns, ignore it.  We don't want to insert an FP_REG_KILL
    // before the return.
    if (!MBB->empty()) {
      MachineBasicBlock::iterator EndI = MBB->end();
      --EndI;
      if (EndI->getDesc().isReturn())
        continue;
    }
    
    // If we find any FP stack code, emit the FP_REG_KILL instruction.
    if (ContainsFPStackCode(MBB, MRI)) {
      BuildMI(*MBB, MBBI->getFirstTerminator(), DebugLoc(),
              MF.getTarget().getInstrInfo()->get(X86::FP_REG_KILL));
      ++NumFPKill;
      Changed = true;
    }
  }

  return Changed;
}
