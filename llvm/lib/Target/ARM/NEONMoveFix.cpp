//===-- NEONMoveFix.cpp - Convert vfp reg-reg moves into neon ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "neon-mov-fix"
#include "ARM.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumVMovs, "Number of reg-reg moves converted");

namespace {
  struct NEONMoveFixPass : public MachineFunctionPass {
    static char ID;
    NEONMoveFixPass() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "NEON reg-reg move conversion";
    }

  private:
    const TargetRegisterInfo *TRI;
    const ARMBaseInstrInfo *TII;
    bool isA8;

    typedef DenseMap<unsigned, const MachineInstr*> RegMap;

    bool InsertMoves(MachineBasicBlock &MBB);

    void TransferImpOps(MachineInstr &Old, MachineInstr &New);
  };
  char NEONMoveFixPass::ID = 0;
}

static bool inNEONDomain(unsigned Domain, bool isA8) {
  return (Domain & ARMII::DomainNEON) ||
    (isA8 && (Domain & ARMII::DomainNEONA8));
}

/// Transfer implicit kill and def operands from Old to New.
void NEONMoveFixPass::TransferImpOps(MachineInstr &Old, MachineInstr &New) {
  for (unsigned i = 0, e = Old.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = Old.getOperand(i);
    if (!MO.isReg() || !MO.isImplicit())
      continue;
    New.addOperand(MO);
  }
}

bool NEONMoveFixPass::InsertMoves(MachineBasicBlock &MBB) {
  RegMap Defs;
  bool Modified = false;

  // Walk over MBB tracking the def points of the registers.
  MachineBasicBlock::iterator MII = MBB.begin(), E = MBB.end();
  MachineBasicBlock::iterator NextMII;
  for (; MII != E; MII = NextMII) {
    NextMII = llvm::next(MII);
    MachineInstr *MI = &*MII;

    if (MI->getOpcode() == ARM::VMOVD &&
        !TII->isPredicated(MI)) {
      unsigned SrcReg = MI->getOperand(1).getReg();
      // If we do not find an instruction defining the reg, this means the
      // register should be live-in for this BB. It's always to better to use
      // NEON reg-reg moves.
      unsigned Domain = ARMII::DomainNEON;
      RegMap::iterator DefMI = Defs.find(SrcReg);
      if (DefMI != Defs.end()) {
        Domain = DefMI->second->getDesc().TSFlags & ARMII::DomainMask;
        // Instructions in general domain are subreg accesses.
        // Map them to NEON reg-reg moves.
        if (Domain == ARMII::DomainGeneral)
          Domain = ARMII::DomainNEON;
      }

      if (inNEONDomain(Domain, isA8)) {
        // Convert VMOVD to VORRd
        unsigned DestReg = MI->getOperand(0).getReg();

        DEBUG({errs() << "vmov convert: "; MI->dump();});

        // We need to preserve imp-defs / imp-uses here. Following passes may
        // use the register scavenger to update liveness.
        MachineInstr *NewMI =
          AddDefaultPred(BuildMI(MBB, *MI, MI->getDebugLoc(),
                                 TII->get(ARM::VORRd), DestReg)
                         .addReg(SrcReg).addReg(SrcReg));
        TransferImpOps(*MI, *NewMI);
        MBB.erase(MI);
        MI = NewMI;

        DEBUG({errs() << "        into: "; MI->dump();});

        Modified = true;
        ++NumVMovs;
      } else {
        assert((Domain & ARMII::DomainVFP) && "Invalid domain!");
        // Do nothing.
      }
    }

    // Update def information.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand& MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned MOReg = MO.getReg();

      Defs[MOReg] = MI;
      // Catch aliases as well.
      for (const unsigned *R = TRI->getAliasSet(MOReg); *R; ++R)
        Defs[*R] = MI;
    }
  }

  return Modified;
}

bool NEONMoveFixPass::runOnMachineFunction(MachineFunction &Fn) {
  ARMFunctionInfo *AFI = Fn.getInfo<ARMFunctionInfo>();
  const TargetMachine &TM = Fn.getTarget();

  if (AFI->isThumb1OnlyFunction())
    return false;

  TRI = TM.getRegisterInfo();
  TII = static_cast<const ARMBaseInstrInfo*>(TM.getInstrInfo());
  isA8 = TM.getSubtarget<ARMSubtarget>().isCortexA8();

  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= InsertMoves(MBB);
  }

  return Modified;
}

/// createNEONMoveFixPass - Returns an instance of the NEON reg-reg moves fix
/// pass.
FunctionPass *llvm::createNEONMoveFixPass() {
  return new NEONMoveFixPass();
}
