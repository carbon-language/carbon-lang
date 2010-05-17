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
    NEONMoveFixPass() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "NEON reg-reg move conversion";
    }

  private:
    const TargetRegisterInfo *TRI;
    const ARMBaseInstrInfo *TII;

    typedef DenseMap<unsigned, const MachineInstr*> RegMap;

    bool InsertMoves(MachineBasicBlock &MBB);
  };
  char NEONMoveFixPass::ID = 0;
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

      if (Domain & ARMII::DomainNEON) {
        // Convert VMOVD to VMOVDneon
        unsigned DestReg = MI->getOperand(0).getReg();

        DEBUG({errs() << "vmov convert: "; MI->dump();});

        // It's safe to ignore imp-defs / imp-uses here, since:
        //  - We're running late, no intelligent condegen passes should be run
        //    afterwards
        //  - The imp-defs / imp-uses are superregs only, we don't care about
        //    them.
        AddDefaultPred(BuildMI(MBB, *MI, MI->getDebugLoc(),
                             TII->get(ARM::VMOVDneon), DestReg).addReg(SrcReg));
        MBB.erase(MI);
        MachineBasicBlock::iterator I = prior(NextMII);
        MI = &*I;

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
      // Catch subregs as well.
      for (const unsigned *R = TRI->getSubRegisters(MOReg); *R; ++R)
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
