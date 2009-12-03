//===-- LowerSubregs.cpp - Subregister Lowering instruction pass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a MachineFunction pass which runs after register
// allocation that turns subreg insert/extract instructions into register
// copies, as needed. This ensures correct codegen even if the coalescer
// isn't able to remove all subreg instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lowersubregs"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct LowerSubregsInstructionPass : public MachineFunctionPass {
  private:
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;

  public:
    static char ID; // Pass identification, replacement for typeid
    LowerSubregsInstructionPass() : MachineFunctionPass(&ID) {}
    
    const char *getPassName() const {
      return "Subregister lowering instruction pass";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - pass entry point
    bool runOnMachineFunction(MachineFunction&);

  private:
    bool LowerExtract(MachineInstr *MI);
    bool LowerInsert(MachineInstr *MI);
    bool LowerSubregToReg(MachineInstr *MI);

    void TransferDeadFlag(MachineInstr *MI, unsigned DstReg,
                          const TargetRegisterInfo *TRI);
    void TransferKillFlag(MachineInstr *MI, unsigned SrcReg,
                          const TargetRegisterInfo *TRI,
                          bool AddIfNotFound = false);
  };

  char LowerSubregsInstructionPass::ID = 0;
}

FunctionPass *llvm::createLowerSubregsPass() { 
  return new LowerSubregsInstructionPass(); 
}

/// TransferDeadFlag - MI is a pseudo-instruction with DstReg dead,
/// and the lowered replacement instructions immediately precede it.
/// Mark the replacement instructions with the dead flag.
void
LowerSubregsInstructionPass::TransferDeadFlag(MachineInstr *MI,
                                              unsigned DstReg,
                                              const TargetRegisterInfo *TRI) {
  for (MachineBasicBlock::iterator MII =
        prior(MachineBasicBlock::iterator(MI)); ; --MII) {
    if (MII->addRegisterDead(DstReg, TRI))
      break;
    assert(MII != MI->getParent()->begin() &&
           "copyRegToReg output doesn't reference destination register!");
  }
}

/// TransferKillFlag - MI is a pseudo-instruction with SrcReg killed,
/// and the lowered replacement instructions immediately precede it.
/// Mark the replacement instructions with the kill flag.
void
LowerSubregsInstructionPass::TransferKillFlag(MachineInstr *MI,
                                              unsigned SrcReg,
                                              const TargetRegisterInfo *TRI,
                                              bool AddIfNotFound) {
  for (MachineBasicBlock::iterator MII =
        prior(MachineBasicBlock::iterator(MI)); ; --MII) {
    if (MII->addRegisterKilled(SrcReg, TRI, AddIfNotFound))
      break;
    assert(MII != MI->getParent()->begin() &&
           "copyRegToReg output doesn't reference source register!");
  }
}

bool LowerSubregsInstructionPass::LowerExtract(MachineInstr *MI) {
  MachineBasicBlock *MBB = MI->getParent();

  assert(MI->getOperand(0).isReg() && MI->getOperand(0).isDef() &&
         MI->getOperand(1).isReg() && MI->getOperand(1).isUse() &&
         MI->getOperand(2).isImm() && "Malformed extract_subreg");

  unsigned DstReg   = MI->getOperand(0).getReg();
  unsigned SuperReg = MI->getOperand(1).getReg();
  unsigned SubIdx   = MI->getOperand(2).getImm();
  unsigned SrcReg   = TRI->getSubReg(SuperReg, SubIdx);

  assert(TargetRegisterInfo::isPhysicalRegister(SuperReg) &&
         "Extract supperg source must be a physical register");
  assert(TargetRegisterInfo::isPhysicalRegister(DstReg) &&
         "Extract destination must be in a physical register");
  assert(SrcReg && "invalid subregister index for register");

  DEBUG(errs() << "subreg: CONVERTING: " << *MI);

  if (SrcReg == DstReg) {
    // No need to insert an identity copy instruction.
    if (MI->getOperand(1).isKill()) {
      // We must make sure the super-register gets killed. Replace the
      // instruction with KILL.
      MI->setDesc(TII->get(TargetInstrInfo::KILL));
      MI->RemoveOperand(2);     // SubIdx
      DEBUG(errs() << "subreg: replace by: " << *MI);
      return true;
    }

    DEBUG(errs() << "subreg: eliminated!");
  } else {
    // Insert copy
    const TargetRegisterClass *TRCS = TRI->getPhysicalRegisterRegClass(DstReg);
    const TargetRegisterClass *TRCD = TRI->getPhysicalRegisterRegClass(SrcReg);
    bool Emitted = TII->copyRegToReg(*MBB, MI, DstReg, SrcReg, TRCD, TRCS);
    (void)Emitted;
    assert(Emitted && "Subreg and Dst must be of compatible register class");
    // Transfer the kill/dead flags, if needed.
    if (MI->getOperand(0).isDead())
      TransferDeadFlag(MI, DstReg, TRI);
    if (MI->getOperand(1).isKill())
      TransferKillFlag(MI, SuperReg, TRI, true);
    DEBUG({
        MachineBasicBlock::iterator dMI = MI;
        errs() << "subreg: " << *(--dMI);
      });
  }

  DEBUG(errs() << '\n');
  MBB->erase(MI);
  return true;
}

bool LowerSubregsInstructionPass::LowerSubregToReg(MachineInstr *MI) {
  MachineBasicBlock *MBB = MI->getParent();
  assert((MI->getOperand(0).isReg() && MI->getOperand(0).isDef()) &&
         MI->getOperand(1).isImm() &&
         (MI->getOperand(2).isReg() && MI->getOperand(2).isUse()) &&
          MI->getOperand(3).isImm() && "Invalid subreg_to_reg");
          
  unsigned DstReg  = MI->getOperand(0).getReg();
  unsigned InsReg  = MI->getOperand(2).getReg();
  unsigned InsSIdx = MI->getOperand(2).getSubReg();
  unsigned SubIdx  = MI->getOperand(3).getImm();

  assert(SubIdx != 0 && "Invalid index for insert_subreg");
  unsigned DstSubReg = TRI->getSubReg(DstReg, SubIdx);

  assert(TargetRegisterInfo::isPhysicalRegister(DstReg) &&
         "Insert destination must be in a physical register");
  assert(TargetRegisterInfo::isPhysicalRegister(InsReg) &&
         "Inserted value must be in a physical register");

  DEBUG(errs() << "subreg: CONVERTING: " << *MI);

  if (DstSubReg == InsReg && InsSIdx == 0) {
    // No need to insert an identify copy instruction.
    // Watch out for case like this:
    // %RAX<def> = ...
    // %RAX<def> = SUBREG_TO_REG 0, %EAX:3<kill>, 3
    // The first def is defining RAX, not EAX so the top bits were not
    // zero extended.
    DEBUG(errs() << "subreg: eliminated!");
  } else {
    // Insert sub-register copy
    const TargetRegisterClass *TRC0= TRI->getPhysicalRegisterRegClass(DstSubReg);
    const TargetRegisterClass *TRC1= TRI->getPhysicalRegisterRegClass(InsReg);
    bool Emitted = TII->copyRegToReg(*MBB, MI, DstSubReg, InsReg, TRC0, TRC1);
    (void)Emitted;
    assert(Emitted && "Subreg and Dst must be of compatible register class");
    // Transfer the kill/dead flags, if needed.
    if (MI->getOperand(0).isDead())
      TransferDeadFlag(MI, DstSubReg, TRI);
    if (MI->getOperand(2).isKill())
      TransferKillFlag(MI, InsReg, TRI);
    DEBUG({
        MachineBasicBlock::iterator dMI = MI;
        errs() << "subreg: " << *(--dMI);
      });
  }

  DEBUG(errs() << '\n');
  MBB->erase(MI);
  return true;
}

bool LowerSubregsInstructionPass::LowerInsert(MachineInstr *MI) {
  MachineBasicBlock *MBB = MI->getParent();
  assert((MI->getOperand(0).isReg() && MI->getOperand(0).isDef()) &&
         (MI->getOperand(1).isReg() && MI->getOperand(1).isUse()) &&
         (MI->getOperand(2).isReg() && MI->getOperand(2).isUse()) &&
          MI->getOperand(3).isImm() && "Invalid insert_subreg");
          
  unsigned DstReg = MI->getOperand(0).getReg();
#ifndef NDEBUG
  unsigned SrcReg = MI->getOperand(1).getReg();
#endif
  unsigned InsReg = MI->getOperand(2).getReg();
  unsigned SubIdx = MI->getOperand(3).getImm();     

  assert(DstReg == SrcReg && "insert_subreg not a two-address instruction?");
  assert(SubIdx != 0 && "Invalid index for insert_subreg");
  unsigned DstSubReg = TRI->getSubReg(DstReg, SubIdx);
  assert(DstSubReg && "invalid subregister index for register");
  assert(TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
         "Insert superreg source must be in a physical register");
  assert(TargetRegisterInfo::isPhysicalRegister(InsReg) &&
         "Inserted value must be in a physical register");

  DEBUG(errs() << "subreg: CONVERTING: " << *MI);

  if (DstSubReg == InsReg) {
    // No need to insert an identity copy instruction. If the SrcReg was
    // <undef>, we need to make sure it is alive by inserting a KILL
    if (MI->getOperand(1).isUndef() && !MI->getOperand(0).isDead()) {
      MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                TII->get(TargetInstrInfo::KILL), DstReg);
      if (MI->getOperand(2).isUndef())
        MIB.addReg(InsReg, RegState::Undef);
      else
        MIB.addReg(InsReg, RegState::Kill);
    } else {
      DEBUG(errs() << "subreg: eliminated!\n");
      MBB->erase(MI);
      return true;
    }
  } else {
    // Insert sub-register copy
    const TargetRegisterClass *TRC0= TRI->getPhysicalRegisterRegClass(DstSubReg);
    const TargetRegisterClass *TRC1= TRI->getPhysicalRegisterRegClass(InsReg);
    if (MI->getOperand(2).isUndef())
      // If the source register being inserted is undef, then this becomes a
      // KILL.
      BuildMI(*MBB, MI, MI->getDebugLoc(),
              TII->get(TargetInstrInfo::KILL), DstSubReg);
    else {
      bool Emitted = TII->copyRegToReg(*MBB, MI, DstSubReg, InsReg, TRC0, TRC1);
      (void)Emitted;
      assert(Emitted && "Subreg and Dst must be of compatible register class");
    }
    MachineBasicBlock::iterator CopyMI = MI;
    --CopyMI;

    // INSERT_SUBREG is a two-address instruction so it implicitly kills SrcReg.
    if (!MI->getOperand(1).isUndef())
      CopyMI->addOperand(MachineOperand::CreateReg(DstReg, false, true, true));

    // Transfer the kill/dead flags, if needed.
    if (MI->getOperand(0).isDead()) {
      TransferDeadFlag(MI, DstSubReg, TRI);
    } else {
      // Make sure the full DstReg is live after this replacement.
      CopyMI->addOperand(MachineOperand::CreateReg(DstReg, true, true));
    }

    // Make sure the inserted register gets killed
    if (MI->getOperand(2).isKill() && !MI->getOperand(2).isUndef())
      TransferKillFlag(MI, InsReg, TRI);
  }

  DEBUG({
      MachineBasicBlock::iterator dMI = MI;
      errs() << "subreg: " << *(--dMI) << "\n";
    });

  MBB->erase(MI);
  return true;
}

/// runOnMachineFunction - Reduce subregister inserts and extracts to register
/// copies.
///
bool LowerSubregsInstructionPass::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(errs() << "Machine Function\n"  
               << "********** LOWERING SUBREG INSTRS **********\n"
               << "********** Function: " 
               << MF.getFunction()->getName() << '\n');
  TRI = MF.getTarget().getRegisterInfo();
  TII = MF.getTarget().getInstrInfo();

  bool MadeChange = false;

  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me;) {
      MachineBasicBlock::iterator nmi = llvm::next(mi);
      MachineInstr *MI = mi;
      if (MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
        MadeChange |= LowerExtract(MI);
      } else if (MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG) {
        MadeChange |= LowerInsert(MI);
      } else if (MI->getOpcode() == TargetInstrInfo::SUBREG_TO_REG) {
        MadeChange |= LowerSubregToReg(MI);
      }
      mi = nmi;
    }
  }

  return MadeChange;
}
