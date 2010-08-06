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
    bool LowerSubregToReg(MachineInstr *MI);
    bool LowerCopy(MachineInstr *MI);

    void TransferDeadFlag(MachineInstr *MI, unsigned DstReg,
                          const TargetRegisterInfo *TRI);
    void TransferKillFlag(MachineInstr *MI, unsigned SrcReg,
                          const TargetRegisterInfo *TRI,
                          bool AddIfNotFound = false);
    void TransferImplicitDefs(MachineInstr *MI);
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
           "copyPhysReg output doesn't reference destination register!");
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
           "copyPhysReg output doesn't reference source register!");
  }
}

/// TransferImplicitDefs - MI is a pseudo-instruction, and the lowered
/// replacement instructions immediately precede it.  Copy any implicit-def
/// operands from MI to the replacement instruction.
void
LowerSubregsInstructionPass::TransferImplicitDefs(MachineInstr *MI) {
  MachineBasicBlock::iterator CopyMI = MI;
  --CopyMI;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isImplicit() || MO.isUse())
      continue;
    CopyMI->addOperand(MachineOperand::CreateReg(MO.getReg(), true, true));
  }
}

bool LowerSubregsInstructionPass::LowerSubregToReg(MachineInstr *MI) {
  MachineBasicBlock *MBB = MI->getParent();
  assert((MI->getOperand(0).isReg() && MI->getOperand(0).isDef()) &&
         MI->getOperand(1).isImm() &&
         (MI->getOperand(2).isReg() && MI->getOperand(2).isUse()) &&
          MI->getOperand(3).isImm() && "Invalid subreg_to_reg");

  unsigned DstReg  = MI->getOperand(0).getReg();
  unsigned InsReg  = MI->getOperand(2).getReg();
  assert(!MI->getOperand(2).getSubReg() && "SubIdx on physreg?");
  unsigned SubIdx  = MI->getOperand(3).getImm();

  assert(SubIdx != 0 && "Invalid index for insert_subreg");
  unsigned DstSubReg = TRI->getSubReg(DstReg, SubIdx);

  assert(TargetRegisterInfo::isPhysicalRegister(DstReg) &&
         "Insert destination must be in a physical register");
  assert(TargetRegisterInfo::isPhysicalRegister(InsReg) &&
         "Inserted value must be in a physical register");

  DEBUG(dbgs() << "subreg: CONVERTING: " << *MI);

  if (DstSubReg == InsReg) {
    // No need to insert an identify copy instruction.
    // Watch out for case like this:
    // %RAX<def> = SUBREG_TO_REG 0, %EAX<kill>, 3
    // We must leave %RAX live.
    if (DstReg != InsReg) {
      MI->setDesc(TII->get(TargetOpcode::KILL));
      MI->RemoveOperand(3);     // SubIdx
      MI->RemoveOperand(1);     // Imm
      DEBUG(dbgs() << "subreg: replace by: " << *MI);
      return true;
    }
    DEBUG(dbgs() << "subreg: eliminated!");
  } else {
    TII->copyPhysReg(*MBB, MI, MI->getDebugLoc(), DstSubReg, InsReg,
                     MI->getOperand(2).isKill());
    // Transfer the kill/dead flags, if needed.
    if (MI->getOperand(0).isDead())
      TransferDeadFlag(MI, DstSubReg, TRI);
    DEBUG({
        MachineBasicBlock::iterator dMI = MI;
        dbgs() << "subreg: " << *(--dMI);
      });
  }

  DEBUG(dbgs() << '\n');
  MBB->erase(MI);
  return true;
}

bool LowerSubregsInstructionPass::LowerCopy(MachineInstr *MI) {
  MachineOperand &DstMO = MI->getOperand(0);
  MachineOperand &SrcMO = MI->getOperand(1);

  if (SrcMO.getReg() == DstMO.getReg()) {
    DEBUG(dbgs() << "identity copy: " << *MI);
    // No need to insert an identity copy instruction, but replace with a KILL
    // if liveness is changed.
    if (DstMO.isDead() || SrcMO.isUndef() || MI->getNumOperands() > 2) {
      // We must make sure the super-register gets killed. Replace the
      // instruction with KILL.
      MI->setDesc(TII->get(TargetOpcode::KILL));
      DEBUG(dbgs() << "replaced by:   " << *MI);
      return true;
    }
    // Vanilla identity copy.
    MI->eraseFromParent();
    return true;
  }

  DEBUG(dbgs() << "real copy:   " << *MI);
  TII->copyPhysReg(*MI->getParent(), MI, MI->getDebugLoc(),
                   DstMO.getReg(), SrcMO.getReg(), SrcMO.isKill());

  if (DstMO.isDead())
    TransferDeadFlag(MI, DstMO.getReg(), TRI);
  if (MI->getNumOperands() > 2)
    TransferImplicitDefs(MI);
  DEBUG({
    MachineBasicBlock::iterator dMI = MI;
    dbgs() << "replaced by: " << *(--dMI);
  });
  MI->eraseFromParent();
  return true;
}

/// runOnMachineFunction - Reduce subregister inserts and extracts to register
/// copies.
///
bool LowerSubregsInstructionPass::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "Machine Function\n"  
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
      assert(!MI->isInsertSubreg() && "INSERT_SUBREG should no longer appear");
      assert(MI->getOpcode() != TargetOpcode::EXTRACT_SUBREG &&
             "EXTRACT_SUBREG should no longer appear");
      if (MI->isSubregToReg()) {
        MadeChange |= LowerSubregToReg(MI);
      } else if (MI->isCopy()) {
        MadeChange |= LowerCopy(MI);
      }
      mi = nmi;
    }
  }

  return MadeChange;
}
