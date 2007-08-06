//===-- LowerSubregs.cpp - Subregister Lowering instruction pass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Christopher Lamb and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lowersubregs"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN LowerSubregsInstructionPass
   : public MachineFunctionPass {
    static char ID; // Pass identification, replacement for typeid
    LowerSubregsInstructionPass() : MachineFunctionPass((intptr_t)&ID) {}
    
    const char *getPassName() const {
      return "Subregister lowering instruction pass";
    }

    /// runOnMachineFunction - pass entry point
    bool runOnMachineFunction(MachineFunction&);
    
    bool LowerExtract(MachineInstr *MI);
    bool LowerInsert(MachineInstr *MI);
  };

  char LowerSubregsInstructionPass::ID = 0;
}

FunctionPass *llvm::createLowerSubregsPass() { 
  return new LowerSubregsInstructionPass(); 
}

// Returns the Register Class of a physical register.
static const TargetRegisterClass *getPhysicalRegisterRegClass(
        const MRegisterInfo &MRI,
        unsigned reg) {
  assert(MRegisterInfo::isPhysicalRegister(reg) &&
         "reg must be a physical register");
  // Pick the register class of the right type that contains this physreg.
  for (MRegisterInfo::regclass_iterator I = MRI.regclass_begin(),
         E = MRI.regclass_end(); I != E; ++I)
    if ((*I)->contains(reg))
      return *I;
  assert(false && "Couldn't find the register class");
  return 0;
}

static bool isSubRegOf(const MRegisterInfo &MRI,
                       unsigned SubReg,
                       unsigned SupReg) {
  const TargetRegisterDesc &RD = MRI[SubReg];
  for (const unsigned *reg = RD.SuperRegs; *reg != 0; ++reg)
    if (*reg == SupReg)
      return true;
      
  return false;
}

bool LowerSubregsInstructionPass::LowerExtract(MachineInstr *MI) {
   MachineBasicBlock *MBB = MI->getParent();
   MachineFunction &MF = *MBB->getParent();
   const MRegisterInfo &MRI = *MF.getTarget().getRegisterInfo();
   
   assert(MI->getOperand(0).isRegister() && MI->getOperand(0).isDef() &&
          MI->getOperand(1).isRegister() && MI->getOperand(1).isUse() &&
          MI->getOperand(2).isImm() && "Malformed extract_subreg");

   unsigned SuperReg = MI->getOperand(1).getReg();
   unsigned SubIdx = MI->getOperand(2).getImm();

   assert(MRegisterInfo::isPhysicalRegister(SuperReg) &&
          "Extract supperg source must be a physical register");
   unsigned SrcReg = MRI.getSubReg(SuperReg, SubIdx);
   unsigned DstReg = MI->getOperand(0).getReg();

   DOUT << "subreg: CONVERTING: " << *MI;

   if (SrcReg != DstReg) {
     const TargetRegisterClass *TRC = 0;
     if (MRegisterInfo::isPhysicalRegister(DstReg)) {
       TRC = getPhysicalRegisterRegClass(MRI, DstReg);
     } else {
       TRC = MF.getSSARegMap()->getRegClass(DstReg);
     }
     assert(TRC == getPhysicalRegisterRegClass(MRI, SrcReg) &&
             "Extract subreg and Dst must be of same register class");

     MRI.copyRegToReg(*MBB, MI, DstReg, SrcReg, TRC);
     MachineBasicBlock::iterator dMI = MI;
     DOUT << "subreg: " << *(--dMI);
   }

   DOUT << "\n";
   MBB->erase(MI);
   return true;
}


bool LowerSubregsInstructionPass::LowerInsert(MachineInstr *MI) {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction &MF = *MBB->getParent();
  const MRegisterInfo &MRI = *MF.getTarget().getRegisterInfo(); 
  unsigned DstReg = 0;
  unsigned SrcReg = 0;
  unsigned InsReg = 0;
  unsigned SubIdx = 0;

  // If only have 3 operands, then the source superreg is undef
  // and we can supress the copy from the undef value
  if (MI->getNumOperands() == 3) {
    assert((MI->getOperand(0).isRegister() && MI->getOperand(0).isDef()) &&
           (MI->getOperand(1).isRegister() && MI->getOperand(1).isUse()) &&
            MI->getOperand(2).isImm() && "Invalid extract_subreg");
    DstReg = MI->getOperand(0).getReg();
    SrcReg = DstReg;
    InsReg = MI->getOperand(1).getReg();
    SubIdx = MI->getOperand(2).getImm();
  } else if (MI->getNumOperands() == 4) {
    assert((MI->getOperand(0).isRegister() && MI->getOperand(0).isDef()) &&
           (MI->getOperand(1).isRegister() && MI->getOperand(1).isUse()) &&
           (MI->getOperand(2).isRegister() && MI->getOperand(2).isUse()) &&
            MI->getOperand(3).isImm() && "Invalid extract_subreg");
    DstReg = MI->getOperand(0).getReg();
    SrcReg = MI->getOperand(1).getReg();
    InsReg = MI->getOperand(2).getReg();
    SubIdx = MI->getOperand(3).getImm();     
  } else 
    assert(0 && "Malformed extract_subreg");

  assert(SubIdx != 0 && "Invalid index for extract_subreg");
  unsigned DstSubReg = MRI.getSubReg(DstReg, SubIdx);

  assert(MRegisterInfo::isPhysicalRegister(SrcReg) &&
         "Insert superreg source must be in a physical register");
  assert(MRegisterInfo::isPhysicalRegister(DstReg) &&
         "Insert destination must be in a physical register");
  assert(MRegisterInfo::isPhysicalRegister(InsReg) &&
         "Inserted value must be in a physical register");

  DOUT << "subreg: CONVERTING: " << *MI;
       
  // If the inserted register is already allocated into a subregister
  // of the destination, we copy the subreg into the source
  // However, this is only safe if the insert instruction is the kill
  // of the source register
  bool revCopyOrder = isSubRegOf(MRI, InsReg, DstReg);    
  if (revCopyOrder) {
    if (MI->getOperand(1).isKill()) {
      DstSubReg = MRI.getSubReg(SrcReg, SubIdx);
      // Insert sub-register copy
      const TargetRegisterClass *TRC1 = 0;
      if (MRegisterInfo::isPhysicalRegister(InsReg)) {
        TRC1 = getPhysicalRegisterRegClass(MRI, InsReg);
      } else {
        TRC1 = MF.getSSARegMap()->getRegClass(InsReg);
      }
    
      MRI.copyRegToReg(*MBB, MI, DstSubReg, InsReg, TRC1);
      MachineBasicBlock::iterator dMI = MI;
      DOUT << "subreg: " << *(--dMI);
    } else {
      assert(0 && "Don't know how to convert this insert");
    }
  }

  if (SrcReg != DstReg) {
    // Insert super-register copy
    const TargetRegisterClass *TRC0 = 0;
    if (MRegisterInfo::isPhysicalRegister(DstReg)) {
      TRC0 = getPhysicalRegisterRegClass(MRI, DstReg);
    } else {
      TRC0 = MF.getSSARegMap()->getRegClass(DstReg);
    }
    assert(TRC0 == getPhysicalRegisterRegClass(MRI, SrcReg) &&
            "Insert superreg and Dst must be of same register class");

    MRI.copyRegToReg(*MBB, MI, DstReg, SrcReg, TRC0);
    MachineBasicBlock::iterator dMI = MI;
    DOUT << "subreg: " << *(--dMI);
  }

  if (!revCopyOrder && InsReg != DstSubReg) {
    // Insert sub-register copy
    const TargetRegisterClass *TRC1 = 0;
    if (MRegisterInfo::isPhysicalRegister(InsReg)) {
      TRC1 = getPhysicalRegisterRegClass(MRI, InsReg);
    } else {
      TRC1 = MF.getSSARegMap()->getRegClass(InsReg);
    }
  
    MRI.copyRegToReg(*MBB, MI, DstSubReg, InsReg, TRC1);
    MachineBasicBlock::iterator dMI = MI;
    DOUT << "subreg: " << *(--dMI);
  }

  DOUT << "\n";
  MBB->erase(MI);
  return true;                    
}

/// runOnMachineFunction - Reduce subregister inserts and extracts to register
/// copies.
///
bool LowerSubregsInstructionPass::runOnMachineFunction(MachineFunction &MF) {
  DOUT << "Machine Function\n";
  
  bool MadeChange = false;

  DOUT << "********** LOWERING SUBREG INSTRS **********\n";
  DOUT << "********** Function: " << MF.getFunction()->getName() << '\n';

  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me;) {
      MachineInstr *MI = mi++;
           
      if (MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
        MadeChange |= LowerExtract(MI);
      } else if (MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG) {
        MadeChange |= LowerInsert(MI);
      }
    }
  }

  return MadeChange;
}
