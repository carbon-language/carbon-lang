//===-- RegAllocSimple.cpp - A simple generic register allocator ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a simple register allocator. *Very* simple: It immediate
// spills every value right after it is computed, and it reloads all used
// operands from the spill area to temporary registers before each instruction.
// It does not keep values in registers across instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> NumStores("ra-simple", "Number of stores added");
  Statistic<> NumLoads ("ra-simple", "Number of loads added");

  class RegAllocSimple : public MachineFunctionPass {
    MachineFunction *MF;
    const TargetMachine *TM;
    const MRegisterInfo *RegInfo;
    
    // StackSlotForVirtReg - Maps SSA Regs => frame index on the stack where
    // these values are spilled
    std::map<unsigned, int> StackSlotForVirtReg;

    // RegsUsed - Keep track of what registers are currently in use.  This is a
    // bitset.
    std::vector<bool> RegsUsed;

    // RegClassIdx - Maps RegClass => which index we can take a register
    // from. Since this is a simple register allocator, when we need a register
    // of a certain class, we just take the next available one.
    std::map<const TargetRegisterClass*, unsigned> RegClassIdx;

  public:
    virtual const char *getPassName() const {
      return "Simple Register Allocator";
    }

    /// runOnMachineFunction - Register allocate the whole function
    bool runOnMachineFunction(MachineFunction &Fn);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(PHIEliminationID);           // Eliminate PHI nodes
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  private:
    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);

    /// getStackSpaceFor - This returns the offset of the specified virtual
    /// register on the stack, allocating space if necessary.
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);

    /// Given a virtual register, return a compatible physical register that is
    /// currently unused.
    ///
    /// Side effect: marks that register as being used until manually cleared
    ///
    unsigned getFreeReg(unsigned virtualReg);

    /// Moves value from memory into that register
    unsigned reloadVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, unsigned VirtReg);

    /// Saves reg value on the stack (maps virtual register to stack value)
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                      unsigned VirtReg, unsigned PhysReg);
  };

}

/// getStackSpaceFor - This allocates space for the specified virtual
/// register to be held on the stack.
int RegAllocSimple::getStackSpaceFor(unsigned VirtReg,
				     const TargetRegisterClass *RC) {
  // Find the location VirtReg would belong...
  std::map<unsigned, int>::iterator I =
    StackSlotForVirtReg.lower_bound(VirtReg);

  if (I != StackSlotForVirtReg.end() && I->first == VirtReg)
    return I->second;          // Already has space allocated?

  // Allocate a new stack object for this spill location...
  int FrameIdx = MF->getFrameInfo()->CreateStackObject(RC);
  
  // Assign the slot...
  StackSlotForVirtReg.insert(I, std::make_pair(VirtReg, FrameIdx));

  return FrameIdx;
}

unsigned RegAllocSimple::getFreeReg(unsigned virtualReg) {
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(virtualReg);
  TargetRegisterClass::iterator RI = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator RE = RC->allocation_order_end(*MF);

  while (1) {
    unsigned regIdx = RegClassIdx[RC]++; 
    assert(RI+regIdx != RE && "Not enough registers!");
    unsigned PhysReg = *(RI+regIdx);
    
    if (!RegsUsed[PhysReg])
      return PhysReg;
  }
}

unsigned RegAllocSimple::reloadVirtReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned VirtReg) {
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(VirtReg);
  int FrameIdx = getStackSpaceFor(VirtReg, RC);
  unsigned PhysReg = getFreeReg(VirtReg);

  // Add move instruction(s)
  ++NumLoads;
  RegInfo->loadRegFromStackSlot(MBB, I, PhysReg, FrameIdx, RC);
  return PhysReg;
}

void RegAllocSimple::spillVirtReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned VirtReg, unsigned PhysReg) {
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(VirtReg);
  int FrameIdx = getStackSpaceFor(VirtReg, RC);

  // Add move instruction(s)
  ++NumStores;
  RegInfo->storeRegToStackSlot(MBB, I, PhysReg, FrameIdx, RC);
}


void RegAllocSimple::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  for (MachineBasicBlock::iterator MI = MBB.begin(); MI != MBB.end(); ++MI) {
    // Made to combat the incorrect allocation of r2 = add r1, r1
    std::map<unsigned, unsigned> Virt2PhysRegMap;

    RegsUsed.resize(RegInfo->getNumRegs());
    
    // a preliminary pass that will invalidate any registers that
    // are used by the instruction (including implicit uses)
    unsigned Opcode = MI->getOpcode();
    const TargetInstrDescriptor &Desc = TM->getInstrInfo().get(Opcode);
    const unsigned *Regs = Desc.ImplicitUses;
    while (*Regs)
      RegsUsed[*Regs++] = true;
    
    Regs = Desc.ImplicitDefs;
    while (*Regs)
      RegsUsed[*Regs++] = true;
    
    // Loop over uses, move from memory into registers
    for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
      MachineOperand &op = MI->getOperand(i);
      
      if (op.isRegister() && op.getReg() &&
          MRegisterInfo::isVirtualRegister(op.getReg())) {
        unsigned virtualReg = (unsigned) op.getReg();
        DEBUG(std::cerr << "op: " << op << "\n");
        DEBUG(std::cerr << "\t inst[" << i << "]: ";
              MI->print(std::cerr, *TM));
        
        // make sure the same virtual register maps to the same physical
        // register in any given instruction
        unsigned physReg = Virt2PhysRegMap[virtualReg];
        if (physReg == 0) {
          if (op.isDef()) {
            if (!TM->getInstrInfo().isTwoAddrInstr(MI->getOpcode()) || i) {
              physReg = getFreeReg(virtualReg);
            } else {
              // must be same register number as the first operand
              // This maps a = b + c into b += c, and saves b into a's spot
              assert(MI->getOperand(1).isRegister()  &&
                     MI->getOperand(1).getReg() &&
                     MI->getOperand(1).isUse() &&
                     "Two address instruction invalid!");

              physReg = MI->getOperand(1).getReg();
              spillVirtReg(MBB, next(MI), virtualReg, physReg);
              MI->getOperand(1).setDef();
              MI->RemoveOperand(0);
              break; // This is the last operand to process
            }
            spillVirtReg(MBB, next(MI), virtualReg, physReg);
          } else {
            physReg = reloadVirtReg(MBB, MI, virtualReg);
            Virt2PhysRegMap[virtualReg] = physReg;
          }
        }
        MI->SetMachineOperandReg(i, physReg);
        DEBUG(std::cerr << "virt: " << virtualReg << 
              ", phys: " << op.getReg() << "\n");
      }
    }
    RegClassIdx.clear();
    RegsUsed.clear();
  }
}


/// runOnMachineFunction - Register allocate the whole function
///
bool RegAllocSimple::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(std::cerr << "Machine Function " << "\n");
  MF = &Fn;
  TM = &MF->getTarget();
  RegInfo = TM->getRegisterInfo();

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  StackSlotForVirtReg.clear();
  return true;
}

FunctionPass *llvm::createSimpleRegisterAllocator() {
  return new RegAllocSimple();
}
