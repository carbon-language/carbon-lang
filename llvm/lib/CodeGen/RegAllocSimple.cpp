//===-- RegAllocSimple.cpp - A simple generic register allocator ----------===//
//
// This file implements a simple register allocator. *Very* simple: It immediate
// spills every value right after it is computed, and it reloads all used
// operands from the spill area to temporary registers before each instruction.
// It does not keep values in registers across instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/FunctionFrameInfo.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Statistic.h"
#include <iostream>

namespace {
  Statistic<> NumSpilled ("ra-simple", "Number of registers spilled");
  Statistic<> NumReloaded("ra-simple", "Number of registers reloaded");

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

  private:
    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);

    /// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions
    /// in predecessor basic blocks.
    void EliminatePHINodes(MachineBasicBlock &MBB);

    /// getStackSpaceFor - This returns the offset of the specified virtual
    /// register on the stack, allocating space if neccesary.
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);

    /// Given a virtual register, return a compatible physical register that is
    /// currently unused.
    ///
    /// Side effect: marks that register as being used until manually cleared
    ///
    unsigned getFreeReg(unsigned virtualReg);

    /// Moves value from memory into that register
    unsigned reloadVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &I, unsigned VirtReg);

    /// Saves reg value on the stack (maps virtual register to stack value)
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
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
  int FrameIdx =
    MF->getFrameInfo()->CreateStackObject(RC->getSize(), RC->getAlignment());
  
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
                                       MachineBasicBlock::iterator &I,
                                       unsigned VirtReg) {
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(VirtReg);
  int FrameIdx = getStackSpaceFor(VirtReg, RC);
  unsigned PhysReg = getFreeReg(VirtReg);

  // Add move instruction(s)
  ++NumReloaded;
  RegInfo->loadRegFromStackSlot(MBB, I, PhysReg, FrameIdx, RC);
  return PhysReg;
}

void RegAllocSimple::spillVirtReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator &I,
                                  unsigned VirtReg, unsigned PhysReg) {
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(VirtReg);
  int FrameIdx = getStackSpaceFor(VirtReg, RC);

  // Add move instruction(s)
  ++NumSpilled;
  RegInfo->storeRegToStackSlot(MBB, I, PhysReg, FrameIdx, RC);
}


/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
///
void RegAllocSimple::EliminatePHINodes(MachineBasicBlock &MBB) {
  const MachineInstrInfo &MII = TM->getInstrInfo();

  while (MBB.front()->getOpcode() == MachineInstrInfo::PHI) {
    MachineInstr *MI = MBB.front();
    // Unlink the PHI node from the basic block... but don't delete the PHI yet
    MBB.erase(MBB.begin());
    
    DEBUG(std::cerr << "num ops: " << MI->getNumOperands() << "\n");
    assert(MI->getOperand(0).isVirtualRegister() &&
           "PHI node doesn't write virt reg?");

    unsigned virtualReg = MI->getOperand(0).getAllocatedRegNum();
    
    for (int i = MI->getNumOperands() - 1; i >= 2; i-=2) {
      MachineOperand &opVal = MI->getOperand(i-1);
      
      // Get the MachineBasicBlock equivalent of the BasicBlock that is the
      // source path the phi
      MachineBasicBlock &opBlock = *MI->getOperand(i).getMachineBasicBlock();

      // Check to make sure we haven't already emitted the copy for this block.
      // This can happen because PHI nodes may have multiple entries for the
      // same basic block.  It doesn't matter which entry we use though, because
      // all incoming values are guaranteed to be the same for a particular bb.
      //
      // Note that this is N^2 in the number of phi node entries, but since the
      // # of entries is tiny, this is not a problem.
      //
      bool HaveNotEmitted = true;
      for (int op = MI->getNumOperands() - 1; op != i; op -= 2)
        if (&opBlock == MI->getOperand(op).getMachineBasicBlock()) {
          HaveNotEmitted = false;
          break;
        }

      if (HaveNotEmitted) {
        MachineBasicBlock::iterator opI = opBlock.end();
        MachineInstr *opMI = *--opI;
        
        // must backtrack over ALL the branches in the previous block
        while (MII.isBranch(opMI->getOpcode()) && opI != opBlock.begin())
          opMI = *--opI;
        
        // move back to the first branch instruction so new instructions
        // are inserted right in front of it and not in front of a non-branch
	//
        if (!MII.isBranch(opMI->getOpcode()))
          ++opI;

        const TargetRegisterClass *RC =
	  MF->getSSARegMap()->getRegClass(virtualReg);

	assert(opVal.isVirtualRegister() &&
	       "Machine PHI Operands must all be virtual registers!");
	RegInfo->copyRegToReg(opBlock, opI, virtualReg, opVal.getReg(), RC);
      }
    }
    
    // really delete the PHI instruction now!
    delete MI;
  }
}


void RegAllocSimple::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
    // Made to combat the incorrect allocation of r2 = add r1, r1
    std::map<unsigned, unsigned> Virt2PhysRegMap;

    MachineInstr *MI = *I;

    RegsUsed.resize(MRegisterInfo::FirstVirtualRegister);
    
    // a preliminary pass that will invalidate any registers that
    // are used by the instruction (including implicit uses)
    unsigned Opcode = MI->getOpcode();
    const MachineInstrDescriptor &Desc = TM->getInstrInfo().get(Opcode);
    if (const unsigned *Regs = Desc.ImplicitUses)
      while (*Regs)
	RegsUsed[*Regs++] = true;
    
    if (const unsigned *Regs = Desc.ImplicitDefs)
      while (*Regs)
	RegsUsed[*Regs++] = true;
    
    // Loop over uses, move from memory into registers
    for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
      MachineOperand &op = MI->getOperand(i);
      
      if (op.isVirtualRegister()) {
        unsigned virtualReg = (unsigned) op.getAllocatedRegNum();
        DEBUG(std::cerr << "op: " << op << "\n");
        DEBUG(std::cerr << "\t inst[" << i << "]: ";
              MI->print(std::cerr, *TM));
        
        // make sure the same virtual register maps to the same physical
        // register in any given instruction
        unsigned physReg = Virt2PhysRegMap[virtualReg];
        if (physReg == 0) {
          if (op.opIsDef()) {
            if (TM->getInstrInfo().isTwoAddrInstr(MI->getOpcode()) && i == 0) {
              // must be same register number as the first operand
              // This maps a = b + c into b += c, and saves b into a's spot
              assert(MI->getOperand(1).isRegister()  &&
                     MI->getOperand(1).getAllocatedRegNum() &&
                     MI->getOperand(1).opIsUse() &&
                     "Two address instruction invalid!");

              physReg = MI->getOperand(1).getAllocatedRegNum();
            } else {
              physReg = getFreeReg(virtualReg);
            }
            ++I;
            spillVirtReg(MBB, I, virtualReg, physReg);
            --I;
          } else {
            physReg = reloadVirtReg(MBB, I, virtualReg);
            Virt2PhysRegMap[virtualReg] = physReg;
          }
        }
        MI->SetMachineOperandReg(i, physReg);
        DEBUG(std::cerr << "virt: " << virtualReg << 
              ", phys: " << op.getAllocatedRegNum() << "\n");
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

  // First pass: eliminate PHI instructions by inserting copies into predecessor
  // blocks.
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    EliminatePHINodes(*MBB);

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  StackSlotForVirtReg.clear();
  return true;
}

Pass *createSimpleRegisterAllocator() {
  return new RegAllocSimple();
}
