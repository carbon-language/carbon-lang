//===-- RegAllocSimple.cpp - A simple generic register allocator ----------===//
//
// This file implements a simple register allocator. *Very* simple.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Statistic.h"
#include <iostream>
#include <set>

namespace {
  Statistic<> NumSpilled ("ra-simple", "Number of registers spilled");
  Statistic<> NumReloaded("ra-simple", "Number of registers reloaded");

  class RegAllocSimple : public FunctionPass {
    TargetMachine &TM;
    MachineFunction *MF;
    const MRegisterInfo *RegInfo;
    unsigned NumBytesAllocated;
    
    // Maps SSA Regs => offsets on the stack where these values are stored
    std::map<unsigned, unsigned> VirtReg2OffsetMap;

    // RegsUsed - Keep track of what registers are currently in use.
    std::set<unsigned> RegsUsed;

    // RegClassIdx - Maps RegClass => which index we can take a register
    // from. Since this is a simple register allocator, when we need a register
    // of a certain class, we just take the next available one.
    std::map<const TargetRegisterClass*, unsigned> RegClassIdx;

  public:

    RegAllocSimple(TargetMachine &tm)
      : TM(tm), RegInfo(tm.getRegisterInfo()) {
      RegsUsed.insert(RegInfo->getFramePointer());
      RegsUsed.insert(RegInfo->getStackPointer());

      cleanupAfterFunction();
    }

    bool runOnFunction(Function &Fn) {
      return runOnMachineFunction(MachineFunction::get(&Fn));
    }

    virtual const char *getPassName() const {
      return "Simple Register Allocator";
    }

  private:
    /// runOnMachineFunction - Register allocate the whole function
    bool runOnMachineFunction(MachineFunction &Fn);

    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);

    /// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions
    /// in predecessor basic blocks.
    void EliminatePHINodes(MachineBasicBlock &MBB);

    /// EmitPrologue/EmitEpilogue - Use the register info object to add a
    /// prologue/epilogue to the function and save/restore any callee saved
    /// registers we are responsible for.
    ///
    void EmitPrologue();
    void EmitEpilogue(MachineBasicBlock &MBB);


    /// getStackSpaceFor - This returns the offset of the specified virtual
    /// register on the stack, allocating space if neccesary.
    unsigned getStackSpaceFor(unsigned VirtReg, 
                              const TargetRegisterClass *regClass);

    /// Given a virtual register, return a compatible physical register that is
    /// currently unused.
    ///
    /// Side effect: marks that register as being used until manually cleared
    ///
    unsigned getFreeReg(unsigned virtualReg);

    /// Returns all `borrowed' registers back to the free pool
    void clearAllRegs() {
      RegClassIdx.clear();
    }

    /// Invalidates any references, real or implicit, to physical registers
    ///
    void invalidatePhysRegs(const MachineInstr *MI) {
      unsigned Opcode = MI->getOpcode();
      const MachineInstrDescriptor &Desc = TM.getInstrInfo().get(Opcode);
      if (const unsigned *regs = Desc.ImplicitUses)
        while (*regs)
          RegsUsed.insert(*regs++);

      if (const unsigned *regs = Desc.ImplicitDefs)
        while (*regs)
          RegsUsed.insert(*regs++);
    }

    void cleanupAfterFunction() {
      VirtReg2OffsetMap.clear();
      NumBytesAllocated = 4;   // FIXME: This is X86 specific
    }

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
unsigned RegAllocSimple::getStackSpaceFor(unsigned VirtReg,
                                          const TargetRegisterClass *regClass) {
  // Find the location VirtReg would belong...
  std::map<unsigned, unsigned>::iterator I =
    VirtReg2OffsetMap.lower_bound(VirtReg);

  if (I != VirtReg2OffsetMap.end() && I->first == VirtReg)
    return I->second;          // Already has space allocated?

  unsigned RegSize = regClass->getDataSize();

  // Align NumBytesAllocated.  We should be using TargetData alignment stuff
  // to determine this, but we don't know the LLVM type associated with the
  // virtual register.  Instead, just align to a multiple of the size for now.
  NumBytesAllocated += RegSize-1;
  NumBytesAllocated = NumBytesAllocated/RegSize*RegSize;
  
  // Assign the slot...
  VirtReg2OffsetMap.insert(I, std::make_pair(VirtReg, NumBytesAllocated));
  
  // Reserve the space!
  NumBytesAllocated += RegSize;
  return NumBytesAllocated-RegSize;
}

unsigned RegAllocSimple::getFreeReg(unsigned virtualReg) {
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(virtualReg);
  
  unsigned regIdx = RegClassIdx[RC]++;
  assert(regIdx < RC->getNumRegs() && "Not enough registers!");
  unsigned physReg = RC->getRegister(regIdx);

  if (RegsUsed.find(physReg) == RegsUsed.end())
    return physReg;
  else
    return getFreeReg(virtualReg);
}

unsigned RegAllocSimple::reloadVirtReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator &I,
                                       unsigned VirtReg) {
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(VirtReg);
  unsigned stackOffset = getStackSpaceFor(VirtReg, RC);
  unsigned PhysReg = getFreeReg(VirtReg);

  // Add move instruction(s)
  ++NumReloaded;
  RegInfo->loadRegOffset2Reg(MBB, I, PhysReg, RegInfo->getFramePointer(),
			     -stackOffset, RC);
  return PhysReg;
}

void RegAllocSimple::spillVirtReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator &I,
                                  unsigned VirtReg, unsigned PhysReg)
{
  const TargetRegisterClass* RC = MF->getSSARegMap()->getRegClass(VirtReg);
  unsigned stackOffset = getStackSpaceFor(VirtReg, RC);

  // Add move instruction(s)
  ++NumSpilled;
  RegInfo->storeReg2RegOffset(MBB, I, PhysReg, RegInfo->getFramePointer(),
			      -stackOffset, RC);
}


/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
///
void RegAllocSimple::EliminatePHINodes(MachineBasicBlock &MBB) {
  const MachineInstrInfo &MII = TM.getInstrInfo();

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

        // Retrieve the constant value from this op, move it to target
        // register of the phi
        if (opVal.isImmediate()) {
          RegInfo->moveImm2Reg(opBlock, opI, virtualReg,
			       (unsigned) opVal.getImmedValue(), RC);
        } else {
          RegInfo->moveReg2Reg(opBlock, opI, virtualReg,
			       opVal.getAllocatedRegNum(), RC);
        }
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
    
    // a preliminary pass that will invalidate any registers that
    // are used by the instruction (including implicit uses)
    invalidatePhysRegs(MI);
    
    // Loop over uses, move from memory into registers
    for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
      MachineOperand &op = MI->getOperand(i);
      
      if (op.isVirtualRegister()) {
        unsigned virtualReg = (unsigned) op.getAllocatedRegNum();
        DEBUG(std::cerr << "op: " << op << "\n");
        DEBUG(std::cerr << "\t inst[" << i << "]: ";
              MI->print(std::cerr, TM));
        
        // make sure the same virtual register maps to the same physical
        // register in any given instruction
        unsigned physReg = Virt2PhysRegMap[virtualReg];
        if (physReg == 0) {
          if (op.opIsDef()) {
            if (TM.getInstrInfo().isTwoAddrInstr(MI->getOpcode()) && i == 0) {
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
    clearAllRegs();
  }
}


/// EmitPrologue - Use the register info object to add a prologue to the
/// function and save any callee saved registers we are responsible for.
///
void RegAllocSimple::EmitPrologue() {
  // Get a list of the callee saved registers, so that we can save them on entry
  // to the function.
  //
  MachineBasicBlock &MBB = MF->front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator I = MBB.begin();

  const unsigned *CSRegs = RegInfo->getCalleeSaveRegs();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    const TargetRegisterClass *RegClass = RegInfo->getRegClass(CSRegs[i]);
    unsigned Offset = getStackSpaceFor(CSRegs[i], RegClass);

    // Insert the spill to the stack frame...
    RegInfo->storeReg2RegOffset(MBB, I,CSRegs[i],RegInfo->getFramePointer(),
				-Offset, RegClass);
    ++NumSpilled;
  }

  // Add prologue to the function...
  RegInfo->emitPrologue(*MF, NumBytesAllocated);
}


/// EmitEpilogue - Use the register info object to add a epilogue to the
/// function and restore any callee saved registers we are responsible for.
///
void RegAllocSimple::EmitEpilogue(MachineBasicBlock &MBB) {
  // Insert instructions before the return.
  MachineBasicBlock::iterator I = MBB.end()-1;

  const unsigned *CSRegs = RegInfo->getCalleeSaveRegs();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    const TargetRegisterClass *RegClass = RegInfo->getRegClass(CSRegs[i]);
    unsigned Offset = getStackSpaceFor(CSRegs[i], RegClass);

    RegInfo->loadRegOffset2Reg(MBB, I, CSRegs[i],RegInfo->getFramePointer(),
			       -Offset, RegClass);
    --I;  // Insert in reverse order
    ++NumReloaded;
  }

  RegInfo->emitEpilogue(MBB, NumBytesAllocated);
}


/// runOnMachineFunction - Register allocate the whole function
///
bool RegAllocSimple::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(std::cerr << "Machine Function " << "\n");
  MF = &Fn;

  // First pass: eliminate PHI instructions by inserting copies into predecessor
  // blocks.
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    EliminatePHINodes(*MBB);

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  // Round stack allocation up to a nice alignment to keep the stack aligned
  // FIXME: This is X86 specific!  Move to frame manager
  NumBytesAllocated = (NumBytesAllocated + 3) & ~3;

  // Emit a prologue for the function...
  EmitPrologue();

  const MachineInstrInfo &MII = TM.getInstrInfo();

  // Add epilogue to restore the callee-save registers in each exiting block
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    // If last instruction is a return instruction, add an epilogue
    if (MII.isReturn(MBB->back()->getOpcode()))
      EmitEpilogue(*MBB);
  }

  cleanupAfterFunction();
  return true;
}

Pass *createSimpleRegisterAllocator(TargetMachine &TM) {
  return new RegAllocSimple(TM);
}
