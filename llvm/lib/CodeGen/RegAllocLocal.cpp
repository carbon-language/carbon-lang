//===-- RegAllocLocal.cpp - A BasicBlock generic register allocator -------===//
//
// This register allocator allocates registers to a basic block at a time,
// attempting to keep values in registers and reusing registers as appropriate.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Statistic.h"
#include <iostream>

/// PhysRegClassMap - Construct a mapping of physical register numbers to their
/// register classes.
///
/// NOTE: This class will eventually be pulled out to somewhere shared.
///
class PhysRegClassMap {
  std::map<unsigned, const TargetRegisterClass*> PhysReg2RegClassMap;
public:
  PhysRegClassMap(const MRegisterInfo *RI) {
    for (MRegisterInfo::const_iterator I = RI->regclass_begin(),
           E = RI->regclass_end(); I != E; ++I)
      for (unsigned i=0; i < (*I)->getNumRegs(); ++i)
        PhysReg2RegClassMap[(*I)->getRegister(i)] = *I;
  }

  const TargetRegisterClass *operator[](unsigned Reg) {
    assert(PhysReg2RegClassMap[Reg] && "Register is not a known physreg!");
    return PhysReg2RegClassMap[Reg];
  }

  const TargetRegisterClass *get(unsigned Reg) { return operator[](Reg); }
};

namespace {
  Statistic<> NumSpilled ("ra-local", "Number of registers spilled");
  Statistic<> NumReloaded("ra-local", "Number of registers reloaded");

  class RA : public FunctionPass {
    TargetMachine &TM;
    MachineFunction *MF;
    const MRegisterInfo *RegInfo;
    unsigned NumBytesAllocated;
    PhysRegClassMap PhysRegClasses;
    
    // Maps SSA Regs => offsets on the stack where these values are stored
    std::map<unsigned, unsigned> VirtReg2OffsetMap;

    // Virt2PhysRegMap - This map contains entries for each virtual register
    // that is currently available in a physical register.
    //
    std::map<unsigned, unsigned> Virt2PhysRegMap;
    
    // PhysRegsUsed - This map contains entries for each physical register that
    // currently has a value (ie, it is in Virt2PhysRegMap).  The value mapped
    // to is the virtual register corresponding to the physical register (the
    // inverse of the Virt2PhysRegMap), or 0.  The value is set to 0 if this
    // register is pinned because it is used by a future instruction.
    //
    std::map<unsigned, unsigned> PhysRegsUsed;

    // PhysRegsUseOrder - This contains a list of the physical registers that
    // currently have a virtual register value in them.  This list provides an
    // ordering of registers, imposing a reallocation order.  This list is only
    // used if all registers are allocated and we have to spill one, in which
    // case we spill the least recently used register.  Entries at the front of
    // the list are the least recently used registers, entries at the back are
    // the most recently used.
    //
    std::vector<unsigned> PhysRegsUseOrder;

    void MarkPhysRegRecentlyUsed(unsigned Reg) {
      assert(std::find(PhysRegsUseOrder.begin(), PhysRegsUseOrder.end(), Reg) !=
             PhysRegsUseOrder.end() && "Register isn't used yet!");
      if (PhysRegsUseOrder.back() != Reg) {
        for (unsigned i = PhysRegsUseOrder.size(); ; --i)
          if (PhysRegsUseOrder[i-1] == Reg) {  // remove from middle
            PhysRegsUseOrder.erase(PhysRegsUseOrder.begin()+i-1);
            PhysRegsUseOrder.push_back(Reg);  // Add it to the end of the list
            return;
          }
      }
    }

  public:

    RA(TargetMachine &tm)
      : TM(tm), RegInfo(tm.getRegisterInfo()), PhysRegClasses(RegInfo) {
      cleanupAfterFunction();
    }

    bool runOnFunction(Function &Fn) {
      return runOnMachineFunction(MachineFunction::get(&Fn));
    }

    virtual const char *getPassName() const {
      return "Local Register Allocator";
    }

  private:
    /// runOnMachineFunction - Register allocate the whole function
    bool runOnMachineFunction(MachineFunction &Fn);

    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);

    /// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions
    /// in predecessor basic blocks.
    void EliminatePHINodes(MachineBasicBlock &MBB);


    /// getStackSpaceFor - This returns the offset of the specified virtual
    /// register on the stack, allocating space if neccesary.
    unsigned getStackSpaceFor(unsigned VirtReg, 
                              const TargetRegisterClass *regClass);

    void cleanupAfterFunction() {
      VirtReg2OffsetMap.clear();
      NumBytesAllocated = 4;   // FIXME: This is X86 specific
    }


    /// spillVirtReg - This method spills the value specified by PhysReg into
    /// the virtual register slot specified by VirtReg.  It then updates the RA
    /// data structures to indicate the fact that PhysReg is now available.
    ///
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                      unsigned VirtReg, unsigned PhysReg);

    void AssignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg);

    
    /// getFreeReg - Find a physical register to hold the specified virtual
    /// register.  If all compatible physical registers are used, this method
    /// spills the last used virtual register to the stack, and uses that
    /// register.
    ///
    unsigned getFreeReg(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &I,
                        unsigned virtualReg);

    /// reloadVirtReg - This method loads the specified virtual register into a
    /// physical register, returning the physical register chosen.  This updates
    /// the regalloc data structures to reflect the fact that the virtual reg is
    /// now alive in a physical register, and the previous one isn't.
    ///
    unsigned reloadVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &I, unsigned VirtReg);
  };

}

/// getStackSpaceFor - This allocates space for the specified virtual
/// register to be held on the stack.
unsigned RA::getStackSpaceFor(unsigned VirtReg,
                              const TargetRegisterClass *RegClass) {
  // Find the location VirtReg would belong...
  std::map<unsigned, unsigned>::iterator I =
    VirtReg2OffsetMap.lower_bound(VirtReg);

  if (I != VirtReg2OffsetMap.end() && I->first == VirtReg)
    return I->second;          // Already has space allocated?

  unsigned RegSize = RegClass->getDataSize();

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

/// spillVirtReg - This method spills the value specified by PhysReg into the
/// virtual register slot specified by VirtReg.  It then updates the RA data
/// structures to indicate the fact that PhysReg is now available.
///
void RA::spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                      unsigned VirtReg, unsigned PhysReg) {
  // If this is just a marker register, we don't need to spill it.
  if (VirtReg != 0) {
    const TargetRegisterClass *RegClass = MF->getRegClass(VirtReg);
    unsigned stackOffset = getStackSpaceFor(VirtReg, RegClass);

    // Add move instruction(s)
    I = RegInfo->storeReg2RegOffset(MBB, I, PhysReg, RegInfo->getFramePointer(),
                                    -stackOffset, RegClass->getDataSize());
    ++NumSpilled;   // Update statistics
    Virt2PhysRegMap.erase(VirtReg);   // VirtReg no longer available
  }
  PhysRegsUsed.erase(PhysReg);      // PhyReg no longer used

  std::vector<unsigned>::iterator It =
    std::find(PhysRegsUseOrder.begin(), PhysRegsUseOrder.end(), PhysReg);
  assert(It != PhysRegsUseOrder.end() &&
         "Spilled a physical register, but it was not in use list!");
  PhysRegsUseOrder.erase(It);
}

/// getFreeReg - Find a physical register to hold the specified virtual
/// register.  If all compatible physical registers are used, this method spills
/// the last used virtual register to the stack, and uses that register.
///
unsigned RA::getFreeReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                        unsigned VirtReg) {
  const TargetRegisterClass *RegClass = MF->getRegClass(VirtReg);
  unsigned PhysReg = 0;
  
  for (TargetRegisterClass::iterator It = RegClass->begin(),E = RegClass->end();
       It != E; ++It) {
    unsigned R = *It;
    if (PhysRegsUsed.find(R) == PhysRegsUsed.end())   // Is reg unused?
      /// FIXME: Hack
      if (R != RegInfo->getFramePointer() && R != RegInfo->getStackPointer() &&
          R != 13 && R != 14) {
        // Found an unused register!
        PhysReg = R;
        break;
      }
  }

  // If we didn't find an unused register, scavange one now!
  if (PhysReg == 0) {
    unsigned i = 0;
    while (PhysRegClasses[PhysRegsUseOrder[i]] != RegClass) {
      ++i;
      assert(i != PhysRegsUseOrder.size() &&
             "Couldn't find a register of the appropriate class!");
    }

    // At this point PhysRegsUseOrder[i] is the least recently used register of
    // compatible register class.  Spill it to memory and reap its remains.
    PhysReg = PhysRegsUseOrder[i];
    spillVirtReg(MBB, I, PhysRegsUsed[PhysReg], PhysReg);
  }

  // Now that we know which register we need to assign this to, do it now!
  AssignVirtToPhysReg(VirtReg, PhysReg);
  return PhysReg;
}

void RA::AssignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg) {
  assert(PhysRegsUsed.find(PhysReg) == PhysRegsUsed.end() &&
         "Phys reg already assigned!");
  // Update information to note the fact that this register was just used, and
  // it holds VirtReg.
  PhysRegsUsed[PhysReg] = VirtReg;
  Virt2PhysRegMap[VirtReg] = PhysReg;
  PhysRegsUseOrder.push_back(PhysReg);   // New use of PhysReg
}


/// reloadVirtReg - This method loads the specified virtual register into a
/// physical register, returning the physical register chosen.  This updates the
/// regalloc data structures to reflect the fact that the virtual reg is now
/// alive in a physical register, and the previous one isn't.
///
unsigned RA::reloadVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &I,
                           unsigned VirtReg) {
  std::map<unsigned, unsigned>::iterator It = Virt2PhysRegMap.find(VirtReg);
  if (It != Virt2PhysRegMap.end()) {
    MarkPhysRegRecentlyUsed(It->second);
    return It->second;               // Already have this value available!
  }

  unsigned PhysReg = getFreeReg(MBB, I, VirtReg);

  const TargetRegisterClass *RegClass = MF->getRegClass(VirtReg);
  unsigned StackOffset = getStackSpaceFor(VirtReg, RegClass);

  // Add move instruction(s)
  I = RegInfo->loadRegOffset2Reg(MBB, I, PhysReg, RegInfo->getFramePointer(),
                                 -StackOffset, RegClass->getDataSize());
  ++NumReloaded;    // Update statistics
  return PhysReg;
}

/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
///
void RA::EliminatePHINodes(MachineBasicBlock &MBB) {
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
        if (!MII.isBranch(opMI->getOpcode()))
          ++opI;

        unsigned dataSize = MF->getRegClass(virtualReg)->getDataSize();

        // Retrieve the constant value from this op, move it to target
        // register of the phi
        if (opVal.isImmediate()) {
          opI = RegInfo->moveImm2Reg(opBlock, opI, virtualReg,
                                     (unsigned) opVal.getImmedValue(),
                                     dataSize);
        } else {
          opI = RegInfo->moveReg2Reg(opBlock, opI, virtualReg,
                                     opVal.getAllocatedRegNum(), dataSize);
        }
      }
    }
    
    // really delete the PHI instruction now!
    delete MI;
  }
}

void RA::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  MachineBasicBlock::iterator I = MBB.begin();
  for (; I != MBB.end(); ++I) {
    MachineInstr *MI = *I;

    // Loop over all of the operands of the instruction, spilling registers that
    // are defined, and marking explicit destinations in the PhysRegsUsed map.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if (MI->getOperand(i).opIsDef() &&
          MI->getOperand(i).isPhysicalRegister()) {
        unsigned Reg  = MI->getOperand(i).getAllocatedRegNum();
        unsigned VMap = PhysRegsUsed[Reg];
        if (VMap) {  // Spill the value in this register...
          spillVirtReg(MBB, I, VMap, Reg);
          PhysRegsUsed[Reg] = 0;  // It's free now, and it's reserved
        }
        PhysRegsUseOrder.push_back(Reg);
      }

    // FIXME: Loop over the implicit defs, spilling them, as above.


    // FIXME: Loop over the implicit uses, making sure that they are at the head
    // of the use order list, so they don't get reallocated.

    // Loop over all of the operands again, getting the used operands into
    // registers.  This has the potiential to spill incoming values because we
    // are out of registers.
    //
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if (MI->getOperand(i).opIsUse() &&
          MI->getOperand(i).isVirtualRegister()) {
        unsigned VirtSrcReg = MI->getOperand(i).getAllocatedRegNum();
        unsigned PhysSrcReg = reloadVirtReg(MBB, I, VirtSrcReg);
        MI->SetMachineOperandReg(i, PhysSrcReg);  // Assign the input register
      }
    
    // Okay, we have allocated all of the source operands and spilled any values
    // that would be destroyed by defs of this instruction.  Loop over the
    // implicit defs and assign them to a register, spilling the incoming value
    // if we need to scavange a register.

    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if (MI->getOperand(i).opIsDef() &&
          !MI->getOperand(i).isPhysicalRegister()) {
        unsigned DestVirtReg = MI->getOperand(i).getAllocatedRegNum();
        unsigned DestPhysReg;

        if (TM.getInstrInfo().isTwoAddrInstr(MI->getOpcode()) && i == 0) {
          // must be same register number as the first operand
          // This maps a = b + c into b += c, and saves b into a's spot
          assert(MI->getOperand(1).isRegister()  &&
                 MI->getOperand(1).getAllocatedRegNum() &&
                 MI->getOperand(1).opIsUse() &&
                 "Two address instruction invalid!");
          DestPhysReg = MI->getOperand(1).getAllocatedRegNum();

          // Spill the incoming value, because we are about to change the
          // register contents.
          spillVirtReg(MBB, I, PhysRegsUsed[DestPhysReg], DestPhysReg);
          AssignVirtToPhysReg(DestVirtReg, DestPhysReg);
        } else {
          DestPhysReg = getFreeReg(MBB, I, DestVirtReg);
        }
        MI->SetMachineOperandReg(i, DestPhysReg);  // Assign the output register
      }
  }

  // Rewind the iterator to point to the first flow control instruction...
  const MachineInstrInfo &MII = TM.getInstrInfo();
  I = MBB.end();
  do {
    --I;
  } while ((MII.isReturn((*I)->getOpcode()) ||
            MII.isBranch((*I)->getOpcode())) && I != MBB.begin());
           
  if (!MII.isReturn((*I)->getOpcode()) && !MII.isBranch((*I)->getOpcode()))
    ++I;

  // Spill all physical registers holding virtual registers now.
  while (!PhysRegsUsed.empty())
    spillVirtReg(MBB, I, PhysRegsUsed.begin()->second,
                 PhysRegsUsed.begin()->first);

  assert(Virt2PhysRegMap.empty() && "Virtual registers still in phys regs?");
  assert(PhysRegsUseOrder.empty() && "Physical regs still allocated?");
}

/// runOnMachineFunction - Register allocate the whole function
///
bool RA::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(std::cerr << "Machine Function " << "\n");
  MF = &Fn;

  // First pass: eliminate PHI instructions by inserting copies into predecessor
  // blocks.
  // FIXME: In this pass, count how many uses of each VReg exist!
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

  // Add prologue to the function...
  RegInfo->emitPrologue(Fn, NumBytesAllocated);

  const MachineInstrInfo &MII = TM.getInstrInfo();

  // Add epilogue to restore the callee-save registers in each exiting block
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    // If last instruction is a return instruction, add an epilogue
    if (MII.isReturn(MBB->back()->getOpcode()))
      RegInfo->emitEpilogue(*MBB, NumBytesAllocated);
  }

  cleanupAfterFunction();
  return true;
}

Pass *createLocalRegisterAllocator(TargetMachine &TM) {
  return new RA(TM);
}
