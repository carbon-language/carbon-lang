//===-- RegAllocLocal.cpp - A BasicBlock generic register allocator -------===//
//
// This register allocator allocates registers to a basic block at a time,
// attempting to keep values in registers and reusing registers as appropriate.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/FunctionFrameInfo.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Statistic.h"
#include "Support/CommandLine.h"
#include <iostream>
#include <set>

namespace {
  Statistic<> NumSpilled ("ra-local", "Number of registers spilled");
  Statistic<> NumReloaded("ra-local", "Number of registers reloaded");
  cl::opt<bool> DisableKill("no-kill", cl::Hidden, 
                            cl::desc("Disable register kill in local-ra"));

  class RA : public MachineFunctionPass {
    const TargetMachine *TM;
    MachineFunction *MF;
    const MRegisterInfo *RegInfo;

    // StackSlotForVirtReg - Maps SSA Regs => frame index where these values are
    // spilled
    std::map<unsigned, int> StackSlotForVirtReg;

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

    // LastUserOf map - This multimap contains the set of registers that each
    // key instruction is the last user of.  If an instruction has an entry in
    // this map, that means that the specified operands are killed after the 
    // instruction is executed, thus they don't need to be spilled into memory
    //
    std::multimap<MachineInstr*, unsigned> LastUserOf;

    void MarkPhysRegRecentlyUsed(unsigned Reg) {
      assert(!PhysRegsUseOrder.empty() && "No registers used!");
      if (PhysRegsUseOrder.back() == Reg) return;  // Already most recently used

      for (unsigned i = PhysRegsUseOrder.size(); i != 0; --i)
	if (areRegsEqual(Reg, PhysRegsUseOrder[i-1])) {
	  unsigned RegMatch = PhysRegsUseOrder[i-1];       // remove from middle
	  PhysRegsUseOrder.erase(PhysRegsUseOrder.begin()+i-1);
	  // Add it to the end of the list
	  PhysRegsUseOrder.push_back(RegMatch);
	  if (RegMatch == Reg) 
	    return;    // Found an exact match, exit early
	}
    }

  public:
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

    /// CalculateLastUseOfVReg - Calculate an approximation of the killing
    /// uses for the virtual registers in the function.  Here we try to capture
    /// registers that are defined and only used within the same basic block.
    /// Because we don't have use-def chains yet, we have to do this the hard
    /// way.
    ///
    void CalculateLastUseOfVReg(MachineBasicBlock &MBB,
                        std::map<unsigned, MachineInstr*> &LastUseOfVReg) const;


    /// areRegsEqual - This method returns true if the specified registers are
    /// related to each other.  To do this, it checks to see if they are equal
    /// or if the first register is in the alias set of the second register.
    ///
    bool areRegsEqual(unsigned R1, unsigned R2) const {
      if (R1 == R2) return true;
      if (const unsigned *AliasSet = RegInfo->getAliasSet(R2))
        for (unsigned i = 0; AliasSet[i]; ++i)
          if (AliasSet[i] == R1) return true;
      return false;
    }

    /// getStackSpaceFor - This returns the frame index of the specified virtual
    /// register on the stack, allocating space if neccesary.
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);

    void removePhysReg(unsigned PhysReg);

    /// spillVirtReg - This method spills the value specified by PhysReg into
    /// the virtual register slot specified by VirtReg.  It then updates the RA
    /// data structures to indicate the fact that PhysReg is now available.
    ///
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                      unsigned VirtReg, unsigned PhysReg);

    /// spillPhysReg - This method spills the specified physical register into
    /// the virtual register slot associated with it.
    //
    void spillPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                      unsigned PhysReg) {
      std::map<unsigned, unsigned>::iterator PI = PhysRegsUsed.find(PhysReg);
      if (PI != PhysRegsUsed.end()) {             // Only spill it if it's used!
        spillVirtReg(MBB, I, PI->second, PhysReg);
      } else if (const unsigned *AliasSet = RegInfo->getAliasSet(PhysReg)) {
        // If the selected register aliases any other registers, we must make
        // sure that one of the aliases isn't alive...
        for (unsigned i = 0; AliasSet[i]; ++i) {
          PI = PhysRegsUsed.find(AliasSet[i]);
          if (PI != PhysRegsUsed.end())     // Spill aliased register...
            spillVirtReg(MBB, I, PI->second, AliasSet[i]);
        }
      }
    }

    void AssignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg);

    /// isPhysRegAvailable - Return true if the specified physical register is
    /// free and available for use.  This also includes checking to see if
    /// aliased registers are all free...
    ///
    bool isPhysRegAvailable(unsigned PhysReg) const;
    
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
int RA::getStackSpaceFor(unsigned VirtReg,
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


/// removePhysReg - This method marks the specified physical register as no 
/// longer being in use.
///
void RA::removePhysReg(unsigned PhysReg) {
  PhysRegsUsed.erase(PhysReg);      // PhyReg no longer used

  std::vector<unsigned>::iterator It =
    std::find(PhysRegsUseOrder.begin(), PhysRegsUseOrder.end(), PhysReg);
  assert(It != PhysRegsUseOrder.end() &&
         "Spilled a physical register, but it was not in use list!");
  PhysRegsUseOrder.erase(It);
}

/// spillVirtReg - This method spills the value specified by PhysReg into the
/// virtual register slot specified by VirtReg.  It then updates the RA data
/// structures to indicate the fact that PhysReg is now available.
///
void RA::spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                      unsigned VirtReg, unsigned PhysReg) {
  // If this is just a marker register, we don't need to spill it.
  if (VirtReg != 0) {
    const TargetRegisterClass *RegClass =
      MF->getSSARegMap()->getRegClass(VirtReg);
    int FrameIndex = getStackSpaceFor(VirtReg, RegClass);

    // Add move instruction(s)
    RegInfo->storeRegToStackSlot(MBB, I, PhysReg, FrameIndex, RegClass);
    ++NumSpilled;   // Update statistics
    Virt2PhysRegMap.erase(VirtReg);   // VirtReg no longer available
  }

  removePhysReg(PhysReg);
}


/// isPhysRegAvailable - Return true if the specified physical register is free
/// and available for use.  This also includes checking to see if aliased
/// registers are all free...
///
bool RA::isPhysRegAvailable(unsigned PhysReg) const {
  if (PhysRegsUsed.count(PhysReg)) return false;

  // If the selected register aliases any other allocated registers, it is
  // not free!
  if (const unsigned *AliasSet = RegInfo->getAliasSet(PhysReg))
    for (unsigned i = 0; AliasSet[i]; ++i)
      if (PhysRegsUsed.count(AliasSet[i])) // Aliased register in use?
        return false;                      // Can't use this reg then.
  return true;
}



/// getFreeReg - Find a physical register to hold the specified virtual
/// register.  If all compatible physical registers are used, this method spills
/// the last used virtual register to the stack, and uses that register.
///
unsigned RA::getFreeReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                        unsigned VirtReg) {
  const TargetRegisterClass *RC = MF->getSSARegMap()->getRegClass(VirtReg);

  // Get iterators defining the range of registers that are valid to allocate in
  // this class, which also specifies the preferred allocation order.
  TargetRegisterClass::iterator RI = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator RE = RC->allocation_order_end(*MF);

  // First check to see if we have a free register of the requested type...
  unsigned PhysReg = 0;
  for (; RI != RE; ++RI) {
    unsigned R = *RI;
    if (isPhysRegAvailable(R)) {       // Is reg unused?
      // Found an unused register!
      PhysReg = R;
      assert(PhysReg != 0 && "Cannot use register!");
      break;
    }
  }

  // If we didn't find an unused register, scavenge one now!
  if (PhysReg == 0) {
    assert(!PhysRegsUseOrder.empty() && "No allocated registers??");

    // Loop over all of the preallocated registers from the least recently used
    // to the most recently used.  When we find one that is capable of holding
    // our register, use it.
    for (unsigned i = 0; PhysReg == 0; ++i) {
      assert(i != PhysRegsUseOrder.size() &&
             "Couldn't find a register of the appropriate class!");
      
      unsigned R = PhysRegsUseOrder[i];
      // If the current register is compatible, use it.
      if (RegInfo->getRegClass(R) == RC) {
	PhysReg = R;
	break;
      } else {
	// If one of the registers aliased to the current register is
	// compatible, use it.
	if (const unsigned *AliasSet = RegInfo->getAliasSet(R))
	  for (unsigned a = 0; AliasSet[a]; ++a)
	    if (RegInfo->getRegClass(AliasSet[a]) == RC) {
	      PhysReg = AliasSet[a];    // Take an aliased register
	      break;
	    }
      }
    }

    assert(PhysReg && "Physical register not assigned!?!?");

    // At this point PhysRegsUseOrder[i] is the least recently used register of
    // compatible register class.  Spill it to memory and reap its remains.
    spillPhysReg(MBB, I, PhysReg);
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

  const TargetRegisterClass *RC = MF->getSSARegMap()->getRegClass(VirtReg);
  int FrameIndex = getStackSpaceFor(VirtReg, RC);

  // Add move instruction(s)
  RegInfo->loadRegFromStackSlot(MBB, I, PhysReg, FrameIndex, RC);
  ++NumReloaded;    // Update statistics
  return PhysReg;
}

/// CalculateLastUseOfVReg - Calculate an approximation of the killing uses for
/// the virtual registers in the function.  Here we try to capture registers 
/// that are defined and only used within the same basic block.  Because we 
/// don't have use-def chains yet, we have to do this the hard way.
///
void RA::CalculateLastUseOfVReg(MachineBasicBlock &MBB, 
                      std::map<unsigned, MachineInstr*> &LastUseOfVReg) const {
  // Calculate the last machine instruction in this basic block that uses the
  // specified virtual register defined in this basic block.
  std::map<unsigned, MachineInstr*> LastLocalUses;

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;++I){
    MachineInstr *MI = *I;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &Op = MI->getOperand(i);
      if (Op.isVirtualRegister()) {
        if (Op.opIsDef()) {   // Definition of a new virtual reg?
          LastLocalUses[Op.getAllocatedRegNum()] = 0;  // Record it
        } else {              // Use of a virtual reg.
          std::map<unsigned, MachineInstr*>::iterator It = 
                               LastLocalUses.find(Op.getAllocatedRegNum());
          if (It != LastLocalUses.end())  // Local use?
            It->second = MI;              // Update last use
          else
            LastUseOfVReg[Op.getAllocatedRegNum()] = 0;
        }
      }
    }
  }

  // Move local uses over... if there are any uses of a local already in the 
  // lastuse map, the newly inserted entry is ignored.
  LastUseOfVReg.insert(LastLocalUses.begin(), LastLocalUses.end());
}
 

/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
///
void RA::EliminatePHINodes(MachineBasicBlock &MBB) {
  const MachineInstrInfo &MII = TM->getInstrInfo();

  while (MBB.front()->getOpcode() == MachineInstrInfo::PHI) {
    MachineInstr *MI = MBB.front();
    // Unlink the PHI node from the basic block... but don't delete the PHI yet
    MBB.erase(MBB.begin());
    
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


void RA::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  MachineBasicBlock::iterator I = MBB.begin();
  for (; I != MBB.end(); ++I) {
    MachineInstr *MI = *I;
    const MachineInstrDescriptor &MID = TM->getInstrInfo().get(MI->getOpcode());

    // Loop over all of the operands of the instruction, spilling registers that
    // are defined, and marking explicit destinations in the PhysRegsUsed map.

    // FIXME: We don't need to spill a register if this is the last use of the
    // value!
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if (MI->getOperand(i).opIsDef() &&
          MI->getOperand(i).isPhysicalRegister()) {
        unsigned Reg = MI->getOperand(i).getAllocatedRegNum();
        spillPhysReg(MBB, I, Reg);
        PhysRegsUsed[Reg] = 0;            // It is free and reserved now
        PhysRegsUseOrder.push_back(Reg);
      }

    // Loop over the implicit defs, spilling them, as above.
    if (const unsigned *ImplicitDefs = MID.ImplicitDefs)
      for (unsigned i = 0; ImplicitDefs[i]; ++i) {
        unsigned Reg = ImplicitDefs[i];

        // We don't want to spill implicit definitions if they were explicitly
        // chosen.  For this reason, check to see now if the register we are
        // to spill has a vreg of 0.
        if (PhysRegsUsed.count(Reg) && PhysRegsUsed[Reg] != 0)
          spillPhysReg(MBB, I, Reg);
	else if (PhysRegsUsed.count(Reg)) {
	  // Remove the entry from PhysRegsUseOrder to avoid having two entries!
	  removePhysReg(Reg);
	}
	PhysRegsUseOrder.push_back(Reg);
	PhysRegsUsed[Reg] = 0;            // It is free and reserved now
      }

    // Loop over the implicit uses, making sure that they are at the head of the
    // use order list, so they don't get reallocated.
    if (const unsigned *ImplicitUses = MID.ImplicitUses)
      for (unsigned i = 0; ImplicitUses[i]; ++i)
        MarkPhysRegRecentlyUsed(ImplicitUses[i]);

    // Loop over all of the operands again, getting the used operands into
    // registers.  This has the potiential to spill incoming values if we are
    // out of registers.
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
    //
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if (MI->getOperand(i).opIsDef() &&
          !MI->getOperand(i).isPhysicalRegister()) {
        unsigned DestVirtReg = MI->getOperand(i).getAllocatedRegNum();
        unsigned DestPhysReg;

        if (TM->getInstrInfo().isTwoAddrInstr(MI->getOpcode()) && i == 0) {
          // must be same register number as the first operand
          // This maps a = b + c into b += c, and saves b into a's spot
          assert(MI->getOperand(1).isRegister()  &&
                 MI->getOperand(1).getAllocatedRegNum() &&
                 MI->getOperand(1).opIsUse() &&
                 "Two address instruction invalid!");
          DestPhysReg = MI->getOperand(1).getAllocatedRegNum();

          // Spill the incoming value, because we are about to change the
          // register contents.
          spillPhysReg(MBB, I, DestPhysReg);
          AssignVirtToPhysReg(DestVirtReg, DestPhysReg);
        } else {
          DestPhysReg = getFreeReg(MBB, I, DestVirtReg);
        }
        MI->SetMachineOperandReg(i, DestPhysReg);  // Assign the output register
      }

    if (!DisableKill) {
      // If this instruction is the last user of anything in registers, kill the
      // value, freeing the register being used, so it doesn't need to be
      // spilled to memory at the end of the block.
      std::multimap<MachineInstr*, unsigned>::iterator LUOI = 
             LastUserOf.lower_bound(MI);
      for (; LUOI != LastUserOf.end() && LUOI->first == MI; ++MI) {
        unsigned VirtReg = LUOI->second;                       // entry found?
        unsigned PhysReg = Virt2PhysRegMap[VirtReg];
        if (PhysReg) {
          DEBUG(std::cout << "V: " << VirtReg << " P: " << PhysReg
		          << " Last use of: " << *MI);
          removePhysReg(PhysReg);
        }
        Virt2PhysRegMap.erase(VirtReg);
      }
    }
  }

  // Rewind the iterator to point to the first flow control instruction...
  const MachineInstrInfo &MII = TM->getInstrInfo();
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
  TM = &Fn.getTarget();
  RegInfo = TM->getRegisterInfo();

  // First pass: eliminate PHI instructions by inserting copies into predecessor
  // blocks, and calculate a simple approximation of killing uses for virtual 
  // registers.
  //
  std::map<unsigned, MachineInstr*> LastUseOfVReg;
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    if (!DisableKill)
      CalculateLastUseOfVReg(*MBB, LastUseOfVReg);
    EliminatePHINodes(*MBB);
  }

  // At this point LastUseOfVReg has been filled in to contain the last 
  // MachineInstr user of the specified virtual register, if that user is 
  // within the same basic block as the definition (otherwise it contains
  // null).  Invert this mapping now:
  if (!DisableKill)
    for (std::map<unsigned, MachineInstr*>::iterator I = LastUseOfVReg.begin(),
         E = LastUseOfVReg.end(); I != E; ++I)
      if (I->second)
        LastUserOf.insert(std::make_pair(I->second, I->first));

  // We're done with the temporary list now.
  LastUseOfVReg.clear();

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  LastUserOf.clear();
  StackSlotForVirtReg.clear();
  return true;
}

Pass *createLocalRegisterAllocator() {
  return new RA();
}
