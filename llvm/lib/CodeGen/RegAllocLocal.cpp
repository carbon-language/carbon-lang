//===-- RegAllocLocal.cpp - A BasicBlock generic register allocator -------===//
//
// This register allocator allocates registers to a basic block at a time,
// attempting to keep values in registers and reusing registers as appropriate.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include <iostream>

namespace {
  Statistic<> NumSpilled ("ra-local", "Number of registers spilled");
  Statistic<> NumReloaded("ra-local", "Number of registers reloaded");
  cl::opt<bool> DisableKill("no-kill", cl::Hidden, 
                            cl::desc("Disable register kill in local-ra"));

  class RA : public MachineFunctionPass {
    const TargetMachine *TM;
    MachineFunction *MF;
    const MRegisterInfo *RegInfo;
    LiveVariables *LV;

    // StackSlotForVirtReg - Maps virtual regs to the frame index where these
    // values are spilled.
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

    // VirtRegModified - This bitset contains information about which virtual
    // registers need to be spilled back to memory when their registers are
    // scavenged.  If a virtual register has simply been rematerialized, there
    // is no reason to spill it to memory when we need the register back.
    //
    std::vector<bool> VirtRegModified;

    void markVirtRegModified(unsigned Reg, bool Val = true) {
      assert(Reg >= MRegisterInfo::FirstVirtualRegister && "Illegal VirtReg!");
      Reg -= MRegisterInfo::FirstVirtualRegister;
      if (VirtRegModified.size() <= Reg) VirtRegModified.resize(Reg+1);
      VirtRegModified[Reg] = Val;
    }

    bool isVirtRegModified(unsigned Reg) const {
      assert(Reg >= MRegisterInfo::FirstVirtualRegister && "Illegal VirtReg!");
      assert(Reg - MRegisterInfo::FirstVirtualRegister < VirtRegModified.size()
	     && "Illegal virtual register!");
      return VirtRegModified[Reg - MRegisterInfo::FirstVirtualRegister];
    }

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

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      if (!DisableKill)
	AU.addRequired<LiveVariables>();
      AU.addRequiredID(PHIEliminationID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    /// runOnMachineFunction - Register allocate the whole function
    bool runOnMachineFunction(MachineFunction &Fn);

    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);


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
    /// register on the stack, allocating space if necessary.
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);

    /// removePhysReg - This method marks the specified physical register as no
    /// longer being in use.
    ///
    void removePhysReg(unsigned PhysReg);

    /// spillVirtReg - This method spills the value specified by PhysReg into
    /// the virtual register slot specified by VirtReg.  It then updates the RA
    /// data structures to indicate the fact that PhysReg is now available.
    ///
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                      unsigned VirtReg, unsigned PhysReg);

    /// spillPhysReg - This method spills the specified physical register into
    /// the virtual register slot associated with it.
    ///
    void spillPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                      unsigned PhysReg);

    /// assignVirtToPhysReg - This method updates local state so that we know
    /// that PhysReg is the proper container for VirtReg now.  The physical
    /// register must not be used for anything else when this is called.
    ///
    void assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg);

    /// liberatePhysReg - Make sure the specified physical register is available
    /// for use.  If there is currently a value in it, it is either moved out of
    /// the way or spilled to memory.
    ///
    void liberatePhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
			 unsigned PhysReg);

    /// isPhysRegAvailable - Return true if the specified physical register is
    /// free and available for use.  This also includes checking to see if
    /// aliased registers are all free...
    ///
    bool isPhysRegAvailable(unsigned PhysReg) const;

    /// getFreeReg - Look to see if there is a free register available in the
    /// specified register class.  If not, return 0.
    ///
    unsigned getFreeReg(const TargetRegisterClass *RC);
    
    /// getReg - Find a physical register to hold the specified virtual
    /// register.  If all compatible physical registers are used, this method
    /// spills the last used virtual register to the stack, and uses that
    /// register.
    ///
    unsigned getReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
		    unsigned VirtReg);

    /// reloadVirtReg - This method loads the specified virtual register into a
    /// physical register, returning the physical register chosen.  This updates
    /// the regalloc data structures to reflect the fact that the virtual reg is
    /// now alive in a physical register, and the previous one isn't.
    ///
    unsigned reloadVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &I, unsigned VirtReg);

    void reloadPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                       unsigned PhysReg);
  };
}


/// getStackSpaceFor - This allocates space for the specified virtual register
/// to be held on the stack.
int RA::getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC) {
  // Find the location Reg would belong...
  std::map<unsigned, int>::iterator I =StackSlotForVirtReg.lower_bound(VirtReg);

  if (I != StackSlotForVirtReg.end() && I->first == VirtReg)
    return I->second;          // Already has space allocated?

  // Allocate a new stack object for this spill location...
  int FrameIdx = MF->getFrameInfo()->CreateStackObject(RC);

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

  DEBUG(std::cerr << "  Spilling register " << RegInfo->getName(PhysReg));
  if (VirtReg == 0) {
    DEBUG(std::cerr << " which corresponds to no vreg, "
                    << "must be spurious physreg: ignoring (WARNING)\n");
  } else {
    DEBUG(std::cerr << " containing %reg" << VirtReg;
          if (!isVirtRegModified(VirtReg))
           std::cerr << " which has not been modified, so no store necessary!");

    // Otherwise, there is a virtual register corresponding to this physical
    // register.  We only need to spill it into its stack slot if it has been
    // modified.
    if (isVirtRegModified(VirtReg)) {
      const TargetRegisterClass *RC = MF->getSSARegMap()->getRegClass(VirtReg);
      int FrameIndex = getStackSpaceFor(VirtReg, RC);
      DEBUG(std::cerr << " to stack slot #" << FrameIndex);
      RegInfo->storeRegToStackSlot(MBB, I, PhysReg, FrameIndex, RC);
      ++NumSpilled;   // Update statistics
    }
    Virt2PhysRegMap.erase(VirtReg);   // VirtReg no longer available
  }

  DEBUG(std::cerr << "\n");
  removePhysReg(PhysReg);
}


/// spillPhysReg - This method spills the specified physical register into the
/// virtual register slot associated with it.
///
void RA::spillPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
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


/// assignVirtToPhysReg - This method updates local state so that we know
/// that PhysReg is the proper container for VirtReg now.  The physical
/// register must not be used for anything else when this is called.
///
void RA::assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg) {
  assert(PhysRegsUsed.find(PhysReg) == PhysRegsUsed.end() &&
         "Phys reg already assigned!");
  // Update information to note the fact that this register was just used, and
  // it holds VirtReg.
  PhysRegsUsed[PhysReg] = VirtReg;
  Virt2PhysRegMap[VirtReg] = PhysReg;
  PhysRegsUseOrder.push_back(PhysReg);   // New use of PhysReg
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


/// getFreeReg - Look to see if there is a free register available in the
/// specified register class.  If not, return 0.
///
unsigned RA::getFreeReg(const TargetRegisterClass *RC) {
  // Get iterators defining the range of registers that are valid to allocate in
  // this class, which also specifies the preferred allocation order.
  TargetRegisterClass::iterator RI = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator RE = RC->allocation_order_end(*MF);

  for (; RI != RE; ++RI)
    if (isPhysRegAvailable(*RI)) {       // Is reg unused?
      assert(*RI != 0 && "Cannot use register!");
      return *RI; // Found an unused register!
    }
  return 0;
}


/// liberatePhysReg - Make sure the specified physical register is available for
/// use.  If there is currently a value in it, it is either moved out of the way
/// or spilled to memory.
///
void RA::liberatePhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
			 unsigned PhysReg) {
  // FIXME: This code checks to see if a register is available, but it really
  // wants to know if a reg is available BEFORE the instruction executes.  If
  // called after killed operands are freed, it runs the risk of reallocating a
  // used operand...
#if 0
  if (isPhysRegAvailable(PhysReg)) return;  // Already available...

  // Check to see if the register is directly used, not indirectly used through
  // aliases.  If aliased registers are the ones actually used, we cannot be
  // sure that we will be able to save the whole thing if we do a reg-reg copy.
  std::map<unsigned, unsigned>::iterator PRUI = PhysRegsUsed.find(PhysReg);
  if (PRUI != PhysRegsUsed.end()) {
    unsigned VirtReg = PRUI->second;   // The virtual register held...

    // Check to see if there is a compatible register available.  If so, we can
    // move the value into the new register...
    //
    const TargetRegisterClass *RC = RegInfo->getRegClass(PhysReg);
    if (unsigned NewReg = getFreeReg(RC)) {
      // Emit the code to copy the value...
      RegInfo->copyRegToReg(MBB, I, NewReg, PhysReg, RC);
      
      // Update our internal state to indicate that PhysReg is available and Reg
      // isn't.
      Virt2PhysRegMap.erase(VirtReg);
      removePhysReg(PhysReg);  // Free the physreg
      
      // Move reference over to new register...
      assignVirtToPhysReg(VirtReg, NewReg);
      return;
    }
  }
#endif
  spillPhysReg(MBB, I, PhysReg);
}


/// getReg - Find a physical register to hold the specified virtual
/// register.  If all compatible physical registers are used, this method spills
/// the last used virtual register to the stack, and uses that register.
///
unsigned RA::getReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
		    unsigned VirtReg) {
  const TargetRegisterClass *RC = MF->getSSARegMap()->getRegClass(VirtReg);

  // First check to see if we have a free register of the requested type...
  unsigned PhysReg = getFreeReg(RC);

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
  assignVirtToPhysReg(VirtReg, PhysReg);
  return PhysReg;
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

  unsigned PhysReg = getReg(MBB, I, VirtReg);

  const TargetRegisterClass *RC = MF->getSSARegMap()->getRegClass(VirtReg);
  int FrameIndex = getStackSpaceFor(VirtReg, RC);

  markVirtRegModified(VirtReg, false);   // Note that this reg was just reloaded

  DEBUG(std::cerr << "  Reloading %reg" << VirtReg << " into "
                  << RegInfo->getName(PhysReg) << "\n");

  // Add move instruction(s)
  RegInfo->loadRegFromStackSlot(MBB, I, PhysReg, FrameIndex, RC);
  ++NumReloaded;    // Update statistics
  return PhysReg;
}



void RA::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  MachineBasicBlock::iterator I = MBB.begin();
  for (; I != MBB.end(); ++I) {
    MachineInstr *MI = *I;
    const TargetInstrDescriptor &TID = TM->getInstrInfo().get(MI->getOpcode());
    DEBUG(std::cerr << "\nStarting RegAlloc of: " << *MI;
          std::cerr << "  Regs have values: ";
          for (std::map<unsigned, unsigned>::const_iterator
                 I = PhysRegsUsed.begin(), E = PhysRegsUsed.end(); I != E; ++I)
             std::cerr << "[" << RegInfo->getName(I->first)
                       << ",%reg" << I->second << "] ";
          std::cerr << "\n");

    // Loop over the implicit uses, making sure that they are at the head of the
    // use order list, so they don't get reallocated.
    if (const unsigned *ImplicitUses = TID.ImplicitUses)
      for (unsigned i = 0; ImplicitUses[i]; ++i)
        MarkPhysRegRecentlyUsed(ImplicitUses[i]);

    // Get the used operands into registers.  This has the potiential to spill
    // incoming values if we are out of registers.  Note that we completely
    // ignore physical register uses here.  We assume that if an explicit
    // physical register is referenced by the instruction, that it is guaranteed
    // to be live-in, or the input is badly hosed.
    //
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if (MI->getOperand(i).opIsUse() && MI->getOperand(i).isVirtualRegister()){
        unsigned VirtSrcReg = MI->getOperand(i).getAllocatedRegNum();
        unsigned PhysSrcReg = reloadVirtReg(MBB, I, VirtSrcReg);
        MI->SetMachineOperandReg(i, PhysSrcReg);  // Assign the input register
      }
    
    if (!DisableKill) {
      // If this instruction is the last user of anything in registers, kill the
      // value, freeing the register being used, so it doesn't need to be
      // spilled to memory.
      //
      for (LiveVariables::killed_iterator KI = LV->killed_begin(MI),
             KE = LV->killed_end(MI); KI != KE; ++KI) {
        unsigned VirtReg = KI->second;
        unsigned PhysReg = VirtReg;
        if (VirtReg >= MRegisterInfo::FirstVirtualRegister) {
          std::map<unsigned, unsigned>::iterator I =
            Virt2PhysRegMap.find(VirtReg);
          assert(I != Virt2PhysRegMap.end());
          PhysReg = I->second;
          Virt2PhysRegMap.erase(I);
        }

        if (PhysReg) {
          DEBUG(std::cerr << "  Last use of " << RegInfo->getName(PhysReg)
                      << "[%reg" << VirtReg <<"], removing it from live set\n");
          // If the physical register was used, but there was no definition of
          // the physical register (we are reading garbage), Live Variables will
          // tell us that this is the last use of the register even though we
          // don't know of anything in the register.  No need to remove it.
          if (VirtReg != PhysReg || PhysRegsUsed.count(PhysReg))
            removePhysReg(PhysReg);
        }
      }
    }

    // Loop over all of the operands of the instruction, spilling registers that
    // are defined, and marking explicit destinations in the PhysRegsUsed map.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if ((MI->getOperand(i).opIsDefOnly() ||
           MI->getOperand(i).opIsDefAndUse()) &&
          MI->getOperand(i).isPhysicalRegister()) {
        unsigned Reg = MI->getOperand(i).getAllocatedRegNum();
        spillPhysReg(MBB, I, Reg);        // Spill any existing value in the reg
        PhysRegsUsed[Reg] = 0;            // It is free and reserved now
        PhysRegsUseOrder.push_back(Reg);
      }

    // Loop over the implicit defs, spilling them as well.
    if (const unsigned *ImplicitDefs = TID.ImplicitDefs)
      for (unsigned i = 0; ImplicitDefs[i]; ++i) {
        unsigned Reg = ImplicitDefs[i];
        spillPhysReg(MBB, I, Reg);
        PhysRegsUseOrder.push_back(Reg);
        PhysRegsUsed[Reg] = 0;            // It is free and reserved now
      }

    // Okay, we have allocated all of the source operands and spilled any values
    // that would be destroyed by defs of this instruction.  Loop over the
    // implicit defs and assign them to a register, spilling incoming values if
    // we need to scavenge a register.
    //
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      if ((MI->getOperand(i).opIsDefOnly() || MI->getOperand(i).opIsDefAndUse())
          && MI->getOperand(i).isVirtualRegister()) {
        unsigned DestVirtReg = MI->getOperand(i).getAllocatedRegNum();
        unsigned DestPhysReg;

        // If DestVirtReg already has a value, forget about it.  Why doesn't
        // getReg do this right?
        std::map<unsigned, unsigned>::iterator DestI =
          Virt2PhysRegMap.find(DestVirtReg);
        if (DestI != Virt2PhysRegMap.end()) {
          unsigned PhysReg = DestI->second;
          Virt2PhysRegMap.erase(DestI);
          removePhysReg(PhysReg);
        }

        if (TM->getInstrInfo().isTwoAddrInstr(MI->getOpcode()) && i == 0) {
          // must be same register number as the first operand
          // This maps a = b + c into b += c, and saves b into a's spot
          assert(MI->getOperand(1).isRegister()  &&
                 MI->getOperand(1).getAllocatedRegNum() &&
                 MI->getOperand(1).opIsUse() &&
                 "Two address instruction invalid!");
          DestPhysReg = MI->getOperand(1).getAllocatedRegNum();

          liberatePhysReg(MBB, I, DestPhysReg);
          assignVirtToPhysReg(DestVirtReg, DestPhysReg);
        } else {
          DestPhysReg = getReg(MBB, I, DestVirtReg);
        }
        markVirtRegModified(DestVirtReg);
        MI->SetMachineOperandReg(i, DestPhysReg);  // Assign the output register
      }

    if (!DisableKill) {
      // If this instruction defines any registers that are immediately dead,
      // kill them now.
      //
      for (LiveVariables::killed_iterator KI = LV->dead_begin(MI),
             KE = LV->dead_end(MI); KI != KE; ++KI) {
        unsigned VirtReg = KI->second;
        unsigned PhysReg = VirtReg;
        if (VirtReg >= MRegisterInfo::FirstVirtualRegister) {
          std::map<unsigned, unsigned>::iterator I =
            Virt2PhysRegMap.find(VirtReg);
          assert(I != Virt2PhysRegMap.end());
          PhysReg = I->second;
          Virt2PhysRegMap.erase(I);
        }

        if (PhysReg) {
          DEBUG(std::cerr << "  Register " << RegInfo->getName(PhysReg)
                          << " [%reg" << VirtReg
                          << "] is never used, removing it frame live list\n");
          removePhysReg(PhysReg);
        }
      }
    }
  }

  // Rewind the iterator to point to the first flow control instruction...
  const TargetInstrInfo &TII = TM->getInstrInfo();
  I = MBB.end();
  while (I != MBB.begin() && TII.isTerminatorInstr((*(I-1))->getOpcode()))
    --I;

  // Spill all physical registers holding virtual registers now.
  while (!PhysRegsUsed.empty())
    spillVirtReg(MBB, I, PhysRegsUsed.begin()->second,
                 PhysRegsUsed.begin()->first);

  for (std::map<unsigned, unsigned>::iterator I = Virt2PhysRegMap.begin(),
         E = Virt2PhysRegMap.end(); I != E; ++I)
    std::cerr << "Register still mapped: " << I->first << " -> "
              << I->second << "\n";

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

  if (!DisableKill)
    LV = &getAnalysis<LiveVariables>();

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  StackSlotForVirtReg.clear();
  VirtRegModified.clear();
  return true;
}

Pass *createLocalRegisterAllocator() {
  return new RA();
}
