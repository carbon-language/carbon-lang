//===-- RegAllocLocal.cpp - A BasicBlock generic register allocator -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
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
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Visibility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
#include <iostream>
using namespace llvm;

namespace {
  static Statistic<> NumStores("ra-local", "Number of stores added");
  static Statistic<> NumLoads ("ra-local", "Number of loads added");
  static Statistic<> NumFolded("ra-local", "Number of loads/stores folded "
			       "into instructions");

  static RegisterRegAlloc
    localRegAlloc("local", "  local register allocator",
                  createLocalRegisterAllocator);


  class VISIBILITY_HIDDEN RA : public MachineFunctionPass {
    const TargetMachine *TM;
    MachineFunction *MF;
    const MRegisterInfo *RegInfo;
    LiveVariables *LV;
    bool *PhysRegsEverUsed;

    // StackSlotForVirtReg - Maps virtual regs to the frame index where these
    // values are spilled.
    std::map<unsigned, int> StackSlotForVirtReg;

    // Virt2PhysRegMap - This map contains entries for each virtual register
    // that is currently available in a physical register.
    DenseMap<unsigned, VirtReg2IndexFunctor> Virt2PhysRegMap;

    unsigned &getVirt2PhysRegMapSlot(unsigned VirtReg) {
      return Virt2PhysRegMap[VirtReg];
    }

    // PhysRegsUsed - This array is effectively a map, containing entries for
    // each physical register that currently has a value (ie, it is in
    // Virt2PhysRegMap).  The value mapped to is the virtual register
    // corresponding to the physical register (the inverse of the
    // Virt2PhysRegMap), or 0.  The value is set to 0 if this register is pinned
    // because it is used by a future instruction.  If the entry for a physical
    // register is -1, then the physical register is "not in the map".
    //
    std::vector<int> PhysRegsUsed;

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
      assert(MRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      Reg -= MRegisterInfo::FirstVirtualRegister;
      if (VirtRegModified.size() <= Reg) VirtRegModified.resize(Reg+1);
      VirtRegModified[Reg] = Val;
    }

    bool isVirtRegModified(unsigned Reg) const {
      assert(MRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      assert(Reg - MRegisterInfo::FirstVirtualRegister < VirtRegModified.size()
             && "Illegal virtual register!");
      return VirtRegModified[Reg - MRegisterInfo::FirstVirtualRegister];
    }

    void MarkPhysRegRecentlyUsed(unsigned Reg) {
      if(PhysRegsUseOrder.empty() ||
         PhysRegsUseOrder.back() == Reg) return;  // Already most recently used

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
      AU.addRequired<LiveVariables>();
      AU.addRequiredID(PHIEliminationID);
      AU.addRequiredID(TwoAddressInstructionPassID);
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
      for (const unsigned *AliasSet = RegInfo->getAliasSet(R2);
           *AliasSet; ++AliasSet) {
        if (*AliasSet == R1) return true;
      }
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
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                      unsigned VirtReg, unsigned PhysReg);

    /// spillPhysReg - This method spills the specified physical register into
    /// the virtual register slot associated with it.  If OnlyVirtRegs is set to
    /// true, then the request is ignored if the physical register does not
    /// contain a virtual register.
    ///
    void spillPhysReg(MachineBasicBlock &MBB, MachineInstr *I,
                      unsigned PhysReg, bool OnlyVirtRegs = false);

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
    unsigned getReg(MachineBasicBlock &MBB, MachineInstr *MI,
                    unsigned VirtReg);

    /// reloadVirtReg - This method transforms the specified specified virtual
    /// register use to refer to a physical register.  This method may do this
    /// in one of several ways: if the register is available in a physical
    /// register already, it uses that physical register.  If the value is not
    /// in a physical register, and if there are physical registers available,
    /// it loads it into a register.  If register pressure is high, and it is
    /// possible, it tries to fold the load of the virtual register into the
    /// instruction itself.  It avoids doing this if register pressure is low to
    /// improve the chance that subsequent instructions can use the reloaded
    /// value.  This method returns the modified instruction.
    ///
    MachineInstr *reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                                unsigned OpNum);


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
  int FrameIdx = MF->getFrameInfo()->CreateStackObject(RC->getSize(),
                                                       RC->getAlignment());

  // Assign the slot...
  StackSlotForVirtReg.insert(I, std::make_pair(VirtReg, FrameIdx));
  return FrameIdx;
}


/// removePhysReg - This method marks the specified physical register as no
/// longer being in use.
///
void RA::removePhysReg(unsigned PhysReg) {
  PhysRegsUsed[PhysReg] = -1;      // PhyReg no longer used

  std::vector<unsigned>::iterator It =
    std::find(PhysRegsUseOrder.begin(), PhysRegsUseOrder.end(), PhysReg);
  if (It != PhysRegsUseOrder.end())
    PhysRegsUseOrder.erase(It);
}


/// spillVirtReg - This method spills the value specified by PhysReg into the
/// virtual register slot specified by VirtReg.  It then updates the RA data
/// structures to indicate the fact that PhysReg is now available.
///
void RA::spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                      unsigned VirtReg, unsigned PhysReg) {
  assert(VirtReg && "Spilling a physical register is illegal!"
         " Must not have appropriate kill for the register or use exists beyond"
         " the intended one.");
  DEBUG(std::cerr << "  Spilling register " << RegInfo->getName(PhysReg);
        std::cerr << " containing %reg" << VirtReg;
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
    ++NumStores;   // Update statistics
  }

  getVirt2PhysRegMapSlot(VirtReg) = 0;   // VirtReg no longer available

  DEBUG(std::cerr << "\n");
  removePhysReg(PhysReg);
}


/// spillPhysReg - This method spills the specified physical register into the
/// virtual register slot associated with it.  If OnlyVirtRegs is set to true,
/// then the request is ignored if the physical register does not contain a
/// virtual register.
///
void RA::spillPhysReg(MachineBasicBlock &MBB, MachineInstr *I,
                      unsigned PhysReg, bool OnlyVirtRegs) {
  if (PhysRegsUsed[PhysReg] != -1) {            // Only spill it if it's used!
    if (PhysRegsUsed[PhysReg] || !OnlyVirtRegs)
      spillVirtReg(MBB, I, PhysRegsUsed[PhysReg], PhysReg);
  } else {
    // If the selected register aliases any other registers, we must make
    // sure that one of the aliases isn't alive...
    for (const unsigned *AliasSet = RegInfo->getAliasSet(PhysReg);
         *AliasSet; ++AliasSet)
      if (PhysRegsUsed[*AliasSet] != -1)     // Spill aliased register...
        if (PhysRegsUsed[*AliasSet] || !OnlyVirtRegs)
          spillVirtReg(MBB, I, PhysRegsUsed[*AliasSet], *AliasSet);
  }
}


/// assignVirtToPhysReg - This method updates local state so that we know
/// that PhysReg is the proper container for VirtReg now.  The physical
/// register must not be used for anything else when this is called.
///
void RA::assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg) {
  assert(PhysRegsUsed[PhysReg] == -1 && "Phys reg already assigned!");
  // Update information to note the fact that this register was just used, and
  // it holds VirtReg.
  PhysRegsUsed[PhysReg] = VirtReg;
  getVirt2PhysRegMapSlot(VirtReg) = PhysReg;
  PhysRegsUseOrder.push_back(PhysReg);   // New use of PhysReg
}


/// isPhysRegAvailable - Return true if the specified physical register is free
/// and available for use.  This also includes checking to see if aliased
/// registers are all free...
///
bool RA::isPhysRegAvailable(unsigned PhysReg) const {
  if (PhysRegsUsed[PhysReg] != -1) return false;

  // If the selected register aliases any other allocated registers, it is
  // not free!
  for (const unsigned *AliasSet = RegInfo->getAliasSet(PhysReg);
       *AliasSet; ++AliasSet)
    if (PhysRegsUsed[*AliasSet] != -1) // Aliased register in use?
      return false;                    // Can't use this reg then.
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
  spillPhysReg(MBB, I, PhysReg);
}


/// getReg - Find a physical register to hold the specified virtual
/// register.  If all compatible physical registers are used, this method spills
/// the last used virtual register to the stack, and uses that register.
///
unsigned RA::getReg(MachineBasicBlock &MBB, MachineInstr *I,
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

      // We can only use this register if it holds a virtual register (ie, it
      // can be spilled).  Do not use it if it is an explicitly allocated
      // physical register!
      assert(PhysRegsUsed[R] != -1 &&
             "PhysReg in PhysRegsUseOrder, but is not allocated?");
      if (PhysRegsUsed[R]) {
        // If the current register is compatible, use it.
        if (RC->contains(R)) {
          PhysReg = R;
          break;
        } else {
          // If one of the registers aliased to the current register is
          // compatible, use it.
          for (const unsigned *AliasSet = RegInfo->getAliasSet(R);
               *AliasSet; ++AliasSet) {
            if (RC->contains(*AliasSet)) {
              PhysReg = *AliasSet;    // Take an aliased register
              break;
            }
          }
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


/// reloadVirtReg - This method transforms the specified specified virtual
/// register use to refer to a physical register.  This method may do this in
/// one of several ways: if the register is available in a physical register
/// already, it uses that physical register.  If the value is not in a physical
/// register, and if there are physical registers available, it loads it into a
/// register.  If register pressure is high, and it is possible, it tries to
/// fold the load of the virtual register into the instruction itself.  It
/// avoids doing this if register pressure is low to improve the chance that
/// subsequent instructions can use the reloaded value.  This method returns the
/// modified instruction.
///
MachineInstr *RA::reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                                unsigned OpNum) {
  unsigned VirtReg = MI->getOperand(OpNum).getReg();

  // If the virtual register is already available, just update the instruction
  // and return.
  if (unsigned PR = getVirt2PhysRegMapSlot(VirtReg)) {
    MarkPhysRegRecentlyUsed(PR);          // Already have this value available!
    MI->getOperand(OpNum).setReg(PR);  // Assign the input register
    return MI;
  }

  // Otherwise, we need to fold it into the current instruction, or reload it.
  // If we have registers available to hold the value, use them.
  const TargetRegisterClass *RC = MF->getSSARegMap()->getRegClass(VirtReg);
  unsigned PhysReg = getFreeReg(RC);
  int FrameIndex = getStackSpaceFor(VirtReg, RC);

  if (PhysReg) {   // Register is available, allocate it!
    assignVirtToPhysReg(VirtReg, PhysReg);
  } else {         // No registers available.
    // If we can fold this spill into this instruction, do so now.
    if (MachineInstr* FMI = RegInfo->foldMemoryOperand(MI, OpNum, FrameIndex)){
      ++NumFolded;
      // Since we changed the address of MI, make sure to update live variables
      // to know that the new instruction has the properties of the old one.
      LV->instructionChanged(MI, FMI);
      return MBB.insert(MBB.erase(MI), FMI);
    }

    // It looks like we can't fold this virtual register load into this
    // instruction.  Force some poor hapless value out of the register file to
    // make room for the new register, and reload it.
    PhysReg = getReg(MBB, MI, VirtReg);
  }

  markVirtRegModified(VirtReg, false);   // Note that this reg was just reloaded

  DEBUG(std::cerr << "  Reloading %reg" << VirtReg << " into "
                  << RegInfo->getName(PhysReg) << "\n");

  // Add move instruction(s)
  RegInfo->loadRegFromStackSlot(MBB, MI, PhysReg, FrameIndex, RC);
  ++NumLoads;    // Update statistics

  PhysRegsEverUsed[PhysReg] = true;
  MI->getOperand(OpNum).setReg(PhysReg);  // Assign the input register
  return MI;
}



void RA::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  MachineBasicBlock::iterator MII = MBB.begin();
  const TargetInstrInfo &TII = *TM->getInstrInfo();
  
  // If this is the first basic block in the machine function, add live-in
  // registers as active.
  if (&MBB == &*MF->begin()) {
    for (MachineFunction::livein_iterator I = MF->livein_begin(),
         E = MF->livein_end(); I != E; ++I) {
      unsigned Reg = I->first;
      PhysRegsEverUsed[Reg] = true;
      PhysRegsUsed[Reg] = 0;            // It is free and reserved now
      PhysRegsUseOrder.push_back(Reg);
      for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
           *AliasSet; ++AliasSet) {
        PhysRegsUseOrder.push_back(*AliasSet);
        PhysRegsUsed[*AliasSet] = 0;  // It is free and reserved now
        PhysRegsEverUsed[*AliasSet] = true;
      }
    }    
  }
  
  // Otherwise, sequentially allocate each instruction in the MBB.
  while (MII != MBB.end()) {
    MachineInstr *MI = MII++;
    const TargetInstrDescriptor &TID = TII.get(MI->getOpcode());
    DEBUG(std::cerr << "\nStarting RegAlloc of: " << *MI;
          std::cerr << "  Regs have values: ";
          for (unsigned i = 0; i != RegInfo->getNumRegs(); ++i)
            if (PhysRegsUsed[i] != -1)
               std::cerr << "[" << RegInfo->getName(i)
                         << ",%reg" << PhysRegsUsed[i] << "] ";
          std::cerr << "\n");

    // Loop over the implicit uses, making sure that they are at the head of the
    // use order list, so they don't get reallocated.
    if (TID.ImplicitUses) {
      for (const unsigned *ImplicitUses = TID.ImplicitUses;
           *ImplicitUses; ++ImplicitUses)
        MarkPhysRegRecentlyUsed(*ImplicitUses);
    }

    // Get the used operands into registers.  This has the potential to spill
    // incoming values if we are out of registers.  Note that we completely
    // ignore physical register uses here.  We assume that if an explicit
    // physical register is referenced by the instruction, that it is guaranteed
    // to be live-in, or the input is badly hosed.
    //
    for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
      MachineOperand& MO = MI->getOperand(i);
      // here we are looking for only used operands (never def&use)
      if (!MO.isDef() && MO.isRegister() && MO.getReg() &&
          MRegisterInfo::isVirtualRegister(MO.getReg()))
        MI = reloadVirtReg(MBB, MI, i);
    }

    // If this instruction is the last user of anything in registers, kill the
    // value, freeing the register being used, so it doesn't need to be
    // spilled to memory.
    //
    for (LiveVariables::killed_iterator KI = LV->killed_begin(MI),
           KE = LV->killed_end(MI); KI != KE; ++KI) {
      unsigned VirtReg = *KI;
      unsigned PhysReg = VirtReg;
      if (MRegisterInfo::isVirtualRegister(VirtReg)) {
        // If the virtual register was never materialized into a register, it
        // might not be in the map, but it won't hurt to zero it out anyway.
        unsigned &PhysRegSlot = getVirt2PhysRegMapSlot(VirtReg);
        PhysReg = PhysRegSlot;
        PhysRegSlot = 0;
      }

      if (PhysReg) {
        DEBUG(std::cerr << "  Last use of " << RegInfo->getName(PhysReg)
              << "[%reg" << VirtReg <<"], removing it from live set\n");
        removePhysReg(PhysReg);
      }
    }

    // Loop over all of the operands of the instruction, spilling registers that
    // are defined, and marking explicit destinations in the PhysRegsUsed map.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = MI->getOperand(i);
      if (MO.isDef() && MO.isRegister() && MO.getReg() &&
          MRegisterInfo::isPhysicalRegister(MO.getReg())) {
        unsigned Reg = MO.getReg();
        PhysRegsEverUsed[Reg] = true;
        spillPhysReg(MBB, MI, Reg, true); // Spill any existing value in the reg
        PhysRegsUsed[Reg] = 0;            // It is free and reserved now
        PhysRegsUseOrder.push_back(Reg);
        for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
             *AliasSet; ++AliasSet) {
          PhysRegsUseOrder.push_back(*AliasSet);
          PhysRegsUsed[*AliasSet] = 0;  // It is free and reserved now
          PhysRegsEverUsed[*AliasSet] = true;
        }
      }
    }

    // Loop over the implicit defs, spilling them as well.
    if (TID.ImplicitDefs) {
      for (const unsigned *ImplicitDefs = TID.ImplicitDefs;
           *ImplicitDefs; ++ImplicitDefs) {
        unsigned Reg = *ImplicitDefs;
        spillPhysReg(MBB, MI, Reg, true);
        PhysRegsUseOrder.push_back(Reg);
        PhysRegsUsed[Reg] = 0;            // It is free and reserved now
        PhysRegsEverUsed[Reg] = true;

        for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
             *AliasSet; ++AliasSet) {
          PhysRegsUseOrder.push_back(*AliasSet);
          PhysRegsUsed[*AliasSet] = 0;  // It is free and reserved now
          PhysRegsEverUsed[*AliasSet] = true;
        }
      }
    }

    // Okay, we have allocated all of the source operands and spilled any values
    // that would be destroyed by defs of this instruction.  Loop over the
    // explicit defs and assign them to a register, spilling incoming values if
    // we need to scavenge a register.
    //
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = MI->getOperand(i);
      if (MO.isDef() && MO.isRegister() && MO.getReg() &&
          MRegisterInfo::isVirtualRegister(MO.getReg())) {
        unsigned DestVirtReg = MO.getReg();
        unsigned DestPhysReg;

        // If DestVirtReg already has a value, use it.
        if (!(DestPhysReg = getVirt2PhysRegMapSlot(DestVirtReg)))
          DestPhysReg = getReg(MBB, MI, DestVirtReg);
        PhysRegsEverUsed[DestPhysReg] = true;
        markVirtRegModified(DestVirtReg);
        MI->getOperand(i).setReg(DestPhysReg);  // Assign the output register
      }
    }

    // If this instruction defines any registers that are immediately dead,
    // kill them now.
    //
    for (LiveVariables::killed_iterator KI = LV->dead_begin(MI),
           KE = LV->dead_end(MI); KI != KE; ++KI) {
      unsigned VirtReg = *KI;
      unsigned PhysReg = VirtReg;
      if (MRegisterInfo::isVirtualRegister(VirtReg)) {
        unsigned &PhysRegSlot = getVirt2PhysRegMapSlot(VirtReg);
        PhysReg = PhysRegSlot;
        assert(PhysReg != 0);
        PhysRegSlot = 0;
      }

      if (PhysReg) {
        DEBUG(std::cerr << "  Register " << RegInfo->getName(PhysReg)
              << " [%reg" << VirtReg
              << "] is never used, removing it frame live list\n");
        removePhysReg(PhysReg);
      }
    }
    
    // Finally, if this is a noop copy instruction, zap it.
    unsigned SrcReg, DstReg;
    if (TII.isMoveInstr(*MI, SrcReg, DstReg) && SrcReg == DstReg)
      MBB.erase(MI);
  }

  MachineBasicBlock::iterator MI = MBB.getFirstTerminator();

  // Spill all physical registers holding virtual registers now.
  for (unsigned i = 0, e = RegInfo->getNumRegs(); i != e; ++i)
    if (PhysRegsUsed[i] != -1)
      if (unsigned VirtReg = PhysRegsUsed[i])
        spillVirtReg(MBB, MI, VirtReg, i);
      else
        removePhysReg(i);

#if 0
  // This checking code is very expensive.
  bool AllOk = true;
  for (unsigned i = MRegisterInfo::FirstVirtualRegister,
           e = MF->getSSARegMap()->getLastVirtReg(); i <= e; ++i)
    if (unsigned PR = Virt2PhysRegMap[i]) {
      std::cerr << "Register still mapped: " << i << " -> " << PR << "\n";
      AllOk = false;
    }
  assert(AllOk && "Virtual registers still in phys regs?");
#endif

  // Clear any physical register which appear live at the end of the basic
  // block, but which do not hold any virtual registers.  e.g., the stack
  // pointer.
  PhysRegsUseOrder.clear();
}


/// runOnMachineFunction - Register allocate the whole function
///
bool RA::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(std::cerr << "Machine Function " << "\n");
  MF = &Fn;
  TM = &Fn.getTarget();
  RegInfo = TM->getRegisterInfo();
  LV = &getAnalysis<LiveVariables>();

  PhysRegsEverUsed = new bool[RegInfo->getNumRegs()];
  std::fill(PhysRegsEverUsed, PhysRegsEverUsed+RegInfo->getNumRegs(), false);
  Fn.setUsedPhysRegs(PhysRegsEverUsed);

  PhysRegsUsed.assign(RegInfo->getNumRegs(), -1);

  // initialize the virtual->physical register map to have a 'null'
  // mapping for all virtual registers
  Virt2PhysRegMap.grow(MF->getSSARegMap()->getLastVirtReg());

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  StackSlotForVirtReg.clear();
  PhysRegsUsed.clear();
  VirtRegModified.clear();
  Virt2PhysRegMap.clear();
  return true;
}

FunctionPass *llvm::createLocalRegisterAllocator() {
  return new RA();
}
