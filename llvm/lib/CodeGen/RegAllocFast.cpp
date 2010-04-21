//===-- RegAllocFast.cpp - A fast register allocator for debug code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This register allocator allocates registers to a basic block at a time,
// attempting to keep values in registers and reusing registers as appropriate.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/BasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumStores, "Number of stores added");
STATISTIC(NumLoads , "Number of loads added");

static RegisterRegAlloc
  fastRegAlloc("fast", "fast register allocator", createFastRegisterAllocator);

namespace {
  class RAFast : public MachineFunctionPass {
  public:
    static char ID;
    RAFast() : MachineFunctionPass(&ID), StackSlotForVirtReg(-1) {}
  private:
    const TargetMachine *TM;
    MachineFunction *MF;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;

    // StackSlotForVirtReg - Maps virtual regs to the frame index where these
    // values are spilled.
    IndexedMap<int, VirtReg2IndexFunctor> StackSlotForVirtReg;

    // Virt2PhysRegMap - This map contains entries for each virtual register
    // that is currently available in a physical register.
    IndexedMap<unsigned, VirtReg2IndexFunctor> Virt2PhysRegMap;

    unsigned &getVirt2PhysRegMapSlot(unsigned VirtReg) {
      return Virt2PhysRegMap[VirtReg];
    }

    // PhysRegsUsed - This array is effectively a map, containing entries for
    // each physical register that currently has a value (ie, it is in
    // Virt2PhysRegMap).  The value mapped to is the virtual register
    // corresponding to the physical register (the inverse of the
    // Virt2PhysRegMap), or 0.  The value is set to 0 if this register is pinned
    // because it is used by a future instruction, and to -2 if it is not
    // allocatable.  If the entry for a physical register is -1, then the
    // physical register is "not in the map".
    //
    std::vector<int> PhysRegsUsed;

    // UsedInInstr - BitVector of physregs that are used in the current
    // instruction, and so cannot be allocated.
    BitVector UsedInInstr;

    // Virt2LastUseMap - This maps each virtual register to its last use
    // (MachineInstr*, operand index pair).
    IndexedMap<std::pair<MachineInstr*, unsigned>, VirtReg2IndexFunctor>
    Virt2LastUseMap;

    std::pair<MachineInstr*,unsigned>& getVirtRegLastUse(unsigned Reg) {
      assert(TargetRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      return Virt2LastUseMap[Reg];
    }

    // VirtRegModified - This bitset contains information about which virtual
    // registers need to be spilled back to memory when their registers are
    // scavenged.  If a virtual register has simply been rematerialized, there
    // is no reason to spill it to memory when we need the register back.
    //
    BitVector VirtRegModified;

    // UsedInMultipleBlocks - Tracks whether a particular register is used in
    // more than one block.
    BitVector UsedInMultipleBlocks;

    void markVirtRegModified(unsigned Reg, bool Val = true) {
      assert(TargetRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      Reg -= TargetRegisterInfo::FirstVirtualRegister;
      if (Val)
        VirtRegModified.set(Reg);
      else
        VirtRegModified.reset(Reg);
    }

    bool isVirtRegModified(unsigned Reg) const {
      assert(TargetRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      assert(Reg - TargetRegisterInfo::FirstVirtualRegister <
             VirtRegModified.size() && "Illegal virtual register!");
      return VirtRegModified[Reg - TargetRegisterInfo::FirstVirtualRegister];
    }

  public:
    virtual const char *getPassName() const {
      return "Fast Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
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
      for (const unsigned *AliasSet = TRI->getAliasSet(R2);
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

    /// isPhysRegAvailable - Return true if the specified physical register is
    /// free and available for use.  This also includes checking to see if
    /// aliased registers are all free...
    ///
    bool isPhysRegAvailable(unsigned PhysReg) const;

    /// isPhysRegSpillable - Can PhysReg be freed by spilling?
    bool isPhysRegSpillable(unsigned PhysReg) const;

    /// getFreeReg - Look to see if there is a free register available in the
    /// specified register class.  If not, return 0.
    ///
    unsigned getFreeReg(const TargetRegisterClass *RC);

    /// getReg - Find a physical register to hold the specified virtual
    /// register.  If all compatible physical registers are used, this method
    /// spills the last used virtual register to the stack, and uses that
    /// register. If NoFree is true, that means the caller knows there isn't
    /// a free register, do not call getFreeReg().
    unsigned getReg(MachineBasicBlock &MBB, MachineInstr *MI,
                    unsigned VirtReg, bool NoFree = false);

    /// reloadVirtReg - This method transforms the specified virtual
    /// register use to refer to a physical register.  This method may do this
    /// in one of several ways: if the register is available in a physical
    /// register already, it uses that physical register.  If the value is not
    /// in a physical register, and if there are physical registers available,
    /// it loads it into a register: PhysReg if that is an available physical
    /// register, otherwise any physical register of the right class.
    /// If register pressure is high, and it is possible, it tries to fold the
    /// load of the virtual register into the instruction itself.  It avoids
    /// doing this if register pressure is low to improve the chance that
    /// subsequent instructions can use the reloaded value.  This method
    /// returns the modified instruction.
    ///
    MachineInstr *reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                                unsigned OpNum, SmallSet<unsigned, 4> &RRegs,
                                unsigned PhysReg);

    void reloadPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                       unsigned PhysReg);
  };
  char RAFast::ID = 0;
}

/// getStackSpaceFor - This allocates space for the specified virtual register
/// to be held on the stack.
int RAFast::getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC) {
  // Find the location Reg would belong...
  int SS = StackSlotForVirtReg[VirtReg];
  if (SS != -1)
    return SS;          // Already has space allocated?

  // Allocate a new stack object for this spill location...
  int FrameIdx = MF->getFrameInfo()->CreateSpillStackObject(RC->getSize(),
                                                            RC->getAlignment());

  // Assign the slot.
  StackSlotForVirtReg[VirtReg] = FrameIdx;
  return FrameIdx;
}


/// removePhysReg - This method marks the specified physical register as no
/// longer being in use.
///
void RAFast::removePhysReg(unsigned PhysReg) {
  PhysRegsUsed[PhysReg] = -1;      // PhyReg no longer used
}


/// spillVirtReg - This method spills the value specified by PhysReg into the
/// virtual register slot specified by VirtReg.  It then updates the RA data
/// structures to indicate the fact that PhysReg is now available.
///
void RAFast::spillVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I,
                           unsigned VirtReg, unsigned PhysReg) {
  assert(VirtReg && "Spilling a physical register is illegal!"
         " Must not have appropriate kill for the register or use exists beyond"
         " the intended one.");
  DEBUG(dbgs() << "  Spilling register " << TRI->getName(PhysReg)
               << " containing %reg" << VirtReg);

  if (!isVirtRegModified(VirtReg)) {
    DEBUG(dbgs() << " which has not been modified, so no store necessary!");
    std::pair<MachineInstr*, unsigned> &LastUse = getVirtRegLastUse(VirtReg);
    if (LastUse.first)
      LastUse.first->getOperand(LastUse.second).setIsKill();
  } else {
    // Otherwise, there is a virtual register corresponding to this physical
    // register.  We only need to spill it into its stack slot if it has been
    // modified.
    const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);
    int FrameIndex = getStackSpaceFor(VirtReg, RC);
    DEBUG(dbgs() << " to stack slot #" << FrameIndex);
    // If the instruction reads the register that's spilled, (e.g. this can
    // happen if it is a move to a physical register), then the spill
    // instruction is not a kill.
    bool isKill = !(I != MBB.end() && I->readsRegister(PhysReg));
    TII->storeRegToStackSlot(MBB, I, PhysReg, isKill, FrameIndex, RC);
    ++NumStores;   // Update statistics
  }

  getVirt2PhysRegMapSlot(VirtReg) = 0;   // VirtReg no longer available

  DEBUG(dbgs() << '\n');
  removePhysReg(PhysReg);
}


/// spillPhysReg - This method spills the specified physical register into the
/// virtual register slot associated with it.  If OnlyVirtRegs is set to true,
/// then the request is ignored if the physical register does not contain a
/// virtual register.
///
void RAFast::spillPhysReg(MachineBasicBlock &MBB, MachineInstr *I,
                           unsigned PhysReg, bool OnlyVirtRegs) {
  if (PhysRegsUsed[PhysReg] != -1) {            // Only spill it if it's used!
    assert(PhysRegsUsed[PhysReg] != -2 && "Non allocable reg used!");
    if (PhysRegsUsed[PhysReg] || !OnlyVirtRegs)
      spillVirtReg(MBB, I, PhysRegsUsed[PhysReg], PhysReg);
    return;
  }

  // If the selected register aliases any other registers, we must make
  // sure that one of the aliases isn't alive.
  for (const unsigned *AliasSet = TRI->getAliasSet(PhysReg);
       *AliasSet; ++AliasSet) {
    if (PhysRegsUsed[*AliasSet] == -1 ||     // Spill aliased register.
        PhysRegsUsed[*AliasSet] == -2)       // If allocatable.
      continue;

    if (PhysRegsUsed[*AliasSet])
      spillVirtReg(MBB, I, PhysRegsUsed[*AliasSet], *AliasSet);
  }
}


/// assignVirtToPhysReg - This method updates local state so that we know
/// that PhysReg is the proper container for VirtReg now.  The physical
/// register must not be used for anything else when this is called.
///
void RAFast::assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg) {
  assert(PhysRegsUsed[PhysReg] == -1 && "Phys reg already assigned!");
  // Update information to note the fact that this register was just used, and
  // it holds VirtReg.
  PhysRegsUsed[PhysReg] = VirtReg;
  getVirt2PhysRegMapSlot(VirtReg) = PhysReg;
  UsedInInstr.set(PhysReg);
}


/// isPhysRegAvailable - Return true if the specified physical register is free
/// and available for use.  This also includes checking to see if aliased
/// registers are all free...
///
bool RAFast::isPhysRegAvailable(unsigned PhysReg) const {
  if (PhysRegsUsed[PhysReg] != -1) return false;

  // If the selected register aliases any other allocated registers, it is
  // not free!
  for (const unsigned *AliasSet = TRI->getAliasSet(PhysReg);
       *AliasSet; ++AliasSet)
    if (PhysRegsUsed[*AliasSet] >= 0) // Aliased register in use?
      return false;                    // Can't use this reg then.
  return true;
}

/// isPhysRegSpillable - Return true if the specified physical register can be
/// spilled for use in the current instruction.
///
bool RAFast::isPhysRegSpillable(unsigned PhysReg) const {
  // Test that PhysReg and all aliases are either free or assigned to a VirtReg
  // that is not used in the instruction.
  if (PhysRegsUsed[PhysReg] != -1 &&
      (PhysRegsUsed[PhysReg] <= 0 || UsedInInstr.test(PhysReg)))
    return false;

  for (const unsigned *AliasSet = TRI->getAliasSet(PhysReg);
       *AliasSet; ++AliasSet)
    if (PhysRegsUsed[*AliasSet] != -1 &&
        (PhysRegsUsed[*AliasSet] <= 0 || UsedInInstr.test(*AliasSet)))
      return false;
  return true;
}


/// getFreeReg - Look to see if there is a free register available in the
/// specified register class.  If not, return 0.
///
unsigned RAFast::getFreeReg(const TargetRegisterClass *RC) {
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


/// getReg - Find a physical register to hold the specified virtual
/// register.  If all compatible physical registers are used, this method spills
/// the last used virtual register to the stack, and uses that register.
///
unsigned RAFast::getReg(MachineBasicBlock &MBB, MachineInstr *I,
                         unsigned VirtReg, bool NoFree) {
  const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);

  // First check to see if we have a free register of the requested type...
  unsigned PhysReg = NoFree ? 0 : getFreeReg(RC);

  if (PhysReg != 0) {
    // Assign the register.
    assignVirtToPhysReg(VirtReg, PhysReg);
    return PhysReg;
  }

  // If we didn't find an unused register, scavenge one now! Don't be fancy,
  // just grab the first possible register.
  TargetRegisterClass::iterator RI = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator RE = RC->allocation_order_end(*MF);

  for (; RI != RE; ++RI)
    if (isPhysRegSpillable(*RI)) {
      PhysReg = *RI;
      break;
    }

  assert(PhysReg && "Physical register not assigned!?!?");
  spillPhysReg(MBB, I, PhysReg);
  assignVirtToPhysReg(VirtReg, PhysReg);
  return PhysReg;
}


/// reloadVirtReg - This method transforms the specified virtual
/// register use to refer to a physical register.  This method may do this in
/// one of several ways: if the register is available in a physical register
/// already, it uses that physical register.  If the value is not in a physical
/// register, and if there are physical registers available, it loads it into a
/// register: PhysReg if that is an available physical register, otherwise any
/// register.  If register pressure is high, and it is possible, it tries to
/// fold the load of the virtual register into the instruction itself.  It
/// avoids doing this if register pressure is low to improve the chance that
/// subsequent instructions can use the reloaded value.  This method returns
/// the modified instruction.
///
MachineInstr *RAFast::reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                                     unsigned OpNum,
                                     SmallSet<unsigned, 4> &ReloadedRegs,
                                     unsigned PhysReg) {
  unsigned VirtReg = MI->getOperand(OpNum).getReg();

  // If the virtual register is already available, just update the instruction
  // and return.
  if (unsigned PR = getVirt2PhysRegMapSlot(VirtReg)) {
    MI->getOperand(OpNum).setReg(PR);  // Assign the input register
    if (!MI->isDebugValue()) {
      // Do not do these for DBG_VALUE as they can affect codegen.
      UsedInInstr.set(PR);
      getVirtRegLastUse(VirtReg) = std::make_pair(MI, OpNum);
    }
    return MI;
  }

  // Otherwise, we need to fold it into the current instruction, or reload it.
  // If we have registers available to hold the value, use them.
  const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);
  // If we already have a PhysReg (this happens when the instruction is a
  // reg-to-reg copy with a PhysReg destination) use that.
  if (!PhysReg || !TargetRegisterInfo::isPhysicalRegister(PhysReg) ||
      !isPhysRegAvailable(PhysReg))
    PhysReg = getFreeReg(RC);
  int FrameIndex = getStackSpaceFor(VirtReg, RC);

  if (PhysReg) {   // Register is available, allocate it!
    assignVirtToPhysReg(VirtReg, PhysReg);
  } else {         // No registers available.
    // Force some poor hapless value out of the register file to
    // make room for the new register, and reload it.
    PhysReg = getReg(MBB, MI, VirtReg, true);
  }

  markVirtRegModified(VirtReg, false);   // Note that this reg was just reloaded

  DEBUG(dbgs() << "  Reloading %reg" << VirtReg << " into "
               << TRI->getName(PhysReg) << "\n");

  // Add move instruction(s)
  TII->loadRegFromStackSlot(MBB, MI, PhysReg, FrameIndex, RC);
  ++NumLoads;    // Update statistics

  MF->getRegInfo().setPhysRegUsed(PhysReg);
  MI->getOperand(OpNum).setReg(PhysReg);  // Assign the input register
  getVirtRegLastUse(VirtReg) = std::make_pair(MI, OpNum);

  if (!ReloadedRegs.insert(PhysReg)) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Ran out of registers during register allocation!";
    if (MI->isInlineAsm()) {
      Msg << "\nPlease check your inline asm statement for invalid "
           << "constraints:\n";
      MI->print(Msg, TM);
    }
    report_fatal_error(Msg.str());
  }
  for (const unsigned *SubRegs = TRI->getSubRegisters(PhysReg);
       *SubRegs; ++SubRegs) {
    if (ReloadedRegs.insert(*SubRegs)) continue;

    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Ran out of registers during register allocation!";
    if (MI->isInlineAsm()) {
      Msg << "\nPlease check your inline asm statement for invalid "
           << "constraints:\n";
      MI->print(Msg, TM);
    }
    report_fatal_error(Msg.str());
  }

  return MI;
}

/// isReadModWriteImplicitKill - True if this is an implicit kill for a
/// read/mod/write register, i.e. update partial register.
static bool isReadModWriteImplicitKill(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == Reg && MO.isImplicit() &&
        MO.isDef() && !MO.isDead())
      return true;
  }
  return false;
}

/// isReadModWriteImplicitDef - True if this is an implicit def for a
/// read/mod/write register, i.e. update partial register.
static bool isReadModWriteImplicitDef(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == Reg && MO.isImplicit() &&
        !MO.isDef() && MO.isKill())
      return true;
  }
  return false;
}

void RAFast::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  MachineBasicBlock::iterator MII = MBB.begin();

  DEBUG({
      const BasicBlock *LBB = MBB.getBasicBlock();
      if (LBB)
        dbgs() << "\nStarting RegAlloc of BB: " << LBB->getName();
    });

  // Add live-in registers as active.
  for (MachineBasicBlock::livein_iterator I = MBB.livein_begin(),
         E = MBB.livein_end(); I != E; ++I) {
    unsigned Reg = *I;
    MF->getRegInfo().setPhysRegUsed(Reg);
    PhysRegsUsed[Reg] = 0;            // It is free and reserved now
    for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
         *SubRegs; ++SubRegs) {
      if (PhysRegsUsed[*SubRegs] == -2) continue;
      PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
      MF->getRegInfo().setPhysRegUsed(*SubRegs);
    }
  }

  // Otherwise, sequentially allocate each instruction in the MBB.
  while (MII != MBB.end()) {
    MachineInstr *MI = MII++;
    const TargetInstrDesc &TID = MI->getDesc();
    DEBUG({
        dbgs() << "\nStarting RegAlloc of: " << *MI;
        dbgs() << "  Regs have values: ";
        for (unsigned i = 0; i != TRI->getNumRegs(); ++i)
          if (PhysRegsUsed[i] != -1 && PhysRegsUsed[i] != -2)
            dbgs() << "[" << TRI->getName(i)
                   << ",%reg" << PhysRegsUsed[i] << "] ";
        dbgs() << '\n';
      });

    // Track registers used by instruction.
    UsedInInstr.reset();

    // Determine whether this is a copy instruction.  The cases where the
    // source or destination are phys regs are handled specially.
    unsigned SrcCopyReg, DstCopyReg, SrcCopySubReg, DstCopySubReg;
    unsigned SrcCopyPhysReg = 0U;
    bool isCopy = TII->isMoveInstr(*MI, SrcCopyReg, DstCopyReg,
                                   SrcCopySubReg, DstCopySubReg);
    if (isCopy && TargetRegisterInfo::isVirtualRegister(SrcCopyReg))
      SrcCopyPhysReg = getVirt2PhysRegMapSlot(SrcCopyReg);

    // Loop over the implicit uses, making sure they don't get reallocated.
    if (TID.ImplicitUses) {
      for (const unsigned *ImplicitUses = TID.ImplicitUses;
           *ImplicitUses; ++ImplicitUses)
        UsedInInstr.set(*ImplicitUses);
    }

    SmallVector<unsigned, 8> Kills;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isKill()) continue;

      if (!MO.isImplicit())
        Kills.push_back(MO.getReg());
      else if (!isReadModWriteImplicitKill(MI, MO.getReg()))
        // These are extra physical register kills when a sub-register
        // is defined (def of a sub-register is a read/mod/write of the
        // larger registers). Ignore.
        Kills.push_back(MO.getReg());
    }

    // If any physical regs are earlyclobber, spill any value they might
    // have in them, then mark them unallocatable.
    // If any virtual regs are earlyclobber, allocate them now (before
    // freeing inputs that are killed).
    if (MI->isInlineAsm()) {
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.isDef() || !MO.isEarlyClobber() ||
            !MO.getReg())
          continue;

        if (TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
          unsigned DestVirtReg = MO.getReg();
          unsigned DestPhysReg;

          // If DestVirtReg already has a value, use it.
          if (!(DestPhysReg = getVirt2PhysRegMapSlot(DestVirtReg)))
            DestPhysReg = getReg(MBB, MI, DestVirtReg);
          MF->getRegInfo().setPhysRegUsed(DestPhysReg);
          markVirtRegModified(DestVirtReg);
          getVirtRegLastUse(DestVirtReg) =
                 std::make_pair((MachineInstr*)0, 0);
          DEBUG(dbgs() << "  Assigning " << TRI->getName(DestPhysReg)
                       << " to %reg" << DestVirtReg << "\n");
          MO.setReg(DestPhysReg);  // Assign the earlyclobber register
        } else {
          unsigned Reg = MO.getReg();
          if (PhysRegsUsed[Reg] == -2) continue;  // Something like ESP.
          // These are extra physical register defs when a sub-register
          // is defined (def of a sub-register is a read/mod/write of the
          // larger registers). Ignore.
          if (isReadModWriteImplicitDef(MI, MO.getReg())) continue;

          MF->getRegInfo().setPhysRegUsed(Reg);
          spillPhysReg(MBB, MI, Reg, true); // Spill any existing value in reg
          PhysRegsUsed[Reg] = 0;            // It is free and reserved now

          for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
               *SubRegs; ++SubRegs) {
            if (PhysRegsUsed[*SubRegs] == -2) continue;
            MF->getRegInfo().setPhysRegUsed(*SubRegs);
            PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
          }
        }
      }
    }

    // If a DBG_VALUE says something is located in a spilled register,
    // change the DBG_VALUE to be undef, which prevents the register
    // from being reloaded here.  Doing that would change the generated
    // code, unless another use immediately follows this instruction.
    if (MI->isDebugValue() &&
        MI->getNumOperands()==3 && MI->getOperand(0).isReg()) {
      unsigned VirtReg = MI->getOperand(0).getReg();
      if (VirtReg && TargetRegisterInfo::isVirtualRegister(VirtReg) &&
          !getVirt2PhysRegMapSlot(VirtReg))
        MI->getOperand(0).setReg(0U);
    }

    // Get the used operands into registers.  This has the potential to spill
    // incoming values if we are out of registers.  Note that we completely
    // ignore physical register uses here.  We assume that if an explicit
    // physical register is referenced by the instruction, that it is guaranteed
    // to be live-in, or the input is badly hosed.
    //
    SmallSet<unsigned, 4> ReloadedRegs;
    for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
      MachineOperand &MO = MI->getOperand(i);
      // here we are looking for only used operands (never def&use)
      if (MO.isReg() && !MO.isDef() && MO.getReg() && !MO.isImplicit() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        MI = reloadVirtReg(MBB, MI, i, ReloadedRegs,
                           isCopy ? DstCopyReg : 0);
    }

    // If this instruction is the last user of this register, kill the
    // value, freeing the register being used, so it doesn't need to be
    // spilled to memory.
    //
    for (unsigned i = 0, e = Kills.size(); i != e; ++i) {
      unsigned VirtReg = Kills[i];
      unsigned PhysReg = VirtReg;
      if (TargetRegisterInfo::isVirtualRegister(VirtReg)) {
        // If the virtual register was never materialized into a register, it
        // might not be in the map, but it won't hurt to zero it out anyway.
        unsigned &PhysRegSlot = getVirt2PhysRegMapSlot(VirtReg);
        PhysReg = PhysRegSlot;
        PhysRegSlot = 0;
      } else if (PhysRegsUsed[PhysReg] == -2) {
        // Unallocatable register dead, ignore.
        continue;
      } else {
        assert((!PhysRegsUsed[PhysReg] || PhysRegsUsed[PhysReg] == -1) &&
               "Silently clearing a virtual register?");
      }

      if (!PhysReg) continue;

      DEBUG(dbgs() << "  Last use of " << TRI->getName(PhysReg)
                   << "[%reg" << VirtReg <<"], removing it from live set\n");
      removePhysReg(PhysReg);
      for (const unsigned *SubRegs = TRI->getSubRegisters(PhysReg);
           *SubRegs; ++SubRegs) {
        if (PhysRegsUsed[*SubRegs] != -2) {
          DEBUG(dbgs()  << "  Last use of "
                        << TRI->getName(*SubRegs) << "[%reg" << VirtReg
                        <<"], removing it from live set\n");
          removePhysReg(*SubRegs);
        }
      }
    }

    // Track registers defined by instruction.
    UsedInInstr.reset();

    // Loop over all of the operands of the instruction, spilling registers that
    // are defined, and marking explicit destinations in the PhysRegsUsed map.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() || MO.isImplicit() || !MO.getReg() ||
          MO.isEarlyClobber() ||
          !TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
        continue;

      unsigned Reg = MO.getReg();
      if (PhysRegsUsed[Reg] == -2) continue;  // Something like ESP.
      // These are extra physical register defs when a sub-register
      // is defined (def of a sub-register is a read/mod/write of the
      // larger registers). Ignore.
      if (isReadModWriteImplicitDef(MI, MO.getReg())) continue;

      MF->getRegInfo().setPhysRegUsed(Reg);
      spillPhysReg(MBB, MI, Reg, true); // Spill any existing value in reg
      PhysRegsUsed[Reg] = 0;            // It is free and reserved now

      for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
           *SubRegs; ++SubRegs) {
        if (PhysRegsUsed[*SubRegs] == -2) continue;

        MF->getRegInfo().setPhysRegUsed(*SubRegs);
        PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
      }
    }

    // Loop over the implicit defs, spilling them as well.
    if (TID.ImplicitDefs) {
      for (const unsigned *ImplicitDefs = TID.ImplicitDefs;
           *ImplicitDefs; ++ImplicitDefs) {
        unsigned Reg = *ImplicitDefs;
        if (PhysRegsUsed[Reg] != -2) {
          spillPhysReg(MBB, MI, Reg, true);
          PhysRegsUsed[Reg] = 0;            // It is free and reserved now
        }
        MF->getRegInfo().setPhysRegUsed(Reg);
        for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
             *SubRegs; ++SubRegs) {
          if (PhysRegsUsed[*SubRegs] == -2) continue;

          PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
          MF->getRegInfo().setPhysRegUsed(*SubRegs);
        }
      }
    }

    SmallVector<unsigned, 8> DeadDefs;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDead())
        DeadDefs.push_back(MO.getReg());
    }

    // Okay, we have allocated all of the source operands and spilled any values
    // that would be destroyed by defs of this instruction.  Loop over the
    // explicit defs and assign them to a register, spilling incoming values if
    // we need to scavenge a register.
    //
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() || !MO.getReg() ||
          MO.isEarlyClobber() ||
          !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        continue;

      unsigned DestVirtReg = MO.getReg();
      unsigned DestPhysReg;

      // If DestVirtReg already has a value, use it.
      if (!(DestPhysReg = getVirt2PhysRegMapSlot(DestVirtReg))) {
        // If this is a copy try to reuse the input as the output;
        // that will make the copy go away.
        // If this is a copy, the source reg is a phys reg, and
        // that reg is available, use that phys reg for DestPhysReg.
        // If this is a copy, the source reg is a virtual reg, and
        // the phys reg that was assigned to that virtual reg is now
        // available, use that phys reg for DestPhysReg.  (If it's now
        // available that means this was the last use of the source.)
        if (isCopy &&
            TargetRegisterInfo::isPhysicalRegister(SrcCopyReg) &&
            isPhysRegAvailable(SrcCopyReg)) {
          DestPhysReg = SrcCopyReg;
          assignVirtToPhysReg(DestVirtReg, DestPhysReg);
        } else if (isCopy &&
            TargetRegisterInfo::isVirtualRegister(SrcCopyReg) &&
            SrcCopyPhysReg && isPhysRegAvailable(SrcCopyPhysReg) &&
            MF->getRegInfo().getRegClass(DestVirtReg)->
                             contains(SrcCopyPhysReg)) {
          DestPhysReg = SrcCopyPhysReg;
          assignVirtToPhysReg(DestVirtReg, DestPhysReg);
        } else
          DestPhysReg = getReg(MBB, MI, DestVirtReg);
      }
      MF->getRegInfo().setPhysRegUsed(DestPhysReg);
      markVirtRegModified(DestVirtReg);
      getVirtRegLastUse(DestVirtReg) = std::make_pair((MachineInstr*)0, 0);
      DEBUG(dbgs() << "  Assigning " << TRI->getName(DestPhysReg)
                   << " to %reg" << DestVirtReg << "\n");
      MO.setReg(DestPhysReg);  // Assign the output register
      UsedInInstr.set(DestPhysReg);
    }

    // If this instruction defines any registers that are immediately dead,
    // kill them now.
    //
    for (unsigned i = 0, e = DeadDefs.size(); i != e; ++i) {
      unsigned VirtReg = DeadDefs[i];
      unsigned PhysReg = VirtReg;
      if (TargetRegisterInfo::isVirtualRegister(VirtReg)) {
        unsigned &PhysRegSlot = getVirt2PhysRegMapSlot(VirtReg);
        PhysReg = PhysRegSlot;
        assert(PhysReg != 0);
        PhysRegSlot = 0;
      } else if (PhysRegsUsed[PhysReg] == -2) {
        // Unallocatable register dead, ignore.
        continue;
      } else if (!PhysReg)
        continue;

      DEBUG(dbgs()  << "  Register " << TRI->getName(PhysReg)
                    << " [%reg" << VirtReg
                    << "] is never used, removing it from live set\n");
      removePhysReg(PhysReg);
      for (const unsigned *AliasSet = TRI->getAliasSet(PhysReg);
           *AliasSet; ++AliasSet) {
        if (PhysRegsUsed[*AliasSet] != -2) {
          DEBUG(dbgs()  << "  Register " << TRI->getName(*AliasSet)
                        << " [%reg" << *AliasSet
                        << "] is never used, removing it from live set\n");
          removePhysReg(*AliasSet);
        }
      }
    }

    // Finally, if this is a noop copy instruction, zap it.  (Except that if
    // the copy is dead, it must be kept to avoid messing up liveness info for
    // the register scavenger.  See pr4100.)
    if (TII->isMoveInstr(*MI, SrcCopyReg, DstCopyReg,
                         SrcCopySubReg, DstCopySubReg) &&
        SrcCopyReg == DstCopyReg && DeadDefs.empty())
      MBB.erase(MI);
  }

  MachineBasicBlock::iterator MI = MBB.getFirstTerminator();

  // Spill all physical registers holding virtual registers now.
  for (unsigned i = 0, e = TRI->getNumRegs(); i != e; ++i)
    if (PhysRegsUsed[i] != -1 && PhysRegsUsed[i] != -2) {
      if (unsigned VirtReg = PhysRegsUsed[i])
        spillVirtReg(MBB, MI, VirtReg, i);
      else
        removePhysReg(i);
    }
}

/// runOnMachineFunction - Register allocate the whole function
///
bool RAFast::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(dbgs() << "Machine Function\n");
  MF = &Fn;
  TM = &Fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();

  PhysRegsUsed.assign(TRI->getNumRegs(), -1);
  UsedInInstr.resize(TRI->getNumRegs());

  // At various places we want to efficiently check to see whether a register
  // is allocatable.  To handle this, we mark all unallocatable registers as
  // being pinned down, permanently.
  {
    BitVector Allocable = TRI->getAllocatableSet(Fn);
    for (unsigned i = 0, e = Allocable.size(); i != e; ++i)
      if (!Allocable[i])
        PhysRegsUsed[i] = -2;  // Mark the reg unallocable.
  }

  // initialize the virtual->physical register map to have a 'null'
  // mapping for all virtual registers
  unsigned LastVirtReg = MF->getRegInfo().getLastVirtReg();
  StackSlotForVirtReg.grow(LastVirtReg);
  Virt2PhysRegMap.grow(LastVirtReg);
  Virt2LastUseMap.grow(LastVirtReg);
  VirtRegModified.resize(LastVirtReg+1 -
                         TargetRegisterInfo::FirstVirtualRegister);
  UsedInMultipleBlocks.resize(LastVirtReg+1 -
                              TargetRegisterInfo::FirstVirtualRegister);

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  StackSlotForVirtReg.clear();
  PhysRegsUsed.clear();
  VirtRegModified.clear();
  UsedInMultipleBlocks.clear();
  Virt2PhysRegMap.clear();
  Virt2LastUseMap.clear();
  return true;
}

FunctionPass *llvm::createFastRegisterAllocator() {
  return new RAFast();
}
