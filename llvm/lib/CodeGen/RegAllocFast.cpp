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

static cl::opt<bool> VerifyFastRegalloc("verify-fast-regalloc", cl::Hidden,
    cl::desc("Verify machine code before fast regalloc"));

STATISTIC(NumStores, "Number of stores added");
STATISTIC(NumLoads , "Number of loads added");
STATISTIC(NumCopies, "Number of copies coalesced");

static RegisterRegAlloc
  fastRegAlloc("fast", "fast register allocator", createFastRegisterAllocator);

namespace {
  class RAFast : public MachineFunctionPass {
  public:
    static char ID;
    RAFast() : MachineFunctionPass(&ID), StackSlotForVirtReg(-1),
               isBulkSpilling(false) {}
  private:
    const TargetMachine *TM;
    MachineFunction *MF;
    MachineRegisterInfo *MRI;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;

    // Basic block currently being allocated.
    MachineBasicBlock *MBB;

    // StackSlotForVirtReg - Maps virtual regs to the frame index where these
    // values are spilled.
    IndexedMap<int, VirtReg2IndexFunctor> StackSlotForVirtReg;

    // Everything we know about a live virtual register.
    struct LiveReg {
      MachineInstr *LastUse;    // Last instr to use reg.
      unsigned PhysReg;         // Currently held here.
      unsigned short LastOpNum; // OpNum on LastUse.
      bool Dirty;               // Register needs spill.

      LiveReg(unsigned p=0) : LastUse(0), PhysReg(p), LastOpNum(0),
                              Dirty(false) {}
    };

    typedef DenseMap<unsigned, LiveReg> LiveRegMap;
    typedef LiveRegMap::value_type LiveRegEntry;

    // LiveVirtRegs - This map contains entries for each virtual register
    // that is currently available in a physical register.
    LiveRegMap LiveVirtRegs;

    // RegState - Track the state of a physical register.
    enum RegState {
      // A disabled register is not available for allocation, but an alias may
      // be in use. A register can only be moved out of the disabled state if
      // all aliases are disabled.
      regDisabled,

      // A free register is not currently in use and can be allocated
      // immediately without checking aliases.
      regFree,

      // A reserved register has been assigned expolicitly (e.g., setting up a
      // call parameter), and it remains reserved until it is used.
      regReserved

      // A register state may also be a virtual register number, indication that
      // the physical register is currently allocated to a virtual register. In
      // that case, LiveVirtRegs contains the inverse mapping.
    };

    // PhysRegState - One of the RegState enums, or a virtreg.
    std::vector<unsigned> PhysRegState;

    // UsedInInstr - BitVector of physregs that are used in the current
    // instruction, and so cannot be allocated.
    BitVector UsedInInstr;

    // Allocatable - vector of allocatable physical registers.
    BitVector Allocatable;

    // isBulkSpilling - This flag is set when LiveRegMap will be cleared
    // completely after spilling all live registers. LiveRegMap entries should
    // not be erased.
    bool isBulkSpilling;

  public:
    virtual const char *getPassName() const {
      return "Fast Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(PHIEliminationID);
      AU.addRequiredID(TwoAddressInstructionPassID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    bool runOnMachineFunction(MachineFunction &Fn);
    void AllocateBasicBlock();
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);
    bool isLastUseOfLocalReg(MachineOperand&);

    void addKillFlag(const LiveReg&);
    void killVirtReg(LiveRegMap::iterator);
    void killVirtReg(unsigned VirtReg);
    void spillVirtReg(MachineBasicBlock::iterator MI, LiveRegMap::iterator);
    void spillVirtReg(MachineBasicBlock::iterator MI, unsigned VirtReg);

    void usePhysReg(MachineOperand&);
    void definePhysReg(MachineInstr *MI, unsigned PhysReg, RegState NewState);
    void assignVirtToPhysReg(LiveRegEntry &LRE, unsigned PhysReg);
    void allocVirtReg(MachineInstr *MI, LiveRegEntry &LRE, unsigned Hint);
    unsigned defineVirtReg(MachineInstr *MI, unsigned OpNum,
                           unsigned VirtReg, unsigned Hint);
    unsigned reloadVirtReg(MachineInstr *MI, unsigned OpNum,
                           unsigned VirtReg, unsigned Hint);
    void spillAll(MachineInstr *MI);
    void setPhysReg(MachineOperand &MO, unsigned PhysReg);
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

/// isLastUseOfLocalReg - Return true if MO is the only remaining reference to
/// its virtual register, and it is guaranteed to be a block-local register.
///
bool RAFast::isLastUseOfLocalReg(MachineOperand &MO) {
  // Check for non-debug uses or defs following MO.
  // This is the most likely way to fail - fast path it.
  MachineOperand *Next = &MO;
  while ((Next = Next->getNextOperandForReg()))
    if (!Next->isDebug())
      return false;

  // If the register has ever been spilled or reloaded, we conservatively assume
  // it is a global register used in multiple blocks.
  if (StackSlotForVirtReg[MO.getReg()] != -1)
    return false;

  // Check that the use/def chain has exactly one operand - MO.
  return &MRI->reg_nodbg_begin(MO.getReg()).getOperand() == &MO;
}

/// addKillFlag - Set kill flags on last use of a virtual register.
void RAFast::addKillFlag(const LiveReg &LR) {
  if (!LR.LastUse) return;
  MachineOperand &MO = LR.LastUse->getOperand(LR.LastOpNum);
  if (MO.isDef())
    MO.setIsDead();
  else if (!LR.LastUse->isRegTiedToDefOperand(LR.LastOpNum))
    MO.setIsKill();
}

/// killVirtReg - Mark virtreg as no longer available.
void RAFast::killVirtReg(LiveRegMap::iterator LRI) {
  addKillFlag(LRI->second);
  const LiveReg &LR = LRI->second;
  assert(PhysRegState[LR.PhysReg] == LRI->first && "Broken RegState mapping");
  PhysRegState[LR.PhysReg] = regFree;
  // Erase from LiveVirtRegs unless we're spilling in bulk.
  if (!isBulkSpilling)
    LiveVirtRegs.erase(LRI);
}

/// killVirtReg - Mark virtreg as no longer available.
void RAFast::killVirtReg(unsigned VirtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "killVirtReg needs a virtual register");
  LiveRegMap::iterator LRI = LiveVirtRegs.find(VirtReg);
  if (LRI != LiveVirtRegs.end())
    killVirtReg(LRI);
}

/// spillVirtReg - This method spills the value specified by VirtReg into the
/// corresponding stack slot if needed. If isKill is set, the register is also
/// killed.
void RAFast::spillVirtReg(MachineBasicBlock::iterator MI, unsigned VirtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Spilling a physical register is illegal!");
  LiveRegMap::iterator LRI = LiveVirtRegs.find(VirtReg);
  assert(LRI != LiveVirtRegs.end() && "Spilling unmapped virtual register");
  spillVirtReg(MI, LRI);
}

/// spillVirtReg - Do the actual work of spilling.
void RAFast::spillVirtReg(MachineBasicBlock::iterator MI,
                          LiveRegMap::iterator LRI) {
  LiveReg &LR = LRI->second;
  assert(PhysRegState[LR.PhysReg] == LRI->first && "Broken RegState mapping");

  if (LR.Dirty) {
    // If this physreg is used by the instruction, we want to kill it on the
    // instruction, not on the spill.
    bool SpillKill = LR.LastUse != MI;
    LR.Dirty = false;
    DEBUG(dbgs() << "Spilling %reg" << LRI->first
                 << " in " << TRI->getName(LR.PhysReg));
    const TargetRegisterClass *RC = MRI->getRegClass(LRI->first);
    int FI = getStackSpaceFor(LRI->first, RC);
    DEBUG(dbgs() << " to stack slot #" << FI << "\n");
    TII->storeRegToStackSlot(*MBB, MI, LR.PhysReg, SpillKill, FI, RC, TRI);
    ++NumStores;   // Update statistics

    if (SpillKill)
      LR.LastUse = 0; // Don't kill register again
  }
  killVirtReg(LRI);
}

/// spillAll - Spill all dirty virtregs without killing them.
void RAFast::spillAll(MachineInstr *MI) {
  isBulkSpilling = true;
  for (LiveRegMap::iterator i = LiveVirtRegs.begin(),
       e = LiveVirtRegs.end(); i != e; ++i)
    spillVirtReg(MI, i);
  LiveVirtRegs.clear();
  isBulkSpilling = false;
}

/// usePhysReg - Handle the direct use of a physical register.
/// Check that the register is not used by a virtreg.
/// Kill the physreg, marking it free.
/// This may add implicit kills to MO->getParent() and invalidate MO.
void RAFast::usePhysReg(MachineOperand &MO) {
  unsigned PhysReg = MO.getReg();
  assert(TargetRegisterInfo::isPhysicalRegister(PhysReg) &&
         "Bad usePhysReg operand");

  switch (PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  case regReserved:
    PhysRegState[PhysReg] = regFree;
    // Fall through
  case regFree:
    UsedInInstr.set(PhysReg);
    MO.setIsKill();
    return;
  default:
    // The physreg was allocated to a virtual register. That means to value we
    // wanted has been clobbered.
    llvm_unreachable("Instruction uses an allocated register");
  }

  // Maybe a superregister is reserved?
  for (const unsigned *AS = TRI->getAliasSet(PhysReg);
       unsigned Alias = *AS; ++AS) {
    switch (PhysRegState[Alias]) {
    case regDisabled:
      break;
    case regReserved:
      assert(TRI->isSuperRegister(PhysReg, Alias) &&
             "Instruction is not using a subregister of a reserved register");
      // Leave the superregister in the working set.
      PhysRegState[Alias] = regFree;
      UsedInInstr.set(Alias);
      MO.getParent()->addRegisterKilled(Alias, TRI, true);
      return;
    case regFree:
      if (TRI->isSuperRegister(PhysReg, Alias)) {
        // Leave the superregister in the working set.
        UsedInInstr.set(Alias);
        MO.getParent()->addRegisterKilled(Alias, TRI, true);
        return;
      }
      // Some other alias was in the working set - clear it.
      PhysRegState[Alias] = regDisabled;
      break;
    default:
      llvm_unreachable("Instruction uses an alias of an allocated register");
    }
  }

  // All aliases are disabled, bring register into working set.
  PhysRegState[PhysReg] = regFree;
  UsedInInstr.set(PhysReg);
  MO.setIsKill();
}

/// definePhysReg - Mark PhysReg as reserved or free after spilling any
/// virtregs. This is very similar to defineVirtReg except the physreg is
/// reserved instead of allocated.
void RAFast::definePhysReg(MachineInstr *MI, unsigned PhysReg,
                           RegState NewState) {
  UsedInInstr.set(PhysReg);
  switch (unsigned VirtReg = PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  default:
    spillVirtReg(MI, VirtReg);
    // Fall through.
  case regFree:
  case regReserved:
    PhysRegState[PhysReg] = NewState;
    return;
  }

  // This is a disabled register, disable all aliases.
  PhysRegState[PhysReg] = NewState;
  for (const unsigned *AS = TRI->getAliasSet(PhysReg);
       unsigned Alias = *AS; ++AS) {
    UsedInInstr.set(Alias);
    switch (unsigned VirtReg = PhysRegState[Alias]) {
    case regDisabled:
      break;
    default:
      spillVirtReg(MI, VirtReg);
      // Fall through.
    case regFree:
    case regReserved:
      PhysRegState[Alias] = regDisabled;
      if (TRI->isSuperRegister(PhysReg, Alias))
        return;
      break;
    }
  }
}


/// assignVirtToPhysReg - This method updates local state so that we know
/// that PhysReg is the proper container for VirtReg now.  The physical
/// register must not be used for anything else when this is called.
///
void RAFast::assignVirtToPhysReg(LiveRegEntry &LRE, unsigned PhysReg) {
  DEBUG(dbgs() << "Assigning %reg" << LRE.first << " to "
               << TRI->getName(PhysReg) << "\n");
  PhysRegState[PhysReg] = LRE.first;
  assert(!LRE.second.PhysReg && "Already assigned a physreg");
  LRE.second.PhysReg = PhysReg;
}

/// allocVirtReg - Allocate a physical register for VirtReg.
void RAFast::allocVirtReg(MachineInstr *MI, LiveRegEntry &LRE, unsigned Hint) {
  const unsigned SpillCost = 100;
  const unsigned VirtReg = LRE.first;

  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Can only allocate virtual registers");

  const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);
  TargetRegisterClass::iterator AOB = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator AOE = RC->allocation_order_end(*MF);

  // Ignore invalid hints.
  if (Hint && (!TargetRegisterInfo::isPhysicalRegister(Hint) ||
               !RC->contains(Hint) || UsedInInstr.test(Hint) ||
               !Allocatable.test(Hint)))
    Hint = 0;

  // If there is no hint, peek at the first use of this register.
  if (!Hint && !MRI->use_nodbg_empty(VirtReg)) {
    MachineInstr &MI = *MRI->use_nodbg_begin(VirtReg);
    unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
    // Copy to physreg -> use physreg as hint.
    if (TII->isMoveInstr(MI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
        SrcReg == VirtReg && TargetRegisterInfo::isPhysicalRegister(DstReg) &&
        RC->contains(DstReg) && !UsedInInstr.test(DstReg) &&
        Allocatable.test(DstReg)) {
      Hint = DstReg;
      DEBUG(dbgs() << "%reg" << VirtReg << " gets hint from " << MI);
    }
  }

  // Take hint when possible.
  if (Hint) {
    assert(RC->contains(Hint) && !UsedInInstr.test(Hint) &&
           Allocatable.test(Hint) && "Invalid hint should have been cleared");
    switch(PhysRegState[Hint]) {
    case regDisabled:
    case regReserved:
      break;
    default:
      spillVirtReg(MI, PhysRegState[Hint]);
      // Fall through.
    case regFree:
      return assignVirtToPhysReg(LRE, Hint);
    }
  }

  // First try to find a completely free register.
  unsigned BestCost = 0, BestReg = 0;
  bool hasDisabled = false;
  for (TargetRegisterClass::iterator I = AOB; I != AOE; ++I) {
    unsigned PhysReg = *I;
    switch(PhysRegState[PhysReg]) {
    case regDisabled:
      hasDisabled = true;
    case regReserved:
      continue;
    case regFree:
      if (!UsedInInstr.test(PhysReg))
        return assignVirtToPhysReg(LRE, PhysReg);
      continue;
    default:
      // Grab the first spillable register we meet.
      if (!BestReg && !UsedInInstr.test(PhysReg))
        BestReg = PhysReg, BestCost = SpillCost;
      continue;
    }
  }

  DEBUG(dbgs() << "Allocating %reg" << VirtReg << " from " << RC->getName()
               << " candidate=" << TRI->getName(BestReg) << "\n");

  // Try to extend the working set for RC if there were any disabled registers.
  if (hasDisabled && (!BestReg || BestCost >= SpillCost)) {
    for (TargetRegisterClass::iterator I = AOB; I != AOE; ++I) {
      unsigned PhysReg = *I;
      if (PhysRegState[PhysReg] != regDisabled || UsedInInstr.test(PhysReg))
        continue;

      // Calculate the cost of bringing PhysReg into the working set.
      unsigned Cost=0;
      bool Impossible = false;
      for (const unsigned *AS = TRI->getAliasSet(PhysReg);
      unsigned Alias = *AS; ++AS) {
        if (UsedInInstr.test(Alias)) {
          Impossible = true;
          break;
        }
        switch (PhysRegState[Alias]) {
        case regDisabled:
          break;
        case regReserved:
          Impossible = true;
          break;
        case regFree:
          Cost++;
          break;
        default:
          Cost += SpillCost;
          break;
        }
      }
      if (Impossible) continue;
      DEBUG(dbgs() << "- candidate " << TRI->getName(PhysReg)
        << " cost=" << Cost << "\n");
      if (!BestReg || Cost < BestCost) {
        BestReg = PhysReg;
        BestCost = Cost;
        if (Cost < SpillCost) break;
      }
    }
  }

  if (BestReg) {
    // BestCost is 0 when all aliases are already disabled.
    if (BestCost) {
      if (PhysRegState[BestReg] != regDisabled)
        spillVirtReg(MI, PhysRegState[BestReg]);
      else {
        // Make sure all aliases are disabled.
        for (const unsigned *AS = TRI->getAliasSet(BestReg);
             unsigned Alias = *AS; ++AS) {
          switch (PhysRegState[Alias]) {
          case regDisabled:
            continue;
          case regFree:
            PhysRegState[Alias] = regDisabled;
            break;
          default:
            spillVirtReg(MI, PhysRegState[Alias]);
            PhysRegState[Alias] = regDisabled;
            break;
          }
        }
      }
    }
    return assignVirtToPhysReg(LRE, BestReg);
  }

  // Nothing we can do.
  std::string msg;
  raw_string_ostream Msg(msg);
  Msg << "Ran out of registers during register allocation!";
  if (MI->isInlineAsm()) {
    Msg << "\nPlease check your inline asm statement for "
        << "invalid constraints:\n";
    MI->print(Msg, TM);
  }
  report_fatal_error(Msg.str());
}

/// defineVirtReg - Allocate a register for VirtReg and mark it as dirty.
unsigned RAFast::defineVirtReg(MachineInstr *MI, unsigned OpNum,
                               unsigned VirtReg, unsigned Hint) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Not a virtual register");
  LiveRegMap::iterator LRI;
  bool New;
  tie(LRI, New) = LiveVirtRegs.insert(std::make_pair(VirtReg, LiveReg()));
  LiveReg &LR = LRI->second;
  if (New)
    allocVirtReg(MI, *LRI, Hint);
  else
    addKillFlag(LR); // Kill before redefine.
  assert(LR.PhysReg && "Register not assigned");
  LR.LastUse = MI;
  LR.LastOpNum = OpNum;
  LR.Dirty = true;
  UsedInInstr.set(LR.PhysReg);
  return LR.PhysReg;
}

/// reloadVirtReg - Make sure VirtReg is available in a physreg and return it.
unsigned RAFast::reloadVirtReg(MachineInstr *MI, unsigned OpNum,
                               unsigned VirtReg, unsigned Hint) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Not a virtual register");
  LiveRegMap::iterator LRI;
  bool New;
  tie(LRI, New) = LiveVirtRegs.insert(std::make_pair(VirtReg, LiveReg()));
  LiveReg &LR = LRI->second;
  if (New) {
    allocVirtReg(MI, *LRI, Hint);
    const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);
    int FrameIndex = getStackSpaceFor(VirtReg, RC);
    DEBUG(dbgs() << "Reloading %reg" << VirtReg << " into "
                 << TRI->getName(LR.PhysReg) << "\n");
    TII->loadRegFromStackSlot(*MBB, MI, LR.PhysReg, FrameIndex, RC, TRI);
    ++NumLoads;
  } else if (LR.Dirty) {
    MachineOperand &MO = MI->getOperand(OpNum);
    if (isLastUseOfLocalReg(MO)) {
      DEBUG(dbgs() << "Killing last use: " << MO << "\n");
      MO.setIsKill();
    } else if (MO.isKill()) {
      DEBUG(dbgs() << "Clearing dubious kill: " << MO << "\n");
      MO.setIsKill(false);
    }
  }
  assert(LR.PhysReg && "Register not assigned");
  LR.LastUse = MI;
  LR.LastOpNum = OpNum;
  UsedInInstr.set(LR.PhysReg);
  return LR.PhysReg;
}

// setPhysReg - Change MO the refer the PhysReg, considering subregs.
void RAFast::setPhysReg(MachineOperand &MO, unsigned PhysReg) {
  if (unsigned Idx = MO.getSubReg()) {
    MO.setReg(PhysReg ? TRI->getSubReg(PhysReg, Idx) : 0);
    MO.setSubReg(0);
  } else
    MO.setReg(PhysReg);
}

void RAFast::AllocateBasicBlock() {
  DEBUG(dbgs() << "\nAllocating " << *MBB);

  PhysRegState.assign(TRI->getNumRegs(), regDisabled);
  assert(LiveVirtRegs.empty() && "Mapping not cleared form last block?");

  MachineBasicBlock::iterator MII = MBB->begin();

  // Add live-in registers as live.
  for (MachineBasicBlock::livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I)
    definePhysReg(MII, *I, regReserved);

  SmallVector<unsigned, 8> VirtKills, PhysDefs;
  SmallVector<MachineInstr*, 32> Coalesced;

  // Otherwise, sequentially allocate each instruction in the MBB.
  while (MII != MBB->end()) {
    MachineInstr *MI = MII++;
    const TargetInstrDesc &TID = MI->getDesc();
    DEBUG({
        dbgs() << "\n>> " << *MI << "Regs:";
        for (unsigned Reg = 1, E = TRI->getNumRegs(); Reg != E; ++Reg) {
          if (PhysRegState[Reg] == regDisabled) continue;
          dbgs() << " " << TRI->getName(Reg);
          switch(PhysRegState[Reg]) {
          case regFree:
            break;
          case regReserved:
            dbgs() << "*";
            break;
          default:
            dbgs() << "=%reg" << PhysRegState[Reg];
            if (LiveVirtRegs[PhysRegState[Reg]].Dirty)
              dbgs() << "*";
            assert(LiveVirtRegs[PhysRegState[Reg]].PhysReg == Reg &&
                   "Bad inverse map");
            break;
          }
        }
        dbgs() << '\n';
        // Check that LiveVirtRegs is the inverse.
        for (LiveRegMap::iterator i = LiveVirtRegs.begin(),
             e = LiveVirtRegs.end(); i != e; ++i) {
           assert(TargetRegisterInfo::isVirtualRegister(i->first) &&
                  "Bad map key");
           assert(TargetRegisterInfo::isPhysicalRegister(i->second.PhysReg) &&
                  "Bad map value");
           assert(PhysRegState[i->second.PhysReg] == i->first &&
                  "Bad inverse map");
        }
      });

    // Debug values are not allowed to change codegen in any way.
    if (MI->isDebugValue()) {
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg()) continue;
        unsigned Reg = MO.getReg();
        if (!Reg || TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
        LiveRegMap::iterator LRI = LiveVirtRegs.find(Reg);
        if (LRI != LiveVirtRegs.end())
          setPhysReg(MO, LRI->second.PhysReg);
        else
          MO.setReg(0); // We can't allocate a physreg for a DebugValue, sorry!
      }
      // Next instruction.
      continue;
    }

    // If this is a copy, we may be able to coalesce.
    unsigned CopySrc, CopyDst, CopySrcSub, CopyDstSub;
    if (!TII->isMoveInstr(*MI, CopySrc, CopyDst, CopySrcSub, CopyDstSub))
      CopySrc = CopyDst = 0;

    // Track registers used by instruction.
    UsedInInstr.reset();
    PhysDefs.clear();

    // First scan.
    // Mark physreg uses and early clobbers as used.
    // Find the end of the virtreg operands
    unsigned VirtOpEnd = 0;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (!Reg) continue;
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        VirtOpEnd = i+1;
        continue;
      }
      if (!Allocatable.test(Reg)) continue;
      if (MO.isUse()) {
        usePhysReg(MO);
      } else if (MO.isEarlyClobber()) {
        definePhysReg(MI, Reg, MO.isDead() ? regFree : regReserved);
        PhysDefs.push_back(Reg);
      }
    }

    // Second scan.
    // Allocate virtreg uses and early clobbers.
    // Collect VirtKills
    for (unsigned i = 0; i != VirtOpEnd; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (!Reg || TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
      if (MO.isUse()) {
        unsigned PhysReg = reloadVirtReg(MI, i, Reg, CopyDst);
        CopySrc = (CopySrc == Reg || CopySrc == PhysReg) ? PhysReg : 0;
        setPhysReg(MO, PhysReg);
        if (MO.isKill())
          VirtKills.push_back(Reg);
      } else if (MO.isEarlyClobber()) {
        unsigned PhysReg = defineVirtReg(MI, i, Reg, 0);
        setPhysReg(MO, PhysReg);
        PhysDefs.push_back(PhysReg);
      }
    }

    // Process virtreg kills
    for (unsigned i = 0, e = VirtKills.size(); i != e; ++i)
      killVirtReg(VirtKills[i]);
    VirtKills.clear();

    MRI->addPhysRegsUsed(UsedInInstr);

    // Track registers defined by instruction - early clobbers at this point.
    UsedInInstr.reset();
    for (unsigned i = 0, e = PhysDefs.size(); i != e; ++i) {
      unsigned PhysReg = PhysDefs[i];
      UsedInInstr.set(PhysReg);
      for (const unsigned *AS = TRI->getAliasSet(PhysReg);
            unsigned Alias = *AS; ++AS)
        UsedInInstr.set(Alias);
    }

    unsigned DefOpEnd = MI->getNumOperands();
    if (TID.isCall()) {
      // Spill all virtregs before a call. This serves two purposes: 1. If an
      // exception is thrown, the landing pad is going to expect to find registers
      // in their spill slots, and 2. we don't have to wade through all the
      // <imp-def> operands on the call instruction.
      DefOpEnd = VirtOpEnd;
      DEBUG(dbgs() << "  Spilling remaining registers before call.\n");
      spillAll(MI);
    }

    // Third scan.
    // Allocate defs and collect dead defs.
    for (unsigned i = 0; i != DefOpEnd; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() || !MO.getReg()) continue;
      unsigned Reg = MO.getReg();

      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        if (!Allocatable.test(Reg)) continue;
        definePhysReg(MI, Reg, (MO.isImplicit() || MO.isDead()) ?
                               regFree : regReserved);
        continue;
      }
      unsigned PhysReg = defineVirtReg(MI, i, Reg, CopySrc);
      if (MO.isDead()) {
        VirtKills.push_back(Reg);
        CopyDst = 0; // cancel coalescing;
      } else
        CopyDst = (CopyDst == Reg || CopyDst == PhysReg) ? PhysReg : 0;
      setPhysReg(MO, PhysReg);
    }

    // Process virtreg deads.
    for (unsigned i = 0, e = VirtKills.size(); i != e; ++i)
      killVirtReg(VirtKills[i]);
    VirtKills.clear();

    MRI->addPhysRegsUsed(UsedInInstr);

    if (CopyDst && CopyDst == CopySrc && CopyDstSub == CopySrcSub) {
      DEBUG(dbgs() << "-- coalescing: " << *MI);
      Coalesced.push_back(MI);
    } else {
      DEBUG(dbgs() << "<< " << *MI);
    }
  }

  // Spill all physical registers holding virtual registers now.
  DEBUG(dbgs() << "Spilling live registers at end of block.\n");
  spillAll(MBB->getFirstTerminator());

  // Erase all the coalesced copies. We are delaying it until now because
  // LiveVirtRegs might refer to the instrs.
  for (unsigned i = 0, e = Coalesced.size(); i != e; ++i)
    MBB->erase(Coalesced[i]);
  NumCopies += Coalesced.size();

  DEBUG(MBB->dump());
}

/// runOnMachineFunction - Register allocate the whole function
///
bool RAFast::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(dbgs() << "********** FAST REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << ((Value*)Fn.getFunction())->getName() << '\n');
  if (VerifyFastRegalloc)
    Fn.verify(this, true);
  MF = &Fn;
  MRI = &MF->getRegInfo();
  TM = &Fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();

  UsedInInstr.resize(TRI->getNumRegs());
  Allocatable = TRI->getAllocatableSet(*MF);

  // initialize the virtual->physical register map to have a 'null'
  // mapping for all virtual registers
  unsigned LastVirtReg = MRI->getLastVirtReg();
  StackSlotForVirtReg.grow(LastVirtReg);

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBBi = Fn.begin(), MBBe = Fn.end();
       MBBi != MBBe; ++MBBi) {
    MBB = &*MBBi;
    AllocateBasicBlock();
  }

  // Make sure the set of used physregs is closed under subreg operations.
  MRI->closePhysRegsUsed(*TRI);

  StackSlotForVirtReg.clear();
  return true;
}

FunctionPass *llvm::createFastRegisterAllocator() {
  return new RAFast();
}
