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

    // Everything we know about a live virtual register.
    struct LiveReg {
      MachineInstr *LastUse; // Last instr to use reg.
      unsigned PhysReg;      // Currently held here.
      unsigned LastOpNum;    // OpNum on LastUse.

      LiveReg(unsigned p=0) : LastUse(0), PhysReg(p), LastOpNum(0) {
        assert(p && "Don't create LiveRegs without a PhysReg");
      }
    };

    typedef DenseMap<unsigned, LiveReg> LiveRegMap;

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

    // PhysRegDirty - A bit is set for each physreg that holds a dirty virtual
    // register. Bits for physregs that are not mapped to a virtual register are
    // invalid.
    BitVector PhysRegDirty;

    // ReservedRegs - vector of reserved physical registers.
    BitVector ReservedRegs;

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
    void AllocateBasicBlock(MachineBasicBlock &MBB);
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);
    void killVirtReg(unsigned VirtReg);
    void killVirtReg(LiveRegMap::iterator i);
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                      unsigned VirtReg, bool isKill);
    void killPhysReg(unsigned PhysReg);
    void spillPhysReg(MachineBasicBlock &MBB, MachineInstr *I,
                      unsigned PhysReg, bool isKill);
    LiveRegMap::iterator assignVirtToPhysReg(unsigned VirtReg,
                                             unsigned PhysReg);
    LiveRegMap::iterator allocVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                                      unsigned VirtReg);
    unsigned defineVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                           unsigned OpNum, unsigned VirtReg);
    unsigned reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                           unsigned OpNum, unsigned VirtReg);
    void reservePhysReg(MachineBasicBlock &MBB, MachineInstr *MI,
                        unsigned PhysReg);
    void spillAll(MachineBasicBlock &MBB, MachineInstr *MI);
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

/// killVirtReg - Mark virtreg as no longer available.
void RAFast::killVirtReg(LiveRegMap::iterator i) {
  assert(i != LiveVirtRegs.end() && "Killing unmapped virtual register");
  unsigned VirtReg = i->first;
  const LiveReg &LR = i->second;
  assert(PhysRegState[LR.PhysReg] == VirtReg && "Broken RegState mapping");
  PhysRegState[LR.PhysReg] = regFree;
  if (LR.LastUse) {
    MachineOperand &MO = LR.LastUse->getOperand(LR.LastOpNum);
    if (MO.isUse()) MO.setIsKill();
    else            MO.setIsDead();
    DEBUG(dbgs() << "  - last seen here: " << *LR.LastUse);
  }
  LiveVirtRegs.erase(i);
}

/// killVirtReg - Mark virtreg as no longer available.
void RAFast::killVirtReg(unsigned VirtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "killVirtReg needs a virtual register");
  DEBUG(dbgs() << "  Killing %reg" << VirtReg << "\n");
  LiveRegMap::iterator i = LiveVirtRegs.find(VirtReg);
  if (i != LiveVirtRegs.end())
    killVirtReg(i);
}

/// spillVirtReg - This method spills the value specified by VirtReg into the
/// corresponding stack slot if needed. If isKill is set, the register is also
/// killed.
void RAFast::spillVirtReg(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          unsigned VirtReg, bool isKill) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Spilling a physical register is illegal!");
  LiveRegMap::iterator i = LiveVirtRegs.find(VirtReg);
  assert(i != LiveVirtRegs.end() && "Spilling unmapped virtual register");
  const LiveReg &LR = i->second;
  assert(PhysRegState[LR.PhysReg] == VirtReg && "Broken RegState mapping");

  // If this physreg is used by the instruction, we want to kill it on the
  // instruction, not on the spill.
  bool spillKill = isKill && LR.LastUse != MI;

  if (PhysRegDirty.test(LR.PhysReg)) {
    PhysRegDirty.reset(LR.PhysReg);
    DEBUG(dbgs() << "  Spilling register " << TRI->getName(LR.PhysReg)
      << " containing %reg" << VirtReg);
    const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);
    int FrameIndex = getStackSpaceFor(VirtReg, RC);
    DEBUG(dbgs() << " to stack slot #" << FrameIndex << "\n");
    TII->storeRegToStackSlot(MBB, MI, LR.PhysReg, spillKill,
                             FrameIndex, RC, TRI);
    ++NumStores;   // Update statistics

    if (spillKill)
      i->second.LastUse = 0; // Don't kill register again
    else if (!isKill) {
      MachineInstr *Spill = llvm::prior(MI);
      i->second.LastUse = Spill;
      i->second.LastOpNum = Spill->findRegisterUseOperandIdx(LR.PhysReg);
    }
  }

  if (isKill)
    killVirtReg(i);
}

/// spillAll - Spill all dirty virtregs without killing them.
void RAFast::spillAll(MachineBasicBlock &MBB, MachineInstr *MI) {
  SmallVector<unsigned, 16> Dirty;
  for (LiveRegMap::iterator i = LiveVirtRegs.begin(),
       e = LiveVirtRegs.end(); i != e; ++i)
    if (PhysRegDirty.test(i->second.PhysReg))
      Dirty.push_back(i->first);
  for (unsigned i = 0, e = Dirty.size(); i != e; ++i)
    spillVirtReg(MBB, MI, Dirty[i], false);
}

/// killPhysReg - Kill any virtual register aliased by PhysReg.
void RAFast::killPhysReg(unsigned PhysReg) {
  // Fast path for the normal case.
  switch (unsigned VirtReg = PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  case regFree:
    return;
  case regReserved:
    PhysRegState[PhysReg] = regFree;
    return;
  default:
    killVirtReg(VirtReg);
    return;
  }

  // This is a disabled register, we have to check aliases.
  for (const unsigned *AS = TRI->getAliasSet(PhysReg);
       unsigned Alias = *AS; ++AS) {
    switch (unsigned VirtReg = PhysRegState[Alias]) {
    case regDisabled:
    case regFree:
      break;
    case regReserved:
      PhysRegState[Alias] = regFree;
      break;
    default:
      killVirtReg(VirtReg);
      break;
    }
  }
}

/// spillPhysReg - Spill any dirty virtual registers that aliases PhysReg. If
/// isKill is set, they are also killed.
void RAFast::spillPhysReg(MachineBasicBlock &MBB, MachineInstr *MI,
                           unsigned PhysReg, bool isKill) {
  switch (unsigned VirtReg = PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  case regFree:
    return;
  case regReserved:
    if (isKill)
      PhysRegState[PhysReg] = regFree;
    return;
  default:
    spillVirtReg(MBB, MI, VirtReg, isKill);
    return;
  }

  // This is a disabled register, we have to check aliases.
  for (const unsigned *AS = TRI->getAliasSet(PhysReg);
       unsigned Alias = *AS; ++AS) {
    switch (unsigned VirtReg = PhysRegState[Alias]) {
    case regDisabled:
    case regFree:
      break;
    case regReserved:
      if (isKill)
        PhysRegState[Alias] = regFree;
      break;
    default:
      spillVirtReg(MBB, MI, VirtReg, isKill);
      break;
    }
  }
}

/// assignVirtToPhysReg - This method updates local state so that we know
/// that PhysReg is the proper container for VirtReg now.  The physical
/// register must not be used for anything else when this is called.
///
RAFast::LiveRegMap::iterator
RAFast::assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg) {
  DEBUG(dbgs() << "  Assigning %reg" << VirtReg << " to "
               << TRI->getName(PhysReg) << "\n");
  PhysRegState[PhysReg] = VirtReg;
  return LiveVirtRegs.insert(std::make_pair(VirtReg, PhysReg)).first;
}

/// allocVirtReg - Allocate a physical register for VirtReg.
RAFast::LiveRegMap::iterator RAFast::allocVirtReg(MachineBasicBlock &MBB,
                                                  MachineInstr *MI,
                                                  unsigned VirtReg) {
  const unsigned spillCost = 100;
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Can only allocate virtual registers");

  const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);
  TargetRegisterClass::iterator AOB = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator AOE = RC->allocation_order_end(*MF);

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
        return assignVirtToPhysReg(VirtReg, PhysReg);
      continue;
    default:
      // Grab the first spillable register we meet.
      if (!BestReg && !UsedInInstr.test(PhysReg)) {
        BestReg = PhysReg;
        BestCost = PhysRegDirty.test(PhysReg) ? spillCost : 1;
      }
      continue;
    }
  }

  DEBUG(dbgs() << "  Allocating %reg" << VirtReg << " from " << RC->getName()
               << " candidate=" << TRI->getName(BestReg) << "\n");

  // Try to extend the working set for RC if there were any disabled registers.
  if (hasDisabled && (!BestReg || BestCost >= spillCost)) {
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
          Cost += PhysRegDirty.test(Alias) ? spillCost : 1;
          break;
        }
      }
      if (Impossible) continue;
      DEBUG(dbgs() << "  - candidate " << TRI->getName(PhysReg)
        << " cost=" << Cost << "\n");
      if (!BestReg || Cost < BestCost) {
        BestReg = PhysReg;
        BestCost = Cost;
        if (Cost < spillCost) break;
      }
    }
  }

  if (BestReg) {
    // BestCost is 0 when all aliases are already disabled.
    if (BestCost) {
      if (PhysRegState[BestReg] != regDisabled)
        spillVirtReg(MBB, MI, PhysRegState[BestReg], true);
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
            spillVirtReg(MBB, MI, PhysRegState[Alias], true);
            PhysRegState[Alias] = regDisabled;
            break;
          }
        }
      }
    }
    return assignVirtToPhysReg(VirtReg, BestReg);
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
  return LiveVirtRegs.end();
}

/// defineVirtReg - Allocate a register for VirtReg and mark it as dirty.
unsigned RAFast::defineVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                               unsigned OpNum, unsigned VirtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Not a virtual register");
  LiveRegMap::iterator i = LiveVirtRegs.find(VirtReg);
  if (i == LiveVirtRegs.end())
    i = allocVirtReg(MBB, MI, VirtReg);
  i->second.LastUse = MI;
  i->second.LastOpNum = OpNum;
  UsedInInstr.set(i->second.PhysReg);
  PhysRegDirty.set(i->second.PhysReg);
  return i->second.PhysReg;
}

/// reloadVirtReg - Make sure VirtReg is available in a physreg and return it.
unsigned RAFast::reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                               unsigned OpNum, unsigned VirtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "Not a virtual register");
  LiveRegMap::iterator i = LiveVirtRegs.find(VirtReg);
  if (i == LiveVirtRegs.end()) {
    i = allocVirtReg(MBB, MI, VirtReg);
    PhysRegDirty.reset(i->second.PhysReg);
    const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);
    int FrameIndex = getStackSpaceFor(VirtReg, RC);
    DEBUG(dbgs() << "  Reloading %reg" << VirtReg << " into "
                 << TRI->getName(i->second.PhysReg) << "\n");
    TII->loadRegFromStackSlot(MBB, MI, i->second.PhysReg, FrameIndex, RC, TRI);
    ++NumLoads;
  }
  i->second.LastUse = MI;
  i->second.LastOpNum = OpNum;
  UsedInInstr.set(i->second.PhysReg);
  return i->second.PhysReg;
}

/// reservePhysReg - Mark PhysReg as reserved. This is very similar to
/// defineVirtReg except the physreg is reverved instead of allocated.
void RAFast::reservePhysReg(MachineBasicBlock &MBB, MachineInstr *MI,
                            unsigned PhysReg) {
  UsedInInstr.set(PhysReg);
  switch (unsigned VirtReg = PhysRegState[PhysReg]) {
  case regDisabled:
    break;
  case regFree:
    PhysRegState[PhysReg] = regReserved;
    return;
  case regReserved:
    return;
  default:
    spillVirtReg(MBB, MI, VirtReg, true);
    PhysRegState[PhysReg] = regReserved;
    return;
  }

  // This is a disabled register, disable all aliases.
  for (const unsigned *AS = TRI->getAliasSet(PhysReg);
       unsigned Alias = *AS; ++AS) {
    UsedInInstr.set(Alias);
    switch (unsigned VirtReg = PhysRegState[Alias]) {
    case regDisabled:
    case regFree:
      break;
    case regReserved:
      // is a super register already reserved?
      if (TRI->isSuperRegister(PhysReg, Alias))
        return;
      break;
    default:
      spillVirtReg(MBB, MI, VirtReg, true);
      break;
    }
    PhysRegState[Alias] = regDisabled;
  }
  PhysRegState[PhysReg] = regReserved;
}

// setPhysReg - Change MO the refer the PhysReg, considering subregs.
void RAFast::setPhysReg(MachineOperand &MO, unsigned PhysReg) {
  if (unsigned Idx = MO.getSubReg()) {
    MO.setReg(PhysReg ? TRI->getSubReg(PhysReg, Idx) : 0);
    MO.setSubReg(0);
  } else
    MO.setReg(PhysReg);
}

void RAFast::AllocateBasicBlock(MachineBasicBlock &MBB) {
  DEBUG(dbgs() << "\nBB#" << MBB.getNumber() << ", "<< MBB.getName() << "\n");

  PhysRegState.assign(TRI->getNumRegs(), regDisabled);
  assert(LiveVirtRegs.empty() && "Mapping not cleared form last block?");
  PhysRegDirty.reset();

  MachineBasicBlock::iterator MII = MBB.begin();

  // Add live-in registers as live.
  for (MachineBasicBlock::livein_iterator I = MBB.livein_begin(),
         E = MBB.livein_end(); I != E; ++I)
    reservePhysReg(MBB, MII, *I);

  SmallVector<unsigned, 8> VirtKills, PhysKills, PhysDefs;

  // Otherwise, sequentially allocate each instruction in the MBB.
  while (MII != MBB.end()) {
    MachineInstr *MI = MII++;
    const TargetInstrDesc &TID = MI->getDesc();
    DEBUG({
        dbgs() << "\nStarting RegAlloc of: " << *MI << "Working set:";
        for (unsigned Reg = 1, E = TRI->getNumRegs(); Reg != E; ++Reg) {
          if (PhysRegState[Reg] == regDisabled) continue;
          dbgs() << " " << TRI->getName(Reg);
          switch(PhysRegState[Reg]) {
          case regFree:
            break;
          case regReserved:
            dbgs() << "(resv)";
            break;
          default:
            dbgs() << "=%reg" << PhysRegState[Reg];
            if (PhysRegDirty.test(Reg))
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
        LiveRegMap::iterator i = LiveVirtRegs.find(Reg);
        if (i != LiveVirtRegs.end())
          setPhysReg(MO, i->second.PhysReg);
        else
          MO.setReg(0); // We can't allocate a physreg for a DebugValue, sorry!
      }
      // Next instruction.
      continue;
    }

    // Track registers used by instruction.
    UsedInInstr.reset();
    PhysDefs.clear();

    // First scan.
    // Mark physreg uses and early clobbers as used.
    // Collect PhysKills.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;

      // FIXME: For now, don't trust kill flags
      if (MO.isUse()) MO.setIsKill(false);

      unsigned Reg = MO.getReg();
      if (!Reg || !TargetRegisterInfo::isPhysicalRegister(Reg) ||
          ReservedRegs.test(Reg)) continue;
      if (MO.isUse()) {
        PhysKills.push_back(Reg); // Any clean physreg use is a kill.
        UsedInInstr.set(Reg);
      } else if (MO.isEarlyClobber()) {
        spillPhysReg(MBB, MI, Reg, true);
        UsedInInstr.set(Reg);
        PhysDefs.push_back(Reg);
      }
    }

    // Second scan.
    // Allocate virtreg uses and early clobbers.
    // Collect VirtKills
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (!Reg || TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
      if (MO.isUse()) {
        setPhysReg(MO, reloadVirtReg(MBB, MI, i, Reg));
        if (MO.isKill())
          VirtKills.push_back(Reg);
      } else if (MO.isEarlyClobber()) {
        unsigned PhysReg = defineVirtReg(MBB, MI, i, Reg);
        setPhysReg(MO, PhysReg);
        PhysDefs.push_back(PhysReg);
      }
    }

    // Process virtreg kills
    for (unsigned i = 0, e = VirtKills.size(); i != e; ++i)
      killVirtReg(VirtKills[i]);
    VirtKills.clear();

    // Process physreg kills
    for (unsigned i = 0, e = PhysKills.size(); i != e; ++i)
      killPhysReg(PhysKills[i]);
    PhysKills.clear();

    MF->getRegInfo().addPhysRegsUsed(UsedInInstr);

    // Track registers defined by instruction - early clobbers at this point.
    UsedInInstr.reset();
    for (unsigned i = 0, e = PhysDefs.size(); i != e; ++i) {
      unsigned PhysReg = PhysDefs[i];
      UsedInInstr.set(PhysReg);
      for (const unsigned *AS = TRI->getAliasSet(PhysReg);
            unsigned Alias = *AS; ++AS)
        UsedInInstr.set(Alias);
    }

    // Third scan.
    // Allocate defs and collect dead defs.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() || !MO.getReg()) continue;
      unsigned Reg = MO.getReg();

      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        if (ReservedRegs.test(Reg)) continue;
        if (MO.isImplicit())
          spillPhysReg(MBB, MI, Reg, true);
        else
          reservePhysReg(MBB, MI, Reg);
        if (MO.isDead())
          PhysKills.push_back(Reg);
        continue;
      }
      if (MO.isDead())
        VirtKills.push_back(Reg);
      setPhysReg(MO, defineVirtReg(MBB, MI, i, Reg));
    }

    // Spill all dirty virtregs before a call, in case of an exception.
    if (TID.isCall()) {
      DEBUG(dbgs() << "  Spilling remaining registers before call.\n");
      spillAll(MBB, MI);
    }

    // Process virtreg deads.
    for (unsigned i = 0, e = VirtKills.size(); i != e; ++i)
      killVirtReg(VirtKills[i]);
    VirtKills.clear();

    // Process physreg deads.
    for (unsigned i = 0, e = PhysKills.size(); i != e; ++i)
      killPhysReg(PhysKills[i]);
    PhysKills.clear();

    MF->getRegInfo().addPhysRegsUsed(UsedInInstr);
  }

  // Spill all physical registers holding virtual registers now.
  DEBUG(dbgs() << "Killing live registers at end of block.\n");
  MachineBasicBlock::iterator MI = MBB.getFirstTerminator();
  while (!LiveVirtRegs.empty())
    spillVirtReg(MBB, MI, LiveVirtRegs.begin()->first, true);

  DEBUG(MBB.dump());
}

/// runOnMachineFunction - Register allocate the whole function
///
bool RAFast::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(dbgs() << "Machine Function\n");
  DEBUG(Fn.dump());
  MF = &Fn;
  TM = &Fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();

  PhysRegDirty.resize(TRI->getNumRegs());
  UsedInInstr.resize(TRI->getNumRegs());
  ReservedRegs = TRI->getReservedRegs(*MF);

  // initialize the virtual->physical register map to have a 'null'
  // mapping for all virtual registers
  unsigned LastVirtReg = MF->getRegInfo().getLastVirtReg();
  StackSlotForVirtReg.grow(LastVirtReg);

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  // Make sure the set of used physregs is closed under subreg operations.
  MF->getRegInfo().closePhysRegsUsed(*TRI);

  StackSlotForVirtReg.clear();
  return true;
}

FunctionPass *llvm::createFastRegisterAllocator() {
  return new RAFast();
}
