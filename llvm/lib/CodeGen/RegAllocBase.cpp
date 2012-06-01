//===-- RegAllocBase.cpp - Register Allocator Base Class ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegAllocBase class which provides comon functionality
// for LiveIntervalUnion-based register allocators.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "RegAllocBase.h"
#include "Spiller.h"
#include "VirtRegMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#ifndef NDEBUG
#include "llvm/ADT/SparseBitVector.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"

using namespace llvm;

STATISTIC(NumAssigned     , "Number of registers assigned");
STATISTIC(NumUnassigned   , "Number of registers unassigned");
STATISTIC(NumNewQueued    , "Number of new live ranges queued");

// Temporary verification option until we can put verification inside
// MachineVerifier.
static cl::opt<bool, true>
VerifyRegAlloc("verify-regalloc", cl::location(RegAllocBase::VerifyEnabled),
               cl::desc("Verify during register allocation"));

const char *RegAllocBase::TimerGroupName = "Register Allocation";
bool RegAllocBase::VerifyEnabled = false;

#ifndef NDEBUG
// Verify each LiveIntervalUnion.
void RegAllocBase::verify() {
  LiveVirtRegBitSet VisitedVRegs;
  OwningArrayPtr<LiveVirtRegBitSet>
    unionVRegs(new LiveVirtRegBitSet[PhysReg2LiveUnion.numRegs()]);

  // Verify disjoint unions.
  for (unsigned PhysReg = 0; PhysReg < PhysReg2LiveUnion.numRegs(); ++PhysReg) {
    DEBUG(PhysReg2LiveUnion[PhysReg].print(dbgs(), TRI));
    LiveVirtRegBitSet &VRegs = unionVRegs[PhysReg];
    PhysReg2LiveUnion[PhysReg].verify(VRegs);
    // Union + intersection test could be done efficiently in one pass, but
    // don't add a method to SparseBitVector unless we really need it.
    assert(!VisitedVRegs.intersects(VRegs) && "vreg in multiple unions");
    VisitedVRegs |= VRegs;
  }

  // Verify vreg coverage.
  for (LiveIntervals::iterator liItr = LIS->begin(), liEnd = LIS->end();
       liItr != liEnd; ++liItr) {
    unsigned reg = liItr->first;
    LiveInterval* li = liItr->second;
    if (TargetRegisterInfo::isPhysicalRegister(reg)) continue;
    if (!VRM->hasPhys(reg)) continue; // spilled?
    if (li->empty()) continue; // unionVRegs will only be filled if li is
                               // non-empty
    unsigned PhysReg = VRM->getPhys(reg);
    if (!unionVRegs[PhysReg].test(reg)) {
      dbgs() << "LiveVirtReg " << PrintReg(reg, TRI) << " not in union " <<
        TRI->getName(PhysReg) << "\n";
      llvm_unreachable("unallocated live vreg");
    }
  }
  // FIXME: I'm not sure how to verify spilled intervals.
}
#endif //!NDEBUG

//===----------------------------------------------------------------------===//
//                         RegAllocBase Implementation
//===----------------------------------------------------------------------===//

// Instantiate a LiveIntervalUnion for each physical register.
void RegAllocBase::LiveUnionArray::init(LiveIntervalUnion::Allocator &allocator,
                                        unsigned NRegs) {
  NumRegs = NRegs;
  Array =
    static_cast<LiveIntervalUnion*>(malloc(sizeof(LiveIntervalUnion)*NRegs));
  for (unsigned r = 0; r != NRegs; ++r)
    new(Array + r) LiveIntervalUnion(r, allocator);
}

void RegAllocBase::init(VirtRegMap &vrm, LiveIntervals &lis) {
  NamedRegionTimer T("Initialize", TimerGroupName, TimePassesIsEnabled);
  TRI = &vrm.getTargetRegInfo();
  MRI = &vrm.getRegInfo();
  VRM = &vrm;
  LIS = &lis;
  MRI->freezeReservedRegs(vrm.getMachineFunction());
  RegClassInfo.runOnMachineFunction(vrm.getMachineFunction());

  const unsigned NumRegs = TRI->getNumRegs();
  if (NumRegs != PhysReg2LiveUnion.numRegs()) {
    PhysReg2LiveUnion.init(UnionAllocator, NumRegs);
    // Cache an interferece query for each physical reg
    Queries.reset(new LiveIntervalUnion::Query[PhysReg2LiveUnion.numRegs()]);
  }
}

void RegAllocBase::LiveUnionArray::clear() {
  if (!Array)
    return;
  for (unsigned r = 0; r != NumRegs; ++r)
    Array[r].~LiveIntervalUnion();
  free(Array);
  NumRegs =  0;
  Array = 0;
}

void RegAllocBase::releaseMemory() {
  for (unsigned r = 0, e = PhysReg2LiveUnion.numRegs(); r != e; ++r)
    PhysReg2LiveUnion[r].clear();
}

// Visit all the live registers. If they are already assigned to a physical
// register, unify them with the corresponding LiveIntervalUnion, otherwise push
// them on the priority queue for later assignment.
void RegAllocBase::seedLiveRegs() {
  NamedRegionTimer T("Seed Live Regs", TimerGroupName, TimePassesIsEnabled);
  for (LiveIntervals::iterator I = LIS->begin(), E = LIS->end(); I != E; ++I) {
    unsigned RegNum = I->first;
    LiveInterval &VirtReg = *I->second;
    if (TargetRegisterInfo::isPhysicalRegister(RegNum))
      PhysReg2LiveUnion[RegNum].unify(VirtReg);
    else
      enqueue(&VirtReg);
  }
}

void RegAllocBase::assign(LiveInterval &VirtReg, unsigned PhysReg) {
  DEBUG(dbgs() << "assigning " << PrintReg(VirtReg.reg, TRI)
               << " to " << PrintReg(PhysReg, TRI) << '\n');
  assert(!VRM->hasPhys(VirtReg.reg) && "Duplicate VirtReg assignment");
  VRM->assignVirt2Phys(VirtReg.reg, PhysReg);
  MRI->setPhysRegUsed(PhysReg);
  PhysReg2LiveUnion[PhysReg].unify(VirtReg);
  ++NumAssigned;
}

void RegAllocBase::unassign(LiveInterval &VirtReg, unsigned PhysReg) {
  DEBUG(dbgs() << "unassigning " << PrintReg(VirtReg.reg, TRI)
               << " from " << PrintReg(PhysReg, TRI) << '\n');
  assert(VRM->getPhys(VirtReg.reg) == PhysReg && "Inconsistent unassign");
  PhysReg2LiveUnion[PhysReg].extract(VirtReg);
  VRM->clearVirt(VirtReg.reg);
  ++NumUnassigned;
}

// Top-level driver to manage the queue of unassigned VirtRegs and call the
// selectOrSplit implementation.
void RegAllocBase::allocatePhysRegs() {
  seedLiveRegs();

  // Continue assigning vregs one at a time to available physical registers.
  while (LiveInterval *VirtReg = dequeue()) {
    assert(!VRM->hasPhys(VirtReg->reg) && "Register already assigned");

    // Unused registers can appear when the spiller coalesces snippets.
    if (MRI->reg_nodbg_empty(VirtReg->reg)) {
      DEBUG(dbgs() << "Dropping unused " << *VirtReg << '\n');
      LIS->removeInterval(VirtReg->reg);
      continue;
    }

    // Invalidate all interference queries, live ranges could have changed.
    invalidateVirtRegs();

    // selectOrSplit requests the allocator to return an available physical
    // register if possible and populate a list of new live intervals that
    // result from splitting.
    DEBUG(dbgs() << "\nselectOrSplit "
                 << MRI->getRegClass(VirtReg->reg)->getName()
                 << ':' << *VirtReg << '\n');
    typedef SmallVector<LiveInterval*, 4> VirtRegVec;
    VirtRegVec SplitVRegs;
    unsigned AvailablePhysReg = selectOrSplit(*VirtReg, SplitVRegs);

    if (AvailablePhysReg == ~0u) {
      // selectOrSplit failed to find a register!
      const char *Msg = "ran out of registers during register allocation";
      // Probably caused by an inline asm.
      MachineInstr *MI;
      for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(VirtReg->reg);
           (MI = I.skipInstruction());)
        if (MI->isInlineAsm())
          break;
      if (MI)
        MI->emitError(Msg);
      else
        report_fatal_error(Msg);
      // Keep going after reporting the error.
      VRM->assignVirt2Phys(VirtReg->reg,
                 RegClassInfo.getOrder(MRI->getRegClass(VirtReg->reg)).front());
      continue;
    }

    if (AvailablePhysReg)
      assign(*VirtReg, AvailablePhysReg);

    for (VirtRegVec::iterator I = SplitVRegs.begin(), E = SplitVRegs.end();
         I != E; ++I) {
      LiveInterval *SplitVirtReg = *I;
      assert(!VRM->hasPhys(SplitVirtReg->reg) && "Register already assigned");
      if (MRI->reg_nodbg_empty(SplitVirtReg->reg)) {
        DEBUG(dbgs() << "not queueing unused  " << *SplitVirtReg << '\n');
        LIS->removeInterval(SplitVirtReg->reg);
        continue;
      }
      DEBUG(dbgs() << "queuing new interval: " << *SplitVirtReg << "\n");
      assert(TargetRegisterInfo::isVirtualRegister(SplitVirtReg->reg) &&
             "expect split value in virtual register");
      enqueue(SplitVirtReg);
      ++NumNewQueued;
    }
  }
}

// Check if this live virtual register interferes with a physical register. If
// not, then check for interference on each register that aliases with the
// physical register. Return the interfering register.
unsigned RegAllocBase::checkPhysRegInterference(LiveInterval &VirtReg,
                                                unsigned PhysReg) {
  for (MCRegAliasIterator AI(PhysReg, TRI, true); AI.isValid(); ++AI)
    if (query(VirtReg, *AI).checkInterference())
      return *AI;
  return 0;
}

// Add newly allocated physical registers to the MBB live in sets.
void RegAllocBase::addMBBLiveIns(MachineFunction *MF) {
  NamedRegionTimer T("MBB Live Ins", TimerGroupName, TimePassesIsEnabled);
  SlotIndexes *Indexes = LIS->getSlotIndexes();
  if (MF->size() <= 1)
    return;

  LiveIntervalUnion::SegmentIter SI;
  for (unsigned PhysReg = 0; PhysReg < PhysReg2LiveUnion.numRegs(); ++PhysReg) {
    LiveIntervalUnion &LiveUnion = PhysReg2LiveUnion[PhysReg];
    if (LiveUnion.empty())
      continue;
    DEBUG(dbgs() << PrintReg(PhysReg, TRI) << " live-in:");
    MachineFunction::iterator MBB = llvm::next(MF->begin());
    MachineFunction::iterator MFE = MF->end();
    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(MBB);
    SI.setMap(LiveUnion.getMap());
    SI.find(Start);
    while (SI.valid()) {
      if (SI.start() <= Start) {
        if (!MBB->isLiveIn(PhysReg))
          MBB->addLiveIn(PhysReg);
        DEBUG(dbgs() << "\tBB#" << MBB->getNumber() << ':'
                     << PrintReg(SI.value()->reg, TRI));
      } else if (SI.start() > Stop)
        MBB = Indexes->getMBBFromIndex(SI.start().getPrevIndex());
      if (++MBB == MFE)
        break;
      tie(Start, Stop) = Indexes->getMBBRange(MBB);
      SI.advanceTo(Start);
    }
    DEBUG(dbgs() << '\n');
  }
}

