//===-- llvm/CodeGen/Spiller.cpp -  Spiller -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "spiller"

#include "Spiller.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace llvm;

namespace {
  enum SpillerName { trivial, standard, splitting, inline_ };
}

static cl::opt<SpillerName>
spillerOpt("spiller",
           cl::desc("Spiller to use: (default: standard)"),
           cl::Prefix,
           cl::values(clEnumVal(trivial,   "trivial spiller"),
                      clEnumVal(standard,  "default spiller"),
                      clEnumVal(splitting, "splitting spiller"),
                      clEnumValN(inline_,  "inline", "inline spiller"),
                      clEnumValEnd),
           cl::init(standard));

// Spiller virtual destructor implementation.
Spiller::~Spiller() {}

namespace {

/// Utility class for spillers.
class SpillerBase : public Spiller {
protected:
  MachineFunctionPass *pass;
  MachineFunction *mf;
  VirtRegMap *vrm;
  LiveIntervals *lis;
  MachineFrameInfo *mfi;
  MachineRegisterInfo *mri;
  const TargetInstrInfo *tii;
  const TargetRegisterInfo *tri;

  /// Construct a spiller base.
  SpillerBase(MachineFunctionPass &pass, MachineFunction &mf, VirtRegMap &vrm)
    : pass(&pass), mf(&mf), vrm(&vrm)
  {
    lis = &pass.getAnalysis<LiveIntervals>();
    mfi = mf.getFrameInfo();
    mri = &mf.getRegInfo();
    tii = mf.getTarget().getInstrInfo();
    tri = mf.getTarget().getRegisterInfo();
  }

  /// Add spill ranges for every use/def of the live interval, inserting loads
  /// immediately before each use, and stores after each def. No folding or
  /// remat is attempted.
  void trivialSpillEverywhere(LiveInterval *li,
                              SmallVectorImpl<LiveInterval*> &newIntervals) {
    DEBUG(dbgs() << "Spilling everywhere " << *li << "\n");

    assert(li->weight != HUGE_VALF &&
           "Attempting to spill already spilled value.");

    assert(!li->isStackSlot() &&
           "Trying to spill a stack slot.");

    DEBUG(dbgs() << "Trivial spill everywhere of reg" << li->reg << "\n");

    const TargetRegisterClass *trc = mri->getRegClass(li->reg);
    unsigned ss = vrm->assignVirt2StackSlot(li->reg);

    // Iterate over reg uses/defs.
    for (MachineRegisterInfo::reg_iterator
         regItr = mri->reg_begin(li->reg); regItr != mri->reg_end();) {

      // Grab the use/def instr.
      MachineInstr *mi = &*regItr;

      DEBUG(dbgs() << "  Processing " << *mi);

      // Step regItr to the next use/def instr.
      do {
        ++regItr;
      } while (regItr != mri->reg_end() && (&*regItr == mi));

      // Collect uses & defs for this instr.
      SmallVector<unsigned, 2> indices;
      bool hasUse = false;
      bool hasDef = false;
      for (unsigned i = 0; i != mi->getNumOperands(); ++i) {
        MachineOperand &op = mi->getOperand(i);
        if (!op.isReg() || op.getReg() != li->reg)
          continue;
        hasUse |= mi->getOperand(i).isUse();
        hasDef |= mi->getOperand(i).isDef();
        indices.push_back(i);
      }

      // Create a new vreg & interval for this instr.
      unsigned newVReg = mri->createVirtualRegister(trc);
      vrm->grow();
      vrm->assignVirt2StackSlot(newVReg, ss);
      LiveInterval *newLI = &lis->getOrCreateInterval(newVReg);
      newLI->weight = HUGE_VALF;

      // Update the reg operands & kill flags.
      for (unsigned i = 0; i < indices.size(); ++i) {
        unsigned mopIdx = indices[i];
        MachineOperand &mop = mi->getOperand(mopIdx);
        mop.setReg(newVReg);
        if (mop.isUse() && !mi->isRegTiedToDefOperand(mopIdx)) {
          mop.setIsKill(true);
        }
      }
      assert(hasUse || hasDef);

      // Insert reload if necessary.
      MachineBasicBlock::iterator miItr(mi);
      if (hasUse) {
        tii->loadRegFromStackSlot(*mi->getParent(), miItr, newVReg, ss, trc,
                                  tri);
        MachineInstr *loadInstr(prior(miItr));
        SlotIndex loadIndex =
          lis->InsertMachineInstrInMaps(loadInstr).getDefIndex();
        vrm->addSpillSlotUse(ss, loadInstr);
        SlotIndex endIndex = loadIndex.getNextIndex();
        VNInfo *loadVNI =
          newLI->getNextValue(loadIndex, 0, true, lis->getVNInfoAllocator());
        newLI->addRange(LiveRange(loadIndex, endIndex, loadVNI));
      }

      // Insert store if necessary.
      if (hasDef) {
        tii->storeRegToStackSlot(*mi->getParent(), llvm::next(miItr), newVReg,
                                 true, ss, trc, tri);
        MachineInstr *storeInstr(llvm::next(miItr));
        SlotIndex storeIndex =
          lis->InsertMachineInstrInMaps(storeInstr).getDefIndex();
        vrm->addSpillSlotUse(ss, storeInstr);
        SlotIndex beginIndex = storeIndex.getPrevIndex();
        VNInfo *storeVNI =
          newLI->getNextValue(beginIndex, 0, true, lis->getVNInfoAllocator());
        newLI->addRange(LiveRange(beginIndex, storeIndex, storeVNI));
      }

      newIntervals.push_back(newLI);
    }
  }
};

} // end anonymous namespace

namespace {

/// Spills any live range using the spill-everywhere method with no attempt at
/// folding.
class TrivialSpiller : public SpillerBase {
public:

  TrivialSpiller(MachineFunctionPass &pass, MachineFunction &mf,
                 VirtRegMap &vrm)
    : SpillerBase(pass, mf, vrm) {}

  void spill(LiveInterval *li,
             SmallVectorImpl<LiveInterval*> &newIntervals,
             SmallVectorImpl<LiveInterval*> &) {
    // Ignore spillIs - we don't use it.
    trivialSpillEverywhere(li, newIntervals);
  }
};

} // end anonymous namespace

namespace {

/// Falls back on LiveIntervals::addIntervalsForSpills.
class StandardSpiller : public Spiller {
protected:
  LiveIntervals *lis;
  MachineLoopInfo *loopInfo;
  VirtRegMap *vrm;
public:
  StandardSpiller(MachineFunctionPass &pass, MachineFunction &mf,
                  VirtRegMap &vrm)
    : lis(&pass.getAnalysis<LiveIntervals>()),
      loopInfo(pass.getAnalysisIfAvailable<MachineLoopInfo>()),
      vrm(&vrm) {}

  /// Falls back on LiveIntervals::addIntervalsForSpills.
  void spill(LiveInterval *li,
             SmallVectorImpl<LiveInterval*> &newIntervals,
             SmallVectorImpl<LiveInterval*> &spillIs) {
    std::vector<LiveInterval*> added =
      lis->addIntervalsForSpills(*li, spillIs, loopInfo, *vrm);
    newIntervals.insert(newIntervals.end(), added.begin(), added.end());
  }
};

} // end anonymous namespace

namespace {

/// When a call to spill is placed this spiller will first try to break the
/// interval up into its component values (one new interval per value).
/// If this fails, or if a call is placed to spill a previously split interval
/// then the spiller falls back on the standard spilling mechanism.
class SplittingSpiller : public StandardSpiller {
public:
  SplittingSpiller(MachineFunctionPass &pass, MachineFunction &mf,
                   VirtRegMap &vrm)
    : StandardSpiller(pass, mf, vrm) {
    mri = &mf.getRegInfo();
    tii = mf.getTarget().getInstrInfo();
    tri = mf.getTarget().getRegisterInfo();
  }

  void spill(LiveInterval *li,
             SmallVectorImpl<LiveInterval*> &newIntervals,
             SmallVectorImpl<LiveInterval*> &spillIs) {
    if (worthTryingToSplit(li))
      tryVNISplit(li);
    else
      StandardSpiller::spill(li, newIntervals, spillIs);
  }

private:

  MachineRegisterInfo *mri;
  const TargetInstrInfo *tii;
  const TargetRegisterInfo *tri;
  DenseSet<LiveInterval*> alreadySplit;

  bool worthTryingToSplit(LiveInterval *li) const {
    return (!alreadySplit.count(li) && li->getNumValNums() > 1);
  }

  /// Try to break a LiveInterval into its component values.
  std::vector<LiveInterval*> tryVNISplit(LiveInterval *li) {

    DEBUG(dbgs() << "Trying VNI split of %reg" << *li << "\n");

    std::vector<LiveInterval*> added;
    SmallVector<VNInfo*, 4> vnis;

    std::copy(li->vni_begin(), li->vni_end(), std::back_inserter(vnis));

    for (SmallVectorImpl<VNInfo*>::iterator vniItr = vnis.begin(),
         vniEnd = vnis.end(); vniItr != vniEnd; ++vniItr) {
      VNInfo *vni = *vniItr;

      // Skip unused VNIs.
      if (vni->isUnused())
        continue;

      DEBUG(dbgs() << "  Extracted Val #" << vni->id << " as ");
      LiveInterval *splitInterval = extractVNI(li, vni);

      if (splitInterval != 0) {
        DEBUG(dbgs() << *splitInterval << "\n");
        added.push_back(splitInterval);
        alreadySplit.insert(splitInterval);
      } else {
        DEBUG(dbgs() << "0\n");
      }
    }

    DEBUG(dbgs() << "Original LI: " << *li << "\n");

    // If there original interval still contains some live ranges
    // add it to added and alreadySplit.
    if (!li->empty()) {
      added.push_back(li);
      alreadySplit.insert(li);
    }

    return added;
  }

  /// Extract the given value number from the interval.
  LiveInterval* extractVNI(LiveInterval *li, VNInfo *vni) const {
    assert(vni->isDefAccurate() || vni->isPHIDef());

    // Create a new vreg and live interval, copy VNI ranges over.
    const TargetRegisterClass *trc = mri->getRegClass(li->reg);
    unsigned newVReg = mri->createVirtualRegister(trc);
    vrm->grow();
    LiveInterval *newLI = &lis->getOrCreateInterval(newVReg);
    VNInfo *newVNI = newLI->createValueCopy(vni, lis->getVNInfoAllocator());

    // Start by copying all live ranges in the VN to the new interval.
    for (LiveInterval::iterator rItr = li->begin(), rEnd = li->end();
         rItr != rEnd; ++rItr) {
      if (rItr->valno == vni) {
        newLI->addRange(LiveRange(rItr->start, rItr->end, newVNI));
      }
    }

    // Erase the old VNI & ranges.
    li->removeValNo(vni);

    // Collect all current uses of the register belonging to the given VNI.
    // We'll use this to rename the register after we've dealt with the def.
    std::set<MachineInstr*> uses;
    for (MachineRegisterInfo::use_iterator
         useItr = mri->use_begin(li->reg), useEnd = mri->use_end();
         useItr != useEnd; ++useItr) {
      uses.insert(&*useItr);
    }

    // Process the def instruction for this VNI.
    if (newVNI->isPHIDef()) {
      // Insert a copy at the start of the MBB. The range proceeding the
      // copy will be attached to the original LiveInterval.
      MachineBasicBlock *defMBB = lis->getMBBFromIndex(newVNI->def);
      MachineInstr *copyMI = BuildMI(*defMBB, defMBB->begin(), DebugLoc(),
                                     tii->get(TargetOpcode::COPY), newVReg)
                               .addReg(li->reg, RegState::Kill);
      SlotIndex copyIdx = lis->InsertMachineInstrInMaps(copyMI);
      VNInfo *phiDefVNI = li->getNextValue(lis->getMBBStartIdx(defMBB),
                                           0, false, lis->getVNInfoAllocator());
      phiDefVNI->setIsPHIDef(true);
      li->addRange(LiveRange(phiDefVNI->def, copyIdx.getDefIndex(), phiDefVNI));
      LiveRange *oldPHIDefRange =
        newLI->getLiveRangeContaining(lis->getMBBStartIdx(defMBB));

      // If the old phi def starts in the middle of the range chop it up.
      if (oldPHIDefRange->start < lis->getMBBStartIdx(defMBB)) {
        LiveRange oldPHIDefRange2(copyIdx.getDefIndex(), oldPHIDefRange->end,
                                  oldPHIDefRange->valno);
        oldPHIDefRange->end = lis->getMBBStartIdx(defMBB);
        newLI->addRange(oldPHIDefRange2);
      } else if (oldPHIDefRange->start == lis->getMBBStartIdx(defMBB)) {
        // Otherwise if it's at the start of the range just trim it.
        oldPHIDefRange->start = copyIdx.getDefIndex();
      } else {
        assert(false && "PHI def range doesn't cover PHI def?");
      }

      newVNI->def = copyIdx.getDefIndex();
      newVNI->setCopy(copyMI);
      newVNI->setIsPHIDef(false); // not a PHI def anymore.
      newVNI->setIsDefAccurate(true);
    } else {
      // non-PHI def. Rename the def. If it's two-addr that means renaming the
      // use and inserting a new copy too.
      MachineInstr *defInst = lis->getInstructionFromIndex(newVNI->def);
      // We'll rename this now, so we can remove it from uses.
      uses.erase(defInst);
      unsigned defOpIdx = defInst->findRegisterDefOperandIdx(li->reg);
      bool isTwoAddr = defInst->isRegTiedToUseOperand(defOpIdx),
        twoAddrUseIsUndef = false;

      for (unsigned i = 0; i < defInst->getNumOperands(); ++i) {
        MachineOperand &mo = defInst->getOperand(i);
        if (mo.isReg() && (mo.isDef() || isTwoAddr) && (mo.getReg()==li->reg)) {
          mo.setReg(newVReg);
          if (isTwoAddr && mo.isUse() && mo.isUndef())
            twoAddrUseIsUndef = true;
        }
      }

      SlotIndex defIdx = lis->getInstructionIndex(defInst);
      newVNI->def = defIdx.getDefIndex();

      if (isTwoAddr && !twoAddrUseIsUndef) {
        MachineBasicBlock *defMBB = defInst->getParent();
        MachineInstr *copyMI = BuildMI(*defMBB, defInst, DebugLoc(),
                                       tii->get(TargetOpcode::COPY), newVReg)
                                 .addReg(li->reg, RegState::Kill);
        SlotIndex copyIdx = lis->InsertMachineInstrInMaps(copyMI);
        LiveRange *origUseRange =
          li->getLiveRangeContaining(newVNI->def.getUseIndex());
        origUseRange->end = copyIdx.getDefIndex();
        VNInfo *copyVNI = newLI->getNextValue(copyIdx.getDefIndex(), copyMI,
                                              true, lis->getVNInfoAllocator());
        LiveRange copyRange(copyIdx.getDefIndex(),defIdx.getDefIndex(),copyVNI);
        newLI->addRange(copyRange);
      }
    }

    for (std::set<MachineInstr*>::iterator
         usesItr = uses.begin(), usesEnd = uses.end();
         usesItr != usesEnd; ++usesItr) {
      MachineInstr *useInst = *usesItr;
      SlotIndex useIdx = lis->getInstructionIndex(useInst);
      LiveRange *useRange =
        newLI->getLiveRangeContaining(useIdx.getUseIndex());

      // If this use doesn't belong to the new interval skip it.
      if (useRange == 0)
        continue;

      // This use doesn't belong to the VNI, skip it.
      if (useRange->valno != newVNI)
        continue;

      // Check if this instr is two address.
      unsigned useOpIdx = useInst->findRegisterUseOperandIdx(li->reg);
      bool isTwoAddress = useInst->isRegTiedToDefOperand(useOpIdx);

      // Rename uses (and defs for two-address instrs).
      for (unsigned i = 0; i < useInst->getNumOperands(); ++i) {
        MachineOperand &mo = useInst->getOperand(i);
        if (mo.isReg() && (mo.isUse() || isTwoAddress) &&
            (mo.getReg() == li->reg)) {
          mo.setReg(newVReg);
        }
      }

      // If this is a two address instruction we've got some extra work to do.
      if (isTwoAddress) {
        // We modified the def operand, so we need to copy back to the original
        // reg.
        MachineBasicBlock *useMBB = useInst->getParent();
        MachineBasicBlock::iterator useItr(useInst);
        MachineInstr *copyMI = BuildMI(*useMBB, llvm::next(useItr), DebugLoc(),
                                       tii->get(TargetOpcode::COPY), newVReg)
                                 .addReg(li->reg, RegState::Kill);
        SlotIndex copyIdx = lis->InsertMachineInstrInMaps(copyMI);

        // Change the old two-address defined range & vni to start at
        // (and be defined by) the copy.
        LiveRange *origDefRange =
          li->getLiveRangeContaining(useIdx.getDefIndex());
        origDefRange->start = copyIdx.getDefIndex();
        origDefRange->valno->def = copyIdx.getDefIndex();
        origDefRange->valno->setCopy(copyMI);

        // Insert a new range & vni for the two-address-to-copy value. This
        // will be attached to the new live interval.
        VNInfo *copyVNI =
          newLI->getNextValue(useIdx.getDefIndex(), 0, true,
                              lis->getVNInfoAllocator());
        LiveRange copyRange(useIdx.getDefIndex(),copyIdx.getDefIndex(),copyVNI);
        newLI->addRange(copyRange);
      }
    }

    // Iterate over any PHI kills - we'll need to insert new copies for them.
    for (LiveInterval::iterator LRI = newLI->begin(), LRE = newLI->end();
         LRI != LRE; ++LRI) {
      if (LRI->valno != newVNI || LRI->end.isPHI())
        continue;
      SlotIndex killIdx = LRI->end;
      MachineBasicBlock *killMBB = lis->getMBBFromIndex(killIdx);
      MachineInstr *copyMI = BuildMI(*killMBB, killMBB->getFirstTerminator(),
                                     DebugLoc(), tii->get(TargetOpcode::COPY),
                                     li->reg)
                               .addReg(newVReg, RegState::Kill);
      SlotIndex copyIdx = lis->InsertMachineInstrInMaps(copyMI);

      // Save the current end. We may need it to add a new range if the
      // current range runs of the end of the MBB.
      SlotIndex newKillRangeEnd = LRI->end;
      LRI->end = copyIdx.getDefIndex();

      if (newKillRangeEnd != lis->getMBBEndIdx(killMBB)) {
        assert(newKillRangeEnd > lis->getMBBEndIdx(killMBB) &&
               "PHI kill range doesn't reach kill-block end. Not sane.");
        newLI->addRange(LiveRange(lis->getMBBEndIdx(killMBB),
                                  newKillRangeEnd, newVNI));
      }

      VNInfo *newKillVNI = li->getNextValue(copyIdx.getDefIndex(),
                                            copyMI, true,
                                            lis->getVNInfoAllocator());
      newKillVNI->setHasPHIKill(true);
      li->addRange(LiveRange(copyIdx.getDefIndex(),
                             lis->getMBBEndIdx(killMBB),
                             newKillVNI));
    }
    newVNI->setHasPHIKill(false);

    return newLI;
  }

};

} // end anonymous namespace


namespace llvm {
Spiller *createInlineSpiller(MachineFunctionPass &pass,
                             MachineFunction &mf,
                             VirtRegMap &vrm);
}

llvm::Spiller* llvm::createSpiller(MachineFunctionPass &pass,
                                   MachineFunction &mf,
                                   VirtRegMap &vrm) {
  switch (spillerOpt) {
  default: assert(0 && "unknown spiller");
  case trivial: return new TrivialSpiller(pass, mf, vrm);
  case standard: return new StandardSpiller(pass, mf, vrm);
  case splitting: return new SplittingSpiller(pass, mf, vrm);
  case inline_: return createInlineSpiller(pass, mf, vrm);
  }
}
