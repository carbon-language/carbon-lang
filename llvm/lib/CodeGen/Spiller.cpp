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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace llvm;

namespace {
  enum SpillerName { trivial, standard, splitting };
}

static cl::opt<SpillerName>
spillerOpt("spiller",
           cl::desc("Spiller to use: (default: standard)"),
           cl::Prefix,
           cl::values(clEnumVal(trivial,   "trivial spiller"),
                      clEnumVal(standard,  "default spiller"),
                      clEnumVal(splitting, "splitting spiller"),
                      clEnumValEnd),
           cl::init(standard));

// Spiller virtual destructor implementation.
Spiller::~Spiller() {}

namespace {

/// Utility class for spillers.
class SpillerBase : public Spiller {
protected:
  MachineFunction *mf;
  LiveIntervals *lis;
  MachineFrameInfo *mfi;
  MachineRegisterInfo *mri;
  const TargetInstrInfo *tii;
  VirtRegMap *vrm;
  
  /// Construct a spiller base. 
  SpillerBase(MachineFunction *mf, LiveIntervals *lis, VirtRegMap *vrm)
    : mf(mf), lis(lis), vrm(vrm)
  {
    mfi = mf->getFrameInfo();
    mri = &mf->getRegInfo();
    tii = mf->getTarget().getInstrInfo();
  }

  /// Add spill ranges for every use/def of the live interval, inserting loads
  /// immediately before each use, and stores after each def. No folding or
  /// remat is attempted.
  std::vector<LiveInterval*> trivialSpillEverywhere(LiveInterval *li) {
    DEBUG(dbgs() << "Spilling everywhere " << *li << "\n");

    assert(li->weight != HUGE_VALF &&
           "Attempting to spill already spilled value.");

    assert(!li->isStackSlot() &&
           "Trying to spill a stack slot.");

    DEBUG(dbgs() << "Trivial spill everywhere of reg" << li->reg << "\n");

    std::vector<LiveInterval*> added;
    
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
        tii->loadRegFromStackSlot(*mi->getParent(), miItr, newVReg, ss, trc);
        MachineInstr *loadInstr(prior(miItr));
        SlotIndex loadIndex =
          lis->InsertMachineInstrInMaps(loadInstr).getDefIndex();
        SlotIndex endIndex = loadIndex.getNextIndex();
        VNInfo *loadVNI =
          newLI->getNextValue(loadIndex, 0, true, lis->getVNInfoAllocator());
        loadVNI->addKill(endIndex);
        newLI->addRange(LiveRange(loadIndex, endIndex, loadVNI));
      }

      // Insert store if necessary.
      if (hasDef) {
        tii->storeRegToStackSlot(*mi->getParent(), llvm::next(miItr), newVReg, true,
                                 ss, trc);
        MachineInstr *storeInstr(llvm::next(miItr));
        SlotIndex storeIndex =
          lis->InsertMachineInstrInMaps(storeInstr).getDefIndex();
        SlotIndex beginIndex = storeIndex.getPrevIndex();
        VNInfo *storeVNI =
          newLI->getNextValue(beginIndex, 0, true, lis->getVNInfoAllocator());
        storeVNI->addKill(storeIndex);
        newLI->addRange(LiveRange(beginIndex, storeIndex, storeVNI));
      }

      added.push_back(newLI);
    }

    return added;
  }
};

} // end anonymous namespace

namespace {

/// Spills any live range using the spill-everywhere method with no attempt at
/// folding.
class TrivialSpiller : public SpillerBase {
public:

  TrivialSpiller(MachineFunction *mf, LiveIntervals *lis, VirtRegMap *vrm)
    : SpillerBase(mf, lis, vrm) {}

  std::vector<LiveInterval*> spill(LiveInterval *li,
                                   SmallVectorImpl<LiveInterval*> &spillIs,
                                   SlotIndex*) {
    // Ignore spillIs - we don't use it.
    return trivialSpillEverywhere(li);
  }
};

} // end anonymous namespace

namespace {

/// Falls back on LiveIntervals::addIntervalsForSpills.
class StandardSpiller : public Spiller {
protected:
  LiveIntervals *lis;
  const MachineLoopInfo *loopInfo;
  VirtRegMap *vrm;
public:
  StandardSpiller(LiveIntervals *lis, const MachineLoopInfo *loopInfo,
                  VirtRegMap *vrm)
    : lis(lis), loopInfo(loopInfo), vrm(vrm) {}

  /// Falls back on LiveIntervals::addIntervalsForSpills.
  std::vector<LiveInterval*> spill(LiveInterval *li,
                                   SmallVectorImpl<LiveInterval*> &spillIs,
                                   SlotIndex*) {
    return lis->addIntervalsForSpills(*li, spillIs, loopInfo, *vrm);
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
  SplittingSpiller(MachineFunction *mf, LiveIntervals *lis,
                   const MachineLoopInfo *loopInfo, VirtRegMap *vrm)
    : StandardSpiller(lis, loopInfo, vrm) {

    mri = &mf->getRegInfo();
    tii = mf->getTarget().getInstrInfo();
    tri = mf->getTarget().getRegisterInfo();
  }

  std::vector<LiveInterval*> spill(LiveInterval *li,
                                   SmallVectorImpl<LiveInterval*> &spillIs,
                                   SlotIndex *earliestStart) {
    
    if (worthTryingToSplit(li)) {
      return tryVNISplit(li, earliestStart);
    }
    // else
    return StandardSpiller::spill(li, spillIs, earliestStart);
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
  std::vector<LiveInterval*> tryVNISplit(LiveInterval *li,
                                         SlotIndex *earliestStart) {

    DEBUG(dbgs() << "Trying VNI split of %reg" << *li << "\n");

    std::vector<LiveInterval*> added;
    SmallVector<VNInfo*, 4> vnis;

    std::copy(li->vni_begin(), li->vni_end(), std::back_inserter(vnis));
   
    for (SmallVectorImpl<VNInfo*>::iterator vniItr = vnis.begin(),
         vniEnd = vnis.end(); vniItr != vniEnd; ++vniItr) {
      VNInfo *vni = *vniItr;
      
      // Skip unused VNIs, or VNIs with no kills.
      if (vni->isUnused() || vni->kills.empty())
        continue;

      DEBUG(dbgs() << "  Extracted Val #" << vni->id << " as ");
      LiveInterval *splitInterval = extractVNI(li, vni);
      
      if (splitInterval != 0) {
        DEBUG(dbgs() << *splitInterval << "\n");
        added.push_back(splitInterval);
        alreadySplit.insert(splitInterval);
        if (earliestStart != 0) {
          if (splitInterval->beginIndex() < *earliestStart)
            *earliestStart = splitInterval->beginIndex();
        }
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
      if (earliestStart != 0) {
        if (li->beginIndex() < *earliestStart)
          *earliestStart = li->beginIndex();
      }
    }

    return added;
  }

  /// Extract the given value number from the interval.
  LiveInterval* extractVNI(LiveInterval *li, VNInfo *vni) const {
    assert(vni->isDefAccurate() || vni->isPHIDef());
    assert(!vni->kills.empty());

    // Create a new vreg and live interval, copy VNI kills & ranges over.                                                                                                                                                     
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
      tii->copyRegToReg(*defMBB, defMBB->begin(), newVReg, li->reg, trc, trc);
      MachineInstr *copyMI = defMBB->begin();
      copyMI->addRegisterKilled(li->reg, tri);
      SlotIndex copyIdx = lis->InsertMachineInstrInMaps(copyMI);
      VNInfo *phiDefVNI = li->getNextValue(lis->getMBBStartIdx(defMBB),
                                           0, false, lis->getVNInfoAllocator());
      phiDefVNI->setIsPHIDef(true);
      phiDefVNI->addKill(copyIdx.getDefIndex());
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
      // non-PHI def. Rename the def. If it's two-addr that means renaming the use
      // and inserting a new copy too.
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
        tii->copyRegToReg(*defMBB, defInst, newVReg, li->reg, trc, trc);
        MachineInstr *copyMI = prior(MachineBasicBlock::iterator(defInst));
        SlotIndex copyIdx = lis->InsertMachineInstrInMaps(copyMI);
        copyMI->addRegisterKilled(li->reg, tri);
        LiveRange *origUseRange =
          li->getLiveRangeContaining(newVNI->def.getUseIndex());
        VNInfo *origUseVNI = origUseRange->valno;
        origUseRange->end = copyIdx.getDefIndex();
        bool updatedKills = false;
        for (unsigned k = 0; k < origUseVNI->kills.size(); ++k) {
          if (origUseVNI->kills[k] == defIdx.getDefIndex()) {
            origUseVNI->kills[k] = copyIdx.getDefIndex();
            updatedKills = true;
            break;
          }
        }
        assert(updatedKills && "Failed to update VNI kill list.");
        VNInfo *copyVNI = newLI->getNextValue(copyIdx.getDefIndex(), copyMI,
                                              true, lis->getVNInfoAllocator());
        copyVNI->addKill(defIdx.getDefIndex());
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
        tii->copyRegToReg(*useMBB, next(useItr), li->reg, newVReg, trc, trc);
        MachineInstr *copyMI = next(useItr);
        copyMI->addRegisterKilled(newVReg, tri);
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
        copyVNI->addKill(copyIdx.getDefIndex());
        LiveRange copyRange(useIdx.getDefIndex(),copyIdx.getDefIndex(),copyVNI);
        newLI->addRange(copyRange);
      }
    }
    
    // Iterate over any PHI kills - we'll need to insert new copies for them.
    for (VNInfo::KillSet::iterator
         killItr = newVNI->kills.begin(), killEnd = newVNI->kills.end();
         killItr != killEnd; ++killItr) {
      SlotIndex killIdx(*killItr);
      if (killItr->isPHI()) {
        MachineBasicBlock *killMBB = lis->getMBBFromIndex(killIdx);
        LiveRange *oldKillRange =
          newLI->getLiveRangeContaining(killIdx);

        assert(oldKillRange != 0 && "No kill range?");

        tii->copyRegToReg(*killMBB, killMBB->getFirstTerminator(),
                          li->reg, newVReg, trc, trc);
        MachineInstr *copyMI = prior(killMBB->getFirstTerminator());
        copyMI->addRegisterKilled(newVReg, tri);
        SlotIndex copyIdx = lis->InsertMachineInstrInMaps(copyMI);

        // Save the current end. We may need it to add a new range if the
        // current range runs of the end of the MBB.
        SlotIndex newKillRangeEnd = oldKillRange->end;
        oldKillRange->end = copyIdx.getDefIndex();

        if (newKillRangeEnd != lis->getMBBEndIdx(killMBB)) {
          assert(newKillRangeEnd > lis->getMBBEndIdx(killMBB) &&
                 "PHI kill range doesn't reach kill-block end. Not sane.");
          newLI->addRange(LiveRange(lis->getMBBEndIdx(killMBB),
                                    newKillRangeEnd, newVNI));
        }

        *killItr = oldKillRange->end;
        VNInfo *newKillVNI = li->getNextValue(copyIdx.getDefIndex(),
                                              copyMI, true,
                                              lis->getVNInfoAllocator());
        newKillVNI->addKill(lis->getMBBTerminatorGap(killMBB));
        newKillVNI->setHasPHIKill(true);
        li->addRange(LiveRange(copyIdx.getDefIndex(),
                               lis->getMBBEndIdx(killMBB),
                               newKillVNI));
      }

    }

    newVNI->setHasPHIKill(false);

    return newLI;
  }

};

} // end anonymous namespace


llvm::Spiller* llvm::createSpiller(MachineFunction *mf, LiveIntervals *lis,
                                   const MachineLoopInfo *loopInfo,
                                   VirtRegMap *vrm) {
  switch (spillerOpt) {
  default: assert(0 && "unknown spiller");
  case trivial: return new TrivialSpiller(mf, lis, vrm);
  case standard: return new StandardSpiller(lis, loopInfo, vrm);
  case splitting: return new SplittingSpiller(mf, lis, loopInfo, vrm);
  }
}
