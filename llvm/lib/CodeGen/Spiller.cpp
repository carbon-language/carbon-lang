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

using namespace llvm;

namespace {
  enum SpillerName { trivial, standard };
}

static cl::opt<SpillerName>
spillerOpt("spiller",
           cl::desc("Spiller to use: (default: standard)"),
           cl::Prefix,
           cl::values(clEnumVal(trivial, "trivial spiller"),
                      clEnumVal(standard, "default spiller"),
                      clEnumValEnd),
           cl::init(standard));

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
    DEBUG(errs() << "Spilling everywhere " << *li << "\n");

    assert(li->weight != HUGE_VALF &&
           "Attempting to spill already spilled value.");

    assert(!li->isStackSlot() &&
           "Trying to spill a stack slot.");

    DEBUG(errs() << "Trivial spill everywhere of reg" << li->reg << "\n");

    std::vector<LiveInterval*> added;
    
    const TargetRegisterClass *trc = mri->getRegClass(li->reg);
    unsigned ss = vrm->assignVirt2StackSlot(li->reg);

    // Iterate over reg uses/defs.
    for (MachineRegisterInfo::reg_iterator
         regItr = mri->reg_begin(li->reg); regItr != mri->reg_end();) {

      // Grab the use/def instr.
      MachineInstr *mi = &*regItr;

      DEBUG(errs() << "  Processing " << *mi);

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
        tii->storeRegToStackSlot(*mi->getParent(), next(miItr), newVReg, true,
                                 ss, trc);
        MachineInstr *storeInstr(next(miItr));
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


/// Spills any live range using the spill-everywhere method with no attempt at
/// folding.
class TrivialSpiller : public SpillerBase {
public:

  TrivialSpiller(MachineFunction *mf, LiveIntervals *lis, VirtRegMap *vrm)
    : SpillerBase(mf, lis, vrm) {}

  std::vector<LiveInterval*> spill(LiveInterval *li,
                                   SmallVectorImpl<LiveInterval*> &spillIs) {
    // Ignore spillIs - we don't use it.
    return trivialSpillEverywhere(li);
  }

};

/// Falls back on LiveIntervals::addIntervalsForSpills.
class StandardSpiller : public Spiller {
private:
  LiveIntervals *lis;
  const MachineLoopInfo *loopInfo;
  VirtRegMap *vrm;
public:
  StandardSpiller(MachineFunction *mf, LiveIntervals *lis,
                  const MachineLoopInfo *loopInfo, VirtRegMap *vrm)
    : lis(lis), loopInfo(loopInfo), vrm(vrm) {}

  /// Falls back on LiveIntervals::addIntervalsForSpills.
  std::vector<LiveInterval*> spill(LiveInterval *li,
                                   SmallVectorImpl<LiveInterval*> &spillIs) {
    return lis->addIntervalsForSpills(*li, spillIs, loopInfo, *vrm);
  }

};

}

llvm::Spiller* llvm::createSpiller(MachineFunction *mf, LiveIntervals *lis,
                                   const MachineLoopInfo *loopInfo,
                                   VirtRegMap *vrm) {
  switch (spillerOpt) {
    case trivial: return new TrivialSpiller(mf, lis, vrm); break;
    case standard: return new StandardSpiller(mf, lis, loopInfo, vrm); break;
    default: llvm_unreachable("Unreachable!"); break;
  }
}
