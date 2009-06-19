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
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

Spiller::~Spiller() {}

namespace {

/// Utility class for spillers.
class SpillerBase : public Spiller {
protected:

  MachineFunction *mf;
  LiveIntervals *lis;
  LiveStacks *ls;
  MachineFrameInfo *mfi;
  MachineRegisterInfo *mri;
  const TargetInstrInfo *tii;
  VirtRegMap *vrm;
  
  /// Construct a spiller base. 
  SpillerBase(MachineFunction *mf, LiveIntervals *lis, LiveStacks *ls,
              VirtRegMap *vrm) :
    mf(mf), lis(lis), ls(ls), vrm(vrm)
  {
    mfi = mf->getFrameInfo();
    mri = &mf->getRegInfo();
    tii = mf->getTarget().getInstrInfo();
  }

  /// Ensures there is space before the given machine instruction, returns the
  /// instruction's new number.
  unsigned makeSpaceBefore(MachineInstr *mi) {
    if (!lis->hasGapBeforeInstr(lis->getInstructionIndex(mi))) {
      lis->scaleNumbering(2);
      ls->scaleNumbering(2);
    }

    unsigned miIdx = lis->getInstructionIndex(mi);

    assert(lis->hasGapBeforeInstr(miIdx));
    
    return miIdx;
  }

  /// Ensure there is space after the given machine instruction, returns the
  /// instruction's new number.
  unsigned makeSpaceAfter(MachineInstr *mi) {
    if (!lis->hasGapAfterInstr(lis->getInstructionIndex(mi))) {
      lis->scaleNumbering(2);
      ls->scaleNumbering(2);
    }

    unsigned miIdx = lis->getInstructionIndex(mi);

    assert(lis->hasGapAfterInstr(miIdx));

    return miIdx;
  }  


  /// Insert a store of the given vreg to the given stack slot immediately
  /// after the given instruction. Returns the base index of the inserted
  /// instruction. The caller is responsible for adding an appropriate
  /// LiveInterval to the LiveIntervals analysis.
  unsigned insertStoreFor(MachineInstr *mi, unsigned ss,
                          unsigned vreg,
                          const TargetRegisterClass *trc) {

    MachineBasicBlock::iterator nextInstItr(mi); 
    ++nextInstItr;

    unsigned miIdx = makeSpaceAfter(mi);

    tii->storeRegToStackSlot(*mi->getParent(), nextInstItr, vreg,
                             true, ss, trc);
    MachineBasicBlock::iterator storeInstItr(mi);
    ++storeInstItr;
    MachineInstr *storeInst = &*storeInstItr;
    unsigned storeInstIdx = miIdx + LiveInterval::InstrSlots::NUM;

    assert(lis->getInstructionFromIndex(storeInstIdx) == 0 &&
           "Store inst index already in use.");
    
    lis->InsertMachineInstrInMaps(storeInst, storeInstIdx);

    return storeInstIdx;
  }

  void insertStoreOnInterval(LiveInterval *li,
                             MachineInstr *mi, unsigned ss,
                             unsigned vreg,
                             const TargetRegisterClass *trc) {

    unsigned storeInstIdx = insertStoreFor(mi, ss, vreg, trc);
    unsigned start = lis->getDefIndex(lis->getInstructionIndex(mi)),
             end = lis->getUseIndex(storeInstIdx);

    VNInfo *vni =
      li->getNextValue(storeInstIdx, 0, true, lis->getVNInfoAllocator());
    vni->kills.push_back(storeInstIdx);
    LiveRange lr(start, end, vni);
      
    li->addRange(lr);
  }

  /// Insert a load of the given veg from the given stack slot immediately
  /// before the given instruction. Returns the base index of the inserted
  /// instruction. The caller is responsible for adding an appropriate
  /// LiveInterval to the LiveIntervals analysis.
  unsigned insertLoadFor(MachineInstr *mi, unsigned ss,
                         unsigned vreg,
                         const TargetRegisterClass *trc) {
    MachineBasicBlock::iterator useInstItr(mi);
  
    unsigned miIdx = makeSpaceBefore(mi);
  
    tii->loadRegFromStackSlot(*mi->getParent(), useInstItr, vreg, ss, trc);
    MachineBasicBlock::iterator loadInstItr(mi);
    --loadInstItr;
    MachineInstr *loadInst = &*loadInstItr;
    unsigned loadInstIdx = miIdx - LiveInterval::InstrSlots::NUM;

    assert(lis->getInstructionFromIndex(loadInstIdx) == 0 &&
           "Load inst index already in use.");

    lis->InsertMachineInstrInMaps(loadInst, loadInstIdx);

    return loadInstIdx;
  }

  void insertLoadOnInterval(LiveInterval *li,
                            MachineInstr *mi, unsigned ss, 
                            unsigned vreg,
                            const TargetRegisterClass *trc) {

    unsigned loadInstIdx = insertLoadFor(mi, ss, vreg, trc);
    unsigned start = lis->getDefIndex(loadInstIdx),
             end = lis->getUseIndex(lis->getInstructionIndex(mi));

    VNInfo *vni =
      li->getNextValue(loadInstIdx, 0, true, lis->getVNInfoAllocator());
    vni->kills.push_back(lis->getInstructionIndex(mi));
    LiveRange lr(start, end, vni);

    li->addRange(lr);
  }



  /// Add spill ranges for every use/def of the live interval, inserting loads
  /// immediately before each use, and stores after each def. No folding is
  /// attempted.
  std::vector<LiveInterval*> trivialSpillEverywhere(LiveInterval *li) {
    DOUT << "Spilling everywhere " << *li << "\n";

    assert(li->weight != HUGE_VALF &&
           "Attempting to spill already spilled value.");

    assert(!li->isStackSlot() &&
           "Trying to spill a stack slot.");

    std::vector<LiveInterval*> added;
    
    const TargetRegisterClass *trc = mri->getRegClass(li->reg);
    unsigned ss = vrm->assignVirt2StackSlot(li->reg);

    for (MachineRegisterInfo::reg_iterator
         regItr = mri->reg_begin(li->reg); regItr != mri->reg_end();) {

      MachineInstr *mi = &*regItr;
      do {
        ++regItr;
      } while (regItr != mri->reg_end() && (&*regItr == mi));
      
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

      unsigned newVReg = mri->createVirtualRegister(trc);
      vrm->grow();
      vrm->assignVirt2StackSlot(newVReg, ss);

      LiveInterval *newLI = &lis->getOrCreateInterval(newVReg);
      newLI->weight = HUGE_VALF;
      
      for (unsigned i = 0; i < indices.size(); ++i) {
        mi->getOperand(indices[i]).setReg(newVReg);

        if (mi->getOperand(indices[i]).isUse()) {
          mi->getOperand(indices[i]).setIsKill(true);
        }
      }

      assert(hasUse || hasDef);

      if (hasUse) {
        insertLoadOnInterval(newLI, mi, ss, newVReg, trc);
      }

      if (hasDef) {
        insertStoreOnInterval(newLI, mi, ss, newVReg, trc);
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

  TrivialSpiller(MachineFunction *mf, LiveIntervals *lis, LiveStacks *ls,
                 VirtRegMap *vrm) :
    SpillerBase(mf, lis, ls, vrm) {}

  std::vector<LiveInterval*> spill(LiveInterval *li) {
    return trivialSpillEverywhere(li);
  }

  std::vector<LiveInterval*> intraBlockSplit(LiveInterval *li, VNInfo *valno)  {
    std::vector<LiveInterval*> spillIntervals;
    MachineBasicBlock::iterator storeInsertPoint;

    if (valno->isDefAccurate()) {
      // If we have an accurate def we can just grab an iterator to the instr
      // after the def.
      storeInsertPoint =
        next(MachineBasicBlock::iterator(lis->getInstructionFromIndex(valno->def)));
    } else {
      // If the def info isn't accurate we check if this is a PHI def.
      // If it is then def holds the index of the defining Basic Block, and we
      // can use that to get an insertion point.
      if (valno->isPHIDef()) {

      } else {
        // We have no usable def info. We can't split this value sensibly.
        // FIXME: Need sensible feedback for "failure to split", an empty
        // set of spill intervals could be reasonably returned from a
        // split where both the store and load are folded.
        return spillIntervals;
      }
    }

        

    return spillIntervals;
  }

};

}

llvm::Spiller* llvm::createSpiller(MachineFunction *mf, LiveIntervals *lis,
                                   LiveStacks *ls, VirtRegMap *vrm) {
  return new TrivialSpiller(mf, lis, ls, vrm);
}
