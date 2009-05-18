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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <map>

using namespace llvm;

Spiller::~Spiller() {}

namespace {

class TrivialSpiller : public Spiller {
public:
  TrivialSpiller(MachineFunction *mf, LiveIntervals *lis, VirtRegMap *vrm) :
    mf(mf), lis(lis), vrm(vrm)
  {
    mfi = mf->getFrameInfo();
    mri = &mf->getRegInfo();
    tii = mf->getTarget().getInstrInfo();
  }

  std::vector<LiveInterval*> spill(LiveInterval *li) {

    DOUT << "Trivial spiller spilling " << *li << "\n";

    assert(li->weight != HUGE_VALF &&
           "Attempting to spill already spilled value.");

    assert(!li->isStackSlot() &&
           "Trying to spill a stack slot.");

    std::vector<LiveInterval*> added;
    
    const TargetRegisterClass *trc = mri->getRegClass(li->reg);
    /*unsigned ss = mfi->CreateStackObject(trc->getSize(),
                                         trc->getAlignment());*/
    unsigned ss = vrm->assignVirt2StackSlot(li->reg);

    MachineRegisterInfo::reg_iterator regItr = mri->reg_begin(li->reg);
    
    while (regItr != mri->reg_end()) {

      MachineInstr *mi = &*regItr;

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
      LiveInterval *newLI = &lis->getOrCreateInterval(newVReg);
      newLI->weight = HUGE_VALF;

      vrm->grow();
      vrm->assignVirt2StackSlot(newVReg, ss);

      for (unsigned i = 0; i < indices.size(); ++i) {
        mi->getOperand(indices[i]).setReg(newVReg);

        if (mi->getOperand(indices[i]).isUse()) {
          mi->getOperand(indices[i]).setIsKill(true);
        }
      }

      if (hasUse) {
        unsigned loadInstIdx = insertLoadFor(mi, ss, newVReg, trc);
        unsigned start = lis->getDefIndex(loadInstIdx),
                 end = lis->getUseIndex(lis->getInstructionIndex(mi));

        VNInfo *vni =
          newLI->getNextValue(loadInstIdx, 0, lis->getVNInfoAllocator());
        vni->kills.push_back(lis->getInstructionIndex(mi));
        LiveRange lr(start, end, vni);

        newLI->addRange(lr);
        added.push_back(newLI);
      }

      if (hasDef) {
        unsigned storeInstIdx = insertStoreFor(mi, ss, newVReg, trc);
        unsigned start = lis->getDefIndex(lis->getInstructionIndex(mi)),
                 end = lis->getUseIndex(storeInstIdx);

        VNInfo *vni =
          newLI->getNextValue(storeInstIdx, 0, lis->getVNInfoAllocator());
        vni->kills.push_back(storeInstIdx);
        LiveRange lr(start, end, vni);
      
        newLI->addRange(lr);
        added.push_back(newLI);
      }

      regItr = mri->reg_begin(li->reg);
    }


    return added;
  }


private:

  MachineFunction *mf;
  LiveIntervals *lis;
  MachineFrameInfo *mfi;
  MachineRegisterInfo *mri;
  const TargetInstrInfo *tii;
  VirtRegMap *vrm;



  void makeRoomForInsertBefore(MachineInstr *mi) {
    if (!lis->hasGapBeforeInstr(lis->getInstructionIndex(mi))) {
      lis->computeNumbering();
    }

    assert(lis->hasGapBeforeInstr(lis->getInstructionIndex(mi)));
  }

  unsigned insertStoreFor(MachineInstr *mi, unsigned ss,
                          unsigned newVReg,
                          const TargetRegisterClass *trc) {
    MachineBasicBlock::iterator nextInstItr(mi); 
    ++nextInstItr;

    makeRoomForInsertBefore(&*nextInstItr);

    unsigned miIdx = lis->getInstructionIndex(mi);

    tii->storeRegToStackSlot(*mi->getParent(), nextInstItr, newVReg,
                             true, ss, trc);
    MachineBasicBlock::iterator storeInstItr(mi);
    ++storeInstItr;
    MachineInstr *storeInst = &*storeInstItr;
    unsigned storeInstIdx = miIdx + LiveIntervals::InstrSlots::NUM;

    assert(lis->getInstructionFromIndex(storeInstIdx) == 0 &&
           "Store inst index already in use.");
    
    lis->InsertMachineInstrInMaps(storeInst, storeInstIdx);

    return storeInstIdx;
  }

  unsigned insertLoadFor(MachineInstr *mi, unsigned ss,
                         unsigned newVReg,
                         const TargetRegisterClass *trc) {
    MachineBasicBlock::iterator useInstItr(mi);

    makeRoomForInsertBefore(mi);
 
    unsigned miIdx = lis->getInstructionIndex(mi);
    
    tii->loadRegFromStackSlot(*mi->getParent(), useInstItr, newVReg, ss, trc);
    MachineBasicBlock::iterator loadInstItr(mi);
    --loadInstItr;
    MachineInstr *loadInst = &*loadInstItr;
    unsigned loadInstIdx = miIdx - LiveIntervals::InstrSlots::NUM;

    assert(lis->getInstructionFromIndex(loadInstIdx) == 0 &&
           "Load inst index already in use.");

    lis->InsertMachineInstrInMaps(loadInst, loadInstIdx);

    return loadInstIdx;
  }

};

}


llvm::Spiller* llvm::createSpiller(MachineFunction *mf, LiveIntervals *lis,
                                   VirtRegMap *vrm) {
  return new TrivialSpiller(mf, lis, vrm);
}
