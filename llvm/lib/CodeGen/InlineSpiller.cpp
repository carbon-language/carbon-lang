//===-------- InlineSpiller.cpp - Insert spills and restores inline -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The inline spiller modifies the machine function directly instead of
// inserting spills and restores in VirtRegMap.
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class InlineSpiller : public Spiller {
  MachineFunction &mf_;
  LiveIntervals &lis_;
  VirtRegMap &vrm_;
  MachineFrameInfo &mfi_;
  MachineRegisterInfo &mri_;
  const TargetInstrInfo &tii_;
  const TargetRegisterInfo &tri_;

  ~InlineSpiller() {}

public:
  InlineSpiller(MachineFunction *mf, LiveIntervals *lis, VirtRegMap *vrm)
    : mf_(*mf), lis_(*lis), vrm_(*vrm),
      mfi_(*mf->getFrameInfo()),
      mri_(mf->getRegInfo()),
      tii_(*mf->getTarget().getInstrInfo()),
      tri_(*mf->getTarget().getRegisterInfo()) {}

  void spill(LiveInterval *li,
             std::vector<LiveInterval*> &newIntervals,
             SmallVectorImpl<LiveInterval*> &spillIs,
             SlotIndex *earliestIndex);
};
}

namespace llvm {
Spiller *createInlineSpiller(MachineFunction *mf,
                             LiveIntervals *lis,
                             const MachineLoopInfo *mli,
                             VirtRegMap *vrm) {
  return new InlineSpiller(mf, lis, vrm);
}
}

void InlineSpiller::spill(LiveInterval *li,
                          std::vector<LiveInterval*> &newIntervals,
                          SmallVectorImpl<LiveInterval*> &spillIs,
                          SlotIndex *earliestIndex) {
  DEBUG(dbgs() << "Inline spilling " << *li << "\n");
  assert(li->isSpillable() && "Attempting to spill already spilled value.");
  assert(!li->isStackSlot() && "Trying to spill a stack slot.");

  const TargetRegisterClass *RC = mri_.getRegClass(li->reg);
  unsigned SS = vrm_.assignVirt2StackSlot(li->reg);

  for (MachineRegisterInfo::reg_iterator RI = mri_.reg_begin(li->reg);
       MachineInstr *MI = RI.skipInstruction();) {
    SlotIndex Idx = lis_.getInstructionIndex(MI).getDefIndex();

    // Analyze instruction.
    bool Reads, Writes;
    SmallVector<unsigned, 8> Ops;
    tie(Reads, Writes) = MI->readsWritesVirtualRegister(li->reg, &Ops);

    // Allocate interval around instruction.
    // FIXME: Infer regclass from instruction alone.
    unsigned NewVReg = mri_.createVirtualRegister(RC);
    vrm_.grow();
    LiveInterval &NewLI = lis_.getOrCreateInterval(NewVReg);
    NewLI.markNotSpillable();

    // Reload if instruction reads register.
    if (Reads) {
      MachineBasicBlock::iterator MII = MI;
      tii_.loadRegFromStackSlot(*MI->getParent(), MII, NewVReg, SS, RC, &tri_);
      --MII; // Point to load instruction.
      SlotIndex LoadIdx = lis_.InsertMachineInstrInMaps(MII).getDefIndex();
      vrm_.addSpillSlotUse(SS, MII);
      DEBUG(dbgs() << "\treload:  " << LoadIdx << '\t' << *MII);
      VNInfo *LoadVNI = NewLI.getNextValue(LoadIdx, 0, true,
                                           lis_.getVNInfoAllocator());
      NewLI.addRange(LiveRange(LoadIdx, Idx, LoadVNI));
    }

    // Rewrite instruction operands.
    bool hasLiveDef = false;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(Ops[i]);
      MO.setReg(NewVReg);
      if (MO.isUse()) {
        if (!MI->isRegTiedToDefOperand(Ops[i]))
          MO.setIsKill();
      } else {
        if (!MO.isDead())
          hasLiveDef = true;
      }
    }
    DEBUG(dbgs() << "\trewrite: " << Idx << '\t' << *MI);

    // Spill is instruction writes register.
    // FIXME: Use a second vreg if instruction has no tied ops.
    if (Writes && hasLiveDef) {
      MachineBasicBlock::iterator MII = MI;
      tii_.storeRegToStackSlot(*MI->getParent(), ++MII, NewVReg, true, SS, RC,
                               &tri_);
      --MII; // Point to store instruction.
      SlotIndex StoreIdx = lis_.InsertMachineInstrInMaps(MII).getDefIndex();
      vrm_.addSpillSlotUse(SS, MII);
      DEBUG(dbgs() << "\tspilled: " << StoreIdx << '\t' << *MII);
      VNInfo *StoreVNI = NewLI.getNextValue(Idx, 0, true,
                                            lis_.getVNInfoAllocator());
      NewLI.addRange(LiveRange(Idx, StoreIdx, StoreVNI));
    }

    DEBUG(dbgs() << "\tinterval: " << NewLI << '\n');
    newIntervals.push_back(&NewLI);
  }
}
