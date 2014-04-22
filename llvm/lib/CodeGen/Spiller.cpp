//===-- llvm/CodeGen/Spiller.cpp -  Spiller -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Spiller.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "spiller"

namespace {
  enum SpillerName { trivial, inline_ };
}

static cl::opt<SpillerName>
spillerOpt("spiller",
           cl::desc("Spiller to use: (default: standard)"),
           cl::Prefix,
           cl::values(clEnumVal(trivial,   "trivial spiller"),
                      clEnumValN(inline_,  "inline", "inline spiller"),
                      clEnumValEnd),
           cl::init(trivial));

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
  void trivialSpillEverywhere(LiveRangeEdit& LRE) {
    LiveInterval* li = &LRE.getParent();

    DEBUG(dbgs() << "Spilling everywhere " << *li << "\n");

    assert(li->weight != llvm::huge_valf &&
           "Attempting to spill already spilled value.");

    assert(!TargetRegisterInfo::isStackSlot(li->reg) &&
           "Trying to spill a stack slot.");

    DEBUG(dbgs() << "Trivial spill everywhere of reg" << li->reg << "\n");

    const TargetRegisterClass *trc = mri->getRegClass(li->reg);
    unsigned ss = vrm->assignVirt2StackSlot(li->reg);

    // Iterate over reg uses/defs.
    for (MachineRegisterInfo::reg_instr_iterator
         regItr = mri->reg_instr_begin(li->reg);
         regItr != mri->reg_instr_end();) {

      // Grab the use/def instr.
      MachineInstr *mi = &*regItr;

      DEBUG(dbgs() << "  Processing " << *mi);

      // Step regItr to the next use/def instr.
      ++regItr;

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

      // Create a new virtual register for the load and/or store.
      unsigned NewVReg = LRE.create();

      // Update the reg operands & kill flags.
      for (unsigned i = 0; i < indices.size(); ++i) {
        unsigned mopIdx = indices[i];
        MachineOperand &mop = mi->getOperand(mopIdx);
        mop.setReg(NewVReg);
        if (mop.isUse() && !mi->isRegTiedToDefOperand(mopIdx)) {
          mop.setIsKill(true);
        }
      }
      assert(hasUse || hasDef);

      // Insert reload if necessary.
      MachineBasicBlock::iterator miItr(mi);
      if (hasUse) {
        MachineInstrSpan MIS(miItr);

        tii->loadRegFromStackSlot(*mi->getParent(), miItr, NewVReg, ss, trc,
                                  tri);
        lis->InsertMachineInstrRangeInMaps(MIS.begin(), miItr);
      }

      // Insert store if necessary.
      if (hasDef) {
        MachineInstrSpan MIS(miItr);

        tii->storeRegToStackSlot(*mi->getParent(), std::next(miItr), NewVReg,
                                 true, ss, trc, tri);
        lis->InsertMachineInstrRangeInMaps(std::next(miItr), MIS.end());
      }
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

  void spill(LiveRangeEdit &LRE) override {
    // Ignore spillIs - we don't use it.
    trivialSpillEverywhere(LRE);
  }
};

} // end anonymous namespace

void Spiller::anchor() { }

llvm::Spiller* llvm::createSpiller(MachineFunctionPass &pass,
                                   MachineFunction &mf,
                                   VirtRegMap &vrm) {
  switch (spillerOpt) {
  case trivial: return new TrivialSpiller(pass, mf, vrm);
  case inline_: return createInlineSpiller(pass, mf, vrm);
  }
  llvm_unreachable("Invalid spiller optimization");
}
