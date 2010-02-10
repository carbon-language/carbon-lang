//===------------------------ CalcSpillWeights.cpp ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "calcspillweights"

#include "llvm/Function.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

char CalculateSpillWeights::ID = 0;
static RegisterPass<CalculateSpillWeights> X("calcspillweights",
                                             "Calculate spill weights");

void CalculateSpillWeights::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<LiveIntervals>();
  au.addRequired<MachineLoopInfo>();
  au.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(au);
}

bool CalculateSpillWeights::runOnMachineFunction(MachineFunction &fn) {

  DEBUG(dbgs() << "********** Compute Spill Weights **********\n"
               << "********** Function: "
               << fn.getFunction()->getName() << '\n');

  LiveIntervals *lis = &getAnalysis<LiveIntervals>();
  MachineLoopInfo *loopInfo = &getAnalysis<MachineLoopInfo>();
  const TargetInstrInfo *tii = fn.getTarget().getInstrInfo();
  MachineRegisterInfo *mri = &fn.getRegInfo();

  SmallSet<unsigned, 4> processed;
  for (MachineFunction::iterator mbbi = fn.begin(), mbbe = fn.end();
       mbbi != mbbe; ++mbbi) {
    MachineBasicBlock* mbb = mbbi;
    SlotIndex mbbEnd = lis->getMBBEndIdx(mbb);
    MachineLoop* loop = loopInfo->getLoopFor(mbb);
    unsigned loopDepth = loop ? loop->getLoopDepth() : 0;
    bool isExiting = loop ? loop->isLoopExiting(mbb) : false;

    for (MachineBasicBlock::const_iterator mii = mbb->begin(), mie = mbb->end();
         mii != mie; ++mii) {
      const MachineInstr *mi = mii;
      if (tii->isIdentityCopy(*mi) || mi->isImplicitDef() || mi->isDebugValue())
        continue;

      for (unsigned i = 0, e = mi->getNumOperands(); i != e; ++i) {
        const MachineOperand &mopi = mi->getOperand(i);
        if (!mopi.isReg() || mopi.getReg() == 0)
          continue;
        unsigned reg = mopi.getReg();
        if (!TargetRegisterInfo::isVirtualRegister(mopi.getReg()))
          continue;
        // Multiple uses of reg by the same instruction. It should not
        // contribute to spill weight again.
        if (!processed.insert(reg))
          continue;

        bool hasDef = mopi.isDef();
        bool hasUse = !hasDef;
        for (unsigned j = i+1; j != e; ++j) {
          const MachineOperand &mopj = mi->getOperand(j);
          if (!mopj.isReg() || mopj.getReg() != reg)
            continue;
          hasDef |= mopj.isDef();
          hasUse |= mopj.isUse();
          if (hasDef && hasUse)
            break;
        }

        LiveInterval &regInt = lis->getInterval(reg);
        float weight = lis->getSpillWeight(hasDef, hasUse, loopDepth);
        if (hasDef && isExiting) {
          // Looks like this is a loop count variable update.
          SlotIndex defIdx = lis->getInstructionIndex(mi).getDefIndex();
          const LiveRange *dlr =
            lis->getInterval(reg).getLiveRangeContaining(defIdx);
          if (dlr->end >= mbbEnd)
            weight *= 3.0F;
        }
        regInt.weight += weight;
      }
      processed.clear();
    }
  }

  for (LiveIntervals::iterator I = lis->begin(), E = lis->end(); I != E; ++I) {
    LiveInterval &li = *I->second;
    if (TargetRegisterInfo::isVirtualRegister(li.reg)) {
      // If the live interval length is essentially zero, i.e. in every live
      // range the use follows def immediately, it doesn't make sense to spill
      // it and hope it will be easier to allocate for this li.
      if (isZeroLengthInterval(&li)) {
        li.weight = HUGE_VALF;
        continue;
      }

      bool isLoad = false;
      SmallVector<LiveInterval*, 4> spillIs;
      if (lis->isReMaterializable(li, spillIs, isLoad)) {
        // If all of the definitions of the interval are re-materializable,
        // it is a preferred candidate for spilling. If non of the defs are
        // loads, then it's potentially very cheap to re-materialize.
        // FIXME: this gets much more complicated once we support non-trivial
        // re-materialization.
        if (isLoad)
          li.weight *= 0.9F;
        else
          li.weight *= 0.5F;
      }

      // Slightly prefer live interval that has been assigned a preferred reg.
      std::pair<unsigned, unsigned> Hint = mri->getRegAllocationHint(li.reg);
      if (Hint.first || Hint.second)
        li.weight *= 1.01F;

      // Divide the weight of the interval by its size.  This encourages
      // spilling of intervals that are large and have few uses, and
      // discourages spilling of small intervals with many uses.
      li.weight /= lis->getApproximateInstructionCount(li) * SlotIndex::NUM;
    }
  }
  
  return false;
}

/// Returns true if the given live interval is zero length.
bool CalculateSpillWeights::isZeroLengthInterval(LiveInterval *li) const {
  for (LiveInterval::Ranges::const_iterator
       i = li->ranges.begin(), e = li->ranges.end(); i != e; ++i)
    if (i->end.getPrevIndex() > i->start)
      return false;
  return true;
}
