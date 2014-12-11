//===---- LiveRangeCalc.cpp - Calculate live ranges -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the LiveRangeCalc class.
//
//===----------------------------------------------------------------------===//

#include "LiveRangeCalc.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

void LiveRangeCalc::reset(const MachineFunction *mf,
                          SlotIndexes *SI,
                          MachineDominatorTree *MDT,
                          VNInfo::Allocator *VNIA) {
  MF = mf;
  MRI = &MF->getRegInfo();
  Indexes = SI;
  DomTree = MDT;
  Alloc = VNIA;

  MainLiveOutData.reset(MF->getNumBlockIDs());
  LiveIn.clear();
}


static SlotIndex getDefIndex(const SlotIndexes &Indexes, const MachineInstr &MI,
                             bool EarlyClobber) {
  // PHI defs begin at the basic block start index.
  if (MI.isPHI())
    return Indexes.getMBBStartIdx(MI.getParent());

  // Instructions are either normal 'r', or early clobber 'e'.
  return Indexes.getInstructionIndex(&MI).getRegSlot(EarlyClobber);
}

void LiveRangeCalc::createDeadDefs(LiveInterval &LI) {
  assert(MRI && Indexes && "call reset() first");

  // Visit all def operands. If the same instruction has multiple defs of Reg,
  // LR.createDeadDef() will deduplicate.
  const TargetRegisterInfo &TRI = *MRI->getTargetRegisterInfo();
  unsigned Reg = LI.reg;
  for (const MachineOperand &MO : MRI->def_operands(Reg)) {
    const MachineInstr *MI = MO.getParent();
    SlotIndex Idx = getDefIndex(*Indexes, *MI, MO.isEarlyClobber());
    unsigned SubReg = MO.getSubReg();
    if (LI.hasSubRanges() || (SubReg != 0 && MRI->tracksSubRegLiveness())) {
      unsigned Mask = SubReg != 0 ? TRI.getSubRegIndexLaneMask(SubReg)
                                  : MRI->getMaxLaneMaskForVReg(Reg);

      // If this is the first time we see a subregister def, initialize
      // subranges by creating a copy of the main range.
      if (!LI.hasSubRanges() && !LI.empty()) {
        unsigned ClassMask = MRI->getMaxLaneMaskForVReg(Reg);
        LI.createSubRangeFrom(*Alloc, ClassMask, LI);
      }

      for (LiveInterval::SubRange &S : LI.subranges()) {
        // A Mask for subregs common to the existing subrange and current def.
        unsigned Common = S.LaneMask & Mask;
        if (Common == 0)
          continue;
        // A Mask for subregs covered by the subrange but not the current def.
        unsigned LRest = S.LaneMask & ~Mask;
        LiveInterval::SubRange *CommonRange;
        if (LRest != 0) {
          // Split current subrange into Common and LRest ranges.
          S.LaneMask = LRest;
          CommonRange = LI.createSubRangeFrom(*Alloc, Common, S);
        } else {
          assert(Common == S.LaneMask);
          CommonRange = &S;
        }
        CommonRange->createDeadDef(Idx, *Alloc);
        Mask &= ~Common;
      }
      if (Mask != 0) {
        LiveInterval::SubRange *SubRange = LI.createSubRange(*Alloc, Mask);
        SubRange->createDeadDef(Idx, *Alloc);
      }
    }

    // Create the def in LR. This may find an existing def.
    LI.createDeadDef(Idx, *Alloc);
  }
}


void LiveRangeCalc::createDeadDefs(LiveRange &LR, unsigned Reg) {
  assert(MRI && Indexes && "call reset() first");

  // Visit all def operands. If the same instruction has multiple defs of Reg,
  // LR.createDeadDef() will deduplicate.
  for (MachineOperand &MO : MRI->def_operands(Reg)) {
    const MachineInstr *MI = MO.getParent();
    SlotIndex Idx = getDefIndex(*Indexes, *MI, MO.isEarlyClobber());
    // Create the def in LR. This may find an existing def.
    LR.createDeadDef(Idx, *Alloc);
  }
}


static SlotIndex getUseIndex(const SlotIndexes &Indexes,
                             const MachineOperand &MO) {
  const MachineInstr *MI = MO.getParent();
  unsigned OpNo = (&MO - &MI->getOperand(0));
  if (MI->isPHI()) {
    assert(!MO.isDef() && "Cannot handle PHI def of partial register.");
    // The actual place where a phi operand is used is the end of the pred MBB.
    // PHI operands are paired: (Reg, PredMBB).
    return Indexes.getMBBEndIdx(MI->getOperand(OpNo+1).getMBB());
  }

  // Check for early-clobber redefs.
  bool isEarlyClobber = false;
  unsigned DefIdx;
  if (MO.isDef()) {
    isEarlyClobber = MO.isEarlyClobber();
  } else if (MI->isRegTiedToDefOperand(OpNo, &DefIdx)) {
    // FIXME: This would be a lot easier if tied early-clobber uses also
    // had an early-clobber flag.
    isEarlyClobber = MI->getOperand(DefIdx).isEarlyClobber();
  }
  return Indexes.getInstructionIndex(MI).getRegSlot(isEarlyClobber);
}


void LiveRangeCalc::extendToUses(LiveRange &LR, unsigned Reg) {
  assert(MRI && Indexes && "call reset() first");

  // Visit all operands that read Reg. This may include partial defs.
  for (MachineOperand &MO : MRI->reg_nodbg_operands(Reg)) {
    // Clear all kill flags. They will be reinserted after register allocation
    // by LiveIntervalAnalysis::addKillFlags().
    if (MO.isUse())
      MO.setIsKill(false);
    if (!MO.readsReg())
      continue;
    // MI is reading Reg. We may have visited MI before if it happens to be
    // reading Reg multiple times. That is OK, extend() is idempotent.
    SlotIndex Idx = getUseIndex(*Indexes, MO);
    extend(LR, Idx, Reg, MainLiveOutData);
  }
}


void LiveRangeCalc::extendToUses(LiveInterval &LI) {
  assert(MRI && Indexes && "call reset() first");

  const TargetRegisterInfo &TRI = *MRI->getTargetRegisterInfo();
  SmallVector<LiveOutData,2> LiveOuts;
  unsigned NumSubRanges = 0;
  for (const auto &S : LI.subranges()) {
    (void)S;
    ++NumSubRanges;
    LiveOuts.push_back(LiveOutData());
    LiveOuts.back().reset(MF->getNumBlockIDs());
  }

  // Visit all operands that read Reg. This may include partial defs.
  unsigned Reg = LI.reg;
  for (MachineOperand &MO : MRI->reg_nodbg_operands(Reg)) {
    // Clear all kill flags. They will be reinserted after register allocation
    // by LiveIntervalAnalysis::addKillFlags().
    if (MO.isUse())
      MO.setIsKill(false);
    if (!MO.readsReg())
      continue;
    SlotIndex Idx = getUseIndex(*Indexes, MO);
    unsigned SubReg = MO.getSubReg();
    if (MO.isUse() && (LI.hasSubRanges() ||
                       (MRI->tracksSubRegLiveness() && SubReg != 0))) {
      unsigned Mask = SubReg != 0
        ? TRI.getSubRegIndexLaneMask(SubReg)
        : MRI->getMaxLaneMaskForVReg(Reg);

      // If this is the first time we see a subregister def/use. Initialize
      // subranges by creating a copy of the main range.
      if (!LI.hasSubRanges()) {
        unsigned ClassMask = MRI->getMaxLaneMaskForVReg(Reg);
        LI.createSubRangeFrom(*Alloc, ClassMask, LI);
        LiveOuts.insert(LiveOuts.begin(), LiveOutData());
        LiveOuts.front().reset(MF->getNumBlockIDs());
        ++NumSubRanges;
      }
      unsigned SubRangeIdx = 0;
      for (LiveInterval::subrange_iterator S = LI.subrange_begin(),
           SE = LI.subrange_end(); S != SE; ++S, ++SubRangeIdx) {
        // A Mask for subregs common to the existing subrange and current def.
        unsigned Common = S->LaneMask & Mask;
        if (Common == 0)
          continue;
        // A Mask for subregs covered by the subrange but not the current def.
        unsigned LRest = S->LaneMask & ~Mask;
        LiveInterval::SubRange *CommonRange;
        unsigned CommonRangeIdx;
        if (LRest != 0) {
          // Split current subrange into Common and LRest ranges.
          S->LaneMask = LRest;
          CommonRange = LI.createSubRangeFrom(*Alloc, Common, *S);
          CommonRangeIdx = 0;
          LiveOuts.insert(LiveOuts.begin(), LiveOutData());
          LiveOuts.front().reset(MF->getNumBlockIDs());
          ++NumSubRanges;
          ++SubRangeIdx;
        } else {
          // The subrange and current def lanemasks match completely.
          assert(Common == S->LaneMask);
          CommonRange = &*S;
          CommonRangeIdx = SubRangeIdx;
        }
        extend(*CommonRange, Idx, Reg, LiveOuts[CommonRangeIdx]);
        Mask &= ~Common;
      }
      assert(SubRangeIdx == NumSubRanges);
    }
    extend(LI, Idx, Reg, MainLiveOutData);
  }
}


void LiveRangeCalc::updateFromLiveIns(LiveOutData &LiveOuts) {
  LiveRangeUpdater Updater;
  for (SmallVectorImpl<LiveInBlock>::iterator I = LiveIn.begin(),
         E = LiveIn.end(); I != E; ++I) {
    if (!I->DomNode)
      continue;
    MachineBasicBlock *MBB = I->DomNode->getBlock();
    assert(I->Value && "No live-in value found");
    SlotIndex Start, End;
    std::tie(Start, End) = Indexes->getMBBRange(MBB);

    if (I->Kill.isValid())
      // Value is killed inside this block.
      End = I->Kill;
    else {
      // The value is live-through, update LiveOut as well.
      // Defer the Domtree lookup until it is needed.
      assert(LiveOuts.Seen.test(MBB->getNumber()));
      LiveOuts.Map[MBB] = LiveOutPair(I->Value, nullptr);
    }
    Updater.setDest(&I->LR);
    Updater.add(Start, End, I->Value);
  }
  LiveIn.clear();
}


void LiveRangeCalc::extend(LiveRange &LR, SlotIndex Kill, unsigned PhysReg,
                           LiveOutData &LiveOuts) {
  assert(Kill.isValid() && "Invalid SlotIndex");
  assert(Indexes && "Missing SlotIndexes");
  assert(DomTree && "Missing dominator tree");

  MachineBasicBlock *KillMBB = Indexes->getMBBFromIndex(Kill.getPrevSlot());
  assert(KillMBB && "No MBB at Kill");

  // Is there a def in the same MBB we can extend?
  if (LR.extendInBlock(Indexes->getMBBStartIdx(KillMBB), Kill))
    return;

  // Find the single reaching def, or determine if Kill is jointly dominated by
  // multiple values, and we may need to create even more phi-defs to preserve
  // VNInfo SSA form.  Perform a search for all predecessor blocks where we
  // know the dominating VNInfo.
  if (findReachingDefs(LR, *KillMBB, Kill, PhysReg, LiveOuts))
    return;

  // When there were multiple different values, we may need new PHIs.
  calculateValues(LiveOuts);
}


// This function is called by a client after using the low-level API to add
// live-out and live-in blocks.  The unique value optimization is not
// available, SplitEditor::transferValues handles that case directly anyway.
void LiveRangeCalc::calculateValues(LiveOutData &LiveOuts) {
  assert(Indexes && "Missing SlotIndexes");
  assert(DomTree && "Missing dominator tree");
  updateSSA(LiveOuts);
  updateFromLiveIns(LiveOuts);
}


bool LiveRangeCalc::findReachingDefs(LiveRange &LR, MachineBasicBlock &KillMBB,
                                     SlotIndex Kill, unsigned PhysReg,
                                     LiveOutData &LiveOuts) {
  unsigned KillMBBNum = KillMBB.getNumber();

  // Block numbers where LR should be live-in.
  SmallVector<unsigned, 16> WorkList(1, KillMBBNum);

  // Remember if we have seen more than one value.
  bool UniqueVNI = true;
  VNInfo *TheVNI = nullptr;

  // Using Seen as a visited set, perform a BFS for all reaching defs.
  for (unsigned i = 0; i != WorkList.size(); ++i) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(WorkList[i]);

#ifndef NDEBUG
    if (MBB->pred_empty()) {
      MBB->getParent()->verify();
      llvm_unreachable("Use not jointly dominated by defs.");
    }

    if (TargetRegisterInfo::isPhysicalRegister(PhysReg) &&
        !MBB->isLiveIn(PhysReg)) {
      MBB->getParent()->verify();
      errs() << "The register needs to be live in to BB#" << MBB->getNumber()
             << ", but is missing from the live-in list.\n";
      llvm_unreachable("Invalid global physical register");
    }
#endif

    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
       MachineBasicBlock *Pred = *PI;

       // Is this a known live-out block?
       if (LiveOuts.Seen.test(Pred->getNumber())) {
         if (VNInfo *VNI = LiveOuts.Map[Pred].first) {
           if (TheVNI && TheVNI != VNI)
             UniqueVNI = false;
           TheVNI = VNI;
         }
         continue;
       }

       SlotIndex Start, End;
       std::tie(Start, End) = Indexes->getMBBRange(Pred);

       // First time we see Pred.  Try to determine the live-out value, but set
       // it as null if Pred is live-through with an unknown value.
       VNInfo *VNI = LR.extendInBlock(Start, End);
       LiveOuts.setLiveOutValue(Pred, VNI);
       if (VNI) {
         if (TheVNI && TheVNI != VNI)
           UniqueVNI = false;
         TheVNI = VNI;
         continue;
       }

       // No, we need a live-in value for Pred as well
       if (Pred != &KillMBB)
          WorkList.push_back(Pred->getNumber());
       else
          // Loopback to KillMBB, so value is really live through.
         Kill = SlotIndex();
    }
  }

  LiveIn.clear();

  // Both updateSSA() and LiveRangeUpdater benefit from ordered blocks, but
  // neither require it. Skip the sorting overhead for small updates.
  if (WorkList.size() > 4)
    array_pod_sort(WorkList.begin(), WorkList.end());

  // If a unique reaching def was found, blit in the live ranges immediately.
  if (UniqueVNI) {
    LiveRangeUpdater Updater(&LR);
    for (SmallVectorImpl<unsigned>::const_iterator I = WorkList.begin(),
         E = WorkList.end(); I != E; ++I) {
       SlotIndex Start, End;
       std::tie(Start, End) = Indexes->getMBBRange(*I);
       // Trim the live range in KillMBB.
       if (*I == KillMBBNum && Kill.isValid())
         End = Kill;
       else
         LiveOuts.Map[MF->getBlockNumbered(*I)] =
           LiveOutPair(TheVNI, nullptr);
       Updater.add(Start, End, TheVNI);
    }
    return true;
  }

  // Multiple values were found, so transfer the work list to the LiveIn array
  // where UpdateSSA will use it as a work list.
  LiveIn.reserve(WorkList.size());
  for (SmallVectorImpl<unsigned>::const_iterator
       I = WorkList.begin(), E = WorkList.end(); I != E; ++I) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(*I);
    addLiveInBlock(LR, DomTree->getNode(MBB));
    if (MBB == &KillMBB)
      LiveIn.back().Kill = Kill;
  }

  return false;
}


// This is essentially the same iterative algorithm that SSAUpdater uses,
// except we already have a dominator tree, so we don't have to recompute it.
void LiveRangeCalc::updateSSA(LiveOutData &LiveOuts) {
  assert(Indexes && "Missing SlotIndexes");
  assert(DomTree && "Missing dominator tree");

  // Interate until convergence.
  unsigned Changes;
  do {
    Changes = 0;
    // Propagate live-out values down the dominator tree, inserting phi-defs
    // when necessary.
    for (SmallVectorImpl<LiveInBlock>::iterator I = LiveIn.begin(),
           E = LiveIn.end(); I != E; ++I) {
      MachineDomTreeNode *Node = I->DomNode;
      // Skip block if the live-in value has already been determined.
      if (!Node)
        continue;
      MachineBasicBlock *MBB = Node->getBlock();
      MachineDomTreeNode *IDom = Node->getIDom();
      LiveOutPair IDomValue;

      // We need a live-in value to a block with no immediate dominator?
      // This is probably an unreachable block that has survived somehow.
      bool needPHI = !IDom
                  || !LiveOuts.Seen.test(IDom->getBlock()->getNumber());

      // IDom dominates all of our predecessors, but it may not be their
      // immediate dominator. Check if any of them have live-out values that are
      // properly dominated by IDom. If so, we need a phi-def here.
      if (!needPHI) {
        IDomValue = LiveOuts.Map[IDom->getBlock()];

        // Cache the DomTree node that defined the value.
        if (IDomValue.first && !IDomValue.second)
          LiveOuts.Map[IDom->getBlock()].second = IDomValue.second =
            DomTree->getNode(Indexes->getMBBFromIndex(IDomValue.first->def));

        for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
               PE = MBB->pred_end(); PI != PE; ++PI) {
          LiveOutPair &Value = LiveOuts.Map[*PI];
          if (!Value.first || Value.first == IDomValue.first)
            continue;

          // Cache the DomTree node that defined the value.
          if (!Value.second)
            Value.second =
              DomTree->getNode(Indexes->getMBBFromIndex(Value.first->def));

          // This predecessor is carrying something other than IDomValue.
          // It could be because IDomValue hasn't propagated yet, or it could be
          // because MBB is in the dominance frontier of that value.
          if (DomTree->dominates(IDom, Value.second)) {
            needPHI = true;
            break;
          }
        }
      }

      // The value may be live-through even if Kill is set, as can happen when
      // we are called from extendRange. In that case LiveOutSeen is true, and
      // LiveOut indicates a foreign or missing value.
      LiveOutPair &LOP = LiveOuts.Map[MBB];

      // Create a phi-def if required.
      if (needPHI) {
        ++Changes;
        assert(Alloc && "Need VNInfo allocator to create PHI-defs");
        SlotIndex Start, End;
        std::tie(Start, End) = Indexes->getMBBRange(MBB);
        LiveRange &LR = I->LR;
        VNInfo *VNI = LR.getNextValue(Start, *Alloc);
        I->Value = VNI;
        // This block is done, we know the final value.
        I->DomNode = nullptr;

        // Add liveness since updateFromLiveIns now skips this node.
        if (I->Kill.isValid())
          LR.addSegment(LiveInterval::Segment(Start, I->Kill, VNI));
        else {
          LR.addSegment(LiveInterval::Segment(Start, End, VNI));
          LOP = LiveOutPair(VNI, Node);
        }
      } else if (IDomValue.first) {
        // No phi-def here. Remember incoming value.
        I->Value = IDomValue.first;

        // If the IDomValue is killed in the block, don't propagate through.
        if (I->Kill.isValid())
          continue;

        // Propagate IDomValue if it isn't killed:
        // MBB is live-out and doesn't define its own value.
        if (LOP.first == IDomValue.first)
          continue;
        ++Changes;
        LOP = IDomValue;
      }
    }
  } while (Changes);
}
