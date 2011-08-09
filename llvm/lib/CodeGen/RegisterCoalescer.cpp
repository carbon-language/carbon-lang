//===- RegisterCoalescer.cpp - Generic Register Coalescing Interface -------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic RegisterCoalescer interface which
// is used as the common interface used by all clients and
// implementations of register coalescing.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regcoalescing"
#include "RegisterCoalescer.h"
#include "VirtRegMap.h"
#include "LiveDebugVariables.h"

#include "llvm/Pass.h"
#include "llvm/Value.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(numJoins    , "Number of interval joins performed");
STATISTIC(numCrossRCs , "Number of cross class joins performed");
STATISTIC(numCommutes , "Number of instruction commuting performed");
STATISTIC(numExtends  , "Number of copies extended");
STATISTIC(NumReMats   , "Number of instructions re-materialized");
STATISTIC(numPeep     , "Number of identity moves eliminated after coalescing");
STATISTIC(numAborts   , "Number of times interval joining aborted");

static cl::opt<bool>
EnableJoining("join-liveintervals",
              cl::desc("Coalesce copies (default=true)"),
              cl::init(true));

static cl::opt<bool>
DisableCrossClassJoin("disable-cross-class-join",
               cl::desc("Avoid coalescing cross register class copies"),
               cl::init(false), cl::Hidden);

static cl::opt<bool>
EnablePhysicalJoin("join-physregs",
                   cl::desc("Join physical register copies"),
                   cl::init(false), cl::Hidden);

static cl::opt<bool>
VerifyCoalescing("verify-coalescing",
         cl::desc("Verify machine instrs before and after register coalescing"),
         cl::Hidden);

char &llvm::RegisterCoalescerPassID = RegisterCoalescer::ID;

INITIALIZE_PASS_BEGIN(RegisterCoalescer, "simple-register-coalescing",
                      "Simple Register Coalescing", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(StrongPHIElimination)
INITIALIZE_PASS_DEPENDENCY(PHIElimination)
INITIALIZE_PASS_DEPENDENCY(TwoAddressInstructionPass)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(RegisterCoalescer, "simple-register-coalescing",
                    "Simple Register Coalescing", false, false)

char RegisterCoalescer::ID = 0;

static unsigned compose(const TargetRegisterInfo &tri, unsigned a, unsigned b) {
  if (!a) return b;
  if (!b) return a;
  return tri.composeSubRegIndices(a, b);
}

static bool isMoveInstr(const TargetRegisterInfo &tri, const MachineInstr *MI,
                        unsigned &Src, unsigned &Dst,
                        unsigned &SrcSub, unsigned &DstSub) {
  if (MI->isCopy()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = MI->getOperand(0).getSubReg();
    Src = MI->getOperand(1).getReg();
    SrcSub = MI->getOperand(1).getSubReg();
  } else if (MI->isSubregToReg()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = compose(tri, MI->getOperand(0).getSubReg(),
                     MI->getOperand(3).getImm());
    Src = MI->getOperand(2).getReg();
    SrcSub = MI->getOperand(2).getSubReg();
  } else
    return false;
  return true;
}

bool CoalescerPair::setRegisters(const MachineInstr *MI) {
  srcReg_ = dstReg_ = subIdx_ = 0;
  newRC_ = 0;
  flipped_ = crossClass_ = false;

  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(tri_, MI, Src, Dst, SrcSub, DstSub))
    return false;
  partial_ = SrcSub || DstSub;

  // If one register is a physreg, it must be Dst.
  if (TargetRegisterInfo::isPhysicalRegister(Src)) {
    if (TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
    flipped_ = true;
  }

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();

  if (TargetRegisterInfo::isPhysicalRegister(Dst)) {
    // Eliminate DstSub on a physreg.
    if (DstSub) {
      Dst = tri_.getSubReg(Dst, DstSub);
      if (!Dst) return false;
      DstSub = 0;
    }

    // Eliminate SrcSub by picking a corresponding Dst superregister.
    if (SrcSub) {
      Dst = tri_.getMatchingSuperReg(Dst, SrcSub, MRI.getRegClass(Src));
      if (!Dst) return false;
      SrcSub = 0;
    } else if (!MRI.getRegClass(Src)->contains(Dst)) {
      return false;
    }
  } else {
    // Both registers are virtual.

    // Both registers have subreg indices.
    if (SrcSub && DstSub) {
      // For now we only handle the case of identical indices in commensurate
      // registers: Dreg:ssub_1 + Dreg:ssub_1 -> Dreg
      // FIXME: Handle Qreg:ssub_3 + Dreg:ssub_1 as QReg:dsub_1 + Dreg.
      if (SrcSub != DstSub)
        return false;
      const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
      const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);
      if (!getCommonSubClass(DstRC, SrcRC))
        return false;
      SrcSub = DstSub = 0;
    }

    // There can be no SrcSub.
    if (SrcSub) {
      std::swap(Src, Dst);
      DstSub = SrcSub;
      SrcSub = 0;
      assert(!flipped_ && "Unexpected flip");
      flipped_ = true;
    }

    // Find the new register class.
    const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);
    if (DstSub)
      newRC_ = tri_.getMatchingSuperRegClass(DstRC, SrcRC, DstSub);
    else
      newRC_ = getCommonSubClass(DstRC, SrcRC);
    if (!newRC_)
      return false;
    crossClass_ = newRC_ != DstRC || newRC_ != SrcRC;
  }
  // Check our invariants
  assert(TargetRegisterInfo::isVirtualRegister(Src) && "Src must be virtual");
  assert(!(TargetRegisterInfo::isPhysicalRegister(Dst) && DstSub) &&
         "Cannot have a physical SubIdx");
  srcReg_ = Src;
  dstReg_ = Dst;
  subIdx_ = DstSub;
  return true;
}

bool CoalescerPair::flip() {
  if (subIdx_ || TargetRegisterInfo::isPhysicalRegister(dstReg_))
    return false;
  std::swap(srcReg_, dstReg_);
  flipped_ = !flipped_;
  return true;
}

bool CoalescerPair::isCoalescable(const MachineInstr *MI) const {
  if (!MI)
    return false;
  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(tri_, MI, Src, Dst, SrcSub, DstSub))
    return false;

  // Find the virtual register that is srcReg_.
  if (Dst == srcReg_) {
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
  } else if (Src != srcReg_) {
    return false;
  }

  // Now check that Dst matches dstReg_.
  if (TargetRegisterInfo::isPhysicalRegister(dstReg_)) {
    if (!TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    assert(!subIdx_ && "Inconsistent CoalescerPair state.");
    // DstSub could be set for a physreg from INSERT_SUBREG.
    if (DstSub)
      Dst = tri_.getSubReg(Dst, DstSub);
    // Full copy of Src.
    if (!SrcSub)
      return dstReg_ == Dst;
    // This is a partial register copy. Check that the parts match.
    return tri_.getSubReg(dstReg_, SrcSub) == Dst;
  } else {
    // dstReg_ is virtual.
    if (dstReg_ != Dst)
      return false;
    // Registers match, do the subregisters line up?
    return compose(tri_, subIdx_, SrcSub) == DstSub;
  }
}

void RegisterCoalescer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreservedID(MachineDominatorsID);
  AU.addPreservedID(StrongPHIEliminationID);
  AU.addPreservedID(PHIEliminationID);
  AU.addPreservedID(TwoAddressInstructionPassID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RegisterCoalescer::markAsJoined(MachineInstr *CopyMI) {
  /// Joined copies are not deleted immediately, but kept in JoinedCopies.
  JoinedCopies.insert(CopyMI);

  /// Mark all register operands of CopyMI as <undef> so they won't affect dead
  /// code elimination.
  for (MachineInstr::mop_iterator I = CopyMI->operands_begin(),
       E = CopyMI->operands_end(); I != E; ++I)
    if (I->isReg())
      I->setIsUndef(true);
}

/// AdjustCopiesBackFrom - We found a non-trivially-coalescable copy with IntA
/// being the source and IntB being the dest, thus this defines a value number
/// in IntB.  If the source value number (in IntA) is defined by a copy from B,
/// see if we can merge these two pieces of B into a single value number,
/// eliminating a copy.  For example:
///
///  A3 = B0
///    ...
///  B1 = A3      <- this copy
///
/// In this case, B0 can be extended to where the B1 copy lives, allowing the B1
/// value number to be replaced with B0 (which simplifies the B liveinterval).
///
/// This returns true if an interval was modified.
///
bool RegisterCoalescer::AdjustCopiesBackFrom(const CoalescerPair &CP,
                                                    MachineInstr *CopyMI) {
  // Bail if there is no dst interval - can happen when merging physical subreg
  // operations.
  if (!li_->hasInterval(CP.getDstReg()))
    return false;

  LiveInterval &IntA =
    li_->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
    li_->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());
  SlotIndex CopyIdx = li_->getInstructionIndex(CopyMI).getDefIndex();

  // BValNo is a value number in B that is defined by a copy from A.  'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  if (BLR == IntB.end()) return false;
  VNInfo *BValNo = BLR->valno;

  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we
  // can't process it.
  if (!BValNo->isDefByCopy()) return false;
  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");

  // AValNo is the value number in A that defines the copy, A3 in the example.
  SlotIndex CopyUseIdx = CopyIdx.getUseIndex();
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyUseIdx);
  // The live range might not exist after fun with physreg coalescing.
  if (ALR == IntA.end()) return false;
  VNInfo *AValNo = ALR->valno;
  // If it's re-defined by an early clobber somewhere in the live range, then
  // it's not safe to eliminate the copy. FIXME: This is a temporary workaround.
  // See PR3149:
  // 172     %ECX<def> = MOV32rr %reg1039<kill>
  // 180     INLINEASM <es:subl $5,$1
  //         sbbl $3,$0>, 10, %EAX<def>, 14, %ECX<earlyclobber,def>, 9,
  //         %EAX<kill>,
  // 36, <fi#0>, 1, %reg0, 0, 9, %ECX<kill>, 36, <fi#1>, 1, %reg0, 0
  // 188     %EAX<def> = MOV32rr %EAX<kill>
  // 196     %ECX<def> = MOV32rr %ECX<kill>
  // 204     %ECX<def> = MOV32rr %ECX<kill>
  // 212     %EAX<def> = MOV32rr %EAX<kill>
  // 220     %EAX<def> = MOV32rr %EAX
  // 228     %reg1039<def> = MOV32rr %ECX<kill>
  // The early clobber operand ties ECX input to the ECX def.
  //
  // The live interval of ECX is represented as this:
  // %reg20,inf = [46,47:1)[174,230:0)  0@174-(230) 1@46-(47)
  // The coalescer has no idea there was a def in the middle of [174,230].
  if (AValNo->hasRedefByEC())
    return false;

  // If AValNo is defined as a copy from IntB, we can potentially process this.
  // Get the instruction that defines this value number.
  if (!CP.isCoalescable(AValNo->getCopy()))
    return false;

  // Get the LiveRange in IntB that this value number starts with.
  LiveInterval::iterator ValLR =
    IntB.FindLiveRangeContaining(AValNo->def.getPrevSlot());
  if (ValLR == IntB.end())
    return false;

  // Make sure that the end of the live range is inside the same block as
  // CopyMI.
  MachineInstr *ValLREndInst =
    li_->getInstructionFromIndex(ValLR->end.getPrevSlot());
  if (!ValLREndInst || ValLREndInst->getParent() != CopyMI->getParent())
    return false;

  // Okay, we now know that ValLR ends in the same block that the CopyMI
  // live-range starts.  If there are no intervening live ranges between them in
  // IntB, we can merge them.
  if (ValLR+1 != BLR) return false;

  // If a live interval is a physical register, conservatively check if any
  // of its aliases is overlapping the live interval of the virtual register.
  // If so, do not coalesce.
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg)) {
    for (const unsigned *AS = tri_->getAliasSet(IntB.reg); *AS; ++AS)
      if (li_->hasInterval(*AS) && IntA.overlaps(li_->getInterval(*AS))) {
        DEBUG({
            dbgs() << "\t\tInterfere with alias ";
            li_->getInterval(*AS).print(dbgs(), tri_);
          });
        return false;
      }
  }

  DEBUG({
      dbgs() << "Extending: ";
      IntB.print(dbgs(), tri_);
    });

  SlotIndex FillerStart = ValLR->end, FillerEnd = BLR->start;
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'. Update the valnum with the new defining
  // instruction #.
  BValNo->def  = FillerStart;
  BValNo->setCopy(0);

  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // If the IntB live range is assigned to a physical register, and if that
  // physreg has sub-registers, update their live intervals as well.
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg)) {
    for (const unsigned *SR = tri_->getSubRegisters(IntB.reg); *SR; ++SR) {
      if (!li_->hasInterval(*SR))
        continue;
      LiveInterval &SRLI = li_->getInterval(*SR);
      SRLI.addRange(LiveRange(FillerStart, FillerEnd,
                              SRLI.getNextValue(FillerStart, 0,
                                                li_->getVNInfoAllocator())));
    }
  }

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValLR->valno) {
    // If B1 is killed by a PHI, then the merged live range must also be killed
    // by the same PHI, as B0 and B1 can not overlap.
    bool HasPHIKill = BValNo->hasPHIKill();
    IntB.MergeValueNumberInto(BValNo, ValLR->valno);
    if (HasPHIKill)
      ValLR->valno->setHasPHIKill(true);
  }
  DEBUG({
      dbgs() << "   result = ";
      IntB.print(dbgs(), tri_);
      dbgs() << "\n";
    });

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValLREndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1) {
    ValLREndInst->getOperand(UIdx).setIsKill(false);
  }

  // If the copy instruction was killing the destination register before the
  // merge, find the last use and trim the live range. That will also add the
  // isKill marker.
  if (ALR->end == CopyIdx)
    li_->shrinkToUses(&IntA);

  ++numExtends;
  return true;
}

/// HasOtherReachingDefs - Return true if there are definitions of IntB
/// other than BValNo val# that can reach uses of AValno val# of IntA.
bool RegisterCoalescer::HasOtherReachingDefs(LiveInterval &IntA,
                                                    LiveInterval &IntB,
                                                    VNInfo *AValNo,
                                                    VNInfo *BValNo) {
  for (LiveInterval::iterator AI = IntA.begin(), AE = IntA.end();
       AI != AE; ++AI) {
    if (AI->valno != AValNo) continue;
    LiveInterval::Ranges::iterator BI =
      std::upper_bound(IntB.ranges.begin(), IntB.ranges.end(), AI->start);
    if (BI != IntB.ranges.begin())
      --BI;
    for (; BI != IntB.ranges.end() && AI->end >= BI->start; ++BI) {
      if (BI->valno == BValNo)
        continue;
      if (BI->start <= AI->start && BI->end > AI->start)
        return true;
      if (BI->start > AI->start && BI->start < AI->end)
        return true;
    }
  }
  return false;
}

/// RemoveCopyByCommutingDef - We found a non-trivially-coalescable copy with
/// IntA being the source and IntB being the dest, thus this defines a value
/// number in IntB.  If the source value number (in IntA) is defined by a
/// commutable instruction and its other operand is coalesced to the copy dest
/// register, see if we can transform the copy into a noop by commuting the
/// definition. For example,
///
///  A3 = op A2 B0<kill>
///    ...
///  B1 = A3      <- this copy
///    ...
///     = op A3   <- more uses
///
/// ==>
///
///  B2 = op B0 A2<kill>
///    ...
///  B1 = B2      <- now an identify copy
///    ...
///     = op B2   <- more uses
///
/// This returns true if an interval was modified.
///
bool RegisterCoalescer::RemoveCopyByCommutingDef(const CoalescerPair &CP,
                                                        MachineInstr *CopyMI) {
  // FIXME: For now, only eliminate the copy by commuting its def when the
  // source register is a virtual register. We want to guard against cases
  // where the copy is a back edge copy and commuting the def lengthen the
  // live interval of the source register to the entire loop.
  if (CP.isPhys() && CP.isFlipped())
    return false;

  // Bail if there is no dst interval.
  if (!li_->hasInterval(CP.getDstReg()))
    return false;

  SlotIndex CopyIdx = li_->getInstructionIndex(CopyMI).getDefIndex();

  LiveInterval &IntA =
    li_->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
    li_->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());

  // BValNo is a value number in B that is defined by a copy from A. 'B3' in
  // the example above.
  VNInfo *BValNo = IntB.getVNInfoAt(CopyIdx);
  if (!BValNo || !BValNo->isDefByCopy())
    return false;

  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");

  // AValNo is the value number in A that defines the copy, A3 in the example.
  VNInfo *AValNo = IntA.getVNInfoAt(CopyIdx.getUseIndex());
  assert(AValNo && "COPY source not live");

  // If other defs can reach uses of this def, then it's not safe to perform
  // the optimization.
  if (AValNo->isPHIDef() || AValNo->isUnused() || AValNo->hasPHIKill())
    return false;
  MachineInstr *DefMI = li_->getInstructionFromIndex(AValNo->def);
  if (!DefMI)
    return false;
  const MCInstrDesc &MCID = DefMI->getDesc();
  if (!MCID.isCommutable())
    return false;
  // If DefMI is a two-address instruction then commuting it will change the
  // destination register.
  int DefIdx = DefMI->findRegisterDefOperandIdx(IntA.reg);
  assert(DefIdx != -1);
  unsigned UseOpIdx;
  if (!DefMI->isRegTiedToUseOperand(DefIdx, &UseOpIdx))
    return false;
  unsigned Op1, Op2, NewDstIdx;
  if (!tii_->findCommutedOpIndices(DefMI, Op1, Op2))
    return false;
  if (Op1 == UseOpIdx)
    NewDstIdx = Op2;
  else if (Op2 == UseOpIdx)
    NewDstIdx = Op1;
  else
    return false;

  MachineOperand &NewDstMO = DefMI->getOperand(NewDstIdx);
  unsigned NewReg = NewDstMO.getReg();
  if (NewReg != IntB.reg || !NewDstMO.isKill())
    return false;

  // Make sure there are no other definitions of IntB that would reach the
  // uses which the new definition can reach.
  if (HasOtherReachingDefs(IntA, IntB, AValNo, BValNo))
    return false;

  // Abort if the aliases of IntB.reg have values that are not simply the
  // clobbers from the superreg.
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg))
    for (const unsigned *AS = tri_->getAliasSet(IntB.reg); *AS; ++AS)
      if (li_->hasInterval(*AS) &&
          HasOtherReachingDefs(IntA, li_->getInterval(*AS), AValNo, 0))
        return false;

  // If some of the uses of IntA.reg is already coalesced away, return false.
  // It's not possible to determine whether it's safe to perform the coalescing.
  for (MachineRegisterInfo::use_nodbg_iterator UI = 
         mri_->use_nodbg_begin(IntA.reg), 
       UE = mri_->use_nodbg_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    SlotIndex UseIdx = li_->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end())
      continue;
    if (ULR->valno == AValNo && JoinedCopies.count(UseMI))
      return false;
  }

  DEBUG(dbgs() << "\tRemoveCopyByCommutingDef: " << AValNo->def << '\t'
               << *DefMI);

  // At this point we have decided that it is legal to do this
  // transformation.  Start by commuting the instruction.
  MachineBasicBlock *MBB = DefMI->getParent();
  MachineInstr *NewMI = tii_->commuteInstruction(DefMI);
  if (!NewMI)
    return false;
  if (TargetRegisterInfo::isVirtualRegister(IntA.reg) &&
      TargetRegisterInfo::isVirtualRegister(IntB.reg) &&
      !mri_->constrainRegClass(IntB.reg, mri_->getRegClass(IntA.reg)))
    return false;
  if (NewMI != DefMI) {
    li_->ReplaceMachineInstrInMaps(DefMI, NewMI);
    MBB->insert(DefMI, NewMI);
    MBB->erase(DefMI);
  }
  unsigned OpIdx = NewMI->findRegisterUseOperandIdx(IntA.reg, false);
  NewMI->getOperand(OpIdx).setIsKill();

  // If ALR and BLR overlaps and end of BLR extends beyond end of ALR, e.g.
  // A = or A, B
  // ...
  // B = A
  // ...
  // C = A<kill>
  // ...
  //   = B

  // Update uses of IntA of the specific Val# with IntB.
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(IntA.reg),
         UE = mri_->use_end(); UI != UE;) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    ++UI;
    if (JoinedCopies.count(UseMI))
      continue;
    if (UseMI->isDebugValue()) {
      // FIXME These don't have an instruction index.  Not clear we have enough
      // info to decide whether to do this replacement or not.  For now do it.
      UseMO.setReg(NewReg);
      continue;
    }
    SlotIndex UseIdx = li_->getInstructionIndex(UseMI).getUseIndex();
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end() || ULR->valno != AValNo)
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(NewReg))
      UseMO.substPhysReg(NewReg, *tri_);
    else
      UseMO.setReg(NewReg);
    if (UseMI == CopyMI)
      continue;
    if (!UseMI->isCopy())
      continue;
    if (UseMI->getOperand(0).getReg() != IntB.reg ||
        UseMI->getOperand(0).getSubReg())
      continue;

    // This copy will become a noop. If it's defining a new val#, merge it into
    // BValNo.
    SlotIndex DefIdx = UseIdx.getDefIndex();
    VNInfo *DVNI = IntB.getVNInfoAt(DefIdx);
    if (!DVNI)
      continue;
    DEBUG(dbgs() << "\t\tnoop: " << DefIdx << '\t' << *UseMI);
    assert(DVNI->def == DefIdx);
    BValNo = IntB.MergeValueNumberInto(BValNo, DVNI);
    markAsJoined(UseMI);
  }

  // Extend BValNo by merging in IntA live ranges of AValNo. Val# definition
  // is updated.
  VNInfo *ValNo = BValNo;
  ValNo->def = AValNo->def;
  ValNo->setCopy(0);
  for (LiveInterval::iterator AI = IntA.begin(), AE = IntA.end();
       AI != AE; ++AI) {
    if (AI->valno != AValNo) continue;
    IntB.addRange(LiveRange(AI->start, AI->end, ValNo));
  }
  DEBUG(dbgs() << "\t\textended: " << IntB << '\n');

  IntA.removeValNo(AValNo);
  DEBUG(dbgs() << "\t\ttrimmed:  " << IntA << '\n');
  ++numCommutes;
  return true;
}

/// ReMaterializeTrivialDef - If the source of a copy is defined by a trivial
/// computation, replace the copy by rematerialize the definition.
bool RegisterCoalescer::ReMaterializeTrivialDef(LiveInterval &SrcInt,
                                                       bool preserveSrcInt,
                                                       unsigned DstReg,
                                                       unsigned DstSubIdx,
                                                       MachineInstr *CopyMI) {
  SlotIndex CopyIdx = li_->getInstructionIndex(CopyMI).getUseIndex();
  LiveInterval::iterator SrcLR = SrcInt.FindLiveRangeContaining(CopyIdx);
  assert(SrcLR != SrcInt.end() && "Live range not found!");
  VNInfo *ValNo = SrcLR->valno;
  // If other defs can reach uses of this def, then it's not safe to perform
  // the optimization.
  if (ValNo->isPHIDef() || ValNo->isUnused() || ValNo->hasPHIKill())
    return false;
  MachineInstr *DefMI = li_->getInstructionFromIndex(ValNo->def);
  if (!DefMI)
    return false;
  assert(DefMI && "Defining instruction disappeared");
  const MCInstrDesc &MCID = DefMI->getDesc();
  if (!MCID.isAsCheapAsAMove())
    return false;
  if (!tii_->isTriviallyReMaterializable(DefMI, AA))
    return false;
  bool SawStore = false;
  if (!DefMI->isSafeToMove(tii_, AA, SawStore))
    return false;
  if (MCID.getNumDefs() != 1)
    return false;
  if (!DefMI->isImplicitDef()) {
    // Make sure the copy destination register class fits the instruction
    // definition register class. The mismatch can happen as a result of earlier
    // extract_subreg, insert_subreg, subreg_to_reg coalescing.
    const TargetRegisterClass *RC = tii_->getRegClass(MCID, 0, tri_);
    if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
      if (mri_->getRegClass(DstReg) != RC)
        return false;
    } else if (!RC->contains(DstReg))
      return false;
  }

  // If destination register has a sub-register index on it, make sure it
  // matches the instruction register class.
  if (DstSubIdx) {
    const MCInstrDesc &MCID = DefMI->getDesc();
    if (MCID.getNumDefs() != 1)
      return false;
    const TargetRegisterClass *DstRC = mri_->getRegClass(DstReg);
    const TargetRegisterClass *DstSubRC =
      DstRC->getSubRegisterRegClass(DstSubIdx);
    const TargetRegisterClass *DefRC = tii_->getRegClass(MCID, 0, tri_);
    if (DefRC == DstRC)
      DstSubIdx = 0;
    else if (DefRC != DstSubRC)
      return false;
  }

  RemoveCopyFlag(DstReg, CopyMI);

  MachineBasicBlock *MBB = CopyMI->getParent();
  MachineBasicBlock::iterator MII =
    llvm::next(MachineBasicBlock::iterator(CopyMI));
  tii_->reMaterialize(*MBB, MII, DstReg, DstSubIdx, DefMI, *tri_);
  MachineInstr *NewMI = prior(MII);

  // CopyMI may have implicit operands, transfer them over to the newly
  // rematerialized instruction. And update implicit def interval valnos.
  for (unsigned i = CopyMI->getDesc().getNumOperands(),
         e = CopyMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = CopyMI->getOperand(i);
    if (MO.isReg() && MO.isImplicit())
      NewMI->addOperand(MO);
    if (MO.isDef())
      RemoveCopyFlag(MO.getReg(), CopyMI);
  }

  NewMI->copyImplicitOps(CopyMI);
  li_->ReplaceMachineInstrInMaps(CopyMI, NewMI);
  CopyMI->eraseFromParent();
  ReMatCopies.insert(CopyMI);
  ReMatDefs.insert(DefMI);
  DEBUG(dbgs() << "Remat: " << *NewMI);
  ++NumReMats;

  // The source interval can become smaller because we removed a use.
  if (preserveSrcInt)
    li_->shrinkToUses(&SrcInt);

  return true;
}

/// eliminateUndefCopy - ProcessImpicitDefs may leave some copies of <undef>
/// values, it only removes local variables. When we have a copy like:
///
///   %vreg1 = COPY %vreg2<undef>
///
/// We delete the copy and remove the corresponding value number from %vreg1.
/// Any uses of that value number are marked as <undef>.
bool RegisterCoalescer::eliminateUndefCopy(MachineInstr *CopyMI,
                                           const CoalescerPair &CP) {
  SlotIndex Idx = li_->getInstructionIndex(CopyMI);
  LiveInterval *SrcInt = &li_->getInterval(CP.getSrcReg());
  if (SrcInt->liveAt(Idx))
    return false;
  LiveInterval *DstInt = &li_->getInterval(CP.getDstReg());
  if (DstInt->liveAt(Idx))
    return false;

  // No intervals are live-in to CopyMI - it is undef.
  if (CP.isFlipped())
    DstInt = SrcInt;
  SrcInt = 0;

  VNInfo *DeadVNI = DstInt->getVNInfoAt(Idx.getDefIndex());
  assert(DeadVNI && "No value defined in DstInt");
  DstInt->removeValNo(DeadVNI);

  // Find new undef uses.
  for (MachineRegisterInfo::reg_nodbg_iterator
         I = mri_->reg_nodbg_begin(DstInt->reg), E = mri_->reg_nodbg_end();
       I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    if (MO.isDef() || MO.isUndef())
      continue;
    MachineInstr *MI = MO.getParent();
    SlotIndex Idx = li_->getInstructionIndex(MI);
    if (DstInt->liveAt(Idx))
      continue;
    MO.setIsUndef(true);
    DEBUG(dbgs() << "\tnew undef: " << Idx << '\t' << *MI);
  }
  return true;
}

/// UpdateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
/// update the subregister number if it is not zero. If DstReg is a
/// physical register and the existing subregister number of the def / use
/// being updated is not zero, make sure to set it to the correct physical
/// subregister.
void
RegisterCoalescer::UpdateRegDefsUses(const CoalescerPair &CP) {
  bool DstIsPhys = CP.isPhys();
  unsigned SrcReg = CP.getSrcReg();
  unsigned DstReg = CP.getDstReg();
  unsigned SubIdx = CP.getSubIdx();

  // Update LiveDebugVariables.
  ldv_->renameRegister(SrcReg, DstReg, SubIdx);

  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(SrcReg);
       MachineInstr *UseMI = I.skipInstruction();) {
    // A PhysReg copy that won't be coalesced can perhaps be rematerialized
    // instead.
    if (DstIsPhys) {
      if (UseMI->isCopy() &&
          !UseMI->getOperand(1).getSubReg() &&
          !UseMI->getOperand(0).getSubReg() &&
          UseMI->getOperand(1).getReg() == SrcReg &&
          UseMI->getOperand(0).getReg() != SrcReg &&
          UseMI->getOperand(0).getReg() != DstReg &&
          !JoinedCopies.count(UseMI) &&
          ReMaterializeTrivialDef(li_->getInterval(SrcReg), false,
                                  UseMI->getOperand(0).getReg(), 0, UseMI))
        continue;
    }

    SmallVector<unsigned,8> Ops;
    bool Reads, Writes;
    tie(Reads, Writes) = UseMI->readsWritesVirtualRegister(SrcReg, &Ops);
    bool Kills = false, Deads = false;

    // Replace SrcReg with DstReg in all UseMI operands.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = UseMI->getOperand(Ops[i]);
      Kills |= MO.isKill();
      Deads |= MO.isDead();

      if (DstIsPhys)
        MO.substPhysReg(DstReg, *tri_);
      else
        MO.substVirtReg(DstReg, SubIdx, *tri_);
    }

    // This instruction is a copy that will be removed.
    if (JoinedCopies.count(UseMI))
      continue;

    if (SubIdx) {
      // If UseMI was a simple SrcReg def, make sure we didn't turn it into a
      // read-modify-write of DstReg.
      if (Deads)
        UseMI->addRegisterDead(DstReg, tri_);
      else if (!Reads && Writes)
        UseMI->addRegisterDefined(DstReg, tri_);

      // Kill flags apply to the whole physical register.
      if (DstIsPhys && Kills)
        UseMI->addRegisterKilled(DstReg, tri_);
    }

    DEBUG({
        dbgs() << "\t\tupdated: ";
        if (!UseMI->isDebugValue())
          dbgs() << li_->getInstructionIndex(UseMI) << "\t";
        dbgs() << *UseMI;
      });
  }
}

/// removeIntervalIfEmpty - Check if the live interval of a physical register
/// is empty, if so remove it and also remove the empty intervals of its
/// sub-registers. Return true if live interval is removed.
static bool removeIntervalIfEmpty(LiveInterval &li, LiveIntervals *li_,
                                  const TargetRegisterInfo *tri_) {
  if (li.empty()) {
    if (TargetRegisterInfo::isPhysicalRegister(li.reg))
      for (const unsigned* SR = tri_->getSubRegisters(li.reg); *SR; ++SR) {
        if (!li_->hasInterval(*SR))
          continue;
        LiveInterval &sli = li_->getInterval(*SR);
        if (sli.empty())
          li_->removeInterval(*SR);
      }
    li_->removeInterval(li.reg);
    return true;
  }
  return false;
}

/// RemoveDeadDef - If a def of a live interval is now determined dead, remove
/// the val# it defines. If the live interval becomes empty, remove it as well.
bool RegisterCoalescer::RemoveDeadDef(LiveInterval &li,
                                             MachineInstr *DefMI) {
  SlotIndex DefIdx = li_->getInstructionIndex(DefMI).getDefIndex();
  LiveInterval::iterator MLR = li.FindLiveRangeContaining(DefIdx);
  if (DefIdx != MLR->valno->def)
    return false;
  li.removeValNo(MLR->valno);
  return removeIntervalIfEmpty(li, li_, tri_);
}

void RegisterCoalescer::RemoveCopyFlag(unsigned DstReg,
                                              const MachineInstr *CopyMI) {
  SlotIndex DefIdx = li_->getInstructionIndex(CopyMI).getDefIndex();
  if (li_->hasInterval(DstReg)) {
    LiveInterval &LI = li_->getInterval(DstReg);
    if (const LiveRange *LR = LI.getLiveRangeContaining(DefIdx))
      if (LR->valno->def == DefIdx)
        LR->valno->setCopy(0);
  }
  if (!TargetRegisterInfo::isPhysicalRegister(DstReg))
    return;
  for (const unsigned* AS = tri_->getAliasSet(DstReg); *AS; ++AS) {
    if (!li_->hasInterval(*AS))
      continue;
    LiveInterval &LI = li_->getInterval(*AS);
    if (const LiveRange *LR = LI.getLiveRangeContaining(DefIdx))
      if (LR->valno->def == DefIdx)
        LR->valno->setCopy(0);
  }
}

/// shouldJoinPhys - Return true if a copy involving a physreg should be joined.
/// We need to be careful about coalescing a source physical register with a
/// virtual register. Once the coalescing is done, it cannot be broken and these
/// are not spillable! If the destination interval uses are far away, think
/// twice about coalescing them!
bool RegisterCoalescer::shouldJoinPhys(CoalescerPair &CP) {
  bool Allocatable = li_->isAllocatable(CP.getDstReg());
  LiveInterval &JoinVInt = li_->getInterval(CP.getSrcReg());

  /// Always join simple intervals that are defined by a single copy from a
  /// reserved register. This doesn't increase register pressure, so it is
  /// always beneficial.
  if (!Allocatable && CP.isFlipped() && JoinVInt.containsOneValue())
    return true;

  if (!EnablePhysicalJoin) {
    DEBUG(dbgs() << "\tPhysreg joins disabled.\n");
    return false;
  }

  // Only coalesce to allocatable physreg, we don't want to risk modifying
  // reserved registers.
  if (!Allocatable) {
    DEBUG(dbgs() << "\tRegister is an unallocatable physreg.\n");
    return false;  // Not coalescable.
  }

  // Don't join with physregs that have a ridiculous number of live
  // ranges. The data structure performance is really bad when that
  // happens.
  if (li_->hasInterval(CP.getDstReg()) &&
      li_->getInterval(CP.getDstReg()).ranges.size() > 1000) {
    ++numAborts;
    DEBUG(dbgs()
          << "\tPhysical register live interval too complicated, abort!\n");
    return false;
  }

  // FIXME: Why are we skipping this test for partial copies?
  //        CodeGen/X86/phys_subreg_coalesce-3.ll needs it.
  if (!CP.isPartial()) {
    const TargetRegisterClass *RC = mri_->getRegClass(CP.getSrcReg());
    unsigned Threshold = RegClassInfo.getNumAllocatableRegs(RC) * 2;
    unsigned Length = li_->getApproximateInstructionCount(JoinVInt);
    if (Length > Threshold) {
      ++numAborts;
      DEBUG(dbgs() << "\tMay tie down a physical register, abort!\n");
      return false;
    }
  }
  return true;
}

/// isWinToJoinCrossClass - Return true if it's profitable to coalesce
/// two virtual registers from different register classes.
bool
RegisterCoalescer::isWinToJoinCrossClass(unsigned SrcReg,
                                             unsigned DstReg,
                                             const TargetRegisterClass *SrcRC,
                                             const TargetRegisterClass *DstRC,
                                             const TargetRegisterClass *NewRC) {
  unsigned NewRCCount = RegClassInfo.getNumAllocatableRegs(NewRC);
  // This heuristics is good enough in practice, but it's obviously not *right*.
  // 4 is a magic number that works well enough for x86, ARM, etc. It filter
  // out all but the most restrictive register classes.
  if (NewRCCount > 4 ||
      // Early exit if the function is fairly small, coalesce aggressively if
      // that's the case. For really special register classes with 3 or
      // fewer registers, be a bit more careful.
      (li_->getFuncInstructionCount() / NewRCCount) < 8)
    return true;
  LiveInterval &SrcInt = li_->getInterval(SrcReg);
  LiveInterval &DstInt = li_->getInterval(DstReg);
  unsigned SrcSize = li_->getApproximateInstructionCount(SrcInt);
  unsigned DstSize = li_->getApproximateInstructionCount(DstInt);

  // Coalesce aggressively if the intervals are small compared to the number of
  // registers in the new class. The number 4 is fairly arbitrary, chosen to be
  // less aggressive than the 8 used for the whole function size.
  const unsigned ThresSize = 4 * NewRCCount;
  if (SrcSize <= ThresSize && DstSize <= ThresSize)
    return true;

  // Estimate *register use density*. If it doubles or more, abort.
  unsigned SrcUses = std::distance(mri_->use_nodbg_begin(SrcReg),
                                   mri_->use_nodbg_end());
  unsigned DstUses = std::distance(mri_->use_nodbg_begin(DstReg),
                                   mri_->use_nodbg_end());
  unsigned NewUses = SrcUses + DstUses;
  unsigned NewSize = SrcSize + DstSize;
  if (SrcRC != NewRC && SrcSize > ThresSize) {
    unsigned SrcRCCount = RegClassInfo.getNumAllocatableRegs(SrcRC);
    if (NewUses*SrcSize*SrcRCCount > 2*SrcUses*NewSize*NewRCCount)
      return false;
  }
  if (DstRC != NewRC && DstSize > ThresSize) {
    unsigned DstRCCount = RegClassInfo.getNumAllocatableRegs(DstRC);
    if (NewUses*DstSize*DstRCCount > 2*DstUses*NewSize*NewRCCount)
      return false;
  }
  return true;
}


/// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coalesced away. If it is not currently
/// possible to coalesce this interval, but it may be possible if other
/// things get coalesced, then it returns true by reference in 'Again'.
bool RegisterCoalescer::JoinCopy(MachineInstr *CopyMI, bool &Again) {

  Again = false;
  if (JoinedCopies.count(CopyMI) || ReMatCopies.count(CopyMI))
    return false; // Already done.

  DEBUG(dbgs() << li_->getInstructionIndex(CopyMI) << '\t' << *CopyMI);

  CoalescerPair CP(*tii_, *tri_);
  if (!CP.setRegisters(CopyMI)) {
    DEBUG(dbgs() << "\tNot coalescable.\n");
    return false;
  }

  // If they are already joined we continue.
  if (CP.getSrcReg() == CP.getDstReg()) {
    markAsJoined(CopyMI);
    DEBUG(dbgs() << "\tCopy already coalesced.\n");
    return false;  // Not coalescable.
  }

  // Eliminate undefs.
  if (!CP.isPhys() && eliminateUndefCopy(CopyMI, CP)) {
    markAsJoined(CopyMI);
    DEBUG(dbgs() << "\tEliminated copy of <undef> value.\n");
    return false;  // Not coalescable.
  }

  DEBUG(dbgs() << "\tConsidering merging " << PrintReg(CP.getSrcReg(), tri_)
               << " with " << PrintReg(CP.getDstReg(), tri_, CP.getSubIdx())
               << "\n");

  // Enforce policies.
  if (CP.isPhys()) {
    if (!shouldJoinPhys(CP)) {
      // Before giving up coalescing, if definition of source is defined by
      // trivial computation, try rematerializing it.
      if (!CP.isFlipped() &&
          ReMaterializeTrivialDef(li_->getInterval(CP.getSrcReg()), true,
                                  CP.getDstReg(), 0, CopyMI))
        return true;
      return false;
    }
  } else {
    // Avoid constraining virtual register regclass too much.
    if (CP.isCrossClass()) {
      DEBUG(dbgs() << "\tCross-class to " << CP.getNewRC()->getName() << ".\n");
      if (DisableCrossClassJoin) {
        DEBUG(dbgs() << "\tCross-class joins disabled.\n");
        return false;
      }
      if (!isWinToJoinCrossClass(CP.getSrcReg(), CP.getDstReg(),
                                 mri_->getRegClass(CP.getSrcReg()),
                                 mri_->getRegClass(CP.getDstReg()),
                                 CP.getNewRC())) {
        DEBUG(dbgs() << "\tAvoid coalescing to constrained register class.\n");
        Again = true;  // May be possible to coalesce later.
        return false;
      }
    }

    // When possible, let DstReg be the larger interval.
    if (!CP.getSubIdx() && li_->getInterval(CP.getSrcReg()).ranges.size() >
                           li_->getInterval(CP.getDstReg()).ranges.size())
      CP.flip();
  }

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  if (!JoinIntervals(CP)) {
    // Coalescing failed.

    // If definition of source is defined by trivial computation, try
    // rematerializing it.
    if (!CP.isFlipped() &&
        ReMaterializeTrivialDef(li_->getInterval(CP.getSrcReg()), true,
                                CP.getDstReg(), 0, CopyMI))
      return true;

    // If we can eliminate the copy without merging the live ranges, do so now.
    if (!CP.isPartial()) {
      if (AdjustCopiesBackFrom(CP, CopyMI) ||
          RemoveCopyByCommutingDef(CP, CopyMI)) {
        markAsJoined(CopyMI);
        DEBUG(dbgs() << "\tTrivial!\n");
        return true;
      }
    }

    // Otherwise, we are unable to join the intervals.
    DEBUG(dbgs() << "\tInterference!\n");
    Again = true;  // May be possible to coalesce later.
    return false;
  }

  // Coalescing to a virtual register that is of a sub-register class of the
  // other. Make sure the resulting register is set to the right register class.
  if (CP.isCrossClass()) {
    ++numCrossRCs;
    mri_->setRegClass(CP.getDstReg(), CP.getNewRC());
  }

  // Remember to delete the copy instruction.
  markAsJoined(CopyMI);

  UpdateRegDefsUses(CP);

  // If we have extended the live range of a physical register, make sure we
  // update live-in lists as well.
  if (CP.isPhys()) {
    SmallVector<MachineBasicBlock*, 16> BlockSeq;
    // JoinIntervals invalidates the VNInfos in SrcInt, but we only need the
    // ranges for this, and they are preserved.
    LiveInterval &SrcInt = li_->getInterval(CP.getSrcReg());
    for (LiveInterval::const_iterator I = SrcInt.begin(), E = SrcInt.end();
         I != E; ++I ) {
      li_->findLiveInMBBs(I->start, I->end, BlockSeq);
      for (unsigned idx = 0, size = BlockSeq.size(); idx != size; ++idx) {
        MachineBasicBlock &block = *BlockSeq[idx];
        if (!block.isLiveIn(CP.getDstReg()))
          block.addLiveIn(CP.getDstReg());
      }
      BlockSeq.clear();
    }
  }

  // SrcReg is guarateed to be the register whose live interval that is
  // being merged.
  li_->removeInterval(CP.getSrcReg());

  // Update regalloc hint.
  tri_->UpdateRegAllocHint(CP.getSrcReg(), CP.getDstReg(), *mf_);

  DEBUG({
    LiveInterval &DstInt = li_->getInterval(CP.getDstReg());
    dbgs() << "\tJoined. Result = ";
    DstInt.print(dbgs(), tri_);
    dbgs() << "\n";
  });

  ++numJoins;
  return true;
}

/// ComputeUltimateVN - Assuming we are going to join two live intervals,
/// compute what the resultant value numbers for each value in the input two
/// ranges will be.  This is complicated by copies between the two which can
/// and will commonly cause multiple value numbers to be merged into one.
///
/// VN is the value number that we're trying to resolve.  InstDefiningValue
/// keeps track of the new InstDefiningValue assignment for the result
/// LiveInterval.  ThisFromOther/OtherFromThis are sets that keep track of
/// whether a value in this or other is a copy from the opposite set.
/// ThisValNoAssignments/OtherValNoAssignments keep track of value #'s that have
/// already been assigned.
///
/// ThisFromOther[x] - If x is defined as a copy from the other interval, this
/// contains the value number the copy is from.
///
static unsigned ComputeUltimateVN(VNInfo *VNI,
                                  SmallVector<VNInfo*, 16> &NewVNInfo,
                                  DenseMap<VNInfo*, VNInfo*> &ThisFromOther,
                                  DenseMap<VNInfo*, VNInfo*> &OtherFromThis,
                                  SmallVector<int, 16> &ThisValNoAssignments,
                                  SmallVector<int, 16> &OtherValNoAssignments) {
  unsigned VN = VNI->id;

  // If the VN has already been computed, just return it.
  if (ThisValNoAssignments[VN] >= 0)
    return ThisValNoAssignments[VN];
  assert(ThisValNoAssignments[VN] != -2 && "Cyclic value numbers");

  // If this val is not a copy from the other val, then it must be a new value
  // number in the destination.
  DenseMap<VNInfo*, VNInfo*>::iterator I = ThisFromOther.find(VNI);
  if (I == ThisFromOther.end()) {
    NewVNInfo.push_back(VNI);
    return ThisValNoAssignments[VN] = NewVNInfo.size()-1;
  }
  VNInfo *OtherValNo = I->second;

  // Otherwise, this *is* a copy from the RHS.  If the other side has already
  // been computed, return it.
  if (OtherValNoAssignments[OtherValNo->id] >= 0)
    return ThisValNoAssignments[VN] = OtherValNoAssignments[OtherValNo->id];

  // Mark this value number as currently being computed, then ask what the
  // ultimate value # of the other value is.
  ThisValNoAssignments[VN] = -2;
  unsigned UltimateVN =
    ComputeUltimateVN(OtherValNo, NewVNInfo, OtherFromThis, ThisFromOther,
                      OtherValNoAssignments, ThisValNoAssignments);
  return ThisValNoAssignments[VN] = UltimateVN;
}


// Find out if we have something like
// A = X
// B = X
// if so, we can pretend this is actually
// A = X
// B = A
// which allows us to coalesce A and B.
// VNI is the definition of B. LR is the life range of A that includes
// the slot just before B. If we return true, we add "B = X" to DupCopies.
static bool RegistersDefinedFromSameValue(LiveIntervals &li,
                                          const TargetRegisterInfo &tri,
                                          CoalescerPair &CP,
                                          VNInfo *VNI,
                                          LiveRange *LR,
                                     SmallVector<MachineInstr*, 8> &DupCopies) {
  // FIXME: This is very conservative. For example, we don't handle
  // physical registers.

  MachineInstr *MI = VNI->getCopy();

  if (!MI->isFullCopy() || CP.isPartial() || CP.isPhys())
    return false;

  unsigned Dst = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();

  if (!TargetRegisterInfo::isVirtualRegister(Src) ||
      !TargetRegisterInfo::isVirtualRegister(Dst))
    return false;

  unsigned A = CP.getDstReg();
  unsigned B = CP.getSrcReg();

  if (B == Dst)
    std::swap(A, B);
  assert(Dst == A);

  VNInfo *Other = LR->valno;
  if (!Other->isDefByCopy())
    return false;
  const MachineInstr *OtherMI = Other->getCopy();

  if (!OtherMI->isFullCopy())
    return false;

  unsigned OtherDst = OtherMI->getOperand(0).getReg();
  unsigned OtherSrc = OtherMI->getOperand(1).getReg();

  if (!TargetRegisterInfo::isVirtualRegister(OtherSrc) ||
      !TargetRegisterInfo::isVirtualRegister(OtherDst))
    return false;

  assert(OtherDst == B);

  if (Src != OtherSrc)
    return false;

  // If the copies use two different value numbers of X, we cannot merge
  // A and B.
  LiveInterval &SrcInt = li.getInterval(Src);
  if (SrcInt.getVNInfoAt(Other->def) != SrcInt.getVNInfoAt(VNI->def))
    return false;

  DupCopies.push_back(MI);

  return true;
}

/// JoinIntervals - Attempt to join these two intervals.  On failure, this
/// returns false.
bool RegisterCoalescer::JoinIntervals(CoalescerPair &CP) {
  LiveInterval &RHS = li_->getInterval(CP.getSrcReg());
  DEBUG({ dbgs() << "\t\tRHS = "; RHS.print(dbgs(), tri_); dbgs() << "\n"; });

  // If a live interval is a physical register, check for interference with any
  // aliases. The interference check implemented here is a bit more conservative
  // than the full interfeence check below. We allow overlapping live ranges
  // only when one is a copy of the other.
  if (CP.isPhys()) {
    for (const unsigned *AS = tri_->getAliasSet(CP.getDstReg()); *AS; ++AS){
      if (!li_->hasInterval(*AS))
        continue;
      const LiveInterval &LHS = li_->getInterval(*AS);
      LiveInterval::const_iterator LI = LHS.begin();
      for (LiveInterval::const_iterator RI = RHS.begin(), RE = RHS.end();
           RI != RE; ++RI) {
        LI = std::lower_bound(LI, LHS.end(), RI->start);
        // Does LHS have an overlapping live range starting before RI?
        if ((LI != LHS.begin() && LI[-1].end > RI->start) &&
            (RI->start != RI->valno->def ||
             !CP.isCoalescable(li_->getInstructionFromIndex(RI->start)))) {
          DEBUG({
            dbgs() << "\t\tInterference from alias: ";
            LHS.print(dbgs(), tri_);
            dbgs() << "\n\t\tOverlap at " << RI->start << " and no copy.\n";
          });
          return false;
        }

        // Check that LHS ranges beginning in this range are copies.
        for (; LI != LHS.end() && LI->start < RI->end; ++LI) {
          if (LI->start != LI->valno->def ||
              !CP.isCoalescable(li_->getInstructionFromIndex(LI->start))) {
            DEBUG({
              dbgs() << "\t\tInterference from alias: ";
              LHS.print(dbgs(), tri_);
              dbgs() << "\n\t\tDef at " << LI->start << " is not a copy.\n";
            });
            return false;
          }
        }
      }
    }
  }

  // Compute the final value assignment, assuming that the live ranges can be
  // coalesced.
  SmallVector<int, 16> LHSValNoAssignments;
  SmallVector<int, 16> RHSValNoAssignments;
  DenseMap<VNInfo*, VNInfo*> LHSValsDefinedFromRHS;
  DenseMap<VNInfo*, VNInfo*> RHSValsDefinedFromLHS;
  SmallVector<VNInfo*, 16> NewVNInfo;

  SmallVector<MachineInstr*, 8> DupCopies;

  LiveInterval &LHS = li_->getOrCreateInterval(CP.getDstReg());
  DEBUG({ dbgs() << "\t\tLHS = "; LHS.print(dbgs(), tri_); dbgs() << "\n"; });

  // Loop over the value numbers of the LHS, seeing if any are defined from
  // the RHS.
  for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
       i != e; ++i) {
    VNInfo *VNI = *i;
    if (VNI->isUnused() || !VNI->isDefByCopy())  // Src not defined by a copy?
      continue;

    // Never join with a register that has EarlyClobber redefs.
    if (VNI->hasRedefByEC())
      return false;

    // Figure out the value # from the RHS.
    LiveRange *lr = RHS.getLiveRangeContaining(VNI->def.getPrevSlot());
    // The copy could be to an aliased physreg.
    if (!lr) continue;

    // DstReg is known to be a register in the LHS interval.  If the src is
    // from the RHS interval, we can use its value #.
    MachineInstr *MI = VNI->getCopy();
    if (!CP.isCoalescable(MI) &&
        !RegistersDefinedFromSameValue(*li_, *tri_, CP, VNI, lr, DupCopies))
      continue;

    LHSValsDefinedFromRHS[VNI] = lr->valno;
  }

  // Loop over the value numbers of the RHS, seeing if any are defined from
  // the LHS.
  for (LiveInterval::vni_iterator i = RHS.vni_begin(), e = RHS.vni_end();
       i != e; ++i) {
    VNInfo *VNI = *i;
    if (VNI->isUnused() || !VNI->isDefByCopy())  // Src not defined by a copy?
      continue;

    // Never join with a register that has EarlyClobber redefs.
    if (VNI->hasRedefByEC())
      return false;

    // Figure out the value # from the LHS.
    LiveRange *lr = LHS.getLiveRangeContaining(VNI->def.getPrevSlot());
    // The copy could be to an aliased physreg.
    if (!lr) continue;

    // DstReg is known to be a register in the RHS interval.  If the src is
    // from the LHS interval, we can use its value #.
    MachineInstr *MI = VNI->getCopy();
    if (!CP.isCoalescable(MI) &&
        !RegistersDefinedFromSameValue(*li_, *tri_, CP, VNI, lr, DupCopies))
        continue;

    RHSValsDefinedFromLHS[VNI] = lr->valno;
  }

  LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
  RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
  NewVNInfo.reserve(LHS.getNumValNums() + RHS.getNumValNums());

  for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
       i != e; ++i) {
    VNInfo *VNI = *i;
    unsigned VN = VNI->id;
    if (LHSValNoAssignments[VN] >= 0 || VNI->isUnused())
      continue;
    ComputeUltimateVN(VNI, NewVNInfo,
                      LHSValsDefinedFromRHS, RHSValsDefinedFromLHS,
                      LHSValNoAssignments, RHSValNoAssignments);
  }
  for (LiveInterval::vni_iterator i = RHS.vni_begin(), e = RHS.vni_end();
       i != e; ++i) {
    VNInfo *VNI = *i;
    unsigned VN = VNI->id;
    if (RHSValNoAssignments[VN] >= 0 || VNI->isUnused())
      continue;
    // If this value number isn't a copy from the LHS, it's a new number.
    if (RHSValsDefinedFromLHS.find(VNI) == RHSValsDefinedFromLHS.end()) {
      NewVNInfo.push_back(VNI);
      RHSValNoAssignments[VN] = NewVNInfo.size()-1;
      continue;
    }

    ComputeUltimateVN(VNI, NewVNInfo,
                      RHSValsDefinedFromLHS, LHSValsDefinedFromRHS,
                      RHSValNoAssignments, LHSValNoAssignments);
  }

  // Armed with the mappings of LHS/RHS values to ultimate values, walk the
  // interval lists to see if these intervals are coalescable.
  LiveInterval::const_iterator I = LHS.begin();
  LiveInterval::const_iterator IE = LHS.end();
  LiveInterval::const_iterator J = RHS.begin();
  LiveInterval::const_iterator JE = RHS.end();

  // Skip ahead until the first place of potential sharing.
  if (I != IE && J != JE) {
    if (I->start < J->start) {
      I = std::upper_bound(I, IE, J->start);
      if (I != LHS.begin()) --I;
    } else if (J->start < I->start) {
      J = std::upper_bound(J, JE, I->start);
      if (J != RHS.begin()) --J;
    }
  }

  while (I != IE && J != JE) {
    // Determine if these two live ranges overlap.
    bool Overlaps;
    if (I->start < J->start) {
      Overlaps = I->end > J->start;
    } else {
      Overlaps = J->end > I->start;
    }

    // If so, check value # info to determine if they are really different.
    if (Overlaps) {
      // If the live range overlap will map to the same value number in the
      // result liverange, we can still coalesce them.  If not, we can't.
      if (LHSValNoAssignments[I->valno->id] !=
          RHSValNoAssignments[J->valno->id])
        return false;
      // If it's re-defined by an early clobber somewhere in the live range,
      // then conservatively abort coalescing.
      if (NewVNInfo[LHSValNoAssignments[I->valno->id]]->hasRedefByEC())
        return false;
    }

    if (I->end < J->end)
      ++I;
    else
      ++J;
  }

  // Update kill info. Some live ranges are extended due to copy coalescing.
  for (DenseMap<VNInfo*, VNInfo*>::iterator I = LHSValsDefinedFromRHS.begin(),
         E = LHSValsDefinedFromRHS.end(); I != E; ++I) {
    VNInfo *VNI = I->first;
    unsigned LHSValID = LHSValNoAssignments[VNI->id];
    if (VNI->hasPHIKill())
      NewVNInfo[LHSValID]->setHasPHIKill(true);
  }

  // Update kill info. Some live ranges are extended due to copy coalescing.
  for (DenseMap<VNInfo*, VNInfo*>::iterator I = RHSValsDefinedFromLHS.begin(),
         E = RHSValsDefinedFromLHS.end(); I != E; ++I) {
    VNInfo *VNI = I->first;
    unsigned RHSValID = RHSValNoAssignments[VNI->id];
    if (VNI->hasPHIKill())
      NewVNInfo[RHSValID]->setHasPHIKill(true);
  }

  if (LHSValNoAssignments.empty())
    LHSValNoAssignments.push_back(-1);
  if (RHSValNoAssignments.empty())
    RHSValNoAssignments.push_back(-1);

  SmallVector<unsigned, 8> SourceRegisters;
  for (SmallVector<MachineInstr*, 8>::iterator I = DupCopies.begin(),
         E = DupCopies.end(); I != E; ++I) {
    MachineInstr *MI = *I;

    // We have pretended that the assignment to B in
    // A = X
    // B = X
    // was actually a copy from A. Now that we decided to coalesce A and B,
    // transform the code into
    // A = X
    // X = X
    // and mark the X as coalesced to keep the illusion.
    unsigned Src = MI->getOperand(1).getReg();
    SourceRegisters.push_back(Src);
    MI->getOperand(0).substVirtReg(Src, 0, *tri_);

    markAsJoined(MI);
  }

  // If B = X was the last use of X in a liverange, we have to shrink it now
  // that B = X is gone.
  for (SmallVector<unsigned, 8>::iterator I = SourceRegisters.begin(),
         E = SourceRegisters.end(); I != E; ++I) {
    li_->shrinkToUses(&li_->getInterval(*I));
  }

  // If we get here, we know that we can coalesce the live ranges.  Ask the
  // intervals to coalesce themselves now.
  LHS.join(RHS, &LHSValNoAssignments[0], &RHSValNoAssignments[0], NewVNInfo,
           mri_);
  return true;
}

namespace {
  // DepthMBBCompare - Comparison predicate that sort first based on the loop
  // depth of the basic block (the unsigned), and then on the MBB number.
  struct DepthMBBCompare {
    typedef std::pair<unsigned, MachineBasicBlock*> DepthMBBPair;
    bool operator()(const DepthMBBPair &LHS, const DepthMBBPair &RHS) const {
      // Deeper loops first
      if (LHS.first != RHS.first)
        return LHS.first > RHS.first;

      // Prefer blocks that are more connected in the CFG. This takes care of
      // the most difficult copies first while intervals are short.
      unsigned cl = LHS.second->pred_size() + LHS.second->succ_size();
      unsigned cr = RHS.second->pred_size() + RHS.second->succ_size();
      if (cl != cr)
        return cl > cr;

      // As a last resort, sort by block number.
      return LHS.second->getNumber() < RHS.second->getNumber();
    }
  };
}

void RegisterCoalescer::CopyCoalesceInMBB(MachineBasicBlock *MBB,
                                            std::vector<MachineInstr*> &TryAgain) {
  DEBUG(dbgs() << MBB->getName() << ":\n");

  SmallVector<MachineInstr*, 8> VirtCopies;
  SmallVector<MachineInstr*, 8> PhysCopies;
  SmallVector<MachineInstr*, 8> ImpDefCopies;
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E;) {
    MachineInstr *Inst = MII++;

    // If this isn't a copy nor a extract_subreg, we can't join intervals.
    unsigned SrcReg, DstReg;
    if (Inst->isCopy()) {
      DstReg = Inst->getOperand(0).getReg();
      SrcReg = Inst->getOperand(1).getReg();
    } else if (Inst->isSubregToReg()) {
      DstReg = Inst->getOperand(0).getReg();
      SrcReg = Inst->getOperand(2).getReg();
    } else
      continue;

    bool SrcIsPhys = TargetRegisterInfo::isPhysicalRegister(SrcReg);
    bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
    if (li_->hasInterval(SrcReg) && li_->getInterval(SrcReg).empty())
      ImpDefCopies.push_back(Inst);
    else if (SrcIsPhys || DstIsPhys)
      PhysCopies.push_back(Inst);
    else
      VirtCopies.push_back(Inst);
  }

  // Try coalescing implicit copies and insert_subreg <undef> first,
  // followed by copies to / from physical registers, then finally copies
  // from virtual registers to virtual registers.
  for (unsigned i = 0, e = ImpDefCopies.size(); i != e; ++i) {
    MachineInstr *TheCopy = ImpDefCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = PhysCopies.size(); i != e; ++i) {
    MachineInstr *TheCopy = PhysCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = VirtCopies.size(); i != e; ++i) {
    MachineInstr *TheCopy = VirtCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
}

void RegisterCoalescer::joinIntervals() {
  DEBUG(dbgs() << "********** JOINING INTERVALS ***********\n");

  std::vector<MachineInstr*> TryAgainList;
  if (loopInfo->empty()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      CopyCoalesceInMBB(I, TryAgainList);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.

    // Join intervals in the function prolog first. We want to join physical
    // registers with virtual registers before the intervals got too long.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();I != E;++I){
      MachineBasicBlock *MBB = I;
      MBBs.push_back(std::make_pair(loopInfo->getLoopDepth(MBB), I));
    }

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order.
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      CopyCoalesceInMBB(MBBs[i].second, TryAgainList);
  }

  // Joining intervals can allow other intervals to be joined.  Iteratively join
  // until we make no progress.
  bool ProgressMade = true;
  while (ProgressMade) {
    ProgressMade = false;

    for (unsigned i = 0, e = TryAgainList.size(); i != e; ++i) {
      MachineInstr *&TheCopy = TryAgainList[i];
      if (!TheCopy)
        continue;

      bool Again = false;
      bool Success = JoinCopy(TheCopy, Again);
      if (Success || !Again) {
        TheCopy= 0;   // Mark this one as done.
        ProgressMade = true;
      }
    }
  }
}

void RegisterCoalescer::releaseMemory() {
  JoinedCopies.clear();
  ReMatCopies.clear();
  ReMatDefs.clear();
}

bool RegisterCoalescer::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &fn.getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  li_ = &getAnalysis<LiveIntervals>();
  ldv_ = &getAnalysis<LiveDebugVariables>();
  AA = &getAnalysis<AliasAnalysis>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  DEBUG(dbgs() << "********** SIMPLE REGISTER COALESCING **********\n"
               << "********** Function: "
               << ((Value*)mf_->getFunction())->getName() << '\n');

  if (VerifyCoalescing)
    mf_->verify(this, "Before register coalescing");

  RegClassInfo.runOnMachineFunction(fn);

  // Join (coalesce) intervals if requested.
  if (EnableJoining) {
    joinIntervals();
    DEBUG({
        dbgs() << "********** INTERVALS POST JOINING **********\n";
        for (LiveIntervals::iterator I = li_->begin(), E = li_->end();
             I != E; ++I){
          I->second->print(dbgs(), tri_);
          dbgs() << "\n";
        }
      });
  }

  // Perform a final pass over the instructions and compute spill weights
  // and remove identity moves.
  SmallVector<unsigned, 4> DeadDefs;
  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    MachineBasicBlock* mbb = mbbi;
    for (MachineBasicBlock::iterator mii = mbb->begin(), mie = mbb->end();
         mii != mie; ) {
      MachineInstr *MI = mii;
      if (JoinedCopies.count(MI)) {
        // Delete all coalesced copies.
        bool DoDelete = true;
        assert(MI->isCopyLike() && "Unrecognized copy instruction");
        unsigned SrcReg = MI->getOperand(MI->isSubregToReg() ? 2 : 1).getReg();
        if (TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
            MI->getNumOperands() > 2)
          // Do not delete extract_subreg, insert_subreg of physical
          // registers unless the definition is dead. e.g.
          // %DO<def> = INSERT_SUBREG %D0<undef>, %S0<kill>, 1
          // or else the scavenger may complain. LowerSubregs will
          // delete them later.
          DoDelete = false;

        if (MI->allDefsAreDead()) {
          if (TargetRegisterInfo::isVirtualRegister(SrcReg) &&
              li_->hasInterval(SrcReg))
            li_->shrinkToUses(&li_->getInterval(SrcReg));
          DoDelete = true;
        }
        if (!DoDelete) {
          // We need the instruction to adjust liveness, so make it a KILL.
          if (MI->isSubregToReg()) {
            MI->RemoveOperand(3);
            MI->RemoveOperand(1);
          }
          MI->setDesc(tii_->get(TargetOpcode::KILL));
          mii = llvm::next(mii);
        } else {
          li_->RemoveMachineInstrFromMaps(MI);
          mii = mbbi->erase(mii);
          ++numPeep;
        }
        continue;
      }

      // Now check if this is a remat'ed def instruction which is now dead.
      if (ReMatDefs.count(MI)) {
        bool isDead = true;
        for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
          const MachineOperand &MO = MI->getOperand(i);
          if (!MO.isReg())
            continue;
          unsigned Reg = MO.getReg();
          if (!Reg)
            continue;
          if (TargetRegisterInfo::isVirtualRegister(Reg))
            DeadDefs.push_back(Reg);
          if (MO.isDead())
            continue;
          if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
              !mri_->use_nodbg_empty(Reg)) {
            isDead = false;
            break;
          }
        }
        if (isDead) {
          while (!DeadDefs.empty()) {
            unsigned DeadDef = DeadDefs.back();
            DeadDefs.pop_back();
            RemoveDeadDef(li_->getInterval(DeadDef), MI);
          }
          li_->RemoveMachineInstrFromMaps(mii);
          mii = mbbi->erase(mii);
          continue;
        } else
          DeadDefs.clear();
      }

      ++mii;

      // Check for now unnecessary kill flags.
      if (li_->isNotInMIMap(MI)) continue;
      SlotIndex DefIdx = li_->getInstructionIndex(MI).getDefIndex();
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.isKill()) continue;
        unsigned reg = MO.getReg();
        if (!reg || !li_->hasInterval(reg)) continue;
        if (!li_->getInterval(reg).killedAt(DefIdx)) {
          MO.setIsKill(false);
          continue;
        }
        // When leaving a kill flag on a physreg, check if any subregs should
        // remain alive.
        if (!TargetRegisterInfo::isPhysicalRegister(reg))
          continue;
        for (const unsigned *SR = tri_->getSubRegisters(reg);
             unsigned S = *SR; ++SR)
          if (li_->hasInterval(S) && li_->getInterval(S).liveAt(DefIdx))
            MI->addRegisterDefined(S, tri_);
      }
    }
  }

  DEBUG(dump());
  DEBUG(ldv_->dump());
  if (VerifyCoalescing)
    mf_->verify(this, "After register coalescing");
  return true;
}

/// print - Implement the dump method.
void RegisterCoalescer::print(raw_ostream &O, const Module* m) const {
   li_->print(O, m);
}
