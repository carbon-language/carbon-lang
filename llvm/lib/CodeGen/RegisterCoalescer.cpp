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

#define DEBUG_TYPE "regalloc"
#include "RegisterCoalescer.h"
#include "LiveDebugVariables.h"
#include "RegisterClassInfo.h"
#include "VirtRegMap.h"

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
STATISTIC(NumInflated , "Number of register classes inflated");

static cl::opt<bool>
EnableJoining("join-liveintervals",
              cl::desc("Coalesce copies (default=true)"),
              cl::init(true));

static cl::opt<bool>
EnablePhysicalJoin("join-physregs",
                   cl::desc("Join physical register copies"),
                   cl::init(false), cl::Hidden);

static cl::opt<bool>
VerifyCoalescing("verify-coalescing",
         cl::desc("Verify machine instrs before and after register coalescing"),
         cl::Hidden);

namespace {
  class RegisterCoalescer : public MachineFunctionPass {
    MachineFunction* MF;
    MachineRegisterInfo* MRI;
    const TargetMachine* TM;
    const TargetRegisterInfo* TRI;
    const TargetInstrInfo* TII;
    LiveIntervals *LIS;
    LiveDebugVariables *LDV;
    const MachineLoopInfo* Loops;
    AliasAnalysis *AA;
    RegisterClassInfo RegClassInfo;

    /// JoinedCopies - Keep track of copies eliminated due to coalescing.
    ///
    SmallPtrSet<MachineInstr*, 32> JoinedCopies;

    /// ReMatCopies - Keep track of copies eliminated due to remat.
    ///
    SmallPtrSet<MachineInstr*, 32> ReMatCopies;

    /// ReMatDefs - Keep track of definition instructions which have
    /// been remat'ed.
    SmallPtrSet<MachineInstr*, 8> ReMatDefs;

    /// joinAllIntervals - join compatible live intervals
    void joinAllIntervals();

    /// copyCoalesceInMBB - Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into the "TryAgain" list.
    void copyCoalesceInMBB(MachineBasicBlock *MBB,
                           std::vector<MachineInstr*> &TryAgain);

    /// joinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns
    /// true if the copy was successfully coalesced away. If it is not
    /// currently possible to coalesce this interval, but it may be possible if
    /// other things get coalesced, then it returns true by reference in
    /// 'Again'.
    bool joinCopy(MachineInstr *TheCopy, bool &Again);

    /// joinIntervals - Attempt to join these two intervals.  On failure, this
    /// returns false.  The output "SrcInt" will not have been modified, so we
    /// can use this information below to update aliases.
    bool joinIntervals(CoalescerPair &CP);

    /// adjustCopiesBackFrom - We found a non-trivially-coalescable copy. If
    /// the source value number is defined by a copy from the destination reg
    /// see if we can merge these two destination reg valno# into a single
    /// value number, eliminating a copy.
    bool adjustCopiesBackFrom(const CoalescerPair &CP, MachineInstr *CopyMI);

    /// hasOtherReachingDefs - Return true if there are definitions of IntB
    /// other than BValNo val# that can reach uses of AValno val# of IntA.
    bool hasOtherReachingDefs(LiveInterval &IntA, LiveInterval &IntB,
                              VNInfo *AValNo, VNInfo *BValNo);

    /// removeCopyByCommutingDef - We found a non-trivially-coalescable copy.
    /// If the source value number is defined by a commutable instruction and
    /// its other operand is coalesced to the copy dest register, see if we
    /// can transform the copy into a noop by commuting the definition.
    bool removeCopyByCommutingDef(const CoalescerPair &CP,MachineInstr *CopyMI);

    /// reMaterializeTrivialDef - If the source of a copy is defined by a
    /// trivial computation, replace the copy by rematerialize the definition.
    /// If PreserveSrcInt is true, make sure SrcInt is valid after the call.
    bool reMaterializeTrivialDef(LiveInterval &SrcInt, bool PreserveSrcInt,
                                 unsigned DstReg, MachineInstr *CopyMI);

    /// shouldJoinPhys - Return true if a physreg copy should be joined.
    bool shouldJoinPhys(CoalescerPair &CP);

    /// updateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
    /// update the subregister number if it is not zero. If DstReg is a
    /// physical register and the existing subregister number of the def / use
    /// being updated is not zero, make sure to set it to the correct physical
    /// subregister.
    void updateRegDefsUses(const CoalescerPair &CP);

    /// removeDeadDef - If a def of a live interval is now determined dead,
    /// remove the val# it defines. If the live interval becomes empty, remove
    /// it as well.
    bool removeDeadDef(LiveInterval &li, MachineInstr *DefMI);

    /// markAsJoined - Remember that CopyMI has already been joined.
    void markAsJoined(MachineInstr *CopyMI);

    /// eliminateUndefCopy - Handle copies of undef values.
    bool eliminateUndefCopy(MachineInstr *CopyMI, const CoalescerPair &CP);

  public:
    static char ID; // Class identification, replacement for typeinfo
    RegisterCoalescer() : MachineFunctionPass(ID) {
      initializeRegisterCoalescerPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    /// print - Implement the dump method.
    virtual void print(raw_ostream &O, const Module* = 0) const;
  };
} /// end anonymous namespace

char &llvm::RegisterCoalescerID = RegisterCoalescer::ID;

INITIALIZE_PASS_BEGIN(RegisterCoalescer, "simple-register-coalescing",
                      "Simple Register Coalescing", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
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
  SrcReg = DstReg = SubIdx = 0;
  NewRC = 0;
  Flipped = CrossClass = false;

  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(TRI, MI, Src, Dst, SrcSub, DstSub))
    return false;
  Partial = SrcSub || DstSub;

  // If one register is a physreg, it must be Dst.
  if (TargetRegisterInfo::isPhysicalRegister(Src)) {
    if (TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
    Flipped = true;
  }

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();

  if (TargetRegisterInfo::isPhysicalRegister(Dst)) {
    // Eliminate DstSub on a physreg.
    if (DstSub) {
      Dst = TRI.getSubReg(Dst, DstSub);
      if (!Dst) return false;
      DstSub = 0;
    }

    // Eliminate SrcSub by picking a corresponding Dst superregister.
    if (SrcSub) {
      Dst = TRI.getMatchingSuperReg(Dst, SrcSub, MRI.getRegClass(Src));
      if (!Dst) return false;
      SrcSub = 0;
    } else if (!MRI.getRegClass(Src)->contains(Dst)) {
      return false;
    }
  } else {
    // Both registers are virtual.
    const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);

    // Both registers have subreg indices.
    if (SrcSub && DstSub) {
      unsigned SrcPre, DstPre;
      NewRC = TRI.getCommonSuperRegClass(SrcRC, SrcSub, DstRC, DstSub,
                                         SrcPre, DstPre);
      if (!NewRC)
        return false;

      // We cannot handle the case where both Src and Dst would be a
      // sub-register. Yet.
      if (SrcPre && DstPre) {
        DEBUG(dbgs() << "\tCannot handle " << NewRC->getName()
                     << " with subregs " << TRI.getSubRegIndexName(SrcPre)
                     << " and " << TRI.getSubRegIndexName(DstPre) << '\n');
        return false;
      }

      // One of these will be 0, so one register is a sub-register of the other.
      SrcSub = DstPre;
      DstSub = SrcPre;
    }

    // There can be no SrcSub.
    if (SrcSub) {
      std::swap(Src, Dst);
      std::swap(SrcRC, DstRC);
      DstSub = SrcSub;
      SrcSub = 0;
      assert(!Flipped && "Unexpected flip");
      Flipped = true;
    }

    // Find the new register class.
    if (!NewRC) {
      if (DstSub)
        NewRC = TRI.getMatchingSuperRegClass(DstRC, SrcRC, DstSub);
      else
        NewRC = TRI.getCommonSubClass(DstRC, SrcRC);
    }
    if (!NewRC)
      return false;
    CrossClass = NewRC != DstRC || NewRC != SrcRC;
  }
  // Check our invariants
  assert(TargetRegisterInfo::isVirtualRegister(Src) && "Src must be virtual");
  assert(!(TargetRegisterInfo::isPhysicalRegister(Dst) && DstSub) &&
         "Cannot have a physical SubIdx");
  SrcReg = Src;
  DstReg = Dst;
  SubIdx = DstSub;
  return true;
}

bool CoalescerPair::flip() {
  if (SubIdx || TargetRegisterInfo::isPhysicalRegister(DstReg))
    return false;
  std::swap(SrcReg, DstReg);
  Flipped = !Flipped;
  return true;
}

bool CoalescerPair::isCoalescable(const MachineInstr *MI) const {
  if (!MI)
    return false;
  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(TRI, MI, Src, Dst, SrcSub, DstSub))
    return false;

  // Find the virtual register that is SrcReg.
  if (Dst == SrcReg) {
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
  } else if (Src != SrcReg) {
    return false;
  }

  // Now check that Dst matches DstReg.
  if (TargetRegisterInfo::isPhysicalRegister(DstReg)) {
    if (!TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    assert(!SubIdx && "Inconsistent CoalescerPair state.");
    // DstSub could be set for a physreg from INSERT_SUBREG.
    if (DstSub)
      Dst = TRI.getSubReg(Dst, DstSub);
    // Full copy of Src.
    if (!SrcSub)
      return DstReg == Dst;
    // This is a partial register copy. Check that the parts match.
    return TRI.getSubReg(DstReg, SrcSub) == Dst;
  } else {
    // DstReg is virtual.
    if (DstReg != Dst)
      return false;
    // Registers match, do the subregisters line up?
    return compose(TRI, SubIdx, SrcSub) == DstSub;
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

/// adjustCopiesBackFrom - We found a non-trivially-coalescable copy with IntA
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
bool RegisterCoalescer::adjustCopiesBackFrom(const CoalescerPair &CP,
                                             MachineInstr *CopyMI) {
  // Bail if there is no dst interval - can happen when merging physical subreg
  // operations.
  if (!LIS->hasInterval(CP.getDstReg()))
    return false;

  LiveInterval &IntA =
    LIS->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
    LIS->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());
  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot();

  // BValNo is a value number in B that is defined by a copy from A.  'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  if (BLR == IntB.end()) return false;
  VNInfo *BValNo = BLR->valno;

  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we
  // can't process it.
  if (BValNo->def != CopyIdx) return false;

  // AValNo is the value number in A that defines the copy, A3 in the example.
  SlotIndex CopyUseIdx = CopyIdx.getRegSlot(true);
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyUseIdx);
  // The live range might not exist after fun with physreg coalescing.
  if (ALR == IntA.end()) return false;
  VNInfo *AValNo = ALR->valno;

  // If AValNo is defined as a copy from IntB, we can potentially process this.
  // Get the instruction that defines this value number.
  MachineInstr *ACopyMI = LIS->getInstructionFromIndex(AValNo->def);
  if (!CP.isCoalescable(ACopyMI))
    return false;

  // Get the LiveRange in IntB that this value number starts with.
  LiveInterval::iterator ValLR =
    IntB.FindLiveRangeContaining(AValNo->def.getPrevSlot());
  if (ValLR == IntB.end())
    return false;

  // Make sure that the end of the live range is inside the same block as
  // CopyMI.
  MachineInstr *ValLREndInst =
    LIS->getInstructionFromIndex(ValLR->end.getPrevSlot());
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
    for (const uint16_t *AS = TRI->getAliasSet(IntB.reg); *AS; ++AS)
      if (LIS->hasInterval(*AS) && IntA.overlaps(LIS->getInterval(*AS))) {
        DEBUG({
            dbgs() << "\t\tInterfere with alias ";
            LIS->getInterval(*AS).print(dbgs(), TRI);
          });
        return false;
      }
  }

  DEBUG({
      dbgs() << "Extending: ";
      IntB.print(dbgs(), TRI);
    });

  SlotIndex FillerStart = ValLR->end, FillerEnd = BLR->start;
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'. Update the valnum with the new defining
  // instruction #.
  BValNo->def = FillerStart;

  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // If the IntB live range is assigned to a physical register, and if that
  // physreg has sub-registers, update their live intervals as well.
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg)) {
    for (const uint16_t *SR = TRI->getSubRegisters(IntB.reg); *SR; ++SR) {
      if (!LIS->hasInterval(*SR))
        continue;
      LiveInterval &SRLI = LIS->getInterval(*SR);
      SRLI.addRange(LiveRange(FillerStart, FillerEnd,
                              SRLI.getNextValue(FillerStart,
                                                LIS->getVNInfoAllocator())));
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
      IntB.print(dbgs(), TRI);
      dbgs() << "\n";
    });

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValLREndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1) {
    ValLREndInst->getOperand(UIdx).setIsKill(false);
  }

  // Rewrite the copy. If the copy instruction was killing the destination
  // register before the merge, find the last use and trim the live range. That
  // will also add the isKill marker.
  CopyMI->substituteRegister(IntA.reg, IntB.reg, CP.getSubIdx(),
                             *TRI);
  if (ALR->end == CopyIdx)
    LIS->shrinkToUses(&IntA);

  ++numExtends;
  return true;
}

/// hasOtherReachingDefs - Return true if there are definitions of IntB
/// other than BValNo val# that can reach uses of AValno val# of IntA.
bool RegisterCoalescer::hasOtherReachingDefs(LiveInterval &IntA,
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

/// removeCopyByCommutingDef - We found a non-trivially-coalescable copy with
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
bool RegisterCoalescer::removeCopyByCommutingDef(const CoalescerPair &CP,
                                                 MachineInstr *CopyMI) {
  // FIXME: For now, only eliminate the copy by commuting its def when the
  // source register is a virtual register. We want to guard against cases
  // where the copy is a back edge copy and commuting the def lengthen the
  // live interval of the source register to the entire loop.
  if (CP.isPhys() && CP.isFlipped())
    return false;

  // Bail if there is no dst interval.
  if (!LIS->hasInterval(CP.getDstReg()))
    return false;

  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot();

  LiveInterval &IntA =
    LIS->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
    LIS->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());

  // BValNo is a value number in B that is defined by a copy from A. 'B3' in
  // the example above.
  VNInfo *BValNo = IntB.getVNInfoAt(CopyIdx);
  if (!BValNo || BValNo->def != CopyIdx)
    return false;

  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");

  // AValNo is the value number in A that defines the copy, A3 in the example.
  VNInfo *AValNo = IntA.getVNInfoAt(CopyIdx.getRegSlot(true));
  assert(AValNo && "COPY source not live");

  // If other defs can reach uses of this def, then it's not safe to perform
  // the optimization.
  if (AValNo->isPHIDef() || AValNo->isUnused() || AValNo->hasPHIKill())
    return false;
  MachineInstr *DefMI = LIS->getInstructionFromIndex(AValNo->def);
  if (!DefMI)
    return false;
  if (!DefMI->isCommutable())
    return false;
  // If DefMI is a two-address instruction then commuting it will change the
  // destination register.
  int DefIdx = DefMI->findRegisterDefOperandIdx(IntA.reg);
  assert(DefIdx != -1);
  unsigned UseOpIdx;
  if (!DefMI->isRegTiedToUseOperand(DefIdx, &UseOpIdx))
    return false;
  unsigned Op1, Op2, NewDstIdx;
  if (!TII->findCommutedOpIndices(DefMI, Op1, Op2))
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
  if (hasOtherReachingDefs(IntA, IntB, AValNo, BValNo))
    return false;

  // Abort if the aliases of IntB.reg have values that are not simply the
  // clobbers from the superreg.
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg))
    for (const uint16_t *AS = TRI->getAliasSet(IntB.reg); *AS; ++AS)
      if (LIS->hasInterval(*AS) &&
          hasOtherReachingDefs(IntA, LIS->getInterval(*AS), AValNo, 0))
        return false;

  // If some of the uses of IntA.reg is already coalesced away, return false.
  // It's not possible to determine whether it's safe to perform the coalescing.
  for (MachineRegisterInfo::use_nodbg_iterator UI =
         MRI->use_nodbg_begin(IntA.reg),
       UE = MRI->use_nodbg_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    SlotIndex UseIdx = LIS->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end())
      continue;
    if (ULR->valno == AValNo && JoinedCopies.count(UseMI))
      return false;
  }

  DEBUG(dbgs() << "\tremoveCopyByCommutingDef: " << AValNo->def << '\t'
               << *DefMI);

  // At this point we have decided that it is legal to do this
  // transformation.  Start by commuting the instruction.
  MachineBasicBlock *MBB = DefMI->getParent();
  MachineInstr *NewMI = TII->commuteInstruction(DefMI);
  if (!NewMI)
    return false;
  if (TargetRegisterInfo::isVirtualRegister(IntA.reg) &&
      TargetRegisterInfo::isVirtualRegister(IntB.reg) &&
      !MRI->constrainRegClass(IntB.reg, MRI->getRegClass(IntA.reg)))
    return false;
  if (NewMI != DefMI) {
    LIS->ReplaceMachineInstrInMaps(DefMI, NewMI);
    MachineBasicBlock::iterator Pos = DefMI;
    MBB->insert(Pos, NewMI);
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
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(IntA.reg),
         UE = MRI->use_end(); UI != UE;) {
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
    SlotIndex UseIdx = LIS->getInstructionIndex(UseMI).getRegSlot(true);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end() || ULR->valno != AValNo)
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(NewReg))
      UseMO.substPhysReg(NewReg, *TRI);
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
    SlotIndex DefIdx = UseIdx.getRegSlot();
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

/// reMaterializeTrivialDef - If the source of a copy is defined by a trivial
/// computation, replace the copy by rematerialize the definition.
bool RegisterCoalescer::reMaterializeTrivialDef(LiveInterval &SrcInt,
                                                bool preserveSrcInt,
                                                unsigned DstReg,
                                                MachineInstr *CopyMI) {
  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot(true);
  LiveInterval::iterator SrcLR = SrcInt.FindLiveRangeContaining(CopyIdx);
  assert(SrcLR != SrcInt.end() && "Live range not found!");
  VNInfo *ValNo = SrcLR->valno;
  if (ValNo->isPHIDef() || ValNo->isUnused())
    return false;
  MachineInstr *DefMI = LIS->getInstructionFromIndex(ValNo->def);
  if (!DefMI)
    return false;
  assert(DefMI && "Defining instruction disappeared");
  if (!DefMI->isAsCheapAsAMove())
    return false;
  if (!TII->isTriviallyReMaterializable(DefMI, AA))
    return false;
  bool SawStore = false;
  if (!DefMI->isSafeToMove(TII, AA, SawStore))
    return false;
  const MCInstrDesc &MCID = DefMI->getDesc();
  if (MCID.getNumDefs() != 1)
    return false;
  if (!DefMI->isImplicitDef()) {
    // Make sure the copy destination register class fits the instruction
    // definition register class. The mismatch can happen as a result of earlier
    // extract_subreg, insert_subreg, subreg_to_reg coalescing.
    const TargetRegisterClass *RC = TII->getRegClass(MCID, 0, TRI, *MF);
    if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
      if (MRI->getRegClass(DstReg) != RC)
        return false;
    } else if (!RC->contains(DstReg))
      return false;
  }

  MachineBasicBlock *MBB = CopyMI->getParent();
  MachineBasicBlock::iterator MII =
    llvm::next(MachineBasicBlock::iterator(CopyMI));
  TII->reMaterialize(*MBB, MII, DstReg, 0, DefMI, *TRI);
  MachineInstr *NewMI = prior(MII);

  // NewMI may have dead implicit defs (E.g. EFLAGS for MOV<bits>r0 on X86).
  // We need to remember these so we can add intervals once we insert
  // NewMI into SlotIndexes.
  SmallVector<unsigned, 4> NewMIImplDefs;
  for (unsigned i = NewMI->getDesc().getNumOperands(),
         e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (MO.isReg()) {
      assert(MO.isDef() && MO.isImplicit() && MO.isDead() &&
             TargetRegisterInfo::isPhysicalRegister(MO.getReg()));
      NewMIImplDefs.push_back(MO.getReg());
    }
  }

  // CopyMI may have implicit operands, transfer them over to the newly
  // rematerialized instruction. And update implicit def interval valnos.
  for (unsigned i = CopyMI->getDesc().getNumOperands(),
         e = CopyMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = CopyMI->getOperand(i);
    if (MO.isReg()) {
      assert(MO.isImplicit() && "No explicit operands after implict operands.");
      // Discard VReg implicit defs.
      if (TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
        NewMI->addOperand(MO);
      }
    }
  }

  LIS->ReplaceMachineInstrInMaps(CopyMI, NewMI);

  SlotIndex NewMIIdx = LIS->getInstructionIndex(NewMI);
  for (unsigned i = 0, e = NewMIImplDefs.size(); i != e; ++i) {
    unsigned reg = NewMIImplDefs[i];
    LiveInterval &li = LIS->getInterval(reg);
    VNInfo *DeadDefVN = li.getNextValue(NewMIIdx.getRegSlot(),
                                        LIS->getVNInfoAllocator());
    LiveRange lr(NewMIIdx.getRegSlot(), NewMIIdx.getDeadSlot(), DeadDefVN);
    li.addRange(lr);
  }

  CopyMI->eraseFromParent();
  ReMatCopies.insert(CopyMI);
  ReMatDefs.insert(DefMI);
  DEBUG(dbgs() << "Remat: " << *NewMI);
  ++NumReMats;

  // The source interval can become smaller because we removed a use.
  if (preserveSrcInt)
    LIS->shrinkToUses(&SrcInt);

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
  SlotIndex Idx = LIS->getInstructionIndex(CopyMI);
  LiveInterval *SrcInt = &LIS->getInterval(CP.getSrcReg());
  if (SrcInt->liveAt(Idx))
    return false;
  LiveInterval *DstInt = &LIS->getInterval(CP.getDstReg());
  if (DstInt->liveAt(Idx))
    return false;

  // No intervals are live-in to CopyMI - it is undef.
  if (CP.isFlipped())
    DstInt = SrcInt;
  SrcInt = 0;

  VNInfo *DeadVNI = DstInt->getVNInfoAt(Idx.getRegSlot());
  assert(DeadVNI && "No value defined in DstInt");
  DstInt->removeValNo(DeadVNI);

  // Find new undef uses.
  for (MachineRegisterInfo::reg_nodbg_iterator
         I = MRI->reg_nodbg_begin(DstInt->reg), E = MRI->reg_nodbg_end();
       I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    if (MO.isDef() || MO.isUndef())
      continue;
    MachineInstr *MI = MO.getParent();
    SlotIndex Idx = LIS->getInstructionIndex(MI);
    if (DstInt->liveAt(Idx))
      continue;
    MO.setIsUndef(true);
    DEBUG(dbgs() << "\tnew undef: " << Idx << '\t' << *MI);
  }
  return true;
}

/// updateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
/// update the subregister number if it is not zero. If DstReg is a
/// physical register and the existing subregister number of the def / use
/// being updated is not zero, make sure to set it to the correct physical
/// subregister.
void RegisterCoalescer::updateRegDefsUses(const CoalescerPair &CP) {
  bool DstIsPhys = CP.isPhys();
  unsigned SrcReg = CP.getSrcReg();
  unsigned DstReg = CP.getDstReg();
  unsigned SubIdx = CP.getSubIdx();

  // Update LiveDebugVariables.
  LDV->renameRegister(SrcReg, DstReg, SubIdx);

  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(SrcReg);
       MachineInstr *UseMI = I.skipInstruction();) {
    // A PhysReg copy that won't be coalesced can perhaps be rematerialized
    // instead.
    if (DstIsPhys) {
      if (UseMI->isFullCopy() &&
          UseMI->getOperand(1).getReg() == SrcReg &&
          UseMI->getOperand(0).getReg() != SrcReg &&
          UseMI->getOperand(0).getReg() != DstReg &&
          !JoinedCopies.count(UseMI) &&
          reMaterializeTrivialDef(LIS->getInterval(SrcReg), false,
                                  UseMI->getOperand(0).getReg(), UseMI))
        continue;
    }

    SmallVector<unsigned,8> Ops;
    bool Reads, Writes;
    tie(Reads, Writes) = UseMI->readsWritesVirtualRegister(SrcReg, &Ops);

    // Replace SrcReg with DstReg in all UseMI operands.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = UseMI->getOperand(Ops[i]);

      // Make sure we don't create read-modify-write defs accidentally.  We
      // assume here that a SrcReg def cannot be joined into a live DstReg.  If
      // RegisterCoalescer starts tracking partially live registers, we will
      // need to check the actual LiveInterval to determine if DstReg is live
      // here.
      if (SubIdx && !Reads)
        MO.setIsUndef();

      if (DstIsPhys)
        MO.substPhysReg(DstReg, *TRI);
      else
        MO.substVirtReg(DstReg, SubIdx, *TRI);
    }

    // This instruction is a copy that will be removed.
    if (JoinedCopies.count(UseMI))
      continue;

    DEBUG({
        dbgs() << "\t\tupdated: ";
        if (!UseMI->isDebugValue())
          dbgs() << LIS->getInstructionIndex(UseMI) << "\t";
        dbgs() << *UseMI;
      });
  }
}

/// removeIntervalIfEmpty - Check if the live interval of a physical register
/// is empty, if so remove it and also remove the empty intervals of its
/// sub-registers. Return true if live interval is removed.
static bool removeIntervalIfEmpty(LiveInterval &li, LiveIntervals *LIS,
                                  const TargetRegisterInfo *TRI) {
  if (li.empty()) {
    if (TargetRegisterInfo::isPhysicalRegister(li.reg))
      for (const uint16_t* SR = TRI->getSubRegisters(li.reg); *SR; ++SR) {
        if (!LIS->hasInterval(*SR))
          continue;
        LiveInterval &sli = LIS->getInterval(*SR);
        if (sli.empty())
          LIS->removeInterval(*SR);
      }
    LIS->removeInterval(li.reg);
    return true;
  }
  return false;
}

/// removeDeadDef - If a def of a live interval is now determined dead, remove
/// the val# it defines. If the live interval becomes empty, remove it as well.
bool RegisterCoalescer::removeDeadDef(LiveInterval &li, MachineInstr *DefMI) {
  SlotIndex DefIdx = LIS->getInstructionIndex(DefMI).getRegSlot();
  LiveInterval::iterator MLR = li.FindLiveRangeContaining(DefIdx);
  if (DefIdx != MLR->valno->def)
    return false;
  li.removeValNo(MLR->valno);
  return removeIntervalIfEmpty(li, LIS, TRI);
}

/// shouldJoinPhys - Return true if a copy involving a physreg should be joined.
/// We need to be careful about coalescing a source physical register with a
/// virtual register. Once the coalescing is done, it cannot be broken and these
/// are not spillable! If the destination interval uses are far away, think
/// twice about coalescing them!
bool RegisterCoalescer::shouldJoinPhys(CoalescerPair &CP) {
  bool Allocatable = LIS->isAllocatable(CP.getDstReg());
  LiveInterval &JoinVInt = LIS->getInterval(CP.getSrcReg());

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
  if (LIS->hasInterval(CP.getDstReg()) &&
      LIS->getInterval(CP.getDstReg()).ranges.size() > 1000) {
    ++numAborts;
    DEBUG(dbgs()
          << "\tPhysical register live interval too complicated, abort!\n");
    return false;
  }

  // FIXME: Why are we skipping this test for partial copies?
  //        CodeGen/X86/phys_subreg_coalesce-3.ll needs it.
  if (!CP.isPartial()) {
    const TargetRegisterClass *RC = MRI->getRegClass(CP.getSrcReg());
    unsigned Threshold = RegClassInfo.getNumAllocatableRegs(RC) * 2;
    unsigned Length = LIS->getApproximateInstructionCount(JoinVInt);
    if (Length > Threshold) {
      ++numAborts;
      DEBUG(dbgs() << "\tMay tie down a physical register, abort!\n");
      return false;
    }
  }
  return true;
}


/// joinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coalesced away. If it is not currently
/// possible to coalesce this interval, but it may be possible if other
/// things get coalesced, then it returns true by reference in 'Again'.
bool RegisterCoalescer::joinCopy(MachineInstr *CopyMI, bool &Again) {

  Again = false;
  if (JoinedCopies.count(CopyMI) || ReMatCopies.count(CopyMI))
    return false; // Already done.

  DEBUG(dbgs() << LIS->getInstructionIndex(CopyMI) << '\t' << *CopyMI);

  CoalescerPair CP(*TII, *TRI);
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

  DEBUG(dbgs() << "\tConsidering merging " << PrintReg(CP.getSrcReg(), TRI)
               << " with " << PrintReg(CP.getDstReg(), TRI, CP.getSubIdx())
               << "\n");

  // Enforce policies.
  if (CP.isPhys()) {
    if (!shouldJoinPhys(CP)) {
      // Before giving up coalescing, if definition of source is defined by
      // trivial computation, try rematerializing it.
      if (!CP.isFlipped() &&
          reMaterializeTrivialDef(LIS->getInterval(CP.getSrcReg()), true,
                                  CP.getDstReg(), CopyMI))
        return true;
      return false;
    }
  } else {
    DEBUG({
      if (CP.isCrossClass())
        dbgs() << "\tCross-class to " << CP.getNewRC()->getName() << ".\n";
    });

    // When possible, let DstReg be the larger interval.
    if (!CP.getSubIdx() && LIS->getInterval(CP.getSrcReg()).ranges.size() >
                           LIS->getInterval(CP.getDstReg()).ranges.size())
      CP.flip();
  }

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  if (!joinIntervals(CP)) {
    // Coalescing failed.

    // If definition of source is defined by trivial computation, try
    // rematerializing it.
    if (!CP.isFlipped() &&
        reMaterializeTrivialDef(LIS->getInterval(CP.getSrcReg()), true,
                                CP.getDstReg(), CopyMI))
      return true;

    // If we can eliminate the copy without merging the live ranges, do so now.
    if (!CP.isPartial()) {
      if (adjustCopiesBackFrom(CP, CopyMI) ||
          removeCopyByCommutingDef(CP, CopyMI)) {
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
    MRI->setRegClass(CP.getDstReg(), CP.getNewRC());
  }

  // Remember to delete the copy instruction.
  markAsJoined(CopyMI);

  updateRegDefsUses(CP);

  // If we have extended the live range of a physical register, make sure we
  // update live-in lists as well.
  if (CP.isPhys()) {
    SmallVector<MachineBasicBlock*, 16> BlockSeq;
    // joinIntervals invalidates the VNInfos in SrcInt, but we only need the
    // ranges for this, and they are preserved.
    LiveInterval &SrcInt = LIS->getInterval(CP.getSrcReg());
    for (LiveInterval::const_iterator I = SrcInt.begin(), E = SrcInt.end();
         I != E; ++I ) {
      LIS->findLiveInMBBs(I->start, I->end, BlockSeq);
      for (unsigned idx = 0, size = BlockSeq.size(); idx != size; ++idx) {
        MachineBasicBlock &block = *BlockSeq[idx];
        if (!block.isLiveIn(CP.getDstReg()))
          block.addLiveIn(CP.getDstReg());
      }
      BlockSeq.clear();
    }
  }

  // SrcReg is guaranteed to be the register whose live interval that is
  // being merged.
  LIS->removeInterval(CP.getSrcReg());

  // Update regalloc hint.
  TRI->UpdateRegAllocHint(CP.getSrcReg(), CP.getDstReg(), *MF);

  DEBUG({
    LiveInterval &DstInt = LIS->getInterval(CP.getDstReg());
    dbgs() << "\tJoined. Result = ";
    DstInt.print(dbgs(), TRI);
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
// This implies that A dominates B.
static bool RegistersDefinedFromSameValue(LiveIntervals &li,
                                          const TargetRegisterInfo &tri,
                                          CoalescerPair &CP,
                                          VNInfo *VNI,
                                          LiveRange *LR,
                                     SmallVector<MachineInstr*, 8> &DupCopies) {
  // FIXME: This is very conservative. For example, we don't handle
  // physical registers.

  MachineInstr *MI = li.getInstructionFromIndex(VNI->def);

  if (!MI || !MI->isFullCopy() || CP.isPartial() || CP.isPhys())
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
  const MachineInstr *OtherMI = li.getInstructionFromIndex(Other->def);

  if (!OtherMI || !OtherMI->isFullCopy())
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
  // getVNInfoBefore returns NULL for undef copies. In this case, the
  // optimization is still safe.
  if (SrcInt.getVNInfoBefore(Other->def) != SrcInt.getVNInfoBefore(VNI->def))
    return false;

  DupCopies.push_back(MI);

  return true;
}

/// joinIntervals - Attempt to join these two intervals.  On failure, this
/// returns false.
bool RegisterCoalescer::joinIntervals(CoalescerPair &CP) {
  LiveInterval &RHS = LIS->getInterval(CP.getSrcReg());
  DEBUG({ dbgs() << "\t\tRHS = "; RHS.print(dbgs(), TRI); dbgs() << "\n"; });

  // If a live interval is a physical register, check for interference with any
  // aliases. The interference check implemented here is a bit more conservative
  // than the full interfeence check below. We allow overlapping live ranges
  // only when one is a copy of the other.
  if (CP.isPhys()) {
    // Optimization for reserved registers like ESP.
    // We can only merge with a reserved physreg if RHS has a single value that
    // is a copy of CP.DstReg().  The live range of the reserved register will
    // look like a set of dead defs - we don't properly track the live range of
    // reserved registers.
    if (RegClassInfo.isReserved(CP.getDstReg())) {
      assert(CP.isFlipped() && RHS.containsOneValue() &&
             "Invalid join with reserved register");
      // Deny any overlapping intervals.  This depends on all the reserved
      // register live ranges to look like dead defs.
      for (const uint16_t *AS = TRI->getOverlaps(CP.getDstReg()); *AS; ++AS) {
        if (!LIS->hasInterval(*AS)) {
          // Make sure at least DstReg itself exists before attempting a join.
          if (*AS == CP.getDstReg())
            LIS->getOrCreateInterval(CP.getDstReg());
          continue;
        }
        if (RHS.overlaps(LIS->getInterval(*AS))) {
          DEBUG(dbgs() << "\t\tInterference: " << PrintReg(*AS, TRI) << '\n');
          return false;
        }
      }
      // Skip any value computations, we are not adding new values to the
      // reserved register.  Also skip merging the live ranges, the reserved
      // register live range doesn't need to be accurate as long as all the
      // defs are there.
      return true;
    }

    // Check if a register mask clobbers DstReg.
    BitVector UsableRegs;
    if (LIS->checkRegMaskInterference(RHS, UsableRegs) &&
        !UsableRegs.test(CP.getDstReg())) {
      DEBUG(dbgs() << "\t\tRegister mask interference.\n");
      return false;
    }

    for (const uint16_t *AS = TRI->getAliasSet(CP.getDstReg()); *AS; ++AS){
      if (!LIS->hasInterval(*AS))
        continue;
      const LiveInterval &LHS = LIS->getInterval(*AS);
      LiveInterval::const_iterator LI = LHS.begin();
      for (LiveInterval::const_iterator RI = RHS.begin(), RE = RHS.end();
           RI != RE; ++RI) {
        LI = std::lower_bound(LI, LHS.end(), RI->start);
        // Does LHS have an overlapping live range starting before RI?
        if ((LI != LHS.begin() && LI[-1].end > RI->start) &&
            (RI->start != RI->valno->def ||
             !CP.isCoalescable(LIS->getInstructionFromIndex(RI->start)))) {
          DEBUG({
            dbgs() << "\t\tInterference from alias: ";
            LHS.print(dbgs(), TRI);
            dbgs() << "\n\t\tOverlap at " << RI->start << " and no copy.\n";
          });
          return false;
        }

        // Check that LHS ranges beginning in this range are copies.
        for (; LI != LHS.end() && LI->start < RI->end; ++LI) {
          if (LI->start != LI->valno->def ||
              !CP.isCoalescable(LIS->getInstructionFromIndex(LI->start))) {
            DEBUG({
              dbgs() << "\t\tInterference from alias: ";
              LHS.print(dbgs(), TRI);
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

  LiveInterval &LHS = LIS->getOrCreateInterval(CP.getDstReg());
  DEBUG({ dbgs() << "\t\tLHS = "; LHS.print(dbgs(), TRI); dbgs() << "\n"; });

  // Loop over the value numbers of the LHS, seeing if any are defined from
  // the RHS.
  for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
       i != e; ++i) {
    VNInfo *VNI = *i;
    if (VNI->isUnused() || VNI->isPHIDef())
      continue;
    MachineInstr *MI = LIS->getInstructionFromIndex(VNI->def);
    assert(MI && "Missing def");
    if (!MI->isCopyLike())  // Src not defined by a copy?
      continue;

    // Figure out the value # from the RHS.
    LiveRange *lr = RHS.getLiveRangeContaining(VNI->def.getPrevSlot());
    // The copy could be to an aliased physreg.
    if (!lr) continue;

    // DstReg is known to be a register in the LHS interval.  If the src is
    // from the RHS interval, we can use its value #.
    if (!CP.isCoalescable(MI) &&
        !RegistersDefinedFromSameValue(*LIS, *TRI, CP, VNI, lr, DupCopies))
      continue;

    LHSValsDefinedFromRHS[VNI] = lr->valno;
  }

  // Loop over the value numbers of the RHS, seeing if any are defined from
  // the LHS.
  for (LiveInterval::vni_iterator i = RHS.vni_begin(), e = RHS.vni_end();
       i != e; ++i) {
    VNInfo *VNI = *i;
    if (VNI->isUnused() || VNI->isPHIDef())
      continue;
    MachineInstr *MI = LIS->getInstructionFromIndex(VNI->def);
    assert(MI && "Missing def");
    if (!MI->isCopyLike())  // Src not defined by a copy?
      continue;

    // Figure out the value # from the LHS.
    LiveRange *lr = LHS.getLiveRangeContaining(VNI->def.getPrevSlot());
    // The copy could be to an aliased physreg.
    if (!lr) continue;

    // DstReg is known to be a register in the RHS interval.  If the src is
    // from the LHS interval, we can use its value #.
    if (!CP.isCoalescable(MI) &&
        !RegistersDefinedFromSameValue(*LIS, *TRI, CP, VNI, lr, DupCopies))
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
    MI->getOperand(0).substVirtReg(Src, 0, *TRI);

    markAsJoined(MI);
  }

  // If B = X was the last use of X in a liverange, we have to shrink it now
  // that B = X is gone.
  for (SmallVector<unsigned, 8>::iterator I = SourceRegisters.begin(),
         E = SourceRegisters.end(); I != E; ++I) {
    LIS->shrinkToUses(&LIS->getInterval(*I));
  }

  // If we get here, we know that we can coalesce the live ranges.  Ask the
  // intervals to coalesce themselves now.
  LHS.join(RHS, &LHSValNoAssignments[0], &RHSValNoAssignments[0], NewVNInfo,
           MRI);
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

void
RegisterCoalescer::copyCoalesceInMBB(MachineBasicBlock *MBB,
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
    if (LIS->hasInterval(SrcReg) && LIS->getInterval(SrcReg).empty())
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
    if (!joinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = PhysCopies.size(); i != e; ++i) {
    MachineInstr *TheCopy = PhysCopies[i];
    bool Again = false;
    if (!joinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = VirtCopies.size(); i != e; ++i) {
    MachineInstr *TheCopy = VirtCopies[i];
    bool Again = false;
    if (!joinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
}

void RegisterCoalescer::joinAllIntervals() {
  DEBUG(dbgs() << "********** JOINING INTERVALS ***********\n");

  std::vector<MachineInstr*> TryAgainList;
  if (Loops->empty()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = MF->begin(), E = MF->end();
         I != E; ++I)
      copyCoalesceInMBB(I, TryAgainList);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.

    // Join intervals in the function prolog first. We want to join physical
    // registers with virtual registers before the intervals got too long.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = MF->begin(), E = MF->end();I != E;++I){
      MachineBasicBlock *MBB = I;
      MBBs.push_back(std::make_pair(Loops->getLoopDepth(MBB), I));
    }

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order.
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      copyCoalesceInMBB(MBBs[i].second, TryAgainList);
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
      bool Success = joinCopy(TheCopy, Again);
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
  MF = &fn;
  MRI = &fn.getRegInfo();
  TM = &fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();
  LIS = &getAnalysis<LiveIntervals>();
  LDV = &getAnalysis<LiveDebugVariables>();
  AA = &getAnalysis<AliasAnalysis>();
  Loops = &getAnalysis<MachineLoopInfo>();

  DEBUG(dbgs() << "********** SIMPLE REGISTER COALESCING **********\n"
               << "********** Function: "
               << ((Value*)MF->getFunction())->getName() << '\n');

  if (VerifyCoalescing)
    MF->verify(this, "Before register coalescing");

  RegClassInfo.runOnMachineFunction(fn);

  // Join (coalesce) intervals if requested.
  if (EnableJoining) {
    joinAllIntervals();
    DEBUG({
        dbgs() << "********** INTERVALS POST JOINING **********\n";
        for (LiveIntervals::iterator I = LIS->begin(), E = LIS->end();
             I != E; ++I){
          I->second->print(dbgs(), TRI);
          dbgs() << "\n";
        }
      });
  }

  // Perform a final pass over the instructions and compute spill weights
  // and remove identity moves.
  SmallVector<unsigned, 4> DeadDefs, InflateRegs;
  for (MachineFunction::iterator mbbi = MF->begin(), mbbe = MF->end();
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
        unsigned DstReg = MI->getOperand(0).getReg();

        // Collect candidates for register class inflation.
        if (TargetRegisterInfo::isVirtualRegister(SrcReg) &&
            RegClassInfo.isProperSubClass(MRI->getRegClass(SrcReg)))
          InflateRegs.push_back(SrcReg);
        if (TargetRegisterInfo::isVirtualRegister(DstReg) &&
            RegClassInfo.isProperSubClass(MRI->getRegClass(DstReg)))
          InflateRegs.push_back(DstReg);

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
              LIS->hasInterval(SrcReg))
            LIS->shrinkToUses(&LIS->getInterval(SrcReg));
          DoDelete = true;
        }
        if (!DoDelete) {
          // We need the instruction to adjust liveness, so make it a KILL.
          if (MI->isSubregToReg()) {
            MI->RemoveOperand(3);
            MI->RemoveOperand(1);
          }
          MI->setDesc(TII->get(TargetOpcode::KILL));
          mii = llvm::next(mii);
        } else {
          LIS->RemoveMachineInstrFromMaps(MI);
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
          DeadDefs.push_back(Reg);
          if (TargetRegisterInfo::isVirtualRegister(Reg)) {
            // Remat may also enable register class inflation.
            if (RegClassInfo.isProperSubClass(MRI->getRegClass(Reg)))
              InflateRegs.push_back(Reg);
          }
          if (MO.isDead())
            continue;
          if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
              !MRI->use_nodbg_empty(Reg)) {
            isDead = false;
            break;
          }
        }
        if (isDead) {
          while (!DeadDefs.empty()) {
            unsigned DeadDef = DeadDefs.back();
            DeadDefs.pop_back();
            removeDeadDef(LIS->getInterval(DeadDef), MI);
          }
          LIS->RemoveMachineInstrFromMaps(mii);
          mii = mbbi->erase(mii);
          continue;
        } else
          DeadDefs.clear();
      }

      ++mii;

      // Check for now unnecessary kill flags.
      if (LIS->isNotInMIMap(MI)) continue;
      SlotIndex DefIdx = LIS->getInstructionIndex(MI).getRegSlot();
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.isKill()) continue;
        unsigned reg = MO.getReg();
        if (!reg || !LIS->hasInterval(reg)) continue;
        if (!LIS->getInterval(reg).killedAt(DefIdx)) {
          MO.setIsKill(false);
          continue;
        }
        // When leaving a kill flag on a physreg, check if any subregs should
        // remain alive.
        if (!TargetRegisterInfo::isPhysicalRegister(reg))
          continue;
        for (const uint16_t *SR = TRI->getSubRegisters(reg);
             unsigned S = *SR; ++SR)
          if (LIS->hasInterval(S) && LIS->getInterval(S).liveAt(DefIdx))
            MI->addRegisterDefined(S, TRI);
      }
    }
  }

  // After deleting a lot of copies, register classes may be less constrained.
  // Removing sub-register opreands may alow GR32_ABCD -> GR32 and DPR_VFP2 ->
  // DPR inflation.
  array_pod_sort(InflateRegs.begin(), InflateRegs.end());
  InflateRegs.erase(std::unique(InflateRegs.begin(), InflateRegs.end()),
                    InflateRegs.end());
  DEBUG(dbgs() << "Trying to inflate " << InflateRegs.size() << " regs.\n");
  for (unsigned i = 0, e = InflateRegs.size(); i != e; ++i) {
    unsigned Reg = InflateRegs[i];
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    if (MRI->recomputeRegClass(Reg, *TM)) {
      DEBUG(dbgs() << PrintReg(Reg) << " inflated to "
                   << MRI->getRegClass(Reg)->getName() << '\n');
      ++NumInflated;
    }
  }

  DEBUG(dump());
  DEBUG(LDV->dump());
  if (VerifyCoalescing)
    MF->verify(this, "After register coalescing");
  return true;
}

/// print - Implement the dump method.
void RegisterCoalescer::print(raw_ostream &O, const Module* m) const {
   LIS->print(O, m);
}
