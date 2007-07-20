//===-- SimpleRegisterCoalescing.cpp - Register Coalescing ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register coalescing pass that attempts to
// aggressively coalesce every register copy that it can.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simpleregistercoalescing"
#include "llvm/CodeGen/SimpleRegisterCoalescing.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "VirtRegMap.h"
#include "llvm/Value.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(numJoins    , "Number of interval joins performed");
STATISTIC(numPeep     , "Number of identity moves eliminated after coalescing");
STATISTIC(numAborts   , "Number of times interval joining aborted");

char SimpleRegisterCoalescing::ID = 0;
namespace {
  static cl::opt<bool>
  EnableJoining("join-liveintervals",
                cl::desc("Coalesce copies (default=true)"),
                cl::init(true));

  RegisterPass<SimpleRegisterCoalescing> 
  X("simple-register-coalescing",
    "Simple register coalescing to eliminate all possible register copies");
}

const PassInfo *llvm::SimpleRegisterCoalescingID = X.getPassInfo();

void SimpleRegisterCoalescing::getAnalysisUsage(AnalysisUsage &AU) const {
   //AU.addPreserved<LiveVariables>();
  AU.addPreserved<LiveIntervals>();
  AU.addPreservedID(PHIEliminationID);
  AU.addPreservedID(TwoAddressInstructionPassID);
  AU.addRequired<LiveVariables>();
  AU.addRequired<LiveIntervals>();
  AU.addRequired<LoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
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
bool SimpleRegisterCoalescing::AdjustCopiesBackFrom(LiveInterval &IntA, LiveInterval &IntB,
                                         MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));

  // BValNo is a value number in B that is defined by a copy from A.  'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  unsigned BValNo = BLR->ValId;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  unsigned BValNoDefIdx = IntB.getInstForValNum(BValNo);
  if (BValNoDefIdx == ~0U) return false;
  assert(BValNoDefIdx == CopyIdx &&
         "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A0 in the example.
  LiveInterval::iterator AValLR = IntA.FindLiveRangeContaining(CopyIdx-1);
  unsigned AValNo = AValLR->ValId;
  
  // If AValNo is defined as a copy from IntB, we can potentially process this.
  
  // Get the instruction that defines this value number.
  unsigned SrcReg = IntA.getSrcRegForValNum(AValNo);
  if (!SrcReg) return false;  // Not defined by a copy.
    
  // If the value number is not defined by a copy instruction, ignore it.
    
  // If the source register comes from an interval other than IntB, we can't
  // handle this.
  if (rep(SrcReg) != IntB.reg) return false;
  
  // Get the LiveRange in IntB that this value number starts with.
  unsigned AValNoInstIdx = IntA.getInstForValNum(AValNo);
  LiveInterval::iterator ValLR = IntB.FindLiveRangeContaining(AValNoInstIdx-1);
  
  // Make sure that the end of the live range is inside the same block as
  // CopyMI.
  MachineInstr *ValLREndInst = li_->getInstructionFromIndex(ValLR->end-1);
  if (!ValLREndInst || 
      ValLREndInst->getParent() != CopyMI->getParent()) return false;

  // Okay, we now know that ValLR ends in the same block that the CopyMI
  // live-range starts.  If there are no intervening live ranges between them in
  // IntB, we can merge them.
  if (ValLR+1 != BLR) return false;
  
  DOUT << "\nExtending: "; IntB.print(DOUT, mri_);
  
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'.
  IntB.setValueNumberInfo(BValNo, std::make_pair(~0U, 0));
  
  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  unsigned FillerStart = ValLR->end, FillerEnd = BLR->start;
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // If the IntB live range is assigned to a physical register, and if that
  // physreg has aliases, 
  if (MRegisterInfo::isPhysicalRegister(IntB.reg)) {
    // Update the liveintervals of sub-registers.
    for (const unsigned *AS = mri_->getSubRegisters(IntB.reg); *AS; ++AS) {
      LiveInterval &AliasLI = li_->getInterval(*AS);
      AliasLI.addRange(LiveRange(FillerStart, FillerEnd,
                                 AliasLI.getNextValue(~0U, 0)));
    }
  }

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValLR->ValId)
    IntB.MergeValueNumberInto(BValNo, ValLR->ValId);
  DOUT << "   result = "; IntB.print(DOUT, mri_);
  DOUT << "\n";

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValLREndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1)
    ValLREndInst->getOperand(UIdx).unsetIsKill();
  
  // Finally, delete the copy instruction.
  li_->RemoveMachineInstrFromMaps(CopyMI);
  CopyMI->eraseFromParent();
  ++numPeep;
  return true;
}

/// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coalesced away, or if it is never possible
/// to coalesce this copy, due to register constraints.  It returns
/// false if it is not currently possible to coalesce this interval, but
/// it may be possible if other things get coalesced.
bool SimpleRegisterCoalescing::JoinCopy(MachineInstr *CopyMI,
                             unsigned SrcReg, unsigned DstReg, bool PhysOnly) {
  DOUT << li_->getInstructionIndex(CopyMI) << '\t' << *CopyMI;

  // Get representative registers.
  unsigned repSrcReg = rep(SrcReg);
  unsigned repDstReg = rep(DstReg);
  
  // If they are already joined we continue.
  if (repSrcReg == repDstReg) {
    DOUT << "\tCopy already coalesced.\n";
    return true;  // Not coalescable.
  }
  
  bool SrcIsPhys = MRegisterInfo::isPhysicalRegister(repSrcReg);
  bool DstIsPhys = MRegisterInfo::isPhysicalRegister(repDstReg);
  if (PhysOnly && !SrcIsPhys && !DstIsPhys)
    // Only joining physical registers with virtual registers in this round.
    return true;

  // If they are both physical registers, we cannot join them.
  if (SrcIsPhys && DstIsPhys) {
    DOUT << "\tCan not coalesce physregs.\n";
    return true;  // Not coalescable.
  }
  
  // We only join virtual registers with allocatable physical registers.
  if (SrcIsPhys && !allocatableRegs_[repSrcReg]) {
    DOUT << "\tSrc reg is unallocatable physreg.\n";
    return true;  // Not coalescable.
  }
  if (DstIsPhys && !allocatableRegs_[repDstReg]) {
    DOUT << "\tDst reg is unallocatable physreg.\n";
    return true;  // Not coalescable.
  }
  
  // If they are not of the same register class, we cannot join them.
  if (differingRegisterClasses(repSrcReg, repDstReg)) {
    DOUT << "\tSrc/Dest are different register classes.\n";
    return true;  // Not coalescable.
  }
  
  LiveInterval &SrcInt = li_->getInterval(repSrcReg);
  LiveInterval &DstInt = li_->getInterval(repDstReg);
  assert(SrcInt.reg == repSrcReg && DstInt.reg == repDstReg &&
         "Register mapping is horribly broken!");

  DOUT << "\t\tInspecting "; SrcInt.print(DOUT, mri_);
  DOUT << " and "; DstInt.print(DOUT, mri_);
  DOUT << ": ";

  // Check if it is necessary to propagate "isDead" property before intervals
  // are joined.
  MachineOperand *mopd = CopyMI->findRegisterDefOperand(DstReg);
  bool isDead = mopd->isDead();
  bool isShorten = false;
  unsigned SrcStart = 0, RemoveStart = 0;
  unsigned SrcEnd = 0, RemoveEnd = 0;
  if (isDead) {
    unsigned CopyIdx = li_->getInstructionIndex(CopyMI);
    LiveInterval::iterator SrcLR =
      SrcInt.FindLiveRangeContaining(li_->getUseIndex(CopyIdx));
    RemoveStart = SrcStart = SrcLR->start;
    RemoveEnd   = SrcEnd   = SrcLR->end;
    // The instruction which defines the src is only truly dead if there are
    // no intermediate uses and there isn't a use beyond the copy.
    // FIXME: find the last use, mark is kill and shorten the live range.
    if (SrcEnd > li_->getDefIndex(CopyIdx)) {
      isDead = false;
    } else {
      MachineOperand *MOU;
      MachineInstr *LastUse= lastRegisterUse(SrcStart, CopyIdx, repSrcReg, MOU);
      if (LastUse) {
        // Shorten the liveinterval to the end of last use.
        MOU->setIsKill();
        isDead = false;
        isShorten = true;
        RemoveStart = li_->getDefIndex(li_->getInstructionIndex(LastUse));
        RemoveEnd   = SrcEnd;
      } else {
        MachineInstr *SrcMI = li_->getInstructionFromIndex(SrcStart);
        if (SrcMI) {
          MachineOperand *mops = findDefOperand(SrcMI, repSrcReg);
          if (mops)
            // A dead def should have a single cycle interval.
            ++RemoveStart;
        }
      }
    }
  }

  // We need to be careful about coalescing a source physical register with a
  // virtual register. Once the coalescing is done, it cannot be broken and
  // these are not spillable! If the destination interval uses are far away,
  // think twice about coalescing them!
  if (!mopd->isDead() && (SrcIsPhys || DstIsPhys)) {
    LiveInterval &JoinVInt = SrcIsPhys ? DstInt : SrcInt;
    unsigned JoinVReg = SrcIsPhys ? repDstReg : repSrcReg;
    unsigned JoinPReg = SrcIsPhys ? repSrcReg : repDstReg;
    const TargetRegisterClass *RC = mf_->getSSARegMap()->getRegClass(JoinVReg);
    unsigned Threshold = allocatableRCRegs_[RC].count();

    // If the virtual register live interval is long has it has low use desity,
    // do not join them, instead mark the physical register as its allocation
    // preference.
    unsigned Length = JoinVInt.getSize() / InstrSlots::NUM;
    LiveVariables::VarInfo &vi = lv_->getVarInfo(JoinVReg);
    if (Length > Threshold &&
        (((float)vi.NumUses / Length) < (1.0 / Threshold))) {
      JoinVInt.preference = JoinPReg;
      ++numAborts;
      DOUT << "\tMay tie down a physical register, abort!\n";
      return false;
    }
  }

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  if (JoinIntervals(DstInt, SrcInt)) {
    if (isDead) {
      // Result of the copy is dead. Propagate this property.
      if (SrcStart == 0) {
        assert(MRegisterInfo::isPhysicalRegister(repSrcReg) &&
               "Live-in must be a physical register!");
        // Live-in to the function but dead. Remove it from entry live-in set.
        // JoinIntervals may end up swapping the two intervals.
        mf_->begin()->removeLiveIn(repSrcReg);
      } else {
        MachineInstr *SrcMI = li_->getInstructionFromIndex(SrcStart);
        if (SrcMI) {
          MachineOperand *mops = findDefOperand(SrcMI, repSrcReg);
          if (mops)
            mops->setIsDead();
        }
      }
    }

    if (isShorten || isDead) {
      // Shorten the live interval.
      LiveInterval &LiveInInt = (repSrcReg == DstInt.reg) ? DstInt : SrcInt;
      LiveInInt.removeRange(RemoveStart, RemoveEnd);
    }
  } else {
    // Coalescing failed.
    
    // If we can eliminate the copy without merging the live ranges, do so now.
    if (AdjustCopiesBackFrom(SrcInt, DstInt, CopyMI))
      return true;

    // Otherwise, we are unable to join the intervals.
    DOUT << "Interference!\n";
    return false;
  }

  bool Swapped = repSrcReg == DstInt.reg;
  if (Swapped)
    std::swap(repSrcReg, repDstReg);
  assert(MRegisterInfo::isVirtualRegister(repSrcReg) &&
         "LiveInterval::join didn't work right!");
                               
  // If we're about to merge live ranges into a physical register live range,
  // we have to update any aliased register's live ranges to indicate that they
  // have clobbered values for this range.
  if (MRegisterInfo::isPhysicalRegister(repDstReg)) {
    // Unset unnecessary kills.
    if (!DstInt.containsOneValue()) {
      for (LiveInterval::Ranges::const_iterator I = SrcInt.begin(),
             E = SrcInt.end(); I != E; ++I)
        unsetRegisterKills(I->start, I->end, repDstReg);
    }

    // Update the liveintervals of sub-registers.
    for (const unsigned *AS = mri_->getSubRegisters(repDstReg); *AS; ++AS)
      li_->getInterval(*AS).MergeInClobberRanges(SrcInt);
  } else {
    // Merge use info if the destination is a virtual register.
    LiveVariables::VarInfo& dVI = lv_->getVarInfo(repDstReg);
    LiveVariables::VarInfo& sVI = lv_->getVarInfo(repSrcReg);
    dVI.NumUses += sVI.NumUses;
  }

  DOUT << "\n\t\tJoined.  Result = "; DstInt.print(DOUT, mri_);
  DOUT << "\n";

  // Remember these liveintervals have been joined.
  JoinedLIs.set(repSrcReg - MRegisterInfo::FirstVirtualRegister);
  if (MRegisterInfo::isVirtualRegister(repDstReg))
    JoinedLIs.set(repDstReg - MRegisterInfo::FirstVirtualRegister);

  // If the intervals were swapped by Join, swap them back so that the register
  // mapping (in the r2i map) is correct.
  if (Swapped) SrcInt.swap(DstInt);

  // repSrcReg is guarateed to be the register whose live interval that is
  // being merged.
  li_->removeInterval(repSrcReg);
  r2rMap_[repSrcReg] = repDstReg;

  // Finally, delete the copy instruction.
  li_->RemoveMachineInstrFromMaps(CopyMI);
  CopyMI->eraseFromParent();
  ++numPeep;
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
static unsigned ComputeUltimateVN(unsigned VN,
                                  SmallVector<std::pair<unsigned,
                                                unsigned>, 16> &ValueNumberInfo,
                                  SmallVector<int, 16> &ThisFromOther,
                                  SmallVector<int, 16> &OtherFromThis,
                                  SmallVector<int, 16> &ThisValNoAssignments,
                                  SmallVector<int, 16> &OtherValNoAssignments,
                                  LiveInterval &ThisLI, LiveInterval &OtherLI) {
  // If the VN has already been computed, just return it.
  if (ThisValNoAssignments[VN] >= 0)
    return ThisValNoAssignments[VN];
//  assert(ThisValNoAssignments[VN] != -2 && "Cyclic case?");
  
  // If this val is not a copy from the other val, then it must be a new value
  // number in the destination.
  int OtherValNo = ThisFromOther[VN];
  if (OtherValNo == -1) {
    ValueNumberInfo.push_back(ThisLI.getValNumInfo(VN));
    return ThisValNoAssignments[VN] = ValueNumberInfo.size()-1;
  }

  // Otherwise, this *is* a copy from the RHS.  If the other side has already
  // been computed, return it.
  if (OtherValNoAssignments[OtherValNo] >= 0)
    return ThisValNoAssignments[VN] = OtherValNoAssignments[OtherValNo];
  
  // Mark this value number as currently being computed, then ask what the
  // ultimate value # of the other value is.
  ThisValNoAssignments[VN] = -2;
  unsigned UltimateVN =
    ComputeUltimateVN(OtherValNo, ValueNumberInfo,
                      OtherFromThis, ThisFromOther,
                      OtherValNoAssignments, ThisValNoAssignments,
                      OtherLI, ThisLI);
  return ThisValNoAssignments[VN] = UltimateVN;
}

static bool InVector(unsigned Val, const SmallVector<unsigned, 8> &V) {
  return std::find(V.begin(), V.end(), Val) != V.end();
}

/// SimpleJoin - Attempt to joint the specified interval into this one. The
/// caller of this method must guarantee that the RHS only contains a single
/// value number and that the RHS is not defined by a copy from this
/// interval.  This returns false if the intervals are not joinable, or it
/// joins them and returns true.
bool SimpleRegisterCoalescing::SimpleJoin(LiveInterval &LHS, LiveInterval &RHS) {
  assert(RHS.containsOneValue());
  
  // Some number (potentially more than one) value numbers in the current
  // interval may be defined as copies from the RHS.  Scan the overlapping
  // portions of the LHS and RHS, keeping track of this and looking for
  // overlapping live ranges that are NOT defined as copies.  If these exist, we
  // cannot coalesce.
  
  LiveInterval::iterator LHSIt = LHS.begin(), LHSEnd = LHS.end();
  LiveInterval::iterator RHSIt = RHS.begin(), RHSEnd = RHS.end();
  
  if (LHSIt->start < RHSIt->start) {
    LHSIt = std::upper_bound(LHSIt, LHSEnd, RHSIt->start);
    if (LHSIt != LHS.begin()) --LHSIt;
  } else if (RHSIt->start < LHSIt->start) {
    RHSIt = std::upper_bound(RHSIt, RHSEnd, LHSIt->start);
    if (RHSIt != RHS.begin()) --RHSIt;
  }
  
  SmallVector<unsigned, 8> EliminatedLHSVals;
  
  while (1) {
    // Determine if these live intervals overlap.
    bool Overlaps = false;
    if (LHSIt->start <= RHSIt->start)
      Overlaps = LHSIt->end > RHSIt->start;
    else
      Overlaps = RHSIt->end > LHSIt->start;
    
    // If the live intervals overlap, there are two interesting cases: if the
    // LHS interval is defined by a copy from the RHS, it's ok and we record
    // that the LHS value # is the same as the RHS.  If it's not, then we cannot
    // coalesce these live ranges and we bail out.
    if (Overlaps) {
      // If we haven't already recorded that this value # is safe, check it.
      if (!InVector(LHSIt->ValId, EliminatedLHSVals)) {
        // Copy from the RHS?
        unsigned SrcReg = LHS.getSrcRegForValNum(LHSIt->ValId);
        if (rep(SrcReg) != RHS.reg)
          return false;    // Nope, bail out.
        
        EliminatedLHSVals.push_back(LHSIt->ValId);
      }
      
      // We know this entire LHS live range is okay, so skip it now.
      if (++LHSIt == LHSEnd) break;
      continue;
    }
    
    if (LHSIt->end < RHSIt->end) {
      if (++LHSIt == LHSEnd) break;
    } else {
      // One interesting case to check here.  It's possible that we have
      // something like "X3 = Y" which defines a new value number in the LHS,
      // and is the last use of this liverange of the RHS.  In this case, we
      // want to notice this copy (so that it gets coalesced away) even though
      // the live ranges don't actually overlap.
      if (LHSIt->start == RHSIt->end) {
        if (InVector(LHSIt->ValId, EliminatedLHSVals)) {
          // We already know that this value number is going to be merged in
          // if coalescing succeeds.  Just skip the liverange.
          if (++LHSIt == LHSEnd) break;
        } else {
          // Otherwise, if this is a copy from the RHS, mark it as being merged
          // in.
          if (rep(LHS.getSrcRegForValNum(LHSIt->ValId)) == RHS.reg) {
            EliminatedLHSVals.push_back(LHSIt->ValId);

            // We know this entire LHS live range is okay, so skip it now.
            if (++LHSIt == LHSEnd) break;
          }
        }
      }
      
      if (++RHSIt == RHSEnd) break;
    }
  }
  
  // If we got here, we know that the coalescing will be successful and that
  // the value numbers in EliminatedLHSVals will all be merged together.  Since
  // the most common case is that EliminatedLHSVals has a single number, we
  // optimize for it: if there is more than one value, we merge them all into
  // the lowest numbered one, then handle the interval as if we were merging
  // with one value number.
  unsigned LHSValNo;
  if (EliminatedLHSVals.size() > 1) {
    // Loop through all the equal value numbers merging them into the smallest
    // one.
    unsigned Smallest = EliminatedLHSVals[0];
    for (unsigned i = 1, e = EliminatedLHSVals.size(); i != e; ++i) {
      if (EliminatedLHSVals[i] < Smallest) {
        // Merge the current notion of the smallest into the smaller one.
        LHS.MergeValueNumberInto(Smallest, EliminatedLHSVals[i]);
        Smallest = EliminatedLHSVals[i];
      } else {
        // Merge into the smallest.
        LHS.MergeValueNumberInto(EliminatedLHSVals[i], Smallest);
      }
    }
    LHSValNo = Smallest;
  } else {
    assert(!EliminatedLHSVals.empty() && "No copies from the RHS?");
    LHSValNo = EliminatedLHSVals[0];
  }
  
  // Okay, now that there is a single LHS value number that we're merging the
  // RHS into, update the value number info for the LHS to indicate that the
  // value number is defined where the RHS value number was.
  LHS.setValueNumberInfo(LHSValNo, RHS.getValNumInfo(0));
  
  // Okay, the final step is to loop over the RHS live intervals, adding them to
  // the LHS.
  LHS.MergeRangesInAsValue(RHS, LHSValNo);
  LHS.weight += RHS.weight;
  if (RHS.preference && !LHS.preference)
    LHS.preference = RHS.preference;
  
  return true;
}

/// JoinIntervals - Attempt to join these two intervals.  On failure, this
/// returns false.  Otherwise, if one of the intervals being joined is a
/// physreg, this method always canonicalizes LHS to be it.  The output
/// "RHS" will not have been modified, so we can use this information
/// below to update aliases.
bool SimpleRegisterCoalescing::JoinIntervals(LiveInterval &LHS, LiveInterval &RHS) {
  // Compute the final value assignment, assuming that the live ranges can be
  // coalesced.
  SmallVector<int, 16> LHSValNoAssignments;
  SmallVector<int, 16> RHSValNoAssignments;
  SmallVector<std::pair<unsigned,unsigned>, 16> ValueNumberInfo;
                          
  // If a live interval is a physical register, conservatively check if any
  // of its sub-registers is overlapping the live interval of the virtual
  // register. If so, do not coalesce.
  if (MRegisterInfo::isPhysicalRegister(LHS.reg) &&
      *mri_->getSubRegisters(LHS.reg)) {
    for (const unsigned* SR = mri_->getSubRegisters(LHS.reg); *SR; ++SR)
      if (li_->hasInterval(*SR) && RHS.overlaps(li_->getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(li_->getInterval(*SR).print(DOUT, mri_));
        return false;
      }
  } else if (MRegisterInfo::isPhysicalRegister(RHS.reg) &&
             *mri_->getSubRegisters(RHS.reg)) {
    for (const unsigned* SR = mri_->getSubRegisters(RHS.reg); *SR; ++SR)
      if (li_->hasInterval(*SR) && LHS.overlaps(li_->getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(li_->getInterval(*SR).print(DOUT, mri_));
        return false;
      }
  }
                          
  // Compute ultimate value numbers for the LHS and RHS values.
  if (RHS.containsOneValue()) {
    // Copies from a liveinterval with a single value are simple to handle and
    // very common, handle the special case here.  This is important, because
    // often RHS is small and LHS is large (e.g. a physreg).
    
    // Find out if the RHS is defined as a copy from some value in the LHS.
    int RHSValID = -1;
    std::pair<unsigned,unsigned> RHSValNoInfo;
    unsigned RHSSrcReg = RHS.getSrcRegForValNum(0);
    if ((RHSSrcReg == 0 || rep(RHSSrcReg) != LHS.reg)) {
      // If RHS is not defined as a copy from the LHS, we can use simpler and
      // faster checks to see if the live ranges are coalescable.  This joiner
      // can't swap the LHS/RHS intervals though.
      if (!MRegisterInfo::isPhysicalRegister(RHS.reg)) {
        return SimpleJoin(LHS, RHS);
      } else {
        RHSValNoInfo = RHS.getValNumInfo(0);
      }
    } else {
      // It was defined as a copy from the LHS, find out what value # it is.
      unsigned ValInst = RHS.getInstForValNum(0);
      RHSValID = LHS.getLiveRangeContaining(ValInst-1)->ValId;
      RHSValNoInfo = LHS.getValNumInfo(RHSValID);
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    ValueNumberInfo.resize(LHS.getNumValNums());
    
    // Okay, *all* of the values in LHS that are defined as a copy from RHS
    // should now get updated.
    for (unsigned VN = 0, e = LHS.getNumValNums(); VN != e; ++VN) {
      if (unsigned LHSSrcReg = LHS.getSrcRegForValNum(VN)) {
        if (rep(LHSSrcReg) != RHS.reg) {
          // If this is not a copy from the RHS, its value number will be
          // unmodified by the coalescing.
          ValueNumberInfo[VN] = LHS.getValNumInfo(VN);
          LHSValNoAssignments[VN] = VN;
        } else if (RHSValID == -1) {
          // Otherwise, it is a copy from the RHS, and we don't already have a
          // value# for it.  Keep the current value number, but remember it.
          LHSValNoAssignments[VN] = RHSValID = VN;
          ValueNumberInfo[VN] = RHSValNoInfo;
        } else {
          // Otherwise, use the specified value #.
          LHSValNoAssignments[VN] = RHSValID;
          if (VN != (unsigned)RHSValID)
            ValueNumberInfo[VN].first = ~1U;
          else
            ValueNumberInfo[VN] = RHSValNoInfo;
        }
      } else {
        ValueNumberInfo[VN] = LHS.getValNumInfo(VN);
        LHSValNoAssignments[VN] = VN;
      }
    }
    
    assert(RHSValID != -1 && "Didn't find value #?");
    RHSValNoAssignments[0] = RHSValID;
    
  } else {
    // Loop over the value numbers of the LHS, seeing if any are defined from
    // the RHS.
    SmallVector<int, 16> LHSValsDefinedFromRHS;
    LHSValsDefinedFromRHS.resize(LHS.getNumValNums(), -1);
    for (unsigned VN = 0, e = LHS.getNumValNums(); VN != e; ++VN) {
      unsigned ValSrcReg = LHS.getSrcRegForValNum(VN);
      if (ValSrcReg == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the LHS interval.  If the src is
      // from the RHS interval, we can use its value #.
      if (rep(ValSrcReg) != RHS.reg)
        continue;
      
      // Figure out the value # from the RHS.
      unsigned ValInst = LHS.getInstForValNum(VN);
      LHSValsDefinedFromRHS[VN] = RHS.getLiveRangeContaining(ValInst-1)->ValId;
    }
    
    // Loop over the value numbers of the RHS, seeing if any are defined from
    // the LHS.
    SmallVector<int, 16> RHSValsDefinedFromLHS;
    RHSValsDefinedFromLHS.resize(RHS.getNumValNums(), -1);
    for (unsigned VN = 0, e = RHS.getNumValNums(); VN != e; ++VN) {
      unsigned ValSrcReg = RHS.getSrcRegForValNum(VN);
      if (ValSrcReg == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the RHS interval.  If the src is
      // from the LHS interval, we can use its value #.
      if (rep(ValSrcReg) != LHS.reg)
        continue;
      
      // Figure out the value # from the LHS.
      unsigned ValInst = RHS.getInstForValNum(VN);
      RHSValsDefinedFromLHS[VN] = LHS.getLiveRangeContaining(ValInst-1)->ValId;
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    ValueNumberInfo.reserve(LHS.getNumValNums() + RHS.getNumValNums());
    
    for (unsigned VN = 0, e = LHS.getNumValNums(); VN != e; ++VN) {
      if (LHSValNoAssignments[VN] >= 0 || LHS.getInstForValNum(VN) == ~2U) 
        continue;
      ComputeUltimateVN(VN, ValueNumberInfo,
                        LHSValsDefinedFromRHS, RHSValsDefinedFromLHS,
                        LHSValNoAssignments, RHSValNoAssignments, LHS, RHS);
    }
    for (unsigned VN = 0, e = RHS.getNumValNums(); VN != e; ++VN) {
      if (RHSValNoAssignments[VN] >= 0 || RHS.getInstForValNum(VN) == ~2U)
        continue;
      // If this value number isn't a copy from the LHS, it's a new number.
      if (RHSValsDefinedFromLHS[VN] == -1) {
        ValueNumberInfo.push_back(RHS.getValNumInfo(VN));
        RHSValNoAssignments[VN] = ValueNumberInfo.size()-1;
        continue;
      }
      
      ComputeUltimateVN(VN, ValueNumberInfo,
                        RHSValsDefinedFromLHS, LHSValsDefinedFromRHS,
                        RHSValNoAssignments, LHSValNoAssignments, RHS, LHS);
    }
  }
  
  // Armed with the mappings of LHS/RHS values to ultimate values, walk the
  // interval lists to see if these intervals are coalescable.
  LiveInterval::const_iterator I = LHS.begin();
  LiveInterval::const_iterator IE = LHS.end();
  LiveInterval::const_iterator J = RHS.begin();
  LiveInterval::const_iterator JE = RHS.end();
  
  // Skip ahead until the first place of potential sharing.
  if (I->start < J->start) {
    I = std::upper_bound(I, IE, J->start);
    if (I != LHS.begin()) --I;
  } else if (J->start < I->start) {
    J = std::upper_bound(J, JE, I->start);
    if (J != RHS.begin()) --J;
  }
  
  while (1) {
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
      if (LHSValNoAssignments[I->ValId] != RHSValNoAssignments[J->ValId])
        return false;
    }
    
    if (I->end < J->end) {
      ++I;
      if (I == IE) break;
    } else {
      ++J;
      if (J == JE) break;
    }
  }

  // If we get here, we know that we can coalesce the live ranges.  Ask the
  // intervals to coalesce themselves now.
  LHS.join(RHS, &LHSValNoAssignments[0], &RHSValNoAssignments[0],
           ValueNumberInfo);
  return true;
}

namespace {
  // DepthMBBCompare - Comparison predicate that sort first based on the loop
  // depth of the basic block (the unsigned), and then on the MBB number.
  struct DepthMBBCompare {
    typedef std::pair<unsigned, MachineBasicBlock*> DepthMBBPair;
    bool operator()(const DepthMBBPair &LHS, const DepthMBBPair &RHS) const {
      if (LHS.first > RHS.first) return true;   // Deeper loops first
      return LHS.first == RHS.first &&
        LHS.second->getNumber() < RHS.second->getNumber();
    }
  };
}

void SimpleRegisterCoalescing::CopyCoalesceInMBB(MachineBasicBlock *MBB,
                                std::vector<CopyRec> *TryAgain, bool PhysOnly) {
  DOUT << ((Value*)MBB->getBasicBlock())->getName() << ":\n";
  
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E;) {
    MachineInstr *Inst = MII++;
    
    // If this isn't a copy, we can't join intervals.
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*Inst, SrcReg, DstReg)) continue;
    
    if (TryAgain && !JoinCopy(Inst, SrcReg, DstReg, PhysOnly))
      TryAgain->push_back(getCopyRec(Inst, SrcReg, DstReg));
  }
}

void SimpleRegisterCoalescing::joinIntervals() {
  DOUT << "********** JOINING INTERVALS ***********\n";

  JoinedLIs.resize(li_->getNumIntervals());
  JoinedLIs.reset();

  std::vector<CopyRec> TryAgainList;
  const LoopInfo &LI = getAnalysis<LoopInfo>();
  if (LI.begin() == LI.end()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      CopyCoalesceInMBB(I, &TryAgainList);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.

    // Join intervals in the function prolog first. We want to join physical
    // registers with virtual registers before the intervals got too long.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end(); I != E;++I)
      MBBs.push_back(std::make_pair(LI.getLoopDepth(I->getBasicBlock()), I));

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order.
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      CopyCoalesceInMBB(MBBs[i].second, NULL, true);
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      CopyCoalesceInMBB(MBBs[i].second, &TryAgainList, false);
  }
  
  // Joining intervals can allow other intervals to be joined.  Iteratively join
  // until we make no progress.
  bool ProgressMade = true;
  while (ProgressMade) {
    ProgressMade = false;

    for (unsigned i = 0, e = TryAgainList.size(); i != e; ++i) {
      CopyRec &TheCopy = TryAgainList[i];
      if (TheCopy.MI &&
          JoinCopy(TheCopy.MI, TheCopy.SrcReg, TheCopy.DstReg)) {
        TheCopy.MI = 0;   // Mark this one as done.
        ProgressMade = true;
      }
    }
  }

  // Some live range has been lengthened due to colaescing, eliminate the
  // unnecessary kills.
  int RegNum = JoinedLIs.find_first();
  while (RegNum != -1) {
    unsigned Reg = RegNum + MRegisterInfo::FirstVirtualRegister;
    unsigned repReg = rep(Reg);
    LiveInterval &LI = li_->getInterval(repReg);
    LiveVariables::VarInfo& svi = lv_->getVarInfo(Reg);
    for (unsigned i = 0, e = svi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = svi.Kills[i];
      // Suppose vr1 = op vr2, x
      // and vr1 and vr2 are coalesced. vr2 should still be marked kill
      // unless it is a two-address operand.
      if (li_->isRemoved(Kill) || hasRegisterDef(Kill, repReg))
        continue;
      if (LI.liveAt(li_->getInstructionIndex(Kill) + InstrSlots::NUM))
        unsetRegisterKill(Kill, repReg);
    }
    RegNum = JoinedLIs.find_next(RegNum);
  }
  
  DOUT << "*** Register mapping ***\n";
  for (int i = 0, e = r2rMap_.size(); i != e; ++i)
    if (r2rMap_[i]) {
      DOUT << "  reg " << i << " -> ";
      DEBUG(printRegName(r2rMap_[i]));
      DOUT << "\n";
    }
}

/// Return true if the two specified registers belong to different register
/// classes.  The registers may be either phys or virt regs.
bool SimpleRegisterCoalescing::differingRegisterClasses(unsigned RegA,
                                             unsigned RegB) const {

  // Get the register classes for the first reg.
  if (MRegisterInfo::isPhysicalRegister(RegA)) {
    assert(MRegisterInfo::isVirtualRegister(RegB) &&
           "Shouldn't consider two physregs!");
    return !mf_->getSSARegMap()->getRegClass(RegB)->contains(RegA);
  }

  // Compare against the regclass for the second reg.
  const TargetRegisterClass *RegClass = mf_->getSSARegMap()->getRegClass(RegA);
  if (MRegisterInfo::isVirtualRegister(RegB))
    return RegClass != mf_->getSSARegMap()->getRegClass(RegB);
  else
    return !RegClass->contains(RegB);
}

/// lastRegisterUse - Returns the last use of the specific register between
/// cycles Start and End. It also returns the use operand by reference. It
/// returns NULL if there are no uses.
MachineInstr *
SimpleRegisterCoalescing::lastRegisterUse(unsigned Start, unsigned End, unsigned Reg,
                               MachineOperand *&MOU) {
  int e = (End-1) / InstrSlots::NUM * InstrSlots::NUM;
  int s = Start;
  while (e >= s) {
    // Skip deleted instructions
    MachineInstr *MI = li_->getInstructionFromIndex(e);
    while ((e - InstrSlots::NUM) >= s && !MI) {
      e -= InstrSlots::NUM;
      MI = li_->getInstructionFromIndex(e);
    }
    if (e < s || MI == NULL)
      return NULL;

    for (unsigned i = 0, NumOps = MI->getNumOperands(); i != NumOps; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() && MO.getReg() &&
          mri_->regsOverlap(rep(MO.getReg()), Reg)) {
        MOU = &MO;
        return MI;
      }
    }

    e -= InstrSlots::NUM;
  }

  return NULL;
}


/// findDefOperand - Returns the MachineOperand that is a def of the specific
/// register. It returns NULL if the def is not found.
MachineOperand *SimpleRegisterCoalescing::findDefOperand(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        mri_->regsOverlap(rep(MO.getReg()), Reg))
      return &MO;
  }
  return NULL;
}

/// unsetRegisterKill - Unset IsKill property of all uses of specific register
/// of the specific instruction.
void SimpleRegisterCoalescing::unsetRegisterKill(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isKill() && MO.getReg() &&
        mri_->regsOverlap(rep(MO.getReg()), Reg))
      MO.unsetIsKill();
  }
}

/// unsetRegisterKills - Unset IsKill property of all uses of specific register
/// between cycles Start and End.
void SimpleRegisterCoalescing::unsetRegisterKills(unsigned Start, unsigned End,
                                       unsigned Reg) {
  int e = (End-1) / InstrSlots::NUM * InstrSlots::NUM;
  int s = Start;
  while (e >= s) {
    // Skip deleted instructions
    MachineInstr *MI = li_->getInstructionFromIndex(e);
    while ((e - InstrSlots::NUM) >= s && !MI) {
      e -= InstrSlots::NUM;
      MI = li_->getInstructionFromIndex(e);
    }
    if (e < s || MI == NULL)
      return;

    for (unsigned i = 0, NumOps = MI->getNumOperands(); i != NumOps; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isKill() && MO.getReg() &&
          mri_->regsOverlap(rep(MO.getReg()), Reg)) {
        MO.unsetIsKill();
      }
    }

    e -= InstrSlots::NUM;
  }
}

/// hasRegisterDef - True if the instruction defines the specific register.
///
bool SimpleRegisterCoalescing::hasRegisterDef(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        mri_->regsOverlap(rep(MO.getReg()), Reg))
      return true;
  }
  return false;
}

void SimpleRegisterCoalescing::printRegName(unsigned reg) const {
  if (MRegisterInfo::isPhysicalRegister(reg))
    cerr << mri_->getName(reg);
  else
    cerr << "%reg" << reg;
}

void SimpleRegisterCoalescing::releaseMemory() {
   r2rMap_.clear();
   JoinedLIs.clear();
}

static bool isZeroLengthInterval(LiveInterval *li) {
  for (LiveInterval::Ranges::const_iterator
         i = li->ranges.begin(), e = li->ranges.end(); i != e; ++i)
    if (i->end - i->start > LiveIntervals::InstrSlots::NUM)
      return false;
  return true;
}

bool SimpleRegisterCoalescing::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  tm_ = &fn.getTarget();
  mri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  li_ = &getAnalysis<LiveIntervals>();
  lv_ = &getAnalysis<LiveVariables>();

  DOUT << "********** SIMPLE REGISTER COALESCING **********\n"
       << "********** Function: "
       << ((Value*)mf_->getFunction())->getName() << '\n';

  allocatableRegs_ = mri_->getAllocatableSet(fn);
  for (MRegisterInfo::regclass_iterator I = mri_->regclass_begin(),
         E = mri_->regclass_end(); I != E; ++I)
    allocatableRCRegs_.insert(std::make_pair(*I,mri_->getAllocatableSet(fn, *I)));

  r2rMap_.grow(mf_->getSSARegMap()->getLastVirtReg());

  // Join (coalesce) intervals if requested.
  if (EnableJoining) {
    joinIntervals();
    DOUT << "********** INTERVALS POST JOINING **********\n";
    for (LiveIntervals::iterator I = li_->begin(), E = li_->end(); I != E; ++I) {
      I->second.print(DOUT, mri_);
      DOUT << "\n";
    }
  }

  // perform a final pass over the instructions and compute spill
  // weights, coalesce virtual registers and remove identity moves.
  const LoopInfo &loopInfo = getAnalysis<LoopInfo>();

  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    MachineBasicBlock* mbb = mbbi;
    unsigned loopDepth = loopInfo.getLoopDepth(mbb->getBasicBlock());

    for (MachineBasicBlock::iterator mii = mbb->begin(), mie = mbb->end();
         mii != mie; ) {
      // if the move will be an identity move delete it
      unsigned srcReg, dstReg, RegRep;
      if (tii_->isMoveInstr(*mii, srcReg, dstReg) &&
          (RegRep = rep(srcReg)) == rep(dstReg)) {
        // remove from def list
        LiveInterval &RegInt = li_->getOrCreateInterval(RegRep);
        MachineOperand *MO = mii->findRegisterDefOperand(dstReg);
        // If def of this move instruction is dead, remove its live range from
        // the dstination register's live interval.
        if (MO->isDead()) {
          unsigned MoveIdx = li_->getDefIndex(li_->getInstructionIndex(mii));
          LiveInterval::iterator MLR = RegInt.FindLiveRangeContaining(MoveIdx);
          RegInt.removeRange(MLR->start, MoveIdx+1);
          if (RegInt.empty())
            li_->removeInterval(RegRep);
        }
        li_->RemoveMachineInstrFromMaps(mii);
        mii = mbbi->erase(mii);
        ++numPeep;
      } else {
        SmallSet<unsigned, 4> UniqueUses;
        for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
          const MachineOperand &mop = mii->getOperand(i);
          if (mop.isRegister() && mop.getReg() &&
              MRegisterInfo::isVirtualRegister(mop.getReg())) {
            // replace register with representative register
            unsigned reg = rep(mop.getReg());
            mii->getOperand(i).setReg(reg);

            // Multiple uses of reg by the same instruction. It should not
            // contribute to spill weight again.
            if (UniqueUses.count(reg) != 0)
              continue;
            LiveInterval &RegInt = li_->getInterval(reg);
            float w = (mop.isUse()+mop.isDef()) * powf(10.0F, (float)loopDepth);
            // If the definition instruction is re-materializable, its spill
            // weight is half of what it would have been normally unless it's
            // a load from fixed stack slot.
            int Dummy;
            if (RegInt.remat && !tii_->isLoadFromStackSlot(RegInt.remat, Dummy))
              w /= 2;
            RegInt.weight += w;
            UniqueUses.insert(reg);
          }
        }
        ++mii;
      }
    }
  }

  for (LiveIntervals::iterator I = li_->begin(), E = li_->end(); I != E; ++I) {
    LiveInterval &LI = I->second;
    if (MRegisterInfo::isVirtualRegister(LI.reg)) {
      // If the live interval length is essentially zero, i.e. in every live
      // range the use follows def immediately, it doesn't make sense to spill
      // it and hope it will be easier to allocate for this li.
      if (isZeroLengthInterval(&LI))
        LI.weight = HUGE_VALF;

      // Slightly prefer live interval that has been assigned a preferred reg.
      if (LI.preference)
        LI.weight *= 1.01F;

      // Divide the weight of the interval by its size.  This encourages 
      // spilling of intervals that are large and have few uses, and
      // discourages spilling of small intervals with many uses.
      LI.weight /= LI.getSize();
    }
  }

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void SimpleRegisterCoalescing::print(std::ostream &O, const Module* m) const {
   li_->print(O, m);
}
