//===- llvm/CodeGen/GlobalISel/RegBankSelect.cpp - RegBankSelect -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the RegBankSelect class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define DEBUG_TYPE "regbankselect"

using namespace llvm;

char RegBankSelect::ID = 0;
INITIALIZE_PASS(RegBankSelect, "regbankselect",
                "Assign register bank of generic virtual registers",
                false, false);

RegBankSelect::RegBankSelect()
    : MachineFunctionPass(ID), RBI(nullptr), MRI(nullptr) {
  initializeRegBankSelectPass(*PassRegistry::getPassRegistry());
}

void RegBankSelect::init(MachineFunction &MF) {
  RBI = MF.getSubtarget().getRegBankInfo();
  assert(RBI && "Cannot work without RegisterBankInfo");
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MIRBuilder.setMF(MF);
}

bool RegBankSelect::assignmentMatch(
    unsigned Reg, const RegisterBankInfo::ValueMapping &ValMapping,
    bool &OnlyAssign) const {
  // By default we assume we will have to repair something.
  OnlyAssign = false;
  // Each part of a break down needs to end up in a different register.
  // In other word, Reg assignement does not match.
  if (ValMapping.BreakDown.size() > 1)
    return false;

  const RegisterBank *CurRegBank = RBI->getRegBank(Reg, *MRI, *TRI);
  const RegisterBank *DesiredRegBrank = ValMapping.BreakDown[0].RegBank;
  // Reg is free of assignment, a simple assignment will make the
  // register bank to match.
  OnlyAssign = CurRegBank == nullptr;
  DEBUG(dbgs() << "Does assignment already match: ";
        if (CurRegBank) dbgs() << *CurRegBank; else dbgs() << "none";
        dbgs() << " against ";
        assert(DesiredRegBrank && "The mapping must be valid");
        dbgs() << *DesiredRegBrank << '\n';);
  return CurRegBank == DesiredRegBrank;
}

unsigned
RegBankSelect::repairReg(unsigned Reg,
                         const RegisterBankInfo::ValueMapping &ValMapping,
                         MachineInstr &DefUseMI, bool IsDef) {
  assert(ValMapping.BreakDown.size() == 1 &&
         "Support for complex break down not supported yet");
  const RegisterBankInfo::PartialMapping &PartialMap = ValMapping.BreakDown[0];
  assert(PartialMap.Length ==
             (TargetRegisterInfo::isPhysicalRegister(Reg)
                  ? TRI->getMinimalPhysRegClass(Reg)->getSize() * 8
                  : MRI->getSize(Reg)) &&
         "Repairing other than copy not implemented yet");
  // If the MIRBuilder is configured to insert somewhere else than
  // DefUseMI, we may not use this function like was it first
  // internded (local repairing), so make sure we pay attention before
  // we remove the assert.
  // In particular, it is likely that we will have to properly save
  // the insertion point of the MIRBuilder and restore it at the end
  // of this method.
  assert(&DefUseMI == &(*MIRBuilder.getInsertPt()) &&
         "Need to save and restore the insertion point");
  // For use, we will add a copy just in front of the instruction.
  // For def, we will add a copy just after the instruction.
  // In either case, the insertion point must be valid. In particular,
  // make sure we do not insert in the middle of terminators or phis.
  bool Before = !IsDef;
  setSafeInsertionPoint(DefUseMI, Before);
  if (DefUseMI.isTerminator() && Before) {
    // Check that the insertion point does not happen
    // before the definition of Reg.
    // This can happen if Reg is defined by a terminator
    // and used by another one.
    // In that case the repairing code is actually more involved
    // because we have to split the block.

    // Assert that this is not a physical register.
    // The target independent code does not insert physical registers
    // on terminators, so if we end up in this situation, this is
    // likely a bug in the target.
    assert(!TargetRegisterInfo::isPhysicalRegister(Reg) &&
           "Check for physical register not implemented");
    const MachineInstr *RegDef = MRI->getVRegDef(Reg);
    assert(RegDef && "Reg has more than one definition?");
    // Assert to make the code more readable; Reg is used by DefUseMI, i.e.,
    // (Before == !IsDef == true), so DefUseMI != RegDef otherwise we have
    // a use (that is not a PHI) that is not dominated by its def.
    assert(&DefUseMI != RegDef && "Def does not dominate all of its uses");
    if (RegDef->isTerminator() && RegDef->getParent() == DefUseMI.getParent())
      // By construction, the repairing should happen between two
      // terminators: RegDef and DefUseMI.
      // This is not implemented.
      report_fatal_error("Repairing between terminators not implemented yet");
  }

  // Create a new temporary to hold the repaired value.
  unsigned NewReg = MRI->createGenericVirtualRegister(PartialMap.Length);
  // Set the registers for the source and destination of the copy.
  unsigned Src = Reg, Dst = NewReg;
  // If this is a definition that we repair, the copy will be
  // inverted.
  if (IsDef)
    std::swap(Src, Dst);
  (void)MIRBuilder.buildInstr(TargetOpcode::COPY, Dst, Src);

  DEBUG(dbgs() << "Repair: " << PrintReg(Reg) << " with: "
        << PrintReg(NewReg) << '\n');

  // Restore the insertion point of the MIRBuilder.
  MIRBuilder.setInstr(DefUseMI, Before);
  return NewReg;
}

void RegBankSelect::setSafeInsertionPoint(MachineInstr &InsertPt, bool Before) {
  // Check that we are not looking to insert before a phi.
  // Indeed, we would need more information on what to do.
  // By default that should be all the predecessors, but this is
  // probably not what we want in general.
  assert((!Before || !InsertPt.isPHI()) &&
         "Insertion before phis not implemented");
  // The same kind of observation hold for terminators if we try to
  // insert after them.
  assert((Before || !InsertPt.isTerminator()) &&
         "Insertion after terminatos not implemented");
  if (InsertPt.isPHI()) {
    assert(!Before && "Not supported!!");
    MachineBasicBlock *MBB = InsertPt.getParent();
    assert(MBB && "Insertion point is not in a basic block");
    MachineBasicBlock::iterator FirstNonPHIPt = MBB->getFirstNonPHI();
    if (FirstNonPHIPt == MBB->end()) {
      // If there is not any non-phi instruction, insert at the end of MBB.
      MIRBuilder.setMBB(*MBB, /*Beginning*/ false);
      return;
    }
    // The insertion point before the first non-phi instruction.
    MIRBuilder.setInstr(*FirstNonPHIPt, /*Before*/ true);
    return;
  }
  if (InsertPt.isTerminator()) {
    MachineBasicBlock *MBB = InsertPt.getParent();
    assert(MBB && "Insertion point is not in a basic block");
    MIRBuilder.setInstr(*MBB->getFirstTerminator(), /*Before*/ true);
    return;
  }
  MIRBuilder.setInstr(InsertPt, /*Before*/ Before);
}

void RegBankSelect::assignInstr(MachineInstr &MI) {
  DEBUG(dbgs() << "Assign: " << MI);
  const RegisterBankInfo::InstructionMapping DefaultMapping =
      RBI->getInstrMapping(MI);
  // Make sure the mapping is valid for MI.
  assert(DefaultMapping.verify(MI) && "Invalid instruction mapping");

  DEBUG(dbgs() << "Mapping: " << DefaultMapping << '\n');

  // Set the insertion point before MI.
  // This is where we are going to insert the repairing code if any.
  MIRBuilder.setInstr(MI, /*Before*/ true);

  // For now, do not look for alternative mappings.
  // Alternative mapping may require to rewrite MI and we do not support
  // that yet.
  // Walk the operands and assign then to the chosen mapping, possibly with
  // the insertion of repair code for uses.
  for (unsigned OpIdx = 0, EndIdx = MI.getNumOperands(); OpIdx != EndIdx;
       ++OpIdx) {
    MachineOperand &MO = MI.getOperand(OpIdx);
    // Nothing to be done for non-register operands.
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;

    const RegisterBankInfo::ValueMapping &ValMapping =
        DefaultMapping.getOperandMapping(OpIdx);
    // If Reg is already properly mapped, move on.
    bool OnlyAssign;
    if (assignmentMatch(Reg, ValMapping, OnlyAssign))
      continue;

    // For uses, we may need to create a new temporary.
    // Indeed, if Reg is already assigned a register bank, at this
    // point, we know it is different from the one defined by the
    // chosen mapping, we need to adjust for that.
    // For definitions, changing the register bank will affect all
    // its uses, and in particular the ones we already visited.
    // Although this is correct, since with the RPO traversal of the
    // basic blocks the only uses that we already visisted for this
    // definition are PHIs (i.e., copies), this may not be the best
    // solution according to the cost model.
    // Therefore, create a new temporary for Reg.
    assert(ValMapping.BreakDown.size() == 1 &&
           "Support for complex break down not supported yet");
    if (!OnlyAssign) {
      if (!MO.isDef() && MI.isPHI()) {
        // Phis are already copies, so there is nothing to repair.
        // Note: This will not hold when we support break downs with
        // more than one segment.
        DEBUG(dbgs() << "Skip PHI use\n");
        continue;
      }
      // If MO is a definition, since repairing after a terminator is
      // painful, do not repair. Indeed, this is probably not worse
      // saving the move in the PHIs that will get reassigned.
      if (!MO.isDef() || !MI.isTerminator())
        Reg = repairReg(Reg, ValMapping, MI, MO.isDef());
    }

    // If we end up here, MO should be free of encoding constraints,
    // i.e., we do not have to constrained the RegBank of Reg to
    // the requirement of the operands.
    // If that is not the case, this means the code was broken before
    // hands because we should have found that the assignment match.
    // This will not hold when we will consider alternative mappings.
    DEBUG(dbgs() << "Assign: " << *ValMapping.BreakDown[0].RegBank << " to "
                 << PrintReg(Reg) << '\n');

    MRI->setRegBank(Reg, *ValMapping.BreakDown[0].RegBank);
    MO.setReg(Reg);
  }
  DEBUG(dbgs() << "Assigned: " << MI);
}

bool RegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "Assign register banks for: " << MF.getName() << '\n');
  init(MF);
  // Walk the function and assign register banks to all operands.
  // Use a RPOT to make sure all registers are assigned before we choose
  // the best mapping of the current instruction.
  ReversePostOrderTraversal<MachineFunction*> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    for (MachineInstr &MI : *MBB)
      assignInstr(MI);
  return false;
}

//------------------------------------------------------------------------------
//                  Helper Classes Implementation
//------------------------------------------------------------------------------
RegBankSelect::RepairingPlacement::RepairingPlacement(
    MachineInstr &MI, unsigned OpIdx, const TargetRegisterInfo &TRI, Pass &P,
    RepairingPlacement::RepairingKind Kind)
    // Default is, we are going to insert code to repair OpIdx.
    : Kind(Kind),
      OpIdx(OpIdx),
      CanMaterialize(Kind != RepairingKind::Impossible),
      HasSplit(false),
      P(P) {
  const MachineOperand &MO = MI.getOperand(OpIdx);
  assert(MO.isReg() && "Trying to repair a non-reg operand");

  if (Kind != RepairingKind::Insert)
    return;

  // Repairings for definitions happen after MI, uses happen before.
  bool Before = !MO.isDef();

  // Check if we are done with MI.
  if (!MI.isPHI() && !MI.isTerminator()) {
    addInsertPoint(MI, Before);
    // We are done with the initialization.
    return;
  }

  // Now, look for the special cases.
  if (MI.isPHI()) {
    // - PHI must be the first instructions:
    //   * Before, we have to split the related incoming edge.
    //   * After, move the insertion point past the last phi.
    if (!Before) {
      MachineBasicBlock::iterator It = MI.getParent()->getFirstNonPHI();
      if (It != MI.getParent()->end())
        addInsertPoint(*It, /*Before*/ true);
      else
        addInsertPoint(*(--It), /*Before*/ false);
      return;
    }
    // We repair a use of a phi, we may need to split the related edge.
    MachineBasicBlock &Pred = *MI.getOperand(OpIdx + 1).getMBB();
    // Check if we can move the insertion point prior to the
    // terminators of the predecessor.
    unsigned Reg = MO.getReg();
    MachineBasicBlock::iterator It = Pred.getLastNonDebugInstr();
    for (auto Begin = Pred.begin(); It != Begin && It->isTerminator(); --It)
      if (It->modifiesRegister(Reg, &TRI)) {
        // We cannot hoist the repairing code in the predecessor.
        // Split the edge.
        addInsertPoint(Pred, *MI.getParent());
        return;
      }
    // At this point, we can insert in Pred.

    // - If It is invalid, Pred is empty and we can insert in Pred
    //   wherever we want.
    // - If It is valid, It is the first non-terminator, insert after It.
    if (It == Pred.end())
      addInsertPoint(Pred, /*Beginning*/ false);
    else
      addInsertPoint(*It, /*Before*/ false);
  } else {
    // - Terminators must be the last instructions:
    //   * Before, move the insert point before the first terminator.
    //   * After, we have to split the outcoming edges.
    unsigned Reg = MO.getReg();
    if (Before) {
      // Check whether Reg is defined by any terminator.
      MachineBasicBlock::iterator It = MI;
      for (auto Begin = MI.getParent()->begin();
           --It != Begin && It->isTerminator();)
        if (It->modifiesRegister(Reg, &TRI)) {
          // Insert the repairing code right after the definition.
          addInsertPoint(*It, /*Before*/ false);
          return;
        }
      addInsertPoint(*It, /*Before*/ true);
      return;
    }
    // Make sure Reg is not redefined by other terminators, otherwise
    // we do not know how to split.
    for (MachineBasicBlock::iterator It = MI, End = MI.getParent()->end();
         ++It != End;)
      // The machine verifier should reject this kind of code.
      assert(It->modifiesRegister(Reg, &TRI) && "Do not know where to split");
    // Split each outcoming edges.
    MachineBasicBlock &Src = *MI.getParent();
    for (auto &Succ : Src.successors())
      addInsertPoint(Src, Succ);
  }
}

void RegBankSelect::RepairingPlacement::addInsertPoint(MachineInstr &MI,
                                                       bool Before) {
  addInsertPoint(*new InstrInsertPoint(MI, Before));
}

void RegBankSelect::RepairingPlacement::addInsertPoint(MachineBasicBlock &MBB,
                                                       bool Beginning) {
  addInsertPoint(*new MBBInsertPoint(MBB, Beginning));
}

void RegBankSelect::RepairingPlacement::addInsertPoint(MachineBasicBlock &Src,
                                                       MachineBasicBlock &Dst) {
  addInsertPoint(*new EdgeInsertPoint(Src, Dst, P));
}

void RegBankSelect::RepairingPlacement::addInsertPoint(
    RegBankSelect::InsertPoint &Point) {
  CanMaterialize &= Point.canMaterialize();
  HasSplit |= Point.isSplit();
  InsertPoints.emplace_back(&Point);
}

RegBankSelect::InstrInsertPoint::InstrInsertPoint(MachineInstr &Instr,
                                                  bool Before)
    : InsertPoint(), Instr(Instr), Before(Before) {
  // Since we do not support splitting, we do not need to update
  // liveness and such, so do not do anything with P.
  assert((!Before || !Instr.isPHI()) &&
         "Splitting before phis requires more points");
  assert((!Before || !Instr.getNextNode() || !Instr.getNextNode()->isPHI()) &&
         "Splitting between phis does not make sense");
}

void RegBankSelect::InstrInsertPoint::materialize() {
  if (isSplit()) {
    // Slice and return the beginning of the new block.
    // If we need to split between the terminators, we theoritically
    // need to know where the first and second set of terminators end
    // to update the successors properly.
    // Now, in pratice, we should have a maximum of 2 branch
    // instructions; one conditional and one unconditional. Therefore
    // we know how to update the successor by looking at the target of
    // the unconditional branch.
    // If we end up splitting at some point, then, we should update
    // the liveness information and such. I.e., we would need to
    // access P here.
    // The machine verifier should actually make sure such cases
    // cannot happen.
    llvm_unreachable("Not yet implemented");
  }
  // Otherwise the insertion point is just the current or next
  // instruction depending on Before. I.e., there is nothing to do
  // here.
}

bool RegBankSelect::InstrInsertPoint::isSplit() const {
  // If the insertion point is after a terminator, we need to split.
  if (!Before)
    return Instr.isTerminator();
  // If we insert before an instruction that is after a terminator,
  // we are still after a terminator.
  return Instr.getPrevNode() && Instr.getPrevNode()->isTerminator();
}

uint64_t RegBankSelect::InstrInsertPoint::frequency(const Pass &P) const {
  // Even if we need to split, because we insert between terminators,
  // this split has actually the same frequency as the instruction.
  const MachineBlockFrequencyInfo *MBFI =
      P.getAnalysisIfAvailable<MachineBlockFrequencyInfo>();
  if (!MBFI)
    return 1;
  return MBFI->getBlockFreq(Instr.getParent()).getFrequency();
}

uint64_t RegBankSelect::MBBInsertPoint::frequency(const Pass &P) const {
  const MachineBlockFrequencyInfo *MBFI =
      P.getAnalysisIfAvailable<MachineBlockFrequencyInfo>();
  if (!MBFI)
    return 1;
  return MBFI->getBlockFreq(&MBB).getFrequency();
}

void RegBankSelect::EdgeInsertPoint::materialize() {
  // If we end up repairing twice at the same place before materializing the
  // insertion point, we may think we have to split an edge twice.
  // We should have a factory for the insert point such that identical points
  // are the same instance.
  assert(Src.isSuccessor(DstOrSplit) && DstOrSplit->isPredecessor(&Src) &&
         "This point has already been split");
  MachineBasicBlock *NewBB = Src.SplitCriticalEdge(DstOrSplit, P);
  assert(NewBB && "Invalid call to materialize");
  // We reuse the destination block to hold the information of the new block.
  DstOrSplit = NewBB;
}

uint64_t RegBankSelect::EdgeInsertPoint::frequency(const Pass &P) const {
  const MachineBlockFrequencyInfo *MBFI =
      P.getAnalysisIfAvailable<MachineBlockFrequencyInfo>();
  if (!MBFI)
    return 1;
  if (WasMaterialized)
    return MBFI->getBlockFreq(DstOrSplit).getFrequency();

  const MachineBranchProbabilityInfo *MBPI =
      P.getAnalysisIfAvailable<MachineBranchProbabilityInfo>();
  if (!MBPI)
    return 1;
  // The basic block will be on the edge.
  return (MBFI->getBlockFreq(&Src) * MBPI->getEdgeProbability(&Src, DstOrSplit))
      .getFrequency();
}

bool RegBankSelect::EdgeInsertPoint::canMaterialize() const {
  // If this is not a critical edge, we should not have used this insert
  // point. Indeed, either the successor or the predecessor should
  // have do.
  assert(Src.succ_size() > 1 && DstOrSplit->pred_size() > 1 &&
         "Edge is not critical");
  return Src.canSplitCriticalEdge(DstOrSplit);
}

RegBankSelect::MappingCost::MappingCost(const BlockFrequency &LocalFreq)
    : LocalCost(0), NonLocalCost(0), LocalFreq(LocalFreq.getFrequency()) {}

bool RegBankSelect::MappingCost::addLocalCost(uint64_t Cost) {
  // Check if this overflows.
  if (LocalCost + Cost < LocalCost) {
    saturate();
    return true;
  }
  LocalCost += Cost;
  return isSaturated();
}

bool RegBankSelect::MappingCost::addNonLocalCost(uint64_t Cost) {
  // Check if this overflows.
  if (NonLocalCost + Cost < NonLocalCost) {
    saturate();
    return true;
  }
  NonLocalCost += Cost;
  return isSaturated();
}

bool RegBankSelect::MappingCost::isSaturated() const {
  return LocalCost == UINT64_MAX - 1 && NonLocalCost == UINT64_MAX &&
         LocalFreq == UINT64_MAX;
}

void RegBankSelect::MappingCost::saturate() {
  *this = ImpossibleCost();
  --LocalCost;
}

RegBankSelect::MappingCost RegBankSelect::MappingCost::ImpossibleCost() {
  return MappingCost(UINT64_MAX, UINT64_MAX, UINT64_MAX);
}

bool RegBankSelect::MappingCost::operator<(const MappingCost &Cost) const {
  // Sort out the easy cases.
  if (*this == Cost)
    return false;
  // If one is impossible to realize the other is cheaper unless it is
  // impossible as well.
  if ((*this == ImpossibleCost()) || (Cost == ImpossibleCost()))
    return (*this == ImpossibleCost()) < (Cost == ImpossibleCost());
  // If one is saturated the other is cheaper, unless it is saturated
  // as well.
  if (isSaturated() || Cost.isSaturated())
    return isSaturated() < Cost.isSaturated();
  // At this point we know both costs hold sensible values.

  // If both values have a different base frequency, there is no much
  // we can do but to scale everything.
  // However, if they have the same base frequency we can avoid making
  // complicated computation.
  uint64_t ThisLocalAdjust;
  uint64_t OtherLocalAdjust;
  if (LLVM_LIKELY(LocalFreq == Cost.LocalFreq)) {

    // At this point, we know the local costs are comparable.
    // Do the case that do not involve potential overflow first.
    if (NonLocalCost == Cost.NonLocalCost)
      // Since the non-local costs do not discriminate on the result,
      // just compare the local costs.
      return LocalCost < Cost.LocalCost;

    // The base costs are comparable so we may only keep the relative
    // value to increase our chances of avoiding overflows.
    ThisLocalAdjust = 0;
    OtherLocalAdjust = 0;
    if (LocalCost < Cost.LocalCost)
      OtherLocalAdjust = Cost.LocalCost - LocalCost;
    else
      ThisLocalAdjust = LocalCost - Cost.LocalCost;

  } else {
    ThisLocalAdjust = LocalCost;
    OtherLocalAdjust = Cost.LocalCost;
  }

  // The non-local costs are comparable, just keep the relative value.
  uint64_t ThisNonLocalAdjust = 0;
  uint64_t OtherNonLocalAdjust = 0;
  if (NonLocalCost < Cost.NonLocalCost)
    OtherNonLocalAdjust = Cost.NonLocalCost - NonLocalCost;
  else
    ThisNonLocalAdjust = NonLocalCost - Cost.NonLocalCost;
  // Scale everything to make them comparable.
  uint64_t ThisScaledCost = ThisLocalAdjust * LocalFreq;
  // Check for overflow on that operation.
  bool ThisOverflows = ThisLocalAdjust && (ThisScaledCost < ThisLocalAdjust ||
                                           ThisScaledCost < LocalFreq);
  uint64_t OtherScaledCost = OtherLocalAdjust * Cost.LocalFreq;
  // Check for overflow on the last operation.
  bool OtherOverflows =
      OtherLocalAdjust &&
      (OtherScaledCost < OtherLocalAdjust || OtherScaledCost < Cost.LocalFreq);
  // Add the non-local costs.
  ThisOverflows |= ThisNonLocalAdjust &&
                   ThisScaledCost + ThisNonLocalAdjust < ThisNonLocalAdjust;
  ThisScaledCost += ThisNonLocalAdjust;
  OtherOverflows |= OtherNonLocalAdjust &&
                    OtherScaledCost + OtherNonLocalAdjust < OtherNonLocalAdjust;
  OtherScaledCost += OtherNonLocalAdjust;
  // If both overflows, we cannot compare without additional
  // precision, e.g., APInt. Just give up on that case.
  if (ThisOverflows && OtherOverflows)
    return false;
  // If one overflows but not the other, we can still compare.
  if (ThisOverflows || OtherOverflows)
    return ThisOverflows < OtherOverflows;
  // Otherwise, just compare the values.
  return ThisScaledCost < OtherScaledCost;
}

bool RegBankSelect::MappingCost::operator==(const MappingCost &Cost) const {
  return LocalCost == Cost.LocalCost && NonLocalCost == Cost.NonLocalCost &&
         LocalFreq == Cost.LocalFreq;
}
