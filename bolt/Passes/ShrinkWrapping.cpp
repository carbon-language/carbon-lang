//===--- Passes/ShrinkWrapping.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "ShrinkWrapping.h"
#include <numeric>

#define DEBUG_TYPE "shrinkwrapping"

using namespace llvm;

namespace opts {

extern cl::opt<bool> TimeOpts;
extern cl::OptionCategory BoltOptCategory;

static cl::opt<unsigned> ShrinkWrappingThreshold(
    "shrink-wrapping-threshold",
    cl::desc("Percentage of prologue execution count to use as threshold when"
             " evaluating whether a block is cold enough to be profitable to"
             " move eligible spills there"),
    cl::init(30), cl::ZeroOrMore, cl::cat(BoltOptCategory));
}

namespace llvm {
namespace bolt {

void CalleeSavedAnalysis::analyzeSaves() {
  ReachingDefOrUse</*Def=*/true> &RD = Info.getReachingDefs();
  StackReachingUses &SRU = Info.getStackReachingUses();
  auto &InsnToBB = Info.getInsnToBBMap();
  BitVector BlacklistedRegs(BC.MRI->getNumRegs(), false);

  DEBUG(dbgs() << "Checking spill locations\n");
  for (auto &BB : BF) {
    DEBUG(dbgs() << "\tNow at BB " << BB.getName() << "\n");
    const MCInst *Prev = nullptr;
    for (auto &Inst : BB) {
      if (auto FIE = FA.getFIEFor(Inst)) {
        // Blacklist weird stores we don't understand
        if ((!FIE->IsSimple || FIE->StackOffset >= 0) && FIE->IsStore &&
            FIE->IsStoreFromReg) {
          BlacklistedRegs.set(FIE->RegOrImm);
          CalleeSaved.reset(FIE->RegOrImm);
          Prev = &Inst;
          continue;
        }

        if (!FIE->IsStore || !FIE->IsStoreFromReg ||
            BlacklistedRegs[FIE->RegOrImm]) {
          Prev = &Inst;
          continue;
        }

        // If this reg is defined locally, it is not a callee-saved reg
        if (RD.isReachedBy(FIE->RegOrImm,
                           Prev ? RD.expr_begin(*Prev) : RD.expr_begin(BB))) {
          BlacklistedRegs.set(FIE->RegOrImm);
          CalleeSaved.reset(FIE->RegOrImm);
          Prev = &Inst;
          continue;
        }

        // If this stack position is accessed in another function, we are
        // probably dealing with a parameter passed in a stack -- do not mess
        // with it
        if (SRU.isStoreUsed(*FIE,
                            Prev ? SRU.expr_begin(*Prev) : SRU.expr_begin(BB)),
            /*IncludeLocalAccesses=*/false) {
          BlacklistedRegs.set(FIE->RegOrImm);
          CalleeSaved.reset(FIE->RegOrImm);
          Prev = &Inst;
          continue;
        }

        // If this stack position is loaded elsewhere in another reg, we can't
        // update it, so blacklist it.
        if (SRU.isLoadedInDifferentReg(*FIE, Prev ? SRU.expr_begin(*Prev)
                                                  : SRU.expr_begin(BB))) {
          BlacklistedRegs.set(FIE->RegOrImm);
          CalleeSaved.reset(FIE->RegOrImm);
          Prev = &Inst;
          continue;
        }

        // Ignore regs with multiple saves
        if (CalleeSaved[FIE->RegOrImm]) {
          BlacklistedRegs.set(FIE->RegOrImm);
          CalleeSaved.reset(FIE->RegOrImm);
          Prev = &Inst;
          continue;
        }

        CalleeSaved.set(FIE->RegOrImm);
        SaveFIEByReg[FIE->RegOrImm] = &*FIE;
        SavingCost[FIE->RegOrImm] += InsnToBB[&Inst]->getKnownExecutionCount();
        BC.MIA->addAnnotation(BC.Ctx.get(), Inst, getSaveTag(), FIE->RegOrImm);
        OffsetsByReg[FIE->RegOrImm] = FIE->StackOffset;
        DEBUG(dbgs() << "Logging new candidate for Callee-Saved Reg: "
                     << FIE->RegOrImm << "\n");
      }
      Prev = &Inst;
    }
  }
}

void CalleeSavedAnalysis::analyzeRestores() {
  ReachingDefOrUse</*Def=*/false> &RU = Info.getReachingUses();

  // Now compute all restores of these callee-saved regs
  for (auto &BB : BF) {
    const MCInst *Prev = nullptr;
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
      auto &Inst = *I;
      if (auto FIE = FA.getFIEFor(Inst)) {
        if (!FIE->IsLoad || !CalleeSaved[FIE->RegOrImm]) {
          Prev = &Inst;
          continue;
        }

        // If this reg is used locally after a restore, then we are probably
        // not dealing with a callee-saved reg. Except if this use is by
        // another store, but we don't cover this case yet.
        // Also not callee-saved if this load accesses caller stack or isn't
        // simple.
        if (!FIE->IsSimple || FIE->StackOffset >= 0 ||
            RU.isReachedBy(FIE->RegOrImm,
                           Prev ? RU.expr_begin(*Prev) : RU.expr_begin(BB))) {
          CalleeSaved.reset(FIE->RegOrImm);
          Prev = &Inst;
          continue;
        }
        // If stack offsets between saves/store don't agree with each other,
        // we don't completely understand what's happening here
        if (FIE->StackOffset != OffsetsByReg[FIE->RegOrImm]) {
          CalleeSaved.reset(FIE->RegOrImm);
          DEBUG(dbgs() << "Dismissing Callee-Saved Reg because we found a "
                          "mismatching restore: "
                       << FIE->RegOrImm << "\n");
          Prev = &Inst;
          continue;
        }

        DEBUG(dbgs() << "Adding matching restore for: " << FIE->RegOrImm
                     << "\n");
        if (LoadFIEByReg[FIE->RegOrImm] == nullptr)
          LoadFIEByReg[FIE->RegOrImm] = &*FIE;
        BC.MIA->addAnnotation(BC.Ctx.get(), Inst, getRestoreTag(),
                              FIE->RegOrImm);
        HasRestores.set(FIE->RegOrImm);
      }
      Prev = &Inst;
    }
  }
}

std::vector<MCInst *> CalleeSavedAnalysis::getSavesByReg(uint16_t Reg) {
  std::vector<MCInst *> Results;
  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      if (getSavedReg(Inst) == Reg)
        Results.push_back(&Inst);
    }
  }
  return Results;
}

std::vector<MCInst *> CalleeSavedAnalysis::getRestoresByReg(uint16_t Reg) {
  std::vector<MCInst *> Results;
  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      if (getRestoredReg(Inst) == Reg)
        Results.push_back(&Inst);
    }
  }
  return Results;
}

CalleeSavedAnalysis::~CalleeSavedAnalysis() {
  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      BC.MIA->removeAnnotation(Inst, getSaveTag());
      BC.MIA->removeAnnotation(Inst, getRestoreTag());
    }
  }
}

void StackLayoutModifier::blacklistRegion(int64_t Offset, int64_t Size) {
  if (BlacklistedRegions[Offset] < Size) {
    BlacklistedRegions[Offset] = Size;
  }
}

bool StackLayoutModifier::isRegionBlacklisted(int64_t Offset, int64_t Size) {
  for (auto Elem : BlacklistedRegions) {
    if (Offset + Size > Elem.first && Offset < Elem.first + Elem.second)
      return true;
  }
  return false;
}

bool StackLayoutModifier::blacklistAllInConflictWith(int64_t Offset,
                                                     int64_t Size) {
  bool HasConflict = false;
  for (auto Iter = AvailableRegions.begin(); Iter != AvailableRegions.end();) {
    auto &Elem = *Iter;
    if (Offset + Size > Elem.first && Offset < Elem.first + Elem.second &&
        (Offset != Elem.first || Size != Elem.second)) {
      Iter = AvailableRegions.erase(Iter);
      HasConflict = true;
      continue;
    }
    ++Iter;
  }
  if (HasConflict) {
    blacklistRegion(Offset, Size);
    return true;
  }
  return false;
}

void StackLayoutModifier::checkFramePointerInitialization(MCInst &Point) {
  auto &SPT = Info.getStackPointerTracking();
  if (!BC.MII->get(Point.getOpcode())
           .hasDefOfPhysReg(Point, BC.MIA->getFramePointer(), *BC.MRI))
    return;

  int SPVal, FPVal;
  std::tie(SPVal, FPVal) = *SPT.getStateBefore(Point);
  std::pair<MCPhysReg, int64_t> FP;

  if (FPVal != SPT.EMPTY && FPVal != SPT.SUPERPOSITION)
    FP = std::make_pair(BC.MIA->getFramePointer(), FPVal);
  else
    FP = std::make_pair(0, 0);
  std::pair<MCPhysReg, int64_t> SP;

  if (SPVal != SPT.EMPTY && SPVal != SPT.SUPERPOSITION)
    SP = std::make_pair(BC.MIA->getStackPointer(), SPVal);
  else
    SP = std::make_pair(0, 0);

  int64_t Output;
  if (!BC.MIA->evaluateSimple(Point, Output, SP, FP))
    return;

  // Not your regular frame pointer initialization... bail
  if (Output != SPVal)
    blacklistRegion(0, 0);
}

void StackLayoutModifier::classifyStackAccesses() {
  // Understand when stack slots are being used non-locally
  auto &SRU = Info.getStackReachingUses();

  for (auto &BB : BF) {
    const MCInst *Prev = nullptr;
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
      auto &Inst = *I;
      checkFramePointerInitialization(Inst);
      auto FIEX = FA.getFIEFor(Inst);
      if (!FIEX) {
        Prev = &Inst;
        continue;
      }
      if (!FIEX->IsSimple || (FIEX->IsStore && !FIEX->IsStoreFromReg)) {
        blacklistRegion(FIEX->StackOffset, FIEX->Size);
        Prev = &Inst;
        continue;
      }
      // If this stack position is accessed in another function, we are
      // probably dealing with a parameter passed in a stack -- do not mess
      // with it
      if (SRU.isStoreUsed(*FIEX,
                          Prev ? SRU.expr_begin(*Prev) : SRU.expr_begin(BB),
                          /*IncludeLocalAccesses=*/false)) {
        blacklistRegion(FIEX->StackOffset, FIEX->Size);
        Prev = &Inst;
        continue;
      }
      // Now we have a clear stack slot access. Check if its blacklisted or if
      // it conflicts with another chunk.
      if (isRegionBlacklisted(FIEX->StackOffset, FIEX->Size) ||
          blacklistAllInConflictWith(FIEX->StackOffset, FIEX->Size)) {
        Prev = &Inst;
        continue;
      }
      // We are free to go. Add it as available stack slot which we know how
      // to move it.
      AvailableRegions[FIEX->StackOffset] = FIEX->Size;
      BC.MIA->addAnnotation(BC.Ctx.get(), Inst, getSlotTagName(),
                            FIEX->StackOffset);
      RegionToRegMap[FIEX->StackOffset].insert(FIEX->RegOrImm);
      RegToRegionMap[FIEX->RegOrImm].insert(FIEX->StackOffset);
      DEBUG(dbgs() << "Adding region " << FIEX->StackOffset << " size "
                   << (int)FIEX->Size << "\n");
    }
  }
}

void StackLayoutModifier::classifyCFIs() {
  std::stack<std::pair<int64_t, uint16_t>> CFIStack;
  int64_t CfaOffset{-8};
  uint16_t CfaReg{7};

  auto recordAccess = [&](MCInst *Inst, int64_t Offset) {
    const uint16_t Reg = BC.MRI->getLLVMRegNum(CfaReg, /*isEH=*/false);
    if (Reg == BC.MIA->getStackPointer() || Reg == BC.MIA->getFramePointer()) {
      BC.MIA->addAnnotation(BC.Ctx.get(), *Inst, getSlotTagName(), Offset);
      DEBUG(dbgs() << "Recording CFI " << Offset << "\n");
    } else {
      IsSimple = false;
      return;
    }
  };

  for (auto &BB : BF.layout()) {
    for (auto &Inst : *BB) {
      if (!BC.MIA->isCFI(Inst))
        continue;
      auto *CFI = BF.getCFIFor(Inst);
      switch (CFI->getOperation()) {
      case MCCFIInstruction::OpDefCfa:
        CfaOffset = CFI->getOffset();
        recordAccess(&Inst, CfaOffset);
      // Fall-through
      case MCCFIInstruction::OpDefCfaRegister:
        CfaReg = CFI->getRegister();
        break;
      case MCCFIInstruction::OpDefCfaOffset:
        CfaOffset = CFI->getOffset();
        recordAccess(&Inst, CfaOffset);
        break;
      case MCCFIInstruction::OpOffset:
        recordAccess(&Inst, CFI->getOffset());
        BC.MIA->addAnnotation(BC.Ctx.get(), Inst, getOffsetCFIRegTagName(),
                              BC.MRI->getLLVMRegNum(CFI->getRegister(),
                                                    /*isEH=*/false));
        break;
      case MCCFIInstruction::OpSameValue:
        BC.MIA->addAnnotation(BC.Ctx.get(), Inst, getOffsetCFIRegTagName(),
                              BC.MRI->getLLVMRegNum(CFI->getRegister(),
                                                    /*isEH=*/false));
        break;
      case MCCFIInstruction::OpRememberState:
        CFIStack.push(std::make_pair(CfaOffset, CfaReg));
        break;
      case MCCFIInstruction::OpRestoreState: {
        assert(!CFIStack.empty() && "Corrupt CFI stack");
        auto &Elem = CFIStack.top();
        CFIStack.pop();
        CfaOffset = Elem.first;
        CfaReg = Elem.second;
        break;
      }
      case MCCFIInstruction::OpRelOffset:
      case MCCFIInstruction::OpAdjustCfaOffset:
        llvm_unreachable("Unhandled AdjustCfaOffset");
        break;
      default:
        break;
      }
    }
  }
}

void StackLayoutModifier::scheduleChange(
    MCInst &Inst, StackLayoutModifier::WorklistItem Item) {
  auto &WList = BC.MIA->getOrCreateAnnotationAs<std::vector<WorklistItem>>(
      BC.Ctx.get(), Inst, getTodoTagName());
  WList.push_back(Item);
}

bool StackLayoutModifier::canCollapseRegion(MCInst *DeletedPush) {
  if (!IsSimple || !BC.MIA->isPush(*DeletedPush))
    return false;

  auto FIE = FA.getFIEFor(*DeletedPush);
  if (!FIE)
    return false;

  return canCollapseRegion(FIE->StackOffset);
}

bool StackLayoutModifier::canCollapseRegion(int64_t RegionAddr) {
  if (!IsInitialized)
    initialize();
  if (!IsSimple)
    return false;

  if (CollapsedRegions.count(RegionAddr))
    return true;

  // Check if it is possible to readjust all accesses below RegionAddr
  if (!BlacklistedRegions.empty())
    return false;

  return true;
}

bool StackLayoutModifier::collapseRegion(MCInst *DeletedPush) {
  auto FIE = FA.getFIEFor(*DeletedPush);
  if (!FIE)
    return false;
  int64_t RegionAddr = FIE->StackOffset;
  int64_t RegionSz = FIE->Size;
  return collapseRegion(DeletedPush, RegionAddr, RegionSz);
}

bool StackLayoutModifier::collapseRegion(MCInst *Alloc, int64_t RegionAddr,
                                         int64_t RegionSz) {
  if (!canCollapseRegion(RegionAddr))
    return false;

  assert(IsInitialized);
  auto &SAA = Info.getStackAllocationAnalysis();

  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      if (!BC.MIA->hasAnnotation(Inst, getSlotTagName()))
        continue;
      auto Slot =
          BC.MIA->getAnnotationAs<decltype(FrameIndexEntry::StackOffset)>(
              Inst, getSlotTagName());
      if (!AvailableRegions.count(Slot))
        continue;
      // We need to ensure this access is affected by the deleted push
      if (!(*SAA.getStateBefore(Inst))[SAA.ExprToIdx[Alloc]])
        continue;

      if (BC.MIA->isCFI(Inst)) {
        if (Slot > RegionAddr)
          continue;
        scheduleChange(Inst, WorklistItem(WorklistItem::AdjustCFI, RegionSz));
        continue;
      }

      if (Slot == RegionAddr) {
        BC.MIA->addAnnotation(BC.Ctx.get(), Inst, "AccessesDeletedPos", 0U);
        continue;
      }
      if (BC.MIA->isPush(Inst) || BC.MIA->isPop(Inst)) {
        continue;
      }

      auto FIE = FA.getFIEFor(Inst);
      assert(FIE);
      if (FIE->StackPtrReg == BC.MIA->getStackPointer() && Slot < RegionAddr)
        continue;

      if (FIE->StackPtrReg == BC.MIA->getFramePointer() && Slot > RegionAddr)
        continue;

      scheduleChange(
          Inst, WorklistItem(WorklistItem::AdjustLoadStoreOffset, RegionSz));
    }
  }

  CollapsedRegions.insert(RegionAddr);
  return true;
}

void StackLayoutModifier::setOffsetForCollapsedAccesses(int64_t NewOffset) {
  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      if (!BC.MIA->hasAnnotation(Inst, "AccessesDeletedPos"))
        continue;
      BC.MIA->removeAnnotation(Inst, "AccessesDeletedPos");
      scheduleChange(
          Inst, WorklistItem(WorklistItem::AdjustLoadStoreOffset, NewOffset));
    }
  }
}

bool StackLayoutModifier::canInsertRegion(ProgramPoint P) {
  if (!IsInitialized)
    initialize();
  if (!IsSimple)
    return false;

  auto &SPT = Info.getStackPointerTracking();
  int64_t RegionAddr = SPT.getStateBefore(P)->first;
  if (RegionAddr == SPT.SUPERPOSITION || RegionAddr == SPT.EMPTY)
    return false;

  if (InsertedRegions.count(RegionAddr))
    return true;

  // Check if we are going to screw up stack accesses at call sites that
  // pass parameters via stack
  if (!BlacklistedRegions.empty())
    return false;

  return true;
}

bool StackLayoutModifier::insertRegion(ProgramPoint P, int64_t RegionSz) {
  if (!canInsertRegion(P))
    return false;

  assert(IsInitialized);
  auto &SPT = Info.getStackPointerTracking();
  // This RegionAddr is slightly different from the one seen in collapseRegion
  // This is the value of SP before the allocation the user wants to make.
  int64_t RegionAddr = SPT.getStateBefore(P)->first;
  if (RegionAddr == SPT.SUPERPOSITION || RegionAddr == SPT.EMPTY)
    return false;

  auto &DA = Info.getDominatorAnalysis();

  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      if (!BC.MIA->hasAnnotation(Inst, getSlotTagName()))
        continue;
      auto Slot =
          BC.MIA->getAnnotationAs<decltype(FrameIndexEntry::StackOffset)>(
              Inst, getSlotTagName());
      if (!AvailableRegions.count(Slot))
        continue;

      if (!(DA.doesADominateB(P, Inst)))
        continue;

      if (BC.MIA->isCFI(Inst)) {
        if (Slot >= RegionAddr)
          continue;
        scheduleChange(Inst, WorklistItem(WorklistItem::AdjustCFI, -RegionSz));
        continue;
      }

      auto FIE = FA.getFIEFor(Inst);
      assert(FIE);
      if (FIE->StackPtrReg == BC.MIA->getStackPointer() && Slot < RegionAddr)
        continue;
      if (FIE->StackPtrReg == BC.MIA->getFramePointer() && Slot >= RegionAddr)
        continue;
      if (BC.MIA->isPush(Inst) || BC.MIA->isPop(Inst))
        continue;
      scheduleChange(
          Inst, WorklistItem(WorklistItem::AdjustLoadStoreOffset, -RegionSz));
    }
  }

  InsertedRegions.insert(RegionAddr);
  return true;
}

void StackLayoutModifier::performChanges() {
  std::set<uint32_t> ModifiedCFIIndices;
  for (auto &BB : BF) {
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
      auto &Inst = *I;
      if (BC.MIA->hasAnnotation(Inst, "AccessesDeletedPos")) {
        assert(BC.MIA->isPop(Inst) || BC.MIA->isPush(Inst));
        BC.MIA->removeAnnotation(Inst, "AccessesDeletedPos");
      }
      if (!BC.MIA->hasAnnotation(Inst, getTodoTagName()))
        continue;
      auto &WList = BC.MIA->getAnnotationAs<std::vector<WorklistItem>>(
          Inst, getTodoTagName());
      int64_t Adjustment = 0;
      WorklistItem::ActionType AdjustmentType = WorklistItem::None;
      for (auto &WI : WList) {
        if (WI.Action == WorklistItem::None)
          continue;
        assert(WI.Action == WorklistItem::AdjustLoadStoreOffset ||
               WI.Action == WorklistItem::AdjustCFI);
        assert((AdjustmentType == WorklistItem::None ||
                AdjustmentType == WI.Action) &&
               "Conflicting actions requested at the same program point");
        AdjustmentType = WI.Action;
        Adjustment += WI.OffsetUpdate;
      }
      if (!Adjustment)
        continue;
      if (AdjustmentType != WorklistItem::AdjustLoadStoreOffset) {
        assert(BC.MIA->isCFI(Inst));
        uint32_t CFINum = Inst.getOperand(0).getImm();
        if (ModifiedCFIIndices.count(CFINum))
          continue;
        ModifiedCFIIndices.insert(CFINum);
        MCCFIInstruction *CFI = BF.getCFIFor(Inst);
        DEBUG(dbgs() << "Changing CFI offset from " << CFI->getOffset()
                     << " to " << (CFI->getOffset() + Adjustment) << "\n");
        CFI->setOffset(CFI->getOffset() + Adjustment);
        continue;
      }
      int32_t SrcImm{0};
      MCPhysReg Reg{0};
      MCPhysReg StackPtrReg{0};
      int64_t StackOffset{0};
      bool IsIndexed{false};
      bool IsLoad{false};
      bool IsStore{false};
      bool IsSimple{false};
      bool IsStoreFromReg{false};
      uint8_t Size{0};
      bool Success{false};
      Success = BC.MIA->isStackAccess(*BC.MRI, Inst, IsLoad, IsStore,
                                      IsStoreFromReg, Reg, SrcImm, StackPtrReg,
                                      StackOffset, Size, IsSimple, IsIndexed);
      assert(Success && IsSimple && !IsIndexed && (!IsStore || IsStoreFromReg));
      if (StackPtrReg != BC.MIA->getFramePointer())
        Adjustment = -Adjustment;
      if (IsLoad)
        Success = BC.MIA->createRestoreFromStack(
            Inst, StackPtrReg, StackOffset + Adjustment, Reg, Size);
      else if (IsStore)
        Success = BC.MIA->createSaveToStack(
            Inst, StackPtrReg, StackOffset + Adjustment, Reg, Size);
      DEBUG({
        dbgs() << "Adjusted instruction: ";
        Inst.dump();
      });
      assert(Success);
    }
  }
}

void StackLayoutModifier::initialize() {
  classifyStackAccesses();
  classifyCFIs();
  IsInitialized = true;
}

uint64_t ShrinkWrapping::SpillsMovedRegularMode = 0;
uint64_t ShrinkWrapping::SpillsMovedPushPopMode = 0;

using BBIterTy = BinaryBasicBlock::iterator;

void ShrinkWrapping::classifyCSRUses() {
  auto &DA = Info.getDominatorAnalysis();
  auto &SPT = Info.getStackPointerTracking();
  UsesByReg = std::vector<BitVector>(BC.MRI->getNumRegs(),
                                     BitVector(DA.NumInstrs, false));

  const BitVector &FPAliases =
      BC.MIA->getAliases(BC.MIA->getFramePointer(), *BC.MRI);
  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      if (BC.MIA->isCFI(Inst))
        continue;
      auto BV = BitVector(BC.MRI->getNumRegs(), false);
      BC.MIA->getTouchedRegs(Inst, BV, *BC.MRI);
      BV &= CSA.CalleeSaved;
      for (int I = BV.find_first(); I != -1; I = BV.find_next(I)) {
        if (I == 0)
          continue;
        if (CSA.getSavedReg(Inst) != I && CSA.getRestoredReg(Inst) != I)
          UsesByReg[I].set(DA.ExprToIdx[&Inst]);
      }
      if (!SPT.HasFramePointer || !BC.MIA->isCall(Inst))
        continue;
      BV = CSA.CalleeSaved;
      BV &= FPAliases;
      for (int I = BV.find_first(); I > 0; I = BV.find_next(I)) {
        UsesByReg[I].set(DA.ExprToIdx[&Inst]);
      }
    }
  }
}

void ShrinkWrapping::pruneUnwantedCSRs() {
  BitVector ParamRegs = BC.MIA->getRegsUsedAsParams(*BC.MRI);
  for (unsigned I = 0, E = BC.MRI->getNumRegs(); I != E; ++I) {
    if (!CSA.CalleeSaved[I])
      continue;
    if (ParamRegs[I]) {
      CSA.CalleeSaved.reset(I);
      continue;
    }
    if (UsesByReg[I].empty()) {
      DEBUG(dbgs()
            << "Dismissing Callee-Saved Reg because we found no uses of it:"
            << I << "\n");
      CSA.CalleeSaved.reset(I);
      continue;
    }
    if (!CSA.HasRestores[I]) {
      DEBUG(dbgs() << "Dismissing Callee-Saved Reg because it does not have "
                      "restores:"
                   << I << "\n");
      CSA.CalleeSaved.reset(I);
    }
  }
}

void ShrinkWrapping::computeSaveLocations() {
  SavePos = std::vector<SmallPtrSet<MCInst *, 4>>(BC.MRI->getNumRegs());
  auto &RI = Info.getReachingInsnsBackwards();
  auto &DA = Info.getDominatorAnalysis();
  auto &SPT = Info.getStackPointerTracking();

  DEBUG(dbgs() << "Checking save/restore possibilities\n");
  for (auto &BB : BF) {
    DEBUG(dbgs() << "\tNow at BB " << BB.getName() << "\n");

    MCInst *First = BB.begin() != BB.end() ? &*BB.begin() : nullptr;
    if (!First)
      continue;

    // Use reaching instructions to detect if we are inside a loop - if we
    // are, do not consider this BB as valid placement for saves.
    if (RI.isInLoop(BB))
      continue;

    const auto SPFP = *SPT.getStateBefore(*First);
    // If we don't know stack state at this point, bail
    if ((SPFP.first == SPT.SUPERPOSITION || SPFP.first == SPT.EMPTY) &&
        (SPFP.second == SPT.SUPERPOSITION || SPFP.second == SPT.EMPTY))
      continue;

    for (unsigned I = 0, E = BC.MRI->getNumRegs(); I != E; ++I) {
      if (!CSA.CalleeSaved[I])
        continue;

      auto BBDominatedUses = BitVector(DA.NumInstrs, false);
      for (auto J = UsesByReg[I].find_first(); J > 0;
           J = UsesByReg[I].find_next(J)) {
        if (DA.doesADominateB(*First, J))
          BBDominatedUses.set(J);
      }
      DEBUG(dbgs() << "\t\tBB " << BB.getName() << " dominates "
                   << BBDominatedUses.count() << " uses for reg " << I
                   << ". Total uses for reg is " << UsesByReg[I].count()
                   << "\n");
      BBDominatedUses &= UsesByReg[I];
      if (BBDominatedUses == UsesByReg[I]) {
        DEBUG(dbgs() << "\t\t\tAdded " << BB.getName() << " as a save pos for "
                     << I << "\n");
        SavePos[I].insert(First);
        DEBUG({
          dbgs() << "Dominated uses are:\n";
          for (auto J = UsesByReg[I].find_first(); J > 0;
               J = UsesByReg[I].find_next(J)) {
            dbgs() << "Idx " << J << ": ";
            DA.Expressions[J]->dump();
          }
        });
      }
    }
  }

  BestSaveCount = std::vector<uint64_t>(BC.MRI->getNumRegs(),
                                        std::numeric_limits<uint64_t>::max());
  BestSavePos = std::vector<MCInst *>(BC.MRI->getNumRegs(), nullptr);
  auto &InsnToBB = Info.getInsnToBBMap();
  for (unsigned I = 0, E = BC.MRI->getNumRegs(); I != E; ++I) {
    if (!CSA.CalleeSaved[I])
      continue;

    for (auto *Pos : SavePos[I]) {
      auto *BB = InsnToBB[Pos];
      uint64_t Count = BB->getExecutionCount();
      if (Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
          Count < BestSaveCount[I]) {
        BestSavePos[I] = Pos;
        BestSaveCount[I] = Count;
      }
    }
  }
}

void ShrinkWrapping::computeDomOrder() {
  std::vector<MCPhysReg> Order;
  for (MCPhysReg I = 0, E = BC.MRI->getNumRegs(); I != E; ++I) {
    Order.push_back(I);
  }

  auto &DA = Info.getDominatorAnalysis();
  auto &InsnToBB = Info.getInsnToBBMap();
  std::sort(Order.begin(), Order.end(), [&](const MCPhysReg &A,
                                            const MCPhysReg &B) {
    auto *BBA = BestSavePos[A] ? InsnToBB[BestSavePos[A]] : nullptr;
    auto *BBB = BestSavePos[B] ? InsnToBB[BestSavePos[B]] : nullptr;
    if (BBA == BBB)
      return A < B;
    if (!BBA && BBB)
      return false;
    if (BBA && !BBB)
      return true;
    if (DA.doesADominateB(*BestSavePos[A], *BestSavePos[B]))
      return true;
    if (DA.doesADominateB(*BestSavePos[B], *BestSavePos[A]))
      return false;
    return A < B;
  });

  for (MCPhysReg I = 0, E = BC.MRI->getNumRegs(); I != E; ++I) {
    DomOrder[Order[I]] = I;
  }
}

bool ShrinkWrapping::isBestSavePosCold(unsigned CSR, MCInst *&BestPosSave,
                                       uint64_t &TotalEstimatedWin) {
  const uint64_t CurSavingCost = CSA.SavingCost[CSR];
  if (!CSA.CalleeSaved[CSR])
    return false;

  uint64_t BestCount = BestSaveCount[CSR];
  BestPosSave = BestSavePos[CSR];
  bool ShouldMove{false};
  if (BestCount != std::numeric_limits<uint64_t>::max() &&
      BestCount < (opts::ShrinkWrappingThreshold / 100.0) * CurSavingCost) {
    DEBUG({
      auto &InsnToBB = Info.getInsnToBBMap();
      dbgs() << "Better position for saves found in func " << BF.getPrintName()
             << " count << " << BF.getKnownExecutionCount() << "\n";
      dbgs() << "Reg: " << CSR
             << "; New BB: " << InsnToBB[BestPosSave]->getName()
             << " Freq reduction: " << (CurSavingCost - BestCount) << "\n";
    });
    TotalEstimatedWin += CurSavingCost - BestCount;
    ShouldMove = true;
  }

  if (!ShouldMove)
    return false;
  if (!BestPosSave) {
    DEBUG({
      dbgs() << "Dropping opportunity because we don't know where to put "
                "stores -- total est. freq reduc: "
             << TotalEstimatedWin << "\n";
    });
    return false;
  }
  return true;
}

/// Auxiliar function used to create basic blocks for critical edges and update
/// the dominance frontier with these new locations
void ShrinkWrapping::splitFrontierCritEdges(
    BinaryFunction *Func, SmallVector<ProgramPoint, 4> &Frontier,
    const SmallVector<bool, 4> &IsCritEdge,
    const SmallVector<BinaryBasicBlock *, 4> &From,
    const SmallVector<SmallVector<BinaryBasicBlock *, 4>, 4> &To) {
  DEBUG(dbgs() << "splitFrontierCritEdges: Now handling func "
               << BF.getPrintName() << "\n");
  for (size_t I = 0; I < Frontier.size(); ++I) {
    if (!IsCritEdge[I])
      continue;
    if (To[I].empty())
      continue;
    auto FromBB = From[I];
    DEBUG(dbgs() << " - Now handling FrontierBB " << FromBB->getName() << "\n");
    for (auto DestinationBB : To[I]) {
      DEBUG(dbgs() << "   - Dest : " << DestinationBB->getName() << "\n");
      auto *NewBB = Func->splitEdge(FromBB, DestinationBB);
      // Insert dummy instruction so this BB is never empty (we need this for
      // PredictiveStackPointerTracking to work, since it annotates instructions
      // and not BBs).
      if (NewBB->empty()) {
        MCInst NewInst;
        BC.MIA->createNoop(NewInst);
        NewBB->addInstruction(std::move(NewInst));
        scheduleChange(&*NewBB->begin(), WorklistItem(WorklistItem::Erase, 0));
      }

      // Update frontier
      Frontier[I] = ProgramPoint::getLastPointAt(*NewBB);
    }
  }
}

SmallVector<ProgramPoint, 4>
ShrinkWrapping::doRestorePlacement(MCInst *BestPosSave, unsigned CSR,
                                   uint64_t TotalEstimatedWin) {
  SmallVector<ProgramPoint, 4> Frontier;
  SmallVector<bool, 4> IsCritEdge;
  bool CannotPlace{false};
  auto &DA = Info.getDominatorAnalysis();

  SmallVector<BinaryBasicBlock *, 4> CritEdgesFrom;
  SmallVector<SmallVector<BinaryBasicBlock *, 4>, 4> CritEdgesTo;
  // In case of a critical edge, we need to create extra BBs to host restores
  // into edges transitioning to the dominance frontier, otherwise we pull these
  // restores to inside the dominated area.
  Frontier = DA.getDominanceFrontierFor(*BestPosSave);
  for (auto &PP : Frontier) {
    bool HasCritEdges{false};
    if (PP.isInst() && BC.MIA->isTerminator(*PP.getInst()) &&
        doesInstUsesCSR(*PP.getInst(), CSR)) {
      CannotPlace = true;
    }
    BinaryBasicBlock *FrontierBB = Info.getParentBB(PP);
    CritEdgesFrom.emplace_back(FrontierBB);
    CritEdgesTo.emplace_back(0);
    auto &Dests = CritEdgesTo.back();
    // Check for invoke instructions at the dominance frontier, which indicates
    // the landing pad is not dominated.
    if (PP.isInst() && BC.MIA->isInvoke(*PP.getInst())) {
      DEBUG(dbgs() << "Bailing on restore placement to avoid LP splitting\n");
      Frontier.clear();
      return Frontier;
    }
    doForAllSuccs(*FrontierBB, [&](ProgramPoint P) {
      if (!DA.doesADominateB(*BestPosSave, P)) {
        Dests.emplace_back(Info.getParentBB(P));
        return;
      }
      HasCritEdges = true;
    });
    IsCritEdge.push_back(HasCritEdges);
  }
  if (std::accumulate(IsCritEdge.begin(), IsCritEdge.end(), 0)) {
    DEBUG({
      dbgs() << "Now detected critical edges in the following frontier:\n";
      for (auto &PP : Frontier) {
        if (PP.isBB())
          dbgs() << "  BB: " << PP.getBB()->getName() << "\n";
        else {
          dbgs() << "  Inst: ";
          PP.getInst()->dump();
        }
      }
    });
    splitFrontierCritEdges(&BF, Frontier, IsCritEdge, CritEdgesFrom,
                           CritEdgesTo);
    // BitVectors that represent all insns of the function are invalid now
    // since we changed BBs/Insts. Re-run steps that depend on pointers being
    // valid
    Info.invalidateAll();
    classifyCSRUses();
  }
  if (CannotPlace) {
    DEBUG({
      dbgs() << "Dropping opportunity because restore placement failed"
                " -- total est. freq reduc: "
             << TotalEstimatedWin << "\n";
    });
    Frontier.clear();
    return Frontier;
  }
  return Frontier;
}

bool ShrinkWrapping::validatePushPopsMode(unsigned CSR, MCInst *BestPosSave,
                                          int64_t SaveOffset) {
  if (FA.requiresAlignment(BF)) {
    DEBUG({
      dbgs() << "Reg " << CSR << " is not using push/pops due to function "
                                 "alignment requirements.\n";
    });
    return false;
  }
  for (MCInst *Save : CSA.getSavesByReg(CSR)) {
    if (!SLM.canCollapseRegion(Save)) {
      DEBUG(dbgs() << "Reg " << CSR << " cannot collapse region.\n");
      return false;
    }
  }

  auto &SPT = Info.getStackPointerTracking();
  // Abort if we are inserting a push into an entry BB (offset -8) and this
  // func sets up a frame pointer.
  if (!SLM.canInsertRegion(BestPosSave) ||
      SaveOffset == SPT.SUPERPOSITION || SaveOffset == SPT.EMPTY ||
      (SaveOffset == -8 && SPT.HasFramePointer)) {
    DEBUG({
      dbgs() << "Reg " << CSR << " cannot insert region or we are "
                                 "trying to insert a push into entry bb.\n";
    });
    return false;
  }
  return true;
}

SmallVector<ProgramPoint, 4> ShrinkWrapping::fixPopsPlacements(
    const SmallVector<ProgramPoint, 4> &RestorePoints, int64_t SaveOffset,
    unsigned CSR) {
  SmallVector<ProgramPoint, 4> FixedRestorePoints = RestorePoints;
  // Moving pop locations to the correct sp offset
  auto &RI = Info.getReachingInsnsBackwards();
  auto &SPT = Info.getStackPointerTracking();
  for (auto &PP : FixedRestorePoints) {
    auto *BB = Info.getParentBB(PP);
    auto Found = false;
    if (SPT.getStateAt(ProgramPoint::getLastPointAt(*BB))->first ==
        SaveOffset) {
      BitVector BV = *RI.getStateAt(ProgramPoint::getLastPointAt(*BB));
      BV &= UsesByReg[CSR];
      if (!BV.any()) {
        Found = true;
        PP = BB;
        continue;
      }
    }
    for (auto RIt = BB->rbegin(), End = BB->rend(); RIt != End; ++RIt) {
      if (SPT.getStateBefore(*RIt)->first == SaveOffset) {
        BitVector BV = *RI.getStateAt(*RIt);
        BV &= UsesByReg[CSR];
        if (!BV.any()) {
          Found = true;
          PP = &*RIt;
          break;
        }
      }
    }
    if (!Found) {
      DEBUG({
        dbgs() << "Could not find restore insertion point for " << CSR
               << ", falling back to load/store mode\n";
      });
      FixedRestorePoints.clear();
      return FixedRestorePoints;
    }
  }
  return FixedRestorePoints;
}

void ShrinkWrapping::scheduleOldSaveRestoresRemoval(unsigned CSR,
                                                    bool UsePushPops) {

  for (auto &BB : BF.layout()) {
    std::vector<MCInst *> CFIs;
    for (auto I = BB->rbegin(), E = BB->rend(); I != E; ++I) {
      auto &Inst = *I;
      if (BC.MIA->isCFI(Inst)) {
        // Delete all offset CFIs related to this CSR
        if (SLM.getOffsetCFIReg(Inst) == CSR) {
          HasDeletedOffsetCFIs[CSR] = true;
          scheduleChange(&Inst, WorklistItem(WorklistItem::Erase, CSR));
          continue;
        }
        CFIs.push_back(&Inst);
        continue;
      }

      auto SavedReg = CSA.getSavedReg(Inst);
      auto RestoredReg = CSA.getRestoredReg(Inst);
      if (SavedReg != CSR && RestoredReg != CSR) {
        CFIs.clear();
        continue;
      }

      scheduleChange(&Inst, WorklistItem(UsePushPops
                                             ? WorklistItem::Erase
                                             : WorklistItem::ChangeToAdjustment,
                                         CSR));

      // Delete associated CFIs
      const bool RecordDeletedPushCFIs =
          SavedReg == CSR && DeletedPushCFIs[CSR].empty();
      const bool RecordDeletedPopCFIs =
          RestoredReg == CSR && DeletedPopCFIs[CSR].empty();
      for (MCInst *CFI : CFIs) {
        auto *MCCFI = BF.getCFIFor(*CFI);
        // Do not touch these...
        if (MCCFI->getOperation() == MCCFIInstruction::OpRestoreState ||
            MCCFI->getOperation() == MCCFIInstruction::OpRememberState)
          continue;
        scheduleChange(CFI, WorklistItem(WorklistItem::Erase, CSR));
        if (RecordDeletedPushCFIs) {
          // Do not record this to be replayed later because we are going to
          // rebuild it.
          if (MCCFI->getOperation() == MCCFIInstruction::OpDefCfaOffset)
            continue;
          DeletedPushCFIs[CSR].push_back(CFI->getOperand(0).getImm());
        }
        if (RecordDeletedPopCFIs) {
          if (MCCFI->getOperation() == MCCFIInstruction::OpDefCfaOffset)
            continue;
          DeletedPopCFIs[CSR].push_back(CFI->getOperand(0).getImm());
        }
      }
      CFIs.clear();
    }
  }
}

bool ShrinkWrapping::doesInstUsesCSR(const MCInst &Inst, uint16_t CSR) {
  if (BC.MIA->isCFI(Inst) || CSA.getSavedReg(Inst) == CSR ||
      CSA.getRestoredReg(Inst) == CSR)
    return false;
  BitVector BV = BitVector(BC.MRI->getNumRegs(), false);
  BC.MIA->getTouchedRegs(Inst, BV, *BC.MRI);
  return BV[CSR];
}

void ShrinkWrapping::scheduleSaveRestoreInsertions(
    unsigned CSR, MCInst *BestPosSave,
    SmallVector<ProgramPoint, 4> &RestorePoints, bool UsePushPops) {
  auto &InsnToBB = Info.getInsnToBBMap();
  auto FIESave = CSA.SaveFIEByReg[CSR];
  auto FIELoad = CSA.LoadFIEByReg[CSR];
  assert(FIESave && FIELoad && "Invalid CSR");

  DEBUG({
    dbgs() << "Scheduling save insertion at: ";
    BestPosSave->dump();
  });

  scheduleChange(BestPosSave, UsePushPops ? WorklistItem::InsertPushOrPop
                                          : WorklistItem::InsertLoadOrStore,
                 *FIESave, CSR);

  for (auto &PP : RestorePoints) {
    BinaryBasicBlock *FrontierBB = Info.getParentBB(PP);
    DEBUG({
      dbgs() << "Scheduling restore insertion at: ";
      if (PP.isInst())
        PP.getInst()->dump();
      else {
        dbgs() << PP.getBB()->getName() << "\n";
      }
    });
    MCInst *Term =
        FrontierBB->getTerminatorBefore(PP.isInst() ? PP.getInst() : nullptr);
    if (Term)
      PP = Term;
    if (PP.isInst() && doesInstUsesCSR(*PP.getInst(), CSR)) {
      assert(!InsnToBB[PP.getInst()]->hasTerminatorAfter(PP.getInst()) &&
             "cannot move to end of bb");
      scheduleChange(InsnToBB[PP.getInst()],
                     UsePushPops ? WorklistItem::InsertPushOrPop
                                 : WorklistItem::InsertLoadOrStore,
                     *FIELoad, CSR);
      continue;
    }
    scheduleChange(PP, UsePushPops ? WorklistItem::InsertPushOrPop
                                   : WorklistItem::InsertLoadOrStore,
                   *FIELoad, CSR);
  }
}

void ShrinkWrapping::moveSaveRestores() {
  bool DisablePushPopMode{false};
  bool UsedPushPopMode{false};
  // Keeps info about successfully moved regs: reg index, save position and
  // save size
  std::vector<std::tuple<unsigned, MCInst *, size_t>> MovedRegs;

  for (unsigned I = 0, E = BC.MRI->getNumRegs(); I != E; ++I) {
    MCInst *BestPosSave{nullptr};
    uint64_t TotalEstimatedWin{0};
    if (!isBestSavePosCold(I, BestPosSave, TotalEstimatedWin))
      continue;
    SmallVector<ProgramPoint, 4> RestorePoints =
        doRestorePlacement(BestPosSave, I, TotalEstimatedWin);
    if (RestorePoints.empty())
      continue;

    auto FIESave = CSA.SaveFIEByReg[I];
    auto FIELoad = CSA.LoadFIEByReg[I];
    assert(FIESave && FIELoad);
    auto &SPT = Info.getStackPointerTracking();
    const auto SPFP = *SPT.getStateBefore(*BestPosSave);
    auto SaveOffset = SPFP.first;
    auto SaveSize = FIESave->Size;

    // If we don't know stack state at this point, bail
    if ((SPFP.first == SPT.SUPERPOSITION || SPFP.first == SPT.EMPTY) &&
        (SPFP.second == SPT.SUPERPOSITION || SPFP.second == SPT.EMPTY))
      continue;

    // Operation mode: if true, will insert push/pops instead of loads/restores
    bool UsePushPops = validatePushPopsMode(I, BestPosSave, SaveOffset);

    if (UsePushPops) {
      auto FixedRestorePoints = fixPopsPlacements(RestorePoints, SaveOffset, I);
      if (FixedRestorePoints.empty())
        UsePushPops = false;
      else
        RestorePoints = FixedRestorePoints;
    }

    // Disable push-pop mode for all CSRs in this function
    if (!UsePushPops)
      DisablePushPopMode = true;
    else
      UsedPushPopMode = true;

    scheduleOldSaveRestoresRemoval(I, UsePushPops);
    scheduleSaveRestoreInsertions(I, BestPosSave, RestorePoints, UsePushPops);
    MovedRegs.emplace_back(std::make_tuple(I, BestPosSave, SaveSize));
  }

  // Revert push-pop mode if it failed for a single CSR
  if (DisablePushPopMode && UsedPushPopMode) {
    UsedPushPopMode = false;
    for (auto &BB : BF) {
      auto WRI = Todo.find(&BB);
      if (WRI != Todo.end()) {
        auto &TodoList = WRI->second;
        for (auto &Item : TodoList) {
          if (Item.Action == WorklistItem::InsertPushOrPop)
            Item.Action = WorklistItem::InsertLoadOrStore;
        }
      }
      for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
        auto &Inst = *I;
        auto TodoList = BC.MIA->tryGetAnnotationAs<std::vector<WorklistItem>>(
            Inst, getAnnotationName());
        if (!TodoList)
          continue;
        bool isCFI = BC.MIA->isCFI(Inst);
        for (auto &Item : *TodoList) {
          if (Item.Action == WorklistItem::InsertPushOrPop)
            Item.Action = WorklistItem::InsertLoadOrStore;
          if (!isCFI && Item.Action == WorklistItem::Erase)
            Item.Action = WorklistItem::ChangeToAdjustment;
        }
      }
    }
  }

  // Update statistics
  if (!UsedPushPopMode) {
    SpillsMovedRegularMode += MovedRegs.size();
    return;
  }

  // Schedule modifications to stack-accessing instructions via
  // StackLayoutModifier.
  SpillsMovedPushPopMode += MovedRegs.size();
  for (auto &I : MovedRegs) {
    unsigned RegNdx;
    MCInst *SavePos;
    size_t SaveSize;
    std::tie(RegNdx, SavePos, SaveSize) = I;
    for (MCInst *Save : CSA.getSavesByReg(RegNdx)) {
      SLM.collapseRegion(Save);
    }
    SLM.insertRegion(SavePos, SaveSize);
  }
}

namespace {
/// Helper function to identify whether two basic blocks created by splitting
/// a critical edge have the same contents.
bool isIdenticalSplitEdgeBB(const BinaryBasicBlock &A,
                            const BinaryBasicBlock &B) {
  if (A.succ_size() != B.succ_size())
    return false;
  if (A.succ_size() != 1)
    return false;

  if (*A.succ_begin() != *B.succ_begin())
    return false;

  if (A.size() != B.size())
    return false;

  // Compare instructions
  auto I = A.begin(), E = A.end();
  auto OtherI = B.begin(), OtherE = B.end();
  while (I != E && OtherI != OtherE) {
    if (I->getOpcode() != OtherI->getOpcode())
      return false;
    if (!I->equals(*OtherI,
                   [](const MCSymbol *A, const MCSymbol *B) { return true; }))
      return false;
    ++I;
    ++OtherI;
  }
  return true;
}
}

bool ShrinkWrapping::foldIdenticalSplitEdges() {
  bool Changed{false};
  for (auto Iter = BF.begin(); Iter != BF.end(); ++Iter) {
    BinaryBasicBlock &BB = *Iter;
    if (!BB.getName().startswith(".LSplitEdge"))
      continue;
    for (auto RIter = BF.rbegin(); RIter != BF.rend(); ++RIter) {
      BinaryBasicBlock &RBB = *RIter;
      if (&RBB == &BB)
        break;
      if (!RBB.getName().startswith(".LSplitEdge") ||
          !RBB.isValid() ||
          !isIdenticalSplitEdgeBB(*Iter, RBB))
        continue;
      assert(RBB.pred_size() == 1 && "Invalid split edge BB");
      BinaryBasicBlock *Pred = *RBB.pred_begin();
      uint64_t OrigCount{Pred->branch_info_begin()->Count};
      uint64_t OrigMispreds{Pred->branch_info_begin()->MispredictedCount};
      Pred->replaceSuccessor(&RBB, &BB, OrigCount, OrigMispreds);
      Changed = true;
      // Remove the block from CFG
      RBB.markValid(false);
    }
  }

  return Changed;
}

namespace {

// A special StackPointerTracking that compensates for our future plans
// in removing/adding insn.
class PredictiveStackPointerTracking
    : public StackPointerTrackingBase<PredictiveStackPointerTracking> {
  friend class DataflowAnalysis<PredictiveStackPointerTracking,
                                std::pair<int, int>>;
  decltype(ShrinkWrapping::Todo) &TodoMap;
  DataflowInfoManager &Info;

protected:
  void compNextAux(const MCInst &Point,
                   const std::vector<ShrinkWrapping::WorklistItem> &TodoItems,
                   std::pair<int, int> &Res) {
    for (const auto &Item : TodoItems) {
      if (Item.Action == ShrinkWrapping::WorklistItem::Erase &&
          BC.MIA->isPush(Point)) {
        Res.first += BC.MIA->getPushSize(Point);
        continue;
      }
      if (Item.Action == ShrinkWrapping::WorklistItem::Erase &&
          BC.MIA->isPop(Point)) {
        Res.first -= BC.MIA->getPopSize(Point);
        continue;
      }
      if (Item.Action == ShrinkWrapping::WorklistItem::InsertPushOrPop &&
          Item.FIEToInsert.IsStore) {
        Res.first -= Item.FIEToInsert.Size;
        continue;
      }
      if (Item.Action == ShrinkWrapping::WorklistItem::InsertPushOrPop &&
          Item.FIEToInsert.IsLoad) {
        Res.first += Item.FIEToInsert.Size;
        continue;
      }
    }
  }

  std::pair<int, int> computeNext(const MCInst &Point,
                                  const std::pair<int, int> &Cur) {
    std::pair<int, int> Res =
        StackPointerTrackingBase<PredictiveStackPointerTracking>::computeNext(
            Point, Cur);
    if (Res.first == StackPointerTracking::SUPERPOSITION ||
        Res.first == StackPointerTracking::EMPTY)
      return Res;
    auto TodoItems =
        BC.MIA->tryGetAnnotationAs<std::vector<ShrinkWrapping::WorklistItem>>(
            Point, ShrinkWrapping::getAnnotationName());
    if (TodoItems)
      compNextAux(Point, *TodoItems, Res);
    auto &InsnToBBMap = Info.getInsnToBBMap();
    if (&*InsnToBBMap[&Point]->rbegin() != &Point)
      return Res;
    auto WRI = TodoMap.find(InsnToBBMap[&Point]);
    if (WRI == TodoMap.end())
      return Res;
    compNextAux(Point, WRI->second, Res);
    return Res;
  }

  StringRef getAnnotationName() const {
    return StringRef("PredictiveStackPointerTracking");
  }

public:
  PredictiveStackPointerTracking(const BinaryContext &BC, BinaryFunction &BF,
                                 decltype(ShrinkWrapping::Todo) &TodoMap,
                                 DataflowInfoManager &Info)
      : StackPointerTrackingBase<PredictiveStackPointerTracking>(BC, BF),
        TodoMap(TodoMap), Info(Info) {}

  void run() {
    NamedRegionTimer T1("PSPT", "Dataflow", opts::TimeOpts);
    StackPointerTrackingBase<PredictiveStackPointerTracking>::run();
  }
};

} // end anonymous namespace

void ShrinkWrapping::insertUpdatedCFI(unsigned CSR, int SPValPush,
                                      int SPValPop) {
  MCInst *SavePoint{nullptr};
  for (auto &BB : BF) {
    for (auto InstIter = BB.rbegin(), EndIter = BB.rend(); InstIter != EndIter;
         ++InstIter) {
      int32_t SrcImm{0};
      MCPhysReg Reg{0};
      MCPhysReg StackPtrReg{0};
      int64_t StackOffset{0};
      bool IsIndexed{false};
      bool IsLoad{false};
      bool IsStore{false};
      bool IsSimple{false};
      bool IsStoreFromReg{false};
      uint8_t Size{0};
      if (!BC.MIA->isStackAccess(*BC.MRI, *InstIter, IsLoad, IsStore,
                                 IsStoreFromReg, Reg, SrcImm, StackPtrReg,
                                 StackOffset, Size, IsSimple, IsIndexed))
        continue;
      if (Reg != CSR || !IsStore || !IsSimple)
        continue;
      SavePoint = &*InstIter;
      break;
    }
    if (SavePoint)
      break;
  }
  assert(SavePoint);
  DEBUG({
    dbgs() << "Now using as save point for reg " << CSR << " :";
    SavePoint->dump();
  });
  bool PrevAffectedZone{false};
  BinaryBasicBlock *PrevBB{nullptr};
  auto &DA = Info.getDominatorAnalysis();
  for (auto BB : BF.layout()) {
    if (BB->size() == 0)
      continue;
    const bool InAffectedZoneAtEnd = DA.count(*BB->rbegin(), *SavePoint);
    const bool InAffectedZoneAtBegin =
        (*DA.getStateBefore(*BB->begin()))[DA.ExprToIdx[SavePoint]];
    bool InAffectedZone = InAffectedZoneAtBegin;
    for (auto InstIter = BB->begin(); InstIter != BB->end(); ++InstIter) {
      const bool CurZone = DA.count(*InstIter, *SavePoint);
      if (InAffectedZone != CurZone) {
        auto InsertionIter = InstIter;
        ++InsertionIter;
        InAffectedZone = CurZone;
        if (InAffectedZone) {
          InstIter = --insertCFIsForPushOrPop(*BB, InsertionIter, CSR, true, 0,
                                              SPValPop);
        } else {
          InstIter = --insertCFIsForPushOrPop(*BB, InsertionIter, CSR, false, 0,
                                              SPValPush);
        }
      }
    }
    // Are we at the hot-cold split point?
    if (BF.isSplit() && PrevBB && BB->isCold() != PrevBB->isCold()) {
      if (InAffectedZoneAtBegin) {
        insertCFIsForPushOrPop(*BB, BB->begin(), CSR, true, 0, SPValPush);
      }
    } else {
      if (InAffectedZoneAtBegin != PrevAffectedZone) {
        if (InAffectedZoneAtBegin) {
          insertCFIsForPushOrPop(*PrevBB, PrevBB->end(), CSR, true, 0,
                                 SPValPush);
        } else {
          insertCFIsForPushOrPop(*PrevBB, PrevBB->end(), CSR, false, 0,
                                 SPValPop);
        }
      }
    }
    PrevAffectedZone = InAffectedZoneAtEnd;
    PrevBB = BB;
  }
}

void ShrinkWrapping::rebuildCFIForSP() {
  for (auto &BB : BF) {
    for (auto &Inst : BB) {
      if (!BC.MIA->isCFI(Inst))
        continue;
      auto *CFI = BF.getCFIFor(Inst);
      if (CFI->getOperation() == MCCFIInstruction::OpDefCfaOffset)
        BC.MIA->addAnnotation(BC.Ctx.get(), Inst, "DeleteMe", 0U);
    }
  }

  int PrevSPVal{-8};
  BinaryBasicBlock *PrevBB{nullptr};
  auto &SPT = Info.getStackPointerTracking();
  for (auto BB : BF.layout()) {
    if (BB->size() == 0)
      continue;
    const int SPValAtEnd = SPT.getStateAt(*BB->rbegin())->first;
    const int SPValAtBegin = SPT.getStateBefore(*BB->begin())->first;
    int SPVal = SPValAtBegin;
    for (auto Iter = BB->begin(); Iter != BB->end(); ++Iter) {
      const int CurVal = SPT.getStateAt(*Iter)->first;
      if (SPVal != CurVal) {
        auto InsertionIter = Iter;
        ++InsertionIter;
        Iter = BF.addCFIInstruction(
            BB, InsertionIter,
            MCCFIInstruction::createDefCfaOffset(nullptr, SPVal));
        SPVal = CurVal;
      }
    }
    if (BF.isSplit() && PrevBB && BB->isCold() != PrevBB->isCold()) {
      BF.addCFIInstruction(
          BB, BB->begin(),
          MCCFIInstruction::createDefCfaOffset(nullptr, SPValAtBegin));
    } else {
      if (SPValAtBegin != PrevSPVal) {
        BF.addCFIInstruction(
            PrevBB, PrevBB->end(),
            MCCFIInstruction::createDefCfaOffset(nullptr, SPValAtBegin));
      }
    }
    PrevSPVal = SPValAtEnd;
    PrevBB = BB;
  }

  for (auto &BB : BF)
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I)
      if (BC.MIA->hasAnnotation(*I, "DeleteMe"))
        BB.eraseInstruction(&*I);
}

MCInst ShrinkWrapping::createStackAccess(int SPVal, int FPVal,
                                         const FrameIndexEntry &FIE,
                                         bool CreatePushOrPop) {
  MCInst NewInst;
  if (SPVal != StackPointerTracking::SUPERPOSITION &&
      SPVal != StackPointerTracking::EMPTY) {
    if (FIE.IsLoad) {
      if (!BC.MIA->createRestoreFromStack(NewInst, BC.MIA->getStackPointer(),
                                          FIE.StackOffset - SPVal, FIE.RegOrImm,
                                          FIE.Size)) {
        errs() << "createRestoreFromStack: not supported on this platform\n";
        abort();
      }
    } else {
      if (!BC.MIA->createSaveToStack(NewInst, BC.MIA->getStackPointer(),
                                     FIE.StackOffset - SPVal, FIE.RegOrImm,
                                     FIE.Size)) {
        errs() << "createSaveToStack: not supported on this platform\n";
        abort();
      }
    }
    if (CreatePushOrPop)
      BC.MIA->changeToPushOrPop(NewInst);
    return NewInst;
  }
  assert(FPVal != StackPointerTracking::SUPERPOSITION &&
         FPVal != StackPointerTracking::EMPTY);

  if (FIE.IsLoad) {
    if (!BC.MIA->createRestoreFromStack(NewInst, BC.MIA->getFramePointer(),
                                        FIE.StackOffset - FPVal, FIE.RegOrImm,
                                        FIE.Size)) {
      errs() << "createRestoreFromStack: not supported on this platform\n";
      abort();
    }
  } else {
    if (!BC.MIA->createSaveToStack(NewInst, BC.MIA->getFramePointer(),
                                   FIE.StackOffset - FPVal, FIE.RegOrImm,
                                   FIE.Size)) {
      errs() << "createSaveToStack: not supported on this platform\n";
      abort();
    }
  }
  return NewInst;
}

void ShrinkWrapping::updateCFIInstOffset(MCInst &Inst, int64_t NewOffset) {
  auto *CFI = BF.getCFIFor(Inst);
  if (UpdatedCFIs.count(CFI))
    return;

  switch (CFI->getOperation()) {
  case MCCFIInstruction::OpDefCfa:
  case MCCFIInstruction::OpDefCfaRegister:
  case MCCFIInstruction::OpDefCfaOffset:
    CFI->setOffset(NewOffset);
    break;
  case MCCFIInstruction::OpOffset:
  default:
    break;
  }

  UpdatedCFIs.insert(CFI);
}

BBIterTy ShrinkWrapping::insertCFIsForPushOrPop(BinaryBasicBlock &BB,
                                                BBIterTy Pos, unsigned Reg,
                                                bool isPush, int Sz,
                                                int64_t NewOffset) {
  if (isPush) {
    for (uint32_t Idx : DeletedPushCFIs[Reg]) {
      Pos = BF.addCFIPseudo(&BB, Pos, Idx);
      updateCFIInstOffset(*Pos++, NewOffset);
    }
    if (HasDeletedOffsetCFIs[Reg]) {
      Pos = ++BF.addCFIInstruction(
          &BB, Pos,
          MCCFIInstruction::createOffset(
              nullptr, BC.MRI->getDwarfRegNum(Reg, false), NewOffset));
    }
  } else {
    for (uint32_t Idx : DeletedPopCFIs[Reg]) {
      Pos = BF.addCFIPseudo(&BB, Pos, Idx);
      updateCFIInstOffset(*Pos++, NewOffset);
    }
    if (HasDeletedOffsetCFIs[Reg]) {
      Pos = ++BF.addCFIInstruction(
          &BB, Pos,
          MCCFIInstruction::createSameValue(
              nullptr, BC.MRI->getDwarfRegNum(Reg, false)));
    }
  }
  return Pos;
}

BBIterTy ShrinkWrapping::processInsertion(BBIterTy InsertionPoint,
                                          BinaryBasicBlock *CurBB,
                                          const WorklistItem &Item,
                                          int64_t SPVal, int64_t FPVal) {
  // Trigger CFI reconstruction for this CSR if necessary - writing to
  // PushOffsetByReg/PopOffsetByReg *will* trigger CFI update
  if ((Item.FIEToInsert.IsStore &&
       !DeletedPushCFIs[Item.AffectedReg].empty()) ||
      (Item.FIEToInsert.IsLoad && !DeletedPopCFIs[Item.AffectedReg].empty()) ||
      HasDeletedOffsetCFIs[Item.AffectedReg]) {
    if (Item.Action == WorklistItem::InsertPushOrPop) {
      if (Item.FIEToInsert.IsStore)
        PushOffsetByReg[Item.AffectedReg] = SPVal - Item.FIEToInsert.Size;
      else
        PopOffsetByReg[Item.AffectedReg] = SPVal;
    } else {
      if (Item.FIEToInsert.IsStore)
        PushOffsetByReg[Item.AffectedReg] = Item.FIEToInsert.StackOffset;
      else
        PopOffsetByReg[Item.AffectedReg] = Item.FIEToInsert.StackOffset;
    }
  }

  DEBUG({
    dbgs() << "Creating stack access with SPVal = " << SPVal
           << "; stack offset = " << Item.FIEToInsert.StackOffset
           << " Is push = " << (Item.Action == WorklistItem::InsertPushOrPop)
           << "\n";
  });
  MCInst NewInst =
      createStackAccess(SPVal, FPVal, Item.FIEToInsert,
                        Item.Action == WorklistItem::InsertPushOrPop);
  if (InsertionPoint != CurBB->end()) {
    DEBUG({
      dbgs() << "Adding before Inst: ";
      InsertionPoint->dump();
      dbgs() << "the following inst: ";
      NewInst.dump();
    });
    return ++CurBB->insertInstruction(InsertionPoint, std::move(NewInst));
  }
  CurBB->addInstruction(std::move(NewInst));
  DEBUG(dbgs() << "Adding to BB!\n");
  return CurBB->end();
}

BBIterTy ShrinkWrapping::processInsertionsList(
    BBIterTy InsertionPoint, BinaryBasicBlock *CurBB,
    std::vector<WorklistItem> &TodoList, int64_t SPVal, int64_t FPVal) {
  bool HasInsertions{false};
  for (auto &Item : TodoList) {
    if (Item.Action == WorklistItem::Erase ||
        Item.Action == WorklistItem::ChangeToAdjustment)
      continue;
    HasInsertions = true;
    break;
  }

  if (!HasInsertions)
    return InsertionPoint;

  assert(((SPVal != StackPointerTracking::SUPERPOSITION &&
           SPVal != StackPointerTracking::EMPTY) ||
          (FPVal != StackPointerTracking::SUPERPOSITION &&
           FPVal != StackPointerTracking::EMPTY)) &&
         "Cannot insert if we have no idea of the stack state here");

  // Revert the effect of PSPT for this location, we want SP Value before
  // insertions
  if (InsertionPoint == CurBB->end()) {
    for (auto &Item : TodoList) {
      if (Item.Action != WorklistItem::InsertPushOrPop)
        continue;
      if (Item.FIEToInsert.IsStore)
        SPVal += Item.FIEToInsert.Size;
      if (Item.FIEToInsert.IsLoad)
        SPVal -= Item.FIEToInsert.Size;
    }
  }

  // Reorder POPs to obey the correct dominance relation between them
  std::stable_sort(TodoList.begin(), TodoList.end(), [&](const WorklistItem &A,
                                                         const WorklistItem
                                                             &B) {
    if ((A.Action != WorklistItem::InsertPushOrPop || !A.FIEToInsert.IsLoad) &&
        (B.Action != WorklistItem::InsertPushOrPop || !B.FIEToInsert.IsLoad))
      return false;
    if ((A.Action != WorklistItem::InsertPushOrPop || !A.FIEToInsert.IsLoad))
      return false;
    if ((B.Action != WorklistItem::InsertPushOrPop || !B.FIEToInsert.IsLoad))
      return true;
    return DomOrder[B.AffectedReg] < DomOrder[A.AffectedReg];
  });

  // Process insertions
  for (auto &Item : TodoList) {
    if (Item.Action == WorklistItem::Erase ||
        Item.Action == WorklistItem::ChangeToAdjustment)
      continue;

    InsertionPoint =
        processInsertion(InsertionPoint, CurBB, Item, SPVal, FPVal);
    if (Item.Action == WorklistItem::InsertPushOrPop &&
        Item.FIEToInsert.IsStore) {
      SPVal -= Item.FIEToInsert.Size;
    }
    if (Item.Action == WorklistItem::InsertPushOrPop &&
        Item.FIEToInsert.IsLoad) {
      SPVal += Item.FIEToInsert.Size;
    }
  }
  return InsertionPoint;
}

bool ShrinkWrapping::processInsertions() {
  PredictiveStackPointerTracking PSPT(BC, BF, Todo, Info);
  PSPT.run();

  bool Changes{false};
  for (auto &BB : BF) {
    // Process insertions before some inst.
    for (auto I = BB.begin(); I != BB.end(); ++I) {
      auto &Inst = *I;
      auto TodoList = BC.MIA->tryGetAnnotationAs<std::vector<WorklistItem>>(
          Inst, getAnnotationName());
      if (!TodoList)
        continue;
      Changes = true;
      auto List = *TodoList;
      DEBUG({
        dbgs() << "Now processing insertions in " << BB.getName()
               << " before inst: ";
        Inst.dump();
      });
      auto Iter = I;
      auto SPTState =
          *PSPT.getStateAt(Iter == BB.begin() ? (ProgramPoint)&BB : &*(--Iter));
      I = processInsertionsList(I, &BB, List, SPTState.first, SPTState.second);
    }
    // Process insertions at the end of bb
    auto WRI = Todo.find(&BB);
    if (WRI != Todo.end()) {
      auto SPTState = *PSPT.getStateAt(*BB.rbegin());
      processInsertionsList(BB.end(), &BB, WRI->second, SPTState.first,
                            SPTState.second);
      Changes = true;
    }
  }
  return Changes;
}

void ShrinkWrapping::processDeletions() {
  auto &LA = Info.getLivenessAnalysis();
  for (auto &BB : BF) {
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
      auto &Inst = *I;
      auto TodoList = BC.MIA->tryGetAnnotationAs<std::vector<WorklistItem>>(
          Inst, getAnnotationName());
      if (!TodoList)
        continue;
      // Process all deletions
      for (auto &Item : *TodoList) {
        if (Item.Action != WorklistItem::Erase &&
            Item.Action != WorklistItem::ChangeToAdjustment)
          continue;

        if (Item.Action == WorklistItem::ChangeToAdjustment) {
          // Is flag reg alive across this func?
          bool DontClobberFlags = LA.isAlive(&Inst, BC.MIA->getFlagsReg());
          if (auto Sz = BC.MIA->getPushSize(Inst)) {
            BC.MIA->createStackPointerIncrement(Inst, Sz, DontClobberFlags);
            continue;
          }
          if (auto Sz = BC.MIA->getPopSize(Inst)) {
            BC.MIA->createStackPointerDecrement(Inst, Sz, DontClobberFlags);
            continue;
          }
        }

        DEBUG({
          dbgs() << "Erasing: ";
          Inst.dump();
        });
        BB.eraseInstruction(&Inst);
        break;
      }
    }
  }
}

void ShrinkWrapping::rebuildCFI() {
  const bool FP = Info.getStackPointerTracking().HasFramePointer;
  Info.invalidateAll();
  if (!FP) {
    rebuildCFIForSP();
    Info.invalidateAll();
  }
  for (unsigned I = 0, E = BC.MRI->getNumRegs(); I != E; ++I) {
    if (PushOffsetByReg[I] == 0 || PopOffsetByReg[I] == 0)
      continue;
    const int64_t SPValPush = PushOffsetByReg[I];
    const int64_t SPValPop = PopOffsetByReg[I];
    insertUpdatedCFI(I, SPValPush, SPValPop);
    Info.invalidateAll();
  }
}

void ShrinkWrapping::perform() {
  HasDeletedOffsetCFIs = std::vector<bool>(BC.MRI->getNumRegs(), false);
  PushOffsetByReg = std::vector<int64_t>(BC.MRI->getNumRegs(), 0LL);
  PopOffsetByReg = std::vector<int64_t>(BC.MRI->getNumRegs(), 0LL);
  DomOrder = std::vector<MCPhysReg>(BC.MRI->getNumRegs(), 0);

  SLM.initialize();
  CSA.compute();
  classifyCSRUses();
  pruneUnwantedCSRs();
  computeSaveLocations();
  computeDomOrder();
  moveSaveRestores();
  DEBUG({
    dbgs() << "Func before shrink-wrapping: \n";
    BF.dump();
  });
  SLM.performChanges();
  // Early exit if processInsertions doesn't detect any todo items
  if (!processInsertions())
    return;
  processDeletions();
  if (foldIdenticalSplitEdges()) {
    const auto Stats = BF.eraseInvalidBBs();
    DEBUG(dbgs() << "Deleted " << Stats.first << " redundant split edge BBs ("
                 << Stats.second << " bytes) for " << BF.getPrintName()
                 << "\n");
  }
  rebuildCFI();
  // We may have split edges, creating BBs that need correct branching
  BF.fixBranches();
  DEBUG({
    dbgs() << "Func after shrink-wrapping: \n";
    BF.dump();
  });
}

void ShrinkWrapping::printStats() {
  outs() << "BOLT-INFO: Shrink wrapping moved " << SpillsMovedRegularMode
         << " spills inserting load/stores and " << SpillsMovedPushPopMode
         << " spills inserting push/pops\n";
}

// Operators necessary as a result of using MCAnnotation
raw_ostream &operator<<(raw_ostream &OS,
                        const std::vector<ShrinkWrapping::WorklistItem> &Vec) {
  OS << "SWTodo[";
  auto Sep = "";
  for (const auto &Item : Vec) {
    OS << Sep;
    switch (Item.Action) {
    case ShrinkWrapping::WorklistItem::Erase:
      OS << "Erase";
      break;
    case ShrinkWrapping::WorklistItem::ChangeToAdjustment:
      OS << "ChangeToAdjustment";
      break;
    case ShrinkWrapping::WorklistItem::InsertLoadOrStore:
      OS << "InsertLoadOrStore";
      break;
    case ShrinkWrapping::WorklistItem::InsertPushOrPop:
      OS << "InsertPushOrPop";
      break;
    }
    Sep = ", ";
  }
  OS << "]";
  return OS;
}

raw_ostream &
operator<<(raw_ostream &OS,
           const std::vector<StackLayoutModifier::WorklistItem> &Vec) {
  OS << "SLMTodo[";
  auto Sep = "";
  for (const auto &Item : Vec) {
    OS << Sep;
    switch (Item.Action) {
    case StackLayoutModifier::WorklistItem::None:
      OS << "None";
      break;
    case StackLayoutModifier::WorklistItem::AdjustLoadStoreOffset:
      OS << "AdjustLoadStoreOffset";
      break;
    case StackLayoutModifier::WorklistItem::AdjustCFI:
      OS << "AdjustCFI";
      break;
    }
    Sep = ", ";
  }
  OS << "]";
  return OS;
}

bool operator==(const ShrinkWrapping::WorklistItem &A,
                const ShrinkWrapping::WorklistItem &B) {
  return (A.Action == B.Action && A.AffectedReg == B.AffectedReg &&
          A.Adjustment == B.Adjustment &&
          A.FIEToInsert.IsLoad == B.FIEToInsert.IsLoad &&
          A.FIEToInsert.IsStore == B.FIEToInsert.IsStore &&
          A.FIEToInsert.RegOrImm == B.FIEToInsert.RegOrImm &&
          A.FIEToInsert.Size == B.FIEToInsert.Size &&
          A.FIEToInsert.IsSimple == B.FIEToInsert.IsSimple &&
          A.FIEToInsert.StackOffset == B.FIEToInsert.StackOffset);
}

bool operator==(const StackLayoutModifier::WorklistItem &A,
                const StackLayoutModifier::WorklistItem &B) {
  return (A.Action == B.Action && A.OffsetUpdate == B.OffsetUpdate);
}

} // end namespace bolt
} // end namespace llvm
