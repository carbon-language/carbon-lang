//===--- Passes/LongJmp.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LongJmp.h"

#define DEBUG_TYPE "longjmp"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> UseOldText;
extern cl::opt<unsigned> AlignFunctions;
extern cl::opt<unsigned> AlignFunctionsMaxBytes;
extern cl::opt<bool> HotFunctionsAtEnd;

static cl::opt<bool>
GroupStubs("group-stubs",
  cl::desc("share stubs across functions"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));
}

namespace llvm {
namespace bolt {

namespace {
constexpr unsigned ColdFragAlign = 16;

void relaxStubToShortJmp(BinaryBasicBlock &StubBB, const MCSymbol *Tgt) {
  const BinaryContext &BC = StubBB.getFunction()->getBinaryContext();
  std::vector<MCInst> Seq;
  BC.MIB->createShortJmp(Seq, Tgt, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstructions(Seq.begin(), Seq.end());
}

void relaxStubToLongJmp(BinaryBasicBlock &StubBB, const MCSymbol *Tgt) {
  const BinaryContext &BC = StubBB.getFunction()->getBinaryContext();
  std::vector<MCInst> Seq;
  BC.MIB->createLongJmp(Seq, Tgt, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstructions(Seq.begin(), Seq.end());
}

BinaryBasicBlock *getBBAtHotColdSplitPoint(BinaryFunction &Func) {
  if (!Func.isSplit() || Func.empty())
    return nullptr;

  assert(!(*Func.begin()).isCold() && "Entry cannot be cold");
  for (auto I = Func.layout_begin(), E = Func.layout_end(); I != E; ++I) {
    auto Next = std::next(I);
    if (Next != E && (*Next)->isCold())
      return *I;
  }
  llvm_unreachable("No hot-colt split point found");
}

bool shouldInsertStub(const BinaryContext &BC, const MCInst &Inst) {
  return (BC.MIB->isBranch(Inst) || BC.MIB->isCall(Inst)) &&
         !BC.MIB->isIndirectBranch(Inst) && !BC.MIB->isIndirectCall(Inst);
}

} // end anonymous namespace

std::pair<std::unique_ptr<BinaryBasicBlock>, MCSymbol *>
LongJmpPass::createNewStub(BinaryBasicBlock &SourceBB, const MCSymbol *TgtSym,
                           bool TgtIsFunc, uint64_t AtAddress) {
  BinaryFunction &Func = *SourceBB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  const bool IsCold = SourceBB.isCold();
  auto *StubSym = BC.Ctx->createTempSymbol("Stub", true);
  auto StubBB = Func.createBasicBlock(0, StubSym);
  MCInst Inst;
  BC.MIB->createUncondBranch(Inst, TgtSym, BC.Ctx.get());
  if (TgtIsFunc)
    BC.MIB->convertJmpToTailCall(Inst, BC.Ctx.get());
  StubBB->addInstruction(Inst);
  StubBB->setExecutionCount(0);

  // Register this in stubs maps
  auto registerInMap = [&](StubGroupsTy &Map) {
    auto &StubGroup = Map[TgtSym];
    StubGroup.insert(
        std::lower_bound(
            StubGroup.begin(), StubGroup.end(),
            std::make_pair(AtAddress, nullptr),
            [&](const std::pair<uint64_t, BinaryBasicBlock *> &LHS,
                const std::pair<uint64_t, BinaryBasicBlock *> &RHS) {
              return LHS.first < RHS.first;
            }),
        std::make_pair(AtAddress, StubBB.get()));
  };

  Stubs[&Func].insert(StubBB.get());
  StubBits[StubBB.get()] = BC.MIB->getUncondBranchEncodingSize();
  if (IsCold) {
    registerInMap(ColdLocalStubs[&Func]);
    if (opts::GroupStubs && TgtIsFunc)
      registerInMap(ColdStubGroups);
    ++NumColdStubs;
  } else {
    registerInMap(HotLocalStubs[&Func]);
    if (opts::GroupStubs && TgtIsFunc)
      registerInMap(HotStubGroups);
    ++NumHotStubs;
  }

  return std::make_pair(std::move(StubBB), StubSym);
}

BinaryBasicBlock *LongJmpPass::lookupStubFromGroup(
    const StubGroupsTy &StubGroups, const BinaryFunction &Func,
    const MCInst &Inst, const MCSymbol *TgtSym, uint64_t DotAddress) const {
  const BinaryContext &BC = Func.getBinaryContext();
  auto CandidatesIter = StubGroups.find(TgtSym);
  if (CandidatesIter == StubGroups.end())
    return nullptr;
  auto &Candidates = CandidatesIter->second;
  if (Candidates.empty())
    return nullptr;
  auto Cand = std::lower_bound(
      Candidates.begin(), Candidates.end(), std::make_pair(DotAddress, nullptr),
      [&](const std::pair<uint64_t, BinaryBasicBlock *> &LHS,
          const std::pair<uint64_t, BinaryBasicBlock *> &RHS) {
        return LHS.first < RHS.first;
      });
  if (Cand != Candidates.begin()) {
    auto LeftCand = Cand;
    --LeftCand;
    if (Cand->first - DotAddress >
        DotAddress - LeftCand->first)
      Cand = LeftCand;
  }
  auto BitsAvail = BC.MIB->getPCRelEncodingSize(Inst) - 1;
  uint64_t Mask = ~((1ULL << BitsAvail) - 1);
  uint64_t PCRelTgtAddress = Cand->first;
  PCRelTgtAddress = DotAddress > PCRelTgtAddress ? DotAddress - PCRelTgtAddress
                                                 : PCRelTgtAddress - DotAddress;
  DEBUG({
    if (Candidates.size() > 1)
      dbgs() << "Considering stub group with " << Candidates.size()
             << " candidates. DotAddress is " << Twine::utohexstr(DotAddress)
             << ", chosen candidate address is "
             << Twine::utohexstr(Cand->first) << "\n";
  });
  return PCRelTgtAddress & Mask ? nullptr : Cand->second;
}

BinaryBasicBlock *
LongJmpPass::lookupGlobalStub(const BinaryBasicBlock &SourceBB,
                              const MCInst &Inst, const MCSymbol *TgtSym,
                              uint64_t DotAddress) const {
  const BinaryFunction &Func = *SourceBB.getFunction();
  const StubGroupsTy &StubGroups =
      SourceBB.isCold() ? ColdStubGroups : HotStubGroups;
  return lookupStubFromGroup(StubGroups, Func, Inst, TgtSym,
                             DotAddress);
}

BinaryBasicBlock *LongJmpPass::lookupLocalStub(const BinaryBasicBlock &SourceBB,
                                               const MCInst &Inst,
                                               const MCSymbol *TgtSym,
                                               uint64_t DotAddress) const {
  const BinaryFunction &Func = *SourceBB.getFunction();
  const DenseMap<const BinaryFunction *, StubGroupsTy> &StubGroups =
      SourceBB.isCold() ? ColdLocalStubs : HotLocalStubs;
  const auto Iter = StubGroups.find(&Func);
  if (Iter == StubGroups.end())
    return nullptr;
  return lookupStubFromGroup(Iter->second, Func, Inst, TgtSym, DotAddress);
}

std::unique_ptr<BinaryBasicBlock>
LongJmpPass::replaceTargetWithStub(BinaryBasicBlock &BB, MCInst &Inst,
                                   uint64_t DotAddress,
                                   uint64_t StubCreationAddress) {
  const BinaryFunction &Func = *BB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  std::unique_ptr<BinaryBasicBlock> NewBB;
  auto TgtSym = BC.MIB->getTargetSymbol(Inst);
  assert (TgtSym && "getTargetSymbol failed");

  BinaryBasicBlock::BinaryBranchInfo BI{0, 0};
  auto *TgtBB = BB.getSuccessor(TgtSym, BI);
  auto LocalStubsIter = Stubs.find(&Func);

  // If already using stub and the stub is from another function, create a local
  // stub, since the foreign stub is now out of range
  if (!TgtBB) {
    auto SSIter = SharedStubs.find(TgtSym);
    if (SSIter != SharedStubs.end()) {
      TgtSym = BC.MIB->getTargetSymbol(*SSIter->second->begin());
      --NumSharedStubs;
    }
  } else if (LocalStubsIter != Stubs.end() &&
             LocalStubsIter->second.count(TgtBB)) {
    // If we are replacing a local stub (because it is now out of range),
    // use its target instead of creating a stub to jump to another stub
    TgtSym = BC.MIB->getTargetSymbol(*TgtBB->begin());
    TgtBB = BB.getSuccessor(TgtSym, BI);
  }

  BinaryBasicBlock *StubBB = lookupLocalStub(BB, Inst, TgtSym, DotAddress);
  // If not found, look it up in globally shared stub maps if it is a function
  // call (TgtBB is not set)
  if (!StubBB && !TgtBB) {
    StubBB = lookupGlobalStub(BB, Inst, TgtSym, DotAddress);
    if (StubBB) {
      SharedStubs[StubBB->getLabel()] = StubBB;
      ++NumSharedStubs;
    }
  }
  MCSymbol *StubSymbol = StubBB ? StubBB->getLabel() : nullptr;

  if (!StubBB) {
    std::tie(NewBB, StubSymbol) =
        createNewStub(BB, TgtSym, /*is func?*/ !TgtBB, StubCreationAddress);
    StubBB = NewBB.get();
  }

  // Local branch
  if (TgtBB) {
    uint64_t OrigCount{BI.Count};
    uint64_t OrigMispreds{BI.MispredictedCount};
    BB.replaceSuccessor(TgtBB, StubBB, OrigCount, OrigMispreds);
    StubBB->setExecutionCount(StubBB->getExecutionCount() + OrigCount);
    if (NewBB) {
      StubBB->addSuccessor(TgtBB, OrigCount, OrigMispreds);
      StubBB->setIsCold(BB.isCold());
    }
  // Call / tail call
  } else {
    StubBB->setExecutionCount(StubBB->getExecutionCount() +
                              BB.getExecutionCount());
    if (NewBB) {
      assert(TgtBB == nullptr);
      StubBB->setIsCold(BB.isCold());
      // Set as entry point because this block is valid but we have no preds
      StubBB->setEntryPoint(true);
    }
  }
  BC.MIB->replaceBranchTarget(Inst, StubSymbol, BC.Ctx.get());

  return NewBB;
}

void LongJmpPass::updateStubGroups() {
  auto update = [&](StubGroupsTy &StubGroups) {
    for (auto &KeyVal : StubGroups) {
      for (auto &Elem : KeyVal.second) {
        Elem.first = BBAddresses[Elem.second];
      }
      std::sort(KeyVal.second.begin(), KeyVal.second.end(),
                [&](const std::pair<uint64_t, BinaryBasicBlock *> &LHS,
                    const std::pair<uint64_t, BinaryBasicBlock *> &RHS) {
                  return LHS.first < RHS.first;
                });
    }
  };

  for (auto &KeyVal : HotLocalStubs)
    update(KeyVal.second);
  for (auto &KeyVal : ColdLocalStubs)
    update(KeyVal.second);
  update(HotStubGroups);
  update(ColdStubGroups);
}

void LongJmpPass::tentativeBBLayout(const BinaryFunction &Func) {
  const BinaryContext &BC = Func.getBinaryContext();
  uint64_t HotDot = HotAddresses[&Func];
  uint64_t ColdDot = ColdAddresses[&Func];
  bool Cold{false};
  for (auto *BB : Func.layout()) {
    if (Cold || BB->isCold()) {
      Cold = true;
      BBAddresses[BB] = ColdDot;
      ColdDot += BC.computeCodeSize(BB->begin(), BB->end());
    } else {
      BBAddresses[BB] = HotDot;
      HotDot += BC.computeCodeSize(BB->begin(), BB->end());
    }
  }
}

uint64_t LongJmpPass::tentativeLayoutRelocColdPart(
  const BinaryContext &BC, std::vector<BinaryFunction *> &SortedFunctions,
  uint64_t DotAddress) {
  for (auto Func : SortedFunctions) {
    if (!Func->isSplit())
      continue;
    DotAddress = alignTo(DotAddress, BinaryFunction::MinAlign);
    auto Pad = OffsetToAlignment(DotAddress, opts::AlignFunctions);
    if (Pad <= opts::AlignFunctionsMaxBytes)
      DotAddress += Pad;
    ColdAddresses[Func] = DotAddress;
    DEBUG(dbgs() << Func->getPrintName() << " cold tentative: "
                 << Twine::utohexstr(DotAddress) << "\n");
    DotAddress += Func->estimateColdSize();
    DotAddress += Func->estimateConstantIslandSize();
  }
  return DotAddress;
}

uint64_t LongJmpPass::tentativeLayoutRelocMode(
  const BinaryContext &BC, std::vector<BinaryFunction *> &SortedFunctions,
  uint64_t DotAddress) {

  // Compute hot cold frontier
  uint32_t LastHotIndex = -1u;
  uint32_t CurrentIndex = 0;
  if (opts::HotFunctionsAtEnd) {
    for (auto *BF : SortedFunctions) {
      if (BF->hasValidIndex() && LastHotIndex == -1u) {
        LastHotIndex = CurrentIndex;
      }
      ++CurrentIndex;
    }
  } else {
    for (auto *BF : SortedFunctions) {
      if (!BF->hasValidIndex() && LastHotIndex == -1u) {
        LastHotIndex = CurrentIndex;
      }
      ++CurrentIndex;
    }
  }

  // Hot
  CurrentIndex = 0;
  bool ColdLayoutDone = false;
  for (auto Func : SortedFunctions) {
    if (!ColdLayoutDone && CurrentIndex >= LastHotIndex) {
      DotAddress =
          tentativeLayoutRelocColdPart(BC, SortedFunctions, DotAddress);
      ColdLayoutDone = true;
      if (opts::HotFunctionsAtEnd)
        DotAddress = alignTo(DotAddress, BC.PageAlign);
    }

    DotAddress = alignTo(DotAddress, BinaryFunction::MinAlign);
    auto Pad = OffsetToAlignment(DotAddress, opts::AlignFunctions);
    if (Pad <= opts::AlignFunctionsMaxBytes)
      DotAddress += Pad;
    HotAddresses[Func] = DotAddress;
    DEBUG(dbgs() << Func->getPrintName()
                 << " tentative: " << Twine::utohexstr(DotAddress) << "\n");
    if (!Func->isSplit())
      DotAddress += Func->estimateSize();
    else
      DotAddress += Func->estimateHotSize();
    DotAddress += Func->estimateConstantIslandSize();
    ++CurrentIndex;
  }
  // BBs
  for (auto Func : SortedFunctions)
    tentativeBBLayout(*Func);

  return DotAddress;
}

void LongJmpPass::tentativeLayout(
    const BinaryContext &BC,
    std::vector<BinaryFunction *> &SortedFunctions) {
  uint64_t DotAddress = BC.LayoutStartAddress;

  if (!BC.HasRelocations) {
    for (auto Func : SortedFunctions) {
      HotAddresses[Func] = Func->getAddress();
      DotAddress = alignTo(DotAddress, ColdFragAlign);
      ColdAddresses[Func] = DotAddress;
      if (Func->isSplit())
        DotAddress += Func->estimateColdSize();
      tentativeBBLayout(*Func);
    }

    return;
  }

  // Relocation mode
  auto EstimatedTextSize = tentativeLayoutRelocMode(BC, SortedFunctions, 0);

  // Initial padding
  if (opts::UseOldText && EstimatedTextSize <= BC.OldTextSectionSize) {
    DotAddress = BC.OldTextSectionAddress;
    auto Pad = OffsetToAlignment(DotAddress, BC.PageAlign);
    if (Pad + EstimatedTextSize <= BC.OldTextSectionSize) {
      DotAddress += Pad;
    }
  } else {
    DotAddress = alignTo(BC.LayoutStartAddress, BC.PageAlign);
  }

  tentativeLayoutRelocMode(BC, SortedFunctions, DotAddress);
}

bool LongJmpPass::usesStub(const BinaryFunction &Func,
                           const MCInst &Inst) const {
  auto TgtSym = Func.getBinaryContext().MIB->getTargetSymbol(Inst);
  auto *TgtBB = Func.getBasicBlockForLabel(TgtSym);
  auto Iter = Stubs.find(&Func);
  if (Iter != Stubs.end())
    return Iter->second.count(TgtBB);
  return false;
}

uint64_t LongJmpPass::getSymbolAddress(const BinaryContext &BC,
                                       const MCSymbol *Target,
                                       const BinaryBasicBlock *TgtBB) const {
  if (TgtBB) {
    auto Iter = BBAddresses.find(TgtBB);
    assert (Iter != BBAddresses.end() && "Unrecognized BB");
    return Iter->second;
  }
  auto *TargetFunc = BC.getFunctionForSymbol(Target);
  auto Iter = HotAddresses.find(TargetFunc);
  if (Iter == HotAddresses.end()) {
    // Look at BinaryContext's resolution for this symbol - this is a symbol not
    // mapped to a BinaryFunction
    auto *BD = BC.getBinaryDataByName(Target->getName());
    assert(BD && "Unrecognized symbol");
    return BD ? BD->getAddress() : 0;
  }
  return Iter->second;
}

bool LongJmpPass::relaxStub(BinaryBasicBlock &StubBB) {
  const BinaryFunction &Func = *StubBB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  const auto Bits = StubBits[&StubBB];
  // Already working with the largest range?
  if (Bits == static_cast<int>(BC.AsmInfo->getCodePointerSize() * 8))
    return false;

  const static auto RangeShortJmp = BC.MIB->getShortJmpEncodingSize();
  const static auto RangeSingleInstr = BC.MIB->getUncondBranchEncodingSize();
  const static uint64_t ShortJmpMask = ~((1ULL << RangeShortJmp) - 1);
  const static uint64_t SingleInstrMask =
      ~((1ULL << (RangeSingleInstr - 1)) - 1);

  auto *RealTargetSym = BC.MIB->getTargetSymbol(*StubBB.begin());
  auto *TgtBB = Func.getBasicBlockForLabel(RealTargetSym);
  uint64_t TgtAddress = getSymbolAddress(BC, RealTargetSym, TgtBB);
  uint64_t DotAddress = BBAddresses[&StubBB];
  uint64_t PCRelTgtAddress = DotAddress > TgtAddress ? DotAddress - TgtAddress
                                            : TgtAddress - DotAddress;
  // If it fits in one instruction, do not relax
  if (!(PCRelTgtAddress & SingleInstrMask))
    return false;

  // Fits short jmp
  if (!(PCRelTgtAddress & ShortJmpMask)) {
    if (Bits >= RangeShortJmp)
      return false;

    DEBUG(dbgs() << "Relaxing stub to short jump. PCRelTgtAddress = "
                 << Twine::utohexstr(PCRelTgtAddress)
                 << " RealTargetSym = " << RealTargetSym->getName() << "\n");
    relaxStubToShortJmp(StubBB, RealTargetSym);
    StubBits[&StubBB] = RangeShortJmp;
    return true;
  }

  // Needs a long jmp
  if (Bits > RangeShortJmp)
    return false;

  DEBUG(dbgs() << "Relaxing stub to long jump. PCRelTgtAddress = "
               << Twine::utohexstr(PCRelTgtAddress)
               << " RealTargetSym = " << RealTargetSym->getName() << "\n");
  relaxStubToLongJmp(StubBB, RealTargetSym);
  StubBits[&StubBB] = static_cast<int>(BC.AsmInfo->getCodePointerSize() * 8);
  return true;
}

bool LongJmpPass::needsStub(const BinaryBasicBlock &BB, const MCInst &Inst,
                            uint64_t DotAddress) const {
  const BinaryFunction &Func = *BB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  auto TgtSym = BC.MIB->getTargetSymbol(Inst);
  assert (TgtSym && "getTargetSymbol failed");

  auto *TgtBB = Func.getBasicBlockForLabel(TgtSym);
  // Check for shared stubs from foreign functions
  if (!TgtBB) {
    auto SSIter = SharedStubs.find(TgtSym);
    if (SSIter != SharedStubs.end()) {
      TgtBB = SSIter->second;
    }
  }

  auto BitsAvail = BC.MIB->getPCRelEncodingSize(Inst) - 1;
  uint64_t Mask = ~((1ULL << BitsAvail) - 1);

  uint64_t PCRelTgtAddress = getSymbolAddress(BC, TgtSym, TgtBB);
  PCRelTgtAddress = DotAddress > PCRelTgtAddress ? DotAddress - PCRelTgtAddress
                                                 : PCRelTgtAddress - DotAddress;

  return PCRelTgtAddress & Mask;
}

bool LongJmpPass::relax(BinaryFunction &Func) {
  const BinaryContext &BC = Func.getBinaryContext();
  bool Modified{false};

  assert(BC.isAArch64() && "Unsupported arch");
  constexpr auto InsnSize = 4; // AArch64
  std::vector<std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>>>
      Insertions;

  BinaryBasicBlock *Frontier = getBBAtHotColdSplitPoint(Func);
  uint64_t FrontierAddress = Frontier ? BBAddresses[Frontier] : 0;
  if (FrontierAddress) {
    FrontierAddress += Frontier->getNumNonPseudos() * InsnSize;
  }
  // Add necessary stubs for branch targets we know we can't fit in the
  // instruction
  for (auto &BB : Func) {
    uint64_t DotAddress = BBAddresses[&BB];
    // Stubs themselves are relaxed on the next loop
    if (Stubs[&Func].count(&BB))
      continue;

    for (auto &Inst : BB) {
      if (BC.MII->get(Inst.getOpcode()).isPseudo())
        continue;

      if (!shouldInsertStub(BC, Inst)) {
        DotAddress += InsnSize;
        continue;
      }

      // Check and relax direct branch or call
      if (!needsStub(BB, Inst, DotAddress)) {
        DotAddress += InsnSize;
        continue;
      }
      Modified = true;

      // Insert stubs close to the patched BB if call, but far away from the
      // hot path if a branch, since this branch target is the cold region
      // (but first check that the far away stub will be in range).
      BinaryBasicBlock *InsertionPoint = &BB;
      if (Func.isSimple() && !BC.MIB->isCall(Inst) && FrontierAddress &&
          !BB.isCold()) {
        auto BitsAvail = BC.MIB->getPCRelEncodingSize(Inst) - 1;
        uint64_t Mask = ~((1ULL << BitsAvail) - 1);
        assert(FrontierAddress > DotAddress &&
               "Hot code should be before the frontier");
        uint64_t PCRelTgt = FrontierAddress - DotAddress;
        if (!(PCRelTgt & Mask))
          InsertionPoint = Frontier;
      }
      // Always put stubs at the end of the function if non-simple. We can't
      // change the layout of non-simple functions because it has jump tables
      // that we do not control.
      if (!Func.isSimple())
        InsertionPoint = &*std::prev(Func.end());

      // Create a stub to handle a far-away target
      Insertions.emplace_back(std::make_pair(
          InsertionPoint,
          replaceTargetWithStub(BB, Inst, DotAddress,
                                InsertionPoint == Frontier ? FrontierAddress
                                                           : DotAddress)));
    }
  }

  // Relax stubs if necessary
  for (auto &BB : Func) {
    if (!Stubs[&Func].count(&BB) || !BB.isValid())
      continue;

    Modified |= relaxStub(BB);
  }

  for (auto &Elmt : Insertions) {
    if (!Elmt.second)
      continue;
    std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
    NewBBs.emplace_back(std::move(Elmt.second));
    Func.insertBasicBlocks(Elmt.first, std::move(NewBBs), true);
  }

  return Modified;
}

void LongJmpPass::runOnFunctions(BinaryContext &BC) {
  outs() << "BOLT-INFO: Starting stub-insertion pass\n";
  auto Sorted = BC.getSortedFunctions();
  bool Modified;
  uint32_t Iterations{0};
  do {
    ++Iterations;
    Modified = false;
    tentativeLayout(BC, Sorted);
    updateStubGroups();
    for (auto Func : Sorted) {
      if (relax(*Func)) {
        // Don't ruin non-simple functions, they can't afford to have the layout
        // changed.
        if (Func->isSimple())
          Func->fixBranches();
        Modified = true;
      }
    }
  } while (Modified);
  outs() << "BOLT-INFO: Inserted " << NumHotStubs
         << " stubs in the hot area and " << NumColdStubs
         << " stubs in the cold area. Shared " << NumSharedStubs
         << " times, iterated " << Iterations << " times.\n";
}
}
}
