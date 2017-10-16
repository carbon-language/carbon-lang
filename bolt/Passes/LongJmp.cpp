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
extern cl::opt<bool> Relocs;
extern cl::opt<bool> UseOldText;
extern cl::opt<unsigned> AlignFunctions;
extern cl::opt<unsigned> AlignFunctionsMaxBytes;
}

namespace llvm {
namespace bolt {

namespace {
constexpr unsigned ColdFragAlign = 16;
constexpr unsigned PageAlign = 0x200000;

std::pair<std::unique_ptr<BinaryBasicBlock>, MCSymbol *>
createNewStub(const BinaryContext &BC, BinaryFunction &Func,
              const MCSymbol *TgtSym) {
  auto *StubSym = BC.Ctx->createTempSymbol("Stub", true);
  auto StubBB = Func.createBasicBlock(0, StubSym);
  std::vector<MCInst> Seq;
  BC.MIA->createLongJmp(Seq, TgtSym, BC.Ctx.get());
  StubBB->addInstructions(Seq.begin(), Seq.end());
  StubBB->setExecutionCount(0);
  return std::make_pair(std::move(StubBB), StubSym);
}

void shrinkStubToShortJmp(const BinaryContext &BC, BinaryBasicBlock &StubBB,
                          const MCSymbol *Tgt) {
  std::vector<MCInst> Seq;
  BC.MIA->createShortJmp(Seq, Tgt, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstructions(Seq.begin(), Seq.end());
}

void shrinkStubToSingleInst(const BinaryContext &BC, BinaryBasicBlock &StubBB,
                            const MCSymbol *Tgt, bool TgtIsFunc) {
  MCInst Inst;
  BC.MIA->createUncondBranch(Inst, Tgt, BC.Ctx.get());
  if (TgtIsFunc)
    BC.MIA->convertJmpToTailCall(Inst, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstruction(Inst);
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
}

std::unique_ptr<BinaryBasicBlock>
LongJmpPass::replaceTargetWithStub(const BinaryContext &BC,
                                   BinaryFunction &Func, BinaryBasicBlock &BB,
                                   MCInst &Inst) {
  std::unique_ptr<BinaryBasicBlock> NewBB;
  auto TgtSym = BC.MIA->getTargetSymbol(Inst);
  assert (TgtSym && "getTargetSymbol failed");

  BinaryBasicBlock::BinaryBranchInfo BI{0, 0};
  auto *TgtBB = BB.getSuccessor(TgtSym, BI);
  // Do not issue a long jmp for blocks in the same region
  if (TgtBB && TgtBB->isCold() == BB.isCold())
    return nullptr;

  BinaryBasicBlock *StubBB =
      BB.isCold() ? ColdStubs[&Func][TgtSym] : HotStubs[&Func][TgtSym];
  MCSymbol *StubSymbol = StubBB ? StubBB->getLabel() : nullptr;

  if (!StubBB) {
    std::tie(NewBB, StubSymbol) = createNewStub(BC, Func, TgtSym);
    StubBB = NewBB.get();
    Stubs[&Func].insert(StubBB);
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
  BC.MIA->replaceBranchTarget(Inst, StubSymbol, BC.Ctx.get());
  ++StubRefCount[StubBB];
  StubBits[StubBB] = BC.AsmInfo->getPointerSize() * 8;

  if (NewBB) {
    if (BB.isCold())
      ColdStubs[&Func][TgtSym] = StubBB;
    else
      HotStubs[&Func][TgtSym] = StubBB;
  }

  return NewBB;
}

namespace {

bool shouldInsertStub(const BinaryContext &BC, const MCInst &Inst) {
  return (BC.MIA->isBranch(Inst) || BC.MIA->isCall(Inst)) &&
         !BC.MIA->isIndirectBranch(Inst) && !BC.MIA->isIndirectCall(Inst);
}

}

void LongJmpPass::insertStubs(const BinaryContext &BC, BinaryFunction &Func) {
  std::vector<std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>>>
      Insertions;

  BinaryBasicBlock *Frontier = getBBAtHotColdSplitPoint(Func);

  for (auto &BB : Func) {
    for (auto &Inst : BB) {
      // Only analyze direct branches with target distance constraints
      if (!shouldInsertStub(BC, Inst))
        continue;

      // Insert stubs close to the patched BB if call, but far away from the
      // hot path if a branch, since this branch target is the cold region
      BinaryBasicBlock *InsertionPoint = &BB;
      if (!BC.MIA->isCall(Inst) && Frontier && !BB.isCold())
        InsertionPoint = Frontier;
      // Create a stub to handle a far-away target
      Insertions.emplace_back(std::make_pair(
          InsertionPoint, replaceTargetWithStub(BC, Func, BB, Inst)));
    }
  }

  for (auto &Elmt : Insertions) {
    if (!Elmt.second)
      continue;
    std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
    NewBBs.emplace_back(std::move(Elmt.second));
    Func.insertBasicBlocks(Elmt.first, std::move(NewBBs), true, true);
  }

}

void LongJmpPass::tentativeBBLayout(const BinaryContext &BC,
                                    const BinaryFunction &Func) {
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

uint64_t LongJmpPass::tentativeLayoutRelocMode(
  const BinaryContext &BC, std::vector<BinaryFunction *> &SortedFunctions,
  uint64_t DotAddress) {

  // Hot
  for (auto Func : SortedFunctions) {
    DotAddress = RoundUpToAlignment(DotAddress, BinaryFunction::MinAlign);
    auto Pad = OffsetToAlignment(DotAddress, opts::AlignFunctions);
    if (Pad <= opts::AlignFunctionsMaxBytes)
      DotAddress += Pad;
    HotAddresses[Func] = DotAddress;
    DEBUG(dbgs() << Func->getPrintName()
                 << " tentative: " << Twine::utohexstr(DotAddress) << "\n");
    if (!Func->isSimple()) {
      DotAddress += Func->getMaxSize();
    } else {
      if (!Func->isSplit()) {
        DotAddress += Func->estimateSize();
      } else {
        DotAddress += Func->estimateHotSize();
        DotAddress += Func->estimateConstantIslandSize();
      }
    }
  }
  // Cold
  for (auto Func : SortedFunctions) {
    DotAddress = RoundUpToAlignment(DotAddress, BinaryFunction::MinAlign);
    auto Pad = OffsetToAlignment(DotAddress, opts::AlignFunctions);
    if (Pad <= opts::AlignFunctionsMaxBytes)
      DotAddress += Pad;
    HotAddresses[Func] = Func->getAddress();
    DotAddress = RoundUpToAlignment(DotAddress, ColdFragAlign);
    ColdAddresses[Func] = DotAddress;
    if (Func->isSplit())
      DotAddress += Func->estimateColdSize();
    tentativeBBLayout(BC, *Func);
  }
  return DotAddress;
}

void LongJmpPass::tentativeLayout(
    const BinaryContext &BC,
    std::vector<BinaryFunction *> &SortedFunctions) {
  uint64_t DotAddress = BC.LayoutStartAddress;

  if (!opts::Relocs) {
    for (auto Func : SortedFunctions) {
      HotAddresses[Func] = Func->getAddress();
      DotAddress = RoundUpToAlignment(DotAddress, 16);
      ColdAddresses[Func] = DotAddress;
      if (Func->isSplit())
        DotAddress += Func->estimateColdSize();
      tentativeBBLayout(BC, *Func);
    }

    return;
  }

  // Relocation mode
  auto EstimatedTextSize = tentativeLayoutRelocMode(BC, SortedFunctions, 0);

  // Initial padding
  if (opts::UseOldText && EstimatedTextSize <= BC.OldTextSectionSize) {
    DotAddress = BC.OldTextSectionAddress;
    auto Pad = OffsetToAlignment(DotAddress, PageAlign);
    if (Pad + EstimatedTextSize <= BC.OldTextSectionSize) {
      DotAddress += Pad;
    }
  } else {
    DotAddress = RoundUpToAlignment(BC.LayoutStartAddress, PageAlign);
  }

  tentativeLayoutRelocMode(BC, SortedFunctions, DotAddress);
}

void LongJmpPass::removeStubRef(const BinaryContext &BC,
                                BinaryBasicBlock *BB, MCInst &Inst,
                                BinaryBasicBlock *StubBB,
                                const MCSymbol *Target,
                                BinaryBasicBlock *TgtBB) {
  BC.MIA->replaceBranchTarget(Inst, Target, BC.Ctx.get());

  --StubRefCount[StubBB];
  assert(StubRefCount[StubBB] >= 0 && "Ref count is lost");

  if (TgtBB && BB->isSuccessor(StubBB)) {
    const auto &BI = BB->getBranchInfo(*StubBB);
    uint64_t OrigCount{BI.Count};
    uint64_t OrigMispreds{BI.MispredictedCount};
    BB->replaceSuccessor(StubBB, TgtBB, OrigCount, OrigMispreds);
  }

  if (StubRefCount[StubBB] == 0) {
    // Remove the block from CFG
    StubBB->removeSuccessors(StubBB->succ_begin(), StubBB->succ_end());
    StubBB->markValid(false);
    StubBB->setEntryPoint(false);
  }
}

bool LongJmpPass::usesStub(const BinaryContext &BC, const BinaryFunction &Func,
                           const MCInst &Inst) const {
  auto TgtSym = BC.MIA->getTargetSymbol(Inst);
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
    assert (Iter != BBAddresses.end() && "Unrecognized local BB");
    return Iter->second;
  }
  auto *TargetFunc = BC.getFunctionForSymbol(Target);
  auto Iter = HotAddresses.find(TargetFunc);
  if (Iter == HotAddresses.end()) {
    // Look at BinaryContext's resolution for this symbol - this is a symbol not
    // mapped to a BinaryFunction
    auto SymIter = BC.GlobalSymbols.find(Target->getName());
    assert (SymIter != BC.GlobalSymbols.end() && "Unrecognized symbol");
    return SymIter->second;
  }
  return Iter->second;
}

bool LongJmpPass::removeOrShrinkStubs(const BinaryContext &BC,
                                      BinaryFunction &Func) {
  bool Modified{false};

  assert(BC.TheTriple->getArch() == llvm::Triple::aarch64 &&
         "Unsupported arch");
  constexpr auto InsnSize = 4; // AArch64
  // Remove unnecessary stubs for branch targets we know we can fit in the
  // instruction
  for (auto &BB : Func) {
    uint64_t DotAddress = BBAddresses[&BB];
    for (auto &Inst : BB) {
      if (!shouldInsertStub(BC, Inst) || !usesStub(BC, Func, Inst)) {
        DotAddress += InsnSize;
        continue;
      }

      auto StubSym = BC.MIA->getTargetSymbol(Inst);
      auto *StubBB = Func.getBasicBlockForLabel(StubSym);
      auto *RealTargetSym = BC.MIA->getTargetSymbol(*StubBB->begin());
      auto *TgtBB = Func.getBasicBlockForLabel(RealTargetSym);
      auto BitsAvail = BC.MIA->getPCRelEncodingSize(Inst) - 1;
      uint64_t Mask = ~((1ULL << BitsAvail) - 1);
      uint64_t Offset = getSymbolAddress(BC, RealTargetSym, TgtBB);
      if (DotAddress > Offset)
        Offset = DotAddress - Offset;
      else
        Offset -= DotAddress;
      // If it fits in the original instr, remove the stub
      if (!(Offset & Mask)) {
        removeStubRef(BC, &BB, Inst, StubBB, RealTargetSym, TgtBB);
        Modified = true;
      }
      DotAddress += InsnSize;
    }
  }

  auto RangeShortJmp = BC.MIA->getShortJmpEncodingSize();
  auto RangeSingleInstr = BC.MIA->getUncondBranchEncodingSize();
  uint64_t ShortJmpMask = ~((1ULL << RangeShortJmp) - 1);
  uint64_t SingleInstrMask = ~((1ULL << (RangeSingleInstr - 1)) - 1);
  // Shrink stubs from 64 to 32 or 28 bit whenever possible
  for (auto &BB : Func) {
    if (!Stubs[&Func].count(&BB) || !BB.isValid())
      continue;

    auto Bits = StubBits[&BB];
    // Already working with the tightest range?
    if (Bits == RangeSingleInstr)
      continue;

    // Attempt to tight to short jmp
    auto *RealTargetSym = BC.MIA->getTargetSymbol(*BB.begin());
    auto *TgtBB = Func.getBasicBlockForLabel(RealTargetSym);
    uint64_t DotAddress = BBAddresses[&BB];
    uint64_t TgtAddress = getSymbolAddress(BC, RealTargetSym, TgtBB);
    if (TgtAddress & ShortJmpMask)
      continue;

    // Attempt to tight to pc-relative single-instr branch
    uint64_t PCRelTgtAddress = TgtAddress > DotAddress
                                   ? TgtAddress - DotAddress
                                   : DotAddress - TgtAddress;
    if (PCRelTgtAddress & SingleInstrMask) {
      if (Bits > RangeShortJmp) {
        shrinkStubToShortJmp(BC, BB, RealTargetSym);
        StubBits[&BB] = RangeShortJmp;
        Modified = true;
      }
      continue;
    }

    if (Bits > RangeSingleInstr) {
      shrinkStubToSingleInst(BC, BB, RealTargetSym, /*is func?*/!TgtBB);
      StubBits[&BB] = RangeSingleInstr;
      Modified = true;
    }
  }
  return Modified;
}

void LongJmpPass::runOnFunctions(BinaryContext &BC,
                                 std::map<uint64_t, BinaryFunction> &BFs,
                                 std::set<uint64_t> &LargeFunctions) {
  auto Sorted = BinaryContext::getSortedFunctions(BFs);
  for (auto Func : Sorted) {
    // We are going to remove invalid BBs, so remove any previous marks
    for (auto &BB : *Func) {
      BB.markValid(true);
    }
    insertStubs(BC, *Func);
    Func->fixBranches();
  }

  bool Modified;
  do {
    Modified = false;
    tentativeLayout(BC, Sorted);
    for (auto Func : Sorted) {
      if (removeOrShrinkStubs(BC, *Func)) {
        Func->eraseInvalidBBs();
        Func->fixBranches();
        Modified = true;
      }
    }
  } while (Modified);
}

}
}
