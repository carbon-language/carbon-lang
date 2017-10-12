//===--- BinaryPasses.cpp - Binary-level analysis/optimization passes -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "IndirectCallPromotion.h"
#include "DataflowInfoManager.h"
#include "llvm/Support/Options.h"

#define DEBUG_TYPE "ICP"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<unsigned> Verbosity;
extern bool shouldProcess(const bolt::BinaryFunction &Function);

cl::opt<IndirectCallPromotionType>
IndirectCallPromotion("indirect-call-promotion",
  cl::init(ICP_NONE),
  cl::desc("indirect call promotion"),
  cl::values(
    clEnumValN(ICP_NONE, "none", "do not perform indirect call promotion"),
    clEnumValN(ICP_CALLS, "calls", "perform ICP on indirect calls"),
    clEnumValN(ICP_JUMP_TABLES, "jump-tables", "perform ICP on jump tables"),
    clEnumValN(ICP_ALL, "all", "perform ICP on calls and jump tables"),
    clEnumValEnd),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
IndirectCallPromotionThreshold(
    "indirect-call-promotion-threshold",
    cl::desc("threshold for optimizing a frequently taken indirect call"),
    cl::init(90),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static cl::opt<unsigned>
IndirectCallPromotionMispredictThreshold(
    "indirect-call-promotion-mispredict-threshold",
    cl::desc("misprediction threshold for skipping ICP on an "
             "indirect call"),
    cl::init(2),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static cl::opt<bool>
IndirectCallPromotionUseMispredicts(
    "indirect-call-promotion-use-mispredicts",
    cl::desc("use misprediction frequency for determining whether or not ICP "
             "should be applied at a callsite.  The "
             "-indirect-call-promotion-mispredict-threshold value will be used "
             "by this heuristic"),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static cl::opt<unsigned>
IndirectCallPromotionTopN(
    "indirect-call-promotion-topn",
    cl::desc("number of targets to consider when doing indirect "
                   "call promotion"),
    cl::init(1),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static cl::list<std::string>
ICPFuncsList("icp-funcs",
             cl::CommaSeparated,
             cl::desc("list of functions to enable ICP for"),
             cl::value_desc("func1,func2,func3,..."),
             cl::Hidden,
             cl::cat(BoltOptCategory));

static cl::opt<bool>
ICPOldCodeSequence(
    "icp-old-code-sequence",
    cl::desc("use old code sequence for promoted calls"),
    cl::init(false),
    cl::ZeroOrMore,
    cl::Hidden,
    cl::cat(BoltOptCategory));

static cl::opt<bool> ICPJumpTablesByTarget(
    "icp-jump-tables-targets",
    cl::desc(
        "for jump tables, optimize indirect jmp targets instead of indices"),
    cl::init(false), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

IndirectCallPromotion::Callsite::Callsite(BinaryFunction &BF,
                                          const BranchInfo &BI)
: From(BF.getSymbol()),
  To(uint64_t(BI.To.Offset)),
  Mispreds{uint64_t(BI.Mispreds)},
  Branches{uint64_t(BI.Branches)},
  Histories{BI.Histories} {
  if (BI.To.IsSymbol) {
    auto &BC = BF.getBinaryContext();
    auto Itr = BC.GlobalSymbols.find(BI.To.Name);
    if (Itr != BC.GlobalSymbols.end()) {
      To.IsSymbol = true;
      To.Sym = BC.getOrCreateGlobalSymbol(Itr->second, "FUNCat");
      To.Addr = 0;
      assert(To.Sym);
    }
  }
}

// Get list of targets for a given call sorted by most frequently
// called first.
std::vector<IndirectCallPromotion::Callsite>
IndirectCallPromotion::getCallTargets(
  BinaryFunction &BF,
  const MCInst &Inst
) const {
  auto &BC = BF.getBinaryContext();
  std::vector<Callsite> Targets;

  if (const auto *JT = BF.getJumpTable(Inst)) {
    // Don't support PIC jump tables for now
    if (!opts::ICPJumpTablesByTarget &&
        JT->Type == BinaryFunction::JumpTable::JTT_PIC)
      return Targets;
    const Location From(BF.getSymbol());
    const auto Range = JT->getEntriesForAddress(BC.MIA->getJumpTable(Inst));
    assert(JT->Counts.empty() || JT->Counts.size() >= Range.second);
    BinaryFunction::JumpInfo DefaultJI;
    const auto *JI = JT->Counts.empty() ? &DefaultJI : &JT->Counts[Range.first];
    const size_t JIAdj = JT->Counts.empty() ? 0 : 1;
    assert(JT->Type == BinaryFunction::JumpTable::JTT_PIC ||
           JT->EntrySize == BC.AsmInfo->getPointerSize());
    for (size_t I = Range.first; I < Range.second; ++I, JI += JIAdj) {
      auto *Entry = JT->Entries[I];
      assert(BF.getBasicBlockForLabel(Entry) ||
             Entry == BF.getFunctionEndLabel() ||
             Entry == BF.getFunctionColdEndLabel());
      const Location To(Entry);
      Callsite CS{
          From, To, JI->Mispreds, JI->Count, BranchHistories(),
          I - Range.first};
      Targets.emplace_back(CS);
    }

    // Sort by symbol then addr.
    std::sort(Targets.begin(), Targets.end(),
              [](const Callsite &A, const Callsite &B) {
                if (A.To.IsSymbol && B.To.IsSymbol)
                  return A.To.Sym < B.To.Sym;
                else if (A.To.IsSymbol && !B.To.IsSymbol)
                  return true;
                else if (!A.To.IsSymbol && B.To.IsSymbol)
                  return false;
                else
                  return A.To.Addr < B.To.Addr;
              });

    // Targets may contain multiple entries to the same target, but using
    // different indices. Their profile will report the same number of branches
    // for different indices if the target is the same. That's because we don't
    // profile the index value, but only the target via LBR.
    auto First = Targets.begin();
    auto Last = Targets.end();
    auto Result = First;
    while (++First != Last) {
      auto &A = *Result;
      const auto &B = *First;
      if (A.To.IsSymbol && B.To.IsSymbol && A.To.Sym == B.To.Sym) {
        A.JTIndex.insert(A.JTIndex.end(), B.JTIndex.begin(), B.JTIndex.end());
      } else {
        *(++Result) = *First;
      }
    }
    ++Result;

    DEBUG(if (Targets.end() - Result > 0) {
      dbgs() << "BOLT-INFO: ICP: " << (Targets.end() - Result)
             << " duplicate targets removed\n";
    });

    Targets.erase(Result, Targets.end());
  } else {
    const auto *BranchData = BF.getBranchData();
    assert(BranchData && "expected initialized branch data");
    auto Offset = BC.MIA->getAnnotationAs<uint64_t>(Inst, "Offset");
    for (const auto &BI : BranchData->getBranchRange(Offset)) {
      Callsite Site(BF, BI);
      if (Site.isValid()) {
        Targets.emplace_back(std::move(Site));
      }
    }
  }

  // Sort by most commonly called targets.
  std::sort(Targets.begin(), Targets.end(),
            [](const Callsite &A, const Callsite &B) {
              return A.Branches > B.Branches;
            });

  // Remove non-symbol targets
  auto Last = std::remove_if(Targets.begin(),
                             Targets.end(),
                             [](const Callsite &CS) {
                               return !CS.To.IsSymbol;
                             });
  Targets.erase(Last, Targets.end());

  DEBUG(
    if (BF.getJumpTable(Inst)) {
      uint64_t TotalCount = 0;
      uint64_t TotalMispreds = 0;
      for (const auto &S : Targets) {
        TotalCount += S.Branches;
        TotalMispreds += S.Mispreds;
      }
      if (!TotalCount) TotalCount = 1;
      if (!TotalMispreds) TotalMispreds = 1;

      dbgs() << "BOLT-INFO: ICP: jump table size = " << Targets.size()
             << ", Count = " << TotalCount
             << ", Mispreds = " << TotalMispreds << "\n";

      size_t I = 0;
      for (const auto &S : Targets) {
        dbgs () << "Count[" << I << "] = " << S.Branches << ", "
                << format("%.1f", (100.0*S.Branches)/TotalCount) << ", "
                << "Mispreds[" << I << "] = " << S.Mispreds << ", "
                << format("%.1f", (100.0*S.Mispreds)/TotalMispreds) << "\n";
        ++I;
      }
    });

  return Targets;
}

std::vector<std::pair<MCSymbol *, uint64_t>>
IndirectCallPromotion::findCallTargetSymbols(
  BinaryContext &BC,
  const std::vector<Callsite> &Targets,
  const size_t N
) const {
  std::vector<std::pair<MCSymbol *, uint64_t>> SymTargets;

  size_t TgtIdx = 0;
  for (size_t I = 0; I < N; ++TgtIdx) {
    assert(Targets[TgtIdx].To.IsSymbol && "All ICP targets must be to known symbols");
    if (Targets[TgtIdx].JTIndex.empty()) {
      SymTargets.push_back(std::make_pair(Targets[TgtIdx].To.Sym, 0));
      ++I;
    } else {
      for (auto Idx : Targets[TgtIdx].JTIndex) {
        SymTargets.push_back(std::make_pair(Targets[TgtIdx].To.Sym, Idx));
        ++I;
      }
    }
  }

  return SymTargets;
}

std::vector<std::unique_ptr<BinaryBasicBlock>>
IndirectCallPromotion::rewriteCall(BinaryContext &BC,
                                   BinaryFunction &Function,
                                   BinaryBasicBlock *IndCallBlock,
                                   const MCInst &CallInst,
                                   MCInstrAnalysis::ICPdata &&ICPcode) const {
  // Create new basic blocks with correct code in each one first.
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
  const bool IsTailCallOrJT = (BC.MIA->isTailCall(CallInst) ||
                               Function.getJumpTable(CallInst));

  // Move instructions from the tail of the original call block
  // to the merge block.

  // Remember any pseudo instructions following a tail call.  These
  // must be preserved and moved to the original block.
  std::vector<MCInst> TailInsts;
  const auto *TailInst= &CallInst;
  if (IsTailCallOrJT) {
    while (TailInst + 1 < &(*IndCallBlock->end()) &&
           BC.MII->get((TailInst + 1)->getOpcode()).isPseudo()) {
      TailInsts.push_back(*++TailInst);
    }
  }

  auto MovedInst = IndCallBlock->splitInstructions(&CallInst);

  IndCallBlock->replaceInstruction(&CallInst, ICPcode.front().second);
  IndCallBlock->addInstructions(TailInsts.begin(), TailInsts.end());

  for (auto Itr = ICPcode.begin() + 1; Itr != ICPcode.end(); ++Itr) {
    auto &Sym = Itr->first;
    auto &Insts = Itr->second;
    assert(Sym);
    auto TBB = Function.createBasicBlock(0, Sym);
    for (auto &Inst : Insts) { // sanitize new instructions.
      if (BC.MIA->isCall(Inst))
        BC.MIA->removeAnnotation(Inst, "Offset");
    }
    TBB->addInstructions(Insts.begin(), Insts.end());
    NewBBs.emplace_back(std::move(TBB));
  }

  // Move tail of instructions from after the original call to
  // the merge block.
  if (!IsTailCallOrJT) {
    NewBBs.back()->addInstructions(MovedInst.begin(), MovedInst.end());
  } else {
    // assert(MovedInst.empty()); empty or just CFI
  }

  return NewBBs;
}

BinaryBasicBlock *IndirectCallPromotion::fixCFG(
  BinaryContext &BC,
  BinaryFunction &Function,
  BinaryBasicBlock *IndCallBlock,
  const bool IsTailCall,
  const bool IsJumpTable,
  IndirectCallPromotion::BasicBlocksVector &&NewBBs,
  const std::vector<Callsite> &Targets
) const {
  using BinaryBranchInfo = BinaryBasicBlock::BinaryBranchInfo;
  BinaryBasicBlock *MergeBlock = nullptr;

  auto moveSuccessors = [](BinaryBasicBlock *Old, BinaryBasicBlock *New) {
    std::vector<BinaryBasicBlock*> OldSucc(Old->successors().begin(),
                                           Old->successors().end());
    std::vector<BinaryBranchInfo> BranchInfo(Old->branch_info_begin(),
                                             Old->branch_info_end());

    // Remove all successors from the old block.
    Old->removeSuccessors(OldSucc.begin(), OldSucc.end());
    assert(Old->succ_empty());

    // Move them to the new block.
    New->addSuccessors(OldSucc.begin(),
                       OldSucc.end(),
                       BranchInfo.begin(),
                       BranchInfo.end());

    // Update the execution count on the new block.
    New->setExecutionCount(Old->getExecutionCount());
  };

  // Scale indirect call counts to the execution count of the original
  // basic block containing the indirect call.
  uint64_t TotalIndirectBranches = 0;
  uint64_t TotalIndirectMispreds = 0;
  for (const auto &BI : Targets) {
    TotalIndirectBranches += BI.Branches;
    TotalIndirectMispreds += BI.Mispreds;
  }

  uint64_t TotalCount = 0;
  uint64_t TotalMispreds = 0;

  if (Function.hasValidProfile()) {
    TotalCount = IndCallBlock->getExecutionCount();
    TotalMispreds =
      TotalCount * ((double)TotalIndirectMispreds / TotalIndirectBranches);
    assert(TotalCount != BinaryBasicBlock::COUNT_NO_PROFILE);
  }

  // New BinaryBranchInfo scaled to the execution count of the original BB.
  std::vector<BinaryBranchInfo> BBI;
  for (auto Itr = Targets.begin(); Itr != Targets.end(); ++Itr) {
    const auto BranchPct = (double)Itr->Branches / TotalIndirectBranches;
    const auto MispredPct = (double)Itr->Mispreds / TotalIndirectMispreds;
    if (Itr->JTIndex.empty()) {
      BBI.push_back(BinaryBranchInfo{uint64_t(TotalCount * BranchPct),
                                     uint64_t(TotalMispreds * MispredPct)});
      continue;
    }
    for (size_t I = 0, E = Itr->JTIndex.size(); I != E; ++I) {
      BBI.push_back(
          BinaryBranchInfo{uint64_t(TotalCount * (BranchPct / E)),
                           uint64_t(TotalMispreds * (MispredPct / E))});
    }
  }

  auto BI = BBI.begin();
  auto updateCurrentBranchInfo = [&]{
    assert(BI < BBI.end());
    TotalCount -= BI->Count;
    TotalMispreds -= BI->MispredictedCount;
    ++BI;
  };

  if (IsTailCall || IsJumpTable) {
    if (IsJumpTable) {
      moveSuccessors(IndCallBlock, NewBBs.back().get());
    }

    std::vector<MCSymbol*> SymTargets;
    for (size_t I = 0; I < Targets.size(); ++I) {
      assert(Targets[I].To.IsSymbol);
      if (Targets[I].JTIndex.empty())
        SymTargets.push_back(Targets[I].To.Sym);
      else {
        for (size_t Idx = 0, E = Targets[I].JTIndex.size(); Idx != E; ++Idx) {
          SymTargets.push_back(Targets[I].To.Sym);
        }
      }
    }

    // Fix up successors and execution counts.
    updateCurrentBranchInfo();
    if (IsJumpTable) {
      auto *Succ = Function.getBasicBlockForLabel(SymTargets[0]);
      IndCallBlock->addSuccessor(Succ, BBI[0]);  // cond branch
    }
    IndCallBlock->addSuccessor(NewBBs[0].get(), TotalCount); // fallthru branch

    for (size_t I = 0; I < NewBBs.size() - 1; ++I) {
      assert(TotalCount <= IndCallBlock->getExecutionCount() ||
             TotalCount <= uint64_t(TotalIndirectBranches));
      uint64_t ExecCount = BBI[I+1].Count;
      updateCurrentBranchInfo();
      if (IsJumpTable) {
        auto *Succ = Function.getBasicBlockForLabel(SymTargets[I+1]);
        NewBBs[I]->addSuccessor(Succ, BBI[I+1]);
      }
      NewBBs[I]->addSuccessor(NewBBs[I+1].get(), TotalCount); // fallthru
      ExecCount += TotalCount;
      NewBBs[I]->setCanOutline(IndCallBlock->canOutline());
      NewBBs[I]->setIsCold(IndCallBlock->isCold());
      NewBBs[I]->setExecutionCount(ExecCount);
    }

  } else {
    assert(NewBBs.size() >= 2);
    assert(NewBBs.size() % 2 == 1 || IndCallBlock->succ_empty());
    assert(NewBBs.size() % 2 == 1);

    MergeBlock = NewBBs.back().get();

    moveSuccessors(IndCallBlock, MergeBlock);

    // Fix up successors and execution counts.
    updateCurrentBranchInfo();
    IndCallBlock->addSuccessor(NewBBs[1].get(), TotalCount); // cond branch
    IndCallBlock->addSuccessor(NewBBs[0].get(), BBI[0]); // uncond branch

    for (size_t I = 0; I < NewBBs.size() - 2; ++I) {
      assert(TotalCount <= IndCallBlock->getExecutionCount() ||
             TotalCount <= uint64_t(TotalIndirectBranches));
      uint64_t ExecCount = BBI[(I+1)/2].Count;
      NewBBs[I]->setCanOutline(IndCallBlock->canOutline());
      NewBBs[I]->setIsCold(IndCallBlock->isCold());
      if (I % 2 == 0) {
        NewBBs[I]->addSuccessor(MergeBlock, BBI[(I+1)/2].Count); // uncond
      } else {
        assert(I + 2 < NewBBs.size());
        updateCurrentBranchInfo();
        NewBBs[I]->addSuccessor(NewBBs[I+2].get(), TotalCount); // uncond branch
        NewBBs[I]->addSuccessor(NewBBs[I+1].get(), BBI[(I+1)/2]); // cond. branch
        ExecCount += TotalCount;
      }
      NewBBs[I]->setExecutionCount(ExecCount);
    }

    // Arrange for the MergeBlock to be the fallthrough for the first
    // promoted call block.
    MergeBlock->setCanOutline(IndCallBlock->canOutline());
    MergeBlock->setIsCold(IndCallBlock->isCold());
    std::unique_ptr<BinaryBasicBlock> MBPtr;
    std::swap(MBPtr, NewBBs.back());
    NewBBs.pop_back();
    NewBBs.emplace(NewBBs.begin() + 1, std::move(MBPtr));
    // TODO: is COUNT_FALLTHROUGH_EDGE the right thing here?
    NewBBs.back()->addSuccessor(MergeBlock, TotalCount); // uncond branch
  }

  // cold call block
  // TODO: should be able to outline/cold this block.
  NewBBs.back()->setExecutionCount(TotalCount);
  NewBBs.back()->setCanOutline(IndCallBlock->canOutline());
  NewBBs.back()->setIsCold(IndCallBlock->isCold());

  // update BB and BB layout.
  Function.insertBasicBlocks(IndCallBlock, std::move(NewBBs));
  assert(Function.validateCFG());

  return MergeBlock;
}

size_t
IndirectCallPromotion::canPromoteCallsite(const BinaryBasicBlock *BB,
                                          const MCInst &Inst,
                                          const std::vector<Callsite> &Targets,
                                          uint64_t NumCalls) {
  const bool IsJumpTable = BB->getFunction()->getJumpTable(Inst);

  // If we have no targets (or no calls), skip this callsite.
  if (Targets.empty() || !NumCalls) {
    if (opts::Verbosity >= 1) {
      const auto InstIdx = &Inst - &(*BB->begin());
      outs() << "BOLT-INFO: ICP failed in " << *BB->getFunction() << " @ "
             << InstIdx << " in " << BB->getName()
             << ", calls = " << NumCalls
             << ", targets empty or NumCalls == 0.\n";
    }
    return 0;
  }

  const auto TrialN = std::min(size_t(opts::IndirectCallPromotionTopN),
                               Targets.size());

  if (!opts::ICPFuncsList.empty()) {
    for (auto &Name : opts::ICPFuncsList) {
      if (BB->getFunction()->hasName(Name))
        return TrialN;
    }
    return 0;
  }

  // Pick the top N targets.
  uint64_t TotalCallsTopN = 0;
  uint64_t TotalMispredictsTopN = 0;
  size_t N = 0;

  if (opts::IndirectCallPromotionUseMispredicts &&
      (!IsJumpTable || opts::ICPJumpTablesByTarget)) {
    // Count total number of mispredictions for (at most) the top N targets.
    // We may choose a smaller N (TrialN vs. N) if the frequency threshold
    // is exceeded by fewer targets.
    double Threshold = double(opts::IndirectCallPromotionMispredictThreshold);
    for (size_t I = 0; I < TrialN && Threshold > 0; ++I, ++N) {
      const auto Frequency = (100.0 * Targets[I].Mispreds) / NumCalls;
      TotalMispredictsTopN += Targets[I].Mispreds;
      if (!IsJumpTable)
        TotalNumFrequentCalls += Targets[I].Branches;
      else
        TotalNumFrequentJmps += Targets[I].Branches;
      Threshold -= Frequency;
    }

    // Compute the misprediction frequency of the top N call targets.  If this
    // frequency is greater than the threshold, we should try ICP on this callsite.
    const double TopNFrequency = (100.0 * TotalMispredictsTopN) / NumCalls;

    if (TopNFrequency == 0 ||
        TopNFrequency < opts::IndirectCallPromotionMispredictThreshold) {
      if (opts::Verbosity >= 1) {
        const auto InstIdx = &Inst - &(*BB->begin());
        outs() << "BOLT-INFO: ICP failed in " << *BB->getFunction() << " @ "
               << InstIdx << " in " << BB->getName() << ", calls = "
               << NumCalls << ", top N mis. frequency "
               << format("%.1f", TopNFrequency) << "% < "
               << opts::IndirectCallPromotionMispredictThreshold << "%\n";
      }
      return 0;
    }
  } else {
    // Count total number of calls for (at most) the top N targets.
    // We may choose a smaller N (TrialN vs. N) if the frequency threshold
    // is exceeded by fewer targets.
    double Threshold = double(opts::IndirectCallPromotionThreshold);
    for (size_t I = 0; I < TrialN && Threshold > 0; ++I) {
      if (N + (Targets[I].JTIndex.empty() ? 1 : Targets[I].JTIndex.size()) >
          TrialN)
        break;
      const auto Frequency = (100.0 * Targets[I].Branches) / NumCalls;
      TotalCallsTopN += Targets[I].Branches;
      TotalMispredictsTopN += Targets[I].Mispreds;
      if (!IsJumpTable)
        TotalNumFrequentCalls += Targets[I].Branches;
      else
        TotalNumFrequentJmps += Targets[I].Branches;
      Threshold -= Frequency;
      N += Targets[I].JTIndex.empty() ? 1 : Targets[I].JTIndex.size();
    }

    // Compute the frequency of the top N call targets.  If this frequency
    // is greater than the threshold, we should try ICP on this callsite.
    const double TopNFrequency = (100.0 * TotalCallsTopN) / NumCalls;

    if (TopNFrequency == 0 ||
        TopNFrequency < opts::IndirectCallPromotionThreshold) {
      if (opts::Verbosity >= 1) {
        const auto InstIdx = &Inst - &(*BB->begin());
        outs() << "BOLT-INFO: ICP failed in " << *BB->getFunction() << " @ "
               << InstIdx << " in " << BB->getName() << ", calls = "
               << NumCalls << ", top N frequency "
               << format("%.1f", TopNFrequency) << "% < "
               << opts::IndirectCallPromotionThreshold << "%\n";
      }
      return 0;
    }

    // Don't check misprediction frequency for jump tables -- we don't really
    // care as long as we are saving loads from the jump table.
    if (IsJumpTable && !opts::ICPJumpTablesByTarget)
      return N;

    // Compute the misprediction frequency of the top N call targets.  If
    // this frequency is less than the threshold, we should skip ICP at
    // this callsite.
    const double TopNMispredictFrequency =
      (100.0 * TotalMispredictsTopN) / NumCalls;

    if (TopNMispredictFrequency <
        opts::IndirectCallPromotionMispredictThreshold) {
      if (opts::Verbosity >= 1) {
        const auto InstIdx = &Inst - &(*BB->begin());
        outs() << "BOLT-INFO: ICP failed in " <<  *BB->getFunction() << " @ "
               << InstIdx << " in " << BB->getName() << ", calls = "
               << NumCalls << ", top N mispredict frequency "
               << format("%.1f", TopNMispredictFrequency) << "% < "
               << opts::IndirectCallPromotionMispredictThreshold << "%\n";
      }
      return 0;
    }
  }

  return N;
}

void
IndirectCallPromotion::printCallsiteInfo(const BinaryBasicBlock *BB,
                                         const MCInst &Inst,
                                         const std::vector<Callsite> &Targets,
                                         const size_t N,
                                         uint64_t NumCalls) const {
  auto &BC = BB->getFunction()->getBinaryContext();
  const bool IsTailCall = BC.MIA->isTailCall(Inst);
  const bool IsJumpTable = BB->getFunction()->getJumpTable(Inst);
  const auto InstIdx = &Inst - &(*BB->begin());
  bool Separator = false;

  outs() << "BOLT-INFO: ICP candidate branch info: "
         << *BB->getFunction() << " @ " << InstIdx
         << " in " << BB->getName()
         << " -> calls = " << NumCalls
         << (IsTailCall ? " (tail)" : (IsJumpTable ? " (jump table)" : ""));
  for (size_t I = 0; I < N; I++) {
    const auto Frequency = 100.0 * Targets[I].Branches / NumCalls;
    const auto MisFrequency = 100.0 * Targets[I].Mispreds / NumCalls;
    outs() << (Separator ? " | " : ", ");
    Separator = true;
    if (Targets[I].To.IsSymbol)
      outs() << Targets[I].To.Sym->getName();
    else
      outs() << Targets[I].To.Addr;
    outs() << ", calls = " << Targets[I].Branches
           << ", mispreds = " << Targets[I].Mispreds
           << ", taken freq = " << format("%.1f", Frequency) << "%"
           << ", mis. freq = " << format("%.1f", MisFrequency) << "%";
  }
  outs() << "\n";

  DEBUG({
    dbgs() << "BOLT-INFO: ICP original call instruction:\n";
    BC.printInstruction(dbgs(), Inst, Targets[0].From.Addr, nullptr, true);
  });
}

void IndirectCallPromotion::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &LargeFunctions
) {
  if (opts::IndirectCallPromotion == ICP_NONE)
    return;

  std::unique_ptr<RegAnalysis> RA;
  std::unique_ptr<BinaryFunctionCallGraph> CG;
  if (opts::IndirectCallPromotion >= ICP_JUMP_TABLES) {
    CG.reset(new BinaryFunctionCallGraph(buildCallGraph(BC, BFs)));
    RA.reset(new RegAnalysis(BC, BFs, *CG));
  }

  for (auto &BFIt : BFs) {
    auto &Function = BFIt.second;

    if (!Function.isSimple() || !opts::shouldProcess(Function))
      continue;

    const auto *BranchData = Function.getBranchData();
    if (!BranchData)
      continue;

    const bool HasLayout = !Function.layout_empty();

    // Note: this is not just counting calls.
    TotalCalls += BranchData->ExecutionCount;

    // Total number of indirect calls issued from the current Function.
    // (a fraction of TotalIndirectCalls)
    uint64_t FuncTotalIndirectCalls = 0;
    uint64_t FuncTotalIndirectJmps = 0;

    std::vector<BinaryBasicBlock *> BBs;
    for (auto &BB : Function) {
      // Skip indirect calls in cold blocks.
      if (!HasLayout || !Function.isSplit() || !BB.isCold()) {
        BBs.push_back(&BB);
      }
    }
    if (BBs.empty())
      continue;

    DataflowInfoManager Info(BC, Function, RA.get(), nullptr);
    while (!BBs.empty()) {
      auto *BB = BBs.back();
      BBs.pop_back();

      for (unsigned Idx = 0; Idx < BB->size(); ++Idx) {
        auto &Inst = BB->getInstructionAtIndex(Idx);
        const auto InstIdx = &Inst - &(*BB->begin());
        const bool IsTailCall = BC.MIA->isTailCall(Inst);
        const bool HasBranchData = Function.getBranchData() &&
                                   BC.MIA->hasAnnotation(Inst, "Offset");
        const bool IsJumpTable = Function.getJumpTable(Inst);
        const bool OptimizeCalls =
          (opts::IndirectCallPromotion == ICP_CALLS ||
           opts::IndirectCallPromotion == ICP_ALL);
        const bool OptimizeJumpTables =
          (opts::IndirectCallPromotion == ICP_JUMP_TABLES ||
           opts::IndirectCallPromotion == ICP_ALL);

        if (!((HasBranchData && !IsJumpTable && OptimizeCalls) ||
              (IsJumpTable && OptimizeJumpTables)))
          continue;

        // Ignore direct calls.
        if (BC.MIA->isCall(Inst) && BC.MIA->getTargetSymbol(Inst, 0))
          continue;

        assert(BC.MIA->isCall(Inst) || BC.MIA->isIndirectBranch(Inst));

        if (IsJumpTable)
          ++TotalJumpTableCallsites;
        else
          ++TotalIndirectCallsites;

        const auto Targets = getCallTargets(Function, Inst);

        // Compute the total number of calls from this particular callsite.
        uint64_t NumCalls = 0;
        for (const auto &BInfo : Targets) {
          NumCalls += BInfo.Branches;
        }
        if (!IsJumpTable)
          FuncTotalIndirectCalls += NumCalls;
        else
          FuncTotalIndirectJmps += NumCalls;

        // If FLAGS regs is alive after this jmp site, do not try
        // promoting because we will clobber FLAGS.
        if (IsJumpTable && (*Info.getLivenessAnalysis().getStateBefore(
                               Inst))[BC.MIA->getFlagsReg()]) {
          if (opts::Verbosity >= 1) {
            outs() << "BOLT-INFO: ICP failed in " << Function << " @ "
                   << InstIdx << " in " << BB->getName()
                   << ", calls = " << NumCalls
                   << ", cannot clobber flags reg.\n";
          }
          continue;
        }

        // Should this callsite be optimized?  Return the number of targets
        // to use when promoting this call.  A value of zero means to skip
        // this callsite.
        size_t N = canPromoteCallsite(BB, Inst, Targets, NumCalls);

        if (!N)
          continue;

        if (opts::Verbosity >= 1) {
          printCallsiteInfo(BB, Inst, Targets, N, NumCalls);
        }

        // Find MCSymbols or absolute addresses for each call target.
        const auto SymTargets = findCallTargetSymbols(BC, Targets, N);

        // If we can't resolve any of the target symbols, punt on this callsite.
        if (SymTargets.size() < N) {
          const auto LastTarget = SymTargets.size();
          if (opts::Verbosity >= 1) {
            outs() << "BOLT-INFO: ICP failed in " << Function << " @ "
                   << InstIdx << " in " << BB->getName()
                   << ", calls = " << NumCalls
                   << ", ICP failed to find target symbol for "
                   << Targets[LastTarget].To.Sym->getName() << "\n";
          }
          continue;
        }

        // Generate new promoted call code for this callsite.
        auto ICPcode =
            (IsJumpTable && !opts::ICPJumpTablesByTarget)
                ? BC.MIA->jumpTablePromotion(Inst, SymTargets, BC.Ctx.get())
                : BC.MIA->indirectCallPromotion(
                      Inst, SymTargets, opts::ICPOldCodeSequence, BC.Ctx.get());

        if (ICPcode.empty()) {
          if (opts::Verbosity >= 1) {
            outs() << "BOLT-INFO: ICP failed in " << Function << " @ "
                   << InstIdx << " in " << BB->getName()
                   << ", calls = " << NumCalls
                   << ", unable to generate promoted call code.\n";
          }
          continue;
        }

        DEBUG({
          auto Offset = Targets[0].From.Addr;
          dbgs() << "BOLT-INFO: ICP indirect call code:\n";
          for (const auto &entry : ICPcode) {
            const auto &Sym = entry.first;
            const auto &Insts = entry.second;
            if (Sym) dbgs() << Sym->getName() << ":\n";
            Offset = BC.printInstructions(dbgs(),
                                          Insts.begin(),
                                          Insts.end(),
                                          Offset);
          }
          dbgs() << "---------------------------------------------------\n";
        });

        // Rewrite the CFG with the newly generated ICP code.
        auto NewBBs = rewriteCall(BC, Function, BB, Inst, std::move(ICPcode));

        // Fix the CFG after inserting the new basic blocks.
        auto MergeBlock = fixCFG(BC, Function, BB, IsTailCall, IsJumpTable,
                                 std::move(NewBBs), Targets);

        // Since the tail of the original block was split off and it may contain
        // additional indirect calls, we must add the merge block to the set of
        // blocks to process.
        if (MergeBlock) {
          BBs.push_back(MergeBlock);
        }

        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: ICP succeeded in "
                 << Function << " @ " << InstIdx
                 << " in " << BB->getName()
                 << " -> calls = " << NumCalls << "\n";
        }

        if (IsJumpTable)
          ++TotalOptimizedJumpTableCallsites;
        else
          ++TotalOptimizedIndirectCallsites;

        Modified.insert(&Function);
      }
    }
    TotalIndirectCalls += FuncTotalIndirectCalls;
    TotalIndirectJmps += FuncTotalIndirectJmps;
  }

  outs() << "BOLT-INFO: ICP total indirect callsites = "
         << TotalIndirectCallsites
         << "\n"
         << "BOLT-INFO: ICP total jump table callsites = "
         << TotalJumpTableCallsites
         << "\n"
         << "BOLT-INFO: ICP total number of calls = "
         << TotalCalls
         << "\n"
         << "BOLT-INFO: ICP percentage of calls that are indirect = "
         << format("%.1f", (100.0 * TotalIndirectCalls) / TotalCalls)
         << "%\n"
         << "BOLT-INFO: ICP percentage of indirect calls that can be "
            "optimized = "
         << format("%.1f", (100.0 * TotalNumFrequentCalls) /
                   std::max(TotalIndirectCalls, 1ul))
         << "%\n"
         << "BOLT-INFO: ICP percentage of indirect calls that are optimized = "
         << format("%.1f", (100.0 * TotalOptimizedIndirectCallsites) /
                   std::max(TotalIndirectCallsites, 1ul))
         << "%\n"
         << "BOLT-INFO: ICP percentage of indirect branches that are "
            "optimized = "
         << format("%.1f", (100.0 * TotalNumFrequentJmps) /
                   std::max(TotalIndirectJmps, 1ul))
         << "%\n"
         << "BOLT-INFO: ICP percentage of jump table callsites that are optimized = "
         << format("%.1f", (100.0 * TotalOptimizedJumpTableCallsites) /
                   std::max(TotalJumpTableCallsites, 1ul))
         << "%\n";
}

} // namespace bolt
} // namespace llvm
