//===--- BinaryFunctionProfile.cpp                                 --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "DataReader.h"
#include "Passes/MCF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory AggregatorCategory;
extern cl::OptionCategory BoltOptCategory;

extern cl::opt<unsigned> Verbosity;
extern cl::opt<IndirectCallPromotionType> IndirectCallPromotion;
extern cl::opt<JumpTableSupportLevel> JumpTables;

static cl::opt<bool>
CompatMode("prof-compat-mode",
  cl::desc("maintain bug-level compatibility with old profile"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<MCFCostFunction>
DoMCF("mcf",
  cl::desc("solve a min cost flow problem on the CFG to fix edge counts "
           "(default=disable)"),
  cl::init(MCF_DISABLE),
  cl::values(
    clEnumValN(MCF_DISABLE, "none",
               "disable MCF"),
    clEnumValN(MCF_LINEAR, "linear",
               "cost function is inversely proportional to edge count"),
    clEnumValN(MCF_QUADRATIC, "quadratic",
               "cost function is inversely proportional to edge count squared"),
    clEnumValN(MCF_LOG, "log",
               "cost function is inversely proportional to log of edge count"),
    clEnumValN(MCF_BLAMEFTS, "blamefts",
               "tune cost to blame fall-through edges for surplus flow"),
    clEnumValEnd),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
FixFuncCounts("fix-func-counts",
  cl::desc("adjust function counts based on basic blocks execution count"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

bool BinaryFunction::recordTrace(
    const LBREntry &First,
    const LBREntry &Second,
    uint64_t Count,
    SmallVector<std::pair<uint64_t, uint64_t>, 16> *Branches) {
  if (!isSimple())
    return false;

  assert(CurrentState == State::CFG && "can only record traces in CFG state");

  // Offsets of the trace within this function.
  const auto From = First.To - getAddress();
  const auto To = Second.From - getAddress();

  if (From > To)
    return false;

  auto *FromBB = getBasicBlockContainingOffset(From);
  auto *ToBB = getBasicBlockContainingOffset(To);

  if (!FromBB || !ToBB)
    return false;

  // Fill out information for fall-through edges. The From and To could be
  // within the same basic block, e.g. when two call instructions are in the
  // same block. In this case we skip the processing.
  if (FromBB == ToBB) {
    if (opts::CompatMode)
      return true;

    // If the previous block ended with a call, the destination of a return
    // would be in ToBB basic block. And if the ToBB starts with a control
    // transfer instruction, we will have a 0-length trace that we have to
    // account for as a fall-through edge.
    if (To == ToBB->getOffset()) {
      // External entry point.
      if (ToBB->isEntryPoint() || ToBB->isLandingPad())
        return true;

      // Check that the origin LBR of a trace starts in another function.
      // Otherwise it's an internal branch that was accounted for.
      if (containsAddress(First.From))
        return true;

      auto *PrevBB = BasicBlocksLayout[ToBB->getIndex() - 1];

      // This could be a bad trace.
      if (!PrevBB->getSuccessor(ToBB->getLabel())) {
        DEBUG(dbgs() << "invalid LBR sequence:\n"
                     << "  " << First << '\n'
                     << "  " << Second << '\n');
        return false;
      }

      auto &BI = PrevBB->getBranchInfo(*ToBB);
      BI.Count += Count;
      if (Branches) {
        const auto *Instr = PrevBB->getLastNonPseudoInstr();
        const auto Offset =
          BC.MIA->getAnnotationWithDefault<uint64_t>(*Instr, "Offset");
        Branches->push_back(std::make_pair(Offset, ToBB->getOffset()));
      }
    }

    return true;
  }

  // Process blocks in the original layout order.
  auto *BB = BasicBlocksLayout[FromBB->getIndex()];
  assert(BB == FromBB && "index mismatch");
  while (BB != ToBB) {
    auto *NextBB = BasicBlocksLayout[BB->getIndex() + 1];
    assert((NextBB && NextBB->getOffset() > BB->getOffset()) && "bad layout");

    // Check for bad LBRs.
    if (!BB->getSuccessor(NextBB->getLabel())) {
      DEBUG(dbgs() << "no fall-through for the trace:\n"
                   << "  " << First << '\n'
                   << "  " << Second << '\n');
      return false;
    }

    // To keep backwards compatibility we skip recording fall-throughs that
    // are not a result of a conditional jump.
    if (!opts::CompatMode ||
        (BB->succ_size() == 2 &&
         BB->getConditionalSuccessor(false) == NextBB)) {
      auto &BI = BB->getBranchInfo(*NextBB);
      BI.Count += Count;

      if (Branches) {
        const auto *Instr = BB->getLastNonPseudoInstr();
        // Note: real offset for conditional jump instruction shouldn't be 0.
        const auto Offset =
            BC.MIA->getAnnotationWithDefault<uint64_t>(*Instr, "Offset");
        if (Offset) {
          Branches->push_back(std::make_pair(Offset, NextBB->getOffset()));
        }
      }
    }

    BB = NextBB;
  }

  return true;
}

bool BinaryFunction::recordBranch(uint64_t From, uint64_t To,
                                  uint64_t Count, uint64_t Mispreds) {
  auto *FromBB = getBasicBlockContainingOffset(From);
  auto *ToBB = getBasicBlockContainingOffset(To);

  if (!FromBB || !ToBB) {
    DEBUG(dbgs() << "failed to get block for recorded branch\n");
    return false;
  }

  // Could be bad LBR data. Ignore, or report as a bad profile for backwards
  // compatibility.
  if (From == To) {
    if (!opts::CompatMode)
      return true;
    auto *Instr = getInstructionAtOffset(0);
    if (Instr && BC.MIA->isCall(*Instr))
      return true;
    return false;
  }

  if (FromBB->succ_size() == 0) {
    // Return from a tail call.
    return true;
  }

  // Very rarely we will see ignored branches. Do a linear check.
  for (auto &Branch : IgnoredBranches) {
    if (Branch == std::make_pair(static_cast<uint32_t>(From),
                                 static_cast<uint32_t>(To)))
      return true;
  }

  if (To != ToBB->getOffset()) {
    // "To" could be referring to nop instructions in between 2 basic blocks.
    // While building the CFG we make sure these nops are attributed to the
    // previous basic block, thus we check if the destination belongs to the
    // gap past the last instruction.
    const auto *LastInstr = ToBB->getLastNonPseudoInstr();
    if (LastInstr) {
      const auto LastInstrOffset =
        BC.MIA->getAnnotationWithDefault<uint64_t>(*LastInstr, "Offset");

      // With old .fdata we are getting FT branches for "jcc,jmp" sequences.
      if (To == LastInstrOffset && BC.MIA->isUnconditionalBranch(*LastInstr)) {
        return true;
      }

      if (To <= LastInstrOffset) {
        DEBUG(dbgs() << "branch recorded into the middle of the block" << " in "
                     << *this << " : " << From << " -> " << To << '\n');
        return false;
      }
    }

    // The real destination is the layout successor of the detected ToBB.
    if (ToBB == BasicBlocksLayout.back())
      return false;
    auto *NextBB = BasicBlocksLayout[ToBB->getIndex() + 1];
    assert((NextBB && NextBB->getOffset() > ToBB->getOffset()) && "bad layout");
    ToBB = NextBB;
  }

  // If there's no corresponding instruction for 'From', we have probably
  // discarded it as a FT from __builtin_unreachable.
  auto *FromInstruction = getInstructionAtOffset(From);
  if (!FromInstruction) {
    DEBUG(dbgs() << "no instruction for offset " << From << " in "
                 << *this << '\n');
    return false;
  }

  if (FromBB == ToBB) {
    // Check for a return from a recursive call.
    // Otherwise it's a simple loop.
  }

  if (!FromBB->getSuccessor(ToBB->getLabel())) {
    // Check if this is a recursive call or a return from a recursive call.
    if (ToBB->isEntryPoint()) {
      // Execution count is already accounted for.
      return true;
    }

    DEBUG(dbgs() << "invalid branch in " << *this << '\n'
                 << Twine::utohexstr(From) << " -> "
                 << Twine::utohexstr(To) << '\n');
    return false;
  }

  auto &BI = FromBB->getBranchInfo(*ToBB);
  BI.Count += Count;
  // Only update mispredicted count if it the count was real.
  if (Count) {
    BI.MispredictedCount += Mispreds;
  }

  return true;
}

bool BinaryFunction::recordEntry(uint64_t To, bool Mispred, uint64_t Count) {
  if (To > getSize())
    return false;

  if (!hasProfile())
    ExecutionCount = 0;

  if (To == 0)
    ExecutionCount += Count;

  return true;
}

bool BinaryFunction::recordExit(uint64_t From, bool Mispred, uint64_t Count) {
  if (!isSimple())
    return false;
  assert(From <= getSize() && "wrong From address");

  if (!hasProfile())
    ExecutionCount = 0;

  return true;
}

void BinaryFunction::postProcessProfile() {
  if (!hasValidProfile()) {
    clearProfile();
    return;
  }

  // Check if MCF post-processing was requested.
  if (opts::DoMCF != MCF_DISABLE) {
    removeTagsFromProfile();
    solveMCF(*this, opts::DoMCF);
    return;
  }

  // Is we are using non-LBR sampling there's nothing left to do.
  if (!BranchData)
    return;

  // Bug compatibility with previous version - double accounting for conditional
  // jump into a fall-through block.
  if (opts::CompatMode) {
    for (auto *BB : BasicBlocks) {
      if (BB->succ_size() == 2 &&
          BB->getConditionalSuccessor(false) ==
            BB->getConditionalSuccessor(true)) {
        auto &TakenBI = *BB->branch_info_begin();
        auto &FallThroughBI = *BB->branch_info_rbegin();
        FallThroughBI.Count = TakenBI.Count;
        FallThroughBI.MispredictedCount = 0;
      }
    }
  }

  // Pre-sort branch data.
  std::stable_sort(BranchData->Data.begin(), BranchData->Data.end());

  // If we have at least some branch data for the function indicate that it
  // was executed.
  if (opts::FixFuncCounts && ExecutionCount == 0) {
    ExecutionCount = 1;
  }

  // Compute preliminary execution count for each basic block
  for (auto *BB : BasicBlocks) {
    BB->ExecutionCount = 0;
  }
  for (auto *BB : BasicBlocks) {
    auto SuccBIIter = BB->branch_info_begin();
    for (auto Succ : BB->successors()) {
      if (SuccBIIter->Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        Succ->setExecutionCount(Succ->getExecutionCount() + SuccBIIter->Count);
      ++SuccBIIter;
    }
  }

  // Set entry BBs to zero, we'll update their execution count next with entry
  // data (we maintain a separate data structure for branches to function entry
  // points)
  for (auto *BB : BasicBlocks) {
    if (BB->isEntryPoint())
      BB->ExecutionCount = 0;
  }

  // Update execution counts of landing pad blocks and entry BBs
  // There is a slight skew introduced here as branches originated from RETs
  // may be accounted for in the execution count of an entry block if the last
  // instruction in a predecessor fall-through block is a call. This situation
  // should rarely happen because there are few multiple-entry functions.
  for (const auto &I : BranchData->EntryData) {
    BinaryBasicBlock *BB = getBasicBlockAtOffset(I.To.Offset);
    if (BB && (BB->isEntryPoint() || BB->isLandingPad())) {
      BB->setExecutionCount(BB->getExecutionCount() + I.Branches);
    }
  }

  inferFallThroughCounts();

  // Update profile information for jump tables based on CFG branch data.
  for (auto *BB : BasicBlocks) {
    const auto *LastInstr = BB->getLastNonPseudoInstr();
    if (!LastInstr)
      continue;
    const auto JTAddress = BC.MIA->getJumpTable(*LastInstr);
    if (!JTAddress)
      continue;
    auto *JT = getJumpTableContainingAddress(JTAddress);
    if (!JT)
      continue;

    uint64_t TotalBranchCount = 0;
    for (const auto &BranchInfo : BB->branch_info()) {
      TotalBranchCount += BranchInfo.Count;
    }
    JT->Count += TotalBranchCount;

    if (opts::IndirectCallPromotion < ICP_JUMP_TABLES &&
        opts::JumpTables < JTS_AGGRESSIVE)
      continue;

    if (JT->Counts.empty())
      JT->Counts.resize(JT->Entries.size());
    auto EI = JT->Entries.begin();
    auto Delta = (JTAddress - JT->Address) / JT->EntrySize;
    EI += Delta;
    while (EI != JT->Entries.end()) {
      const auto *TargetBB = getBasicBlockForLabel(*EI);
      if (TargetBB) {
        const auto &BranchInfo = BB->getBranchInfo(*TargetBB);
        assert(Delta < JT->Counts.size());
        JT->Counts[Delta].Count += BranchInfo.Count;
        JT->Counts[Delta].Mispreds += BranchInfo.MispredictedCount;
      }
      ++Delta;
      ++EI;
      // A label marks the start of another jump table.
      if (JT->Labels.count(Delta * JT->EntrySize))
        break;
    }
  }
}

Optional<SmallVector<std::pair<uint64_t, uint64_t>, 16>>
BinaryFunction::getFallthroughsInTrace(const LBREntry &First,
                                       const LBREntry &Second) {
  SmallVector<std::pair<uint64_t, uint64_t>, 16> Res;

  if (!recordTrace(First, Second, 1, &Res))
    return NoneType();

  return Res;
}

void BinaryFunction::readProfile() {
  if (empty())
    return;

  if (!BC.DR.hasLBR()) {
    readSampleData();
    return;
  }

  // Possibly assign/re-assign branch profile data.
  matchProfileData();

  if (!BranchData)
    return;

  uint64_t MismatchedBranches = 0;
  for (const auto &BI : BranchData->Data) {
    if (BI.From.Name != BI.To.Name) {
      continue;
    }

    if (!recordBranch(BI.From.Offset, BI.To.Offset,
                      BI.Branches, BI.Mispreds)) {
      DEBUG(dbgs() << "bad branch : " << BI.From.Offset << " -> "
                   << BI.To.Offset << '\n');
      ++MismatchedBranches;
    }
  }

  // Special profile data propagation is required for conditional tail calls.
  for (auto BB : BasicBlocks) {
    auto *CTCInstr = BB->getLastNonPseudoInstr();
    if (!CTCInstr || !BC.MIA->getConditionalTailCall(*CTCInstr))
      continue;

    auto OffsetOrErr =
      BC.MIA->tryGetAnnotationAs<uint64_t>(*CTCInstr, "Offset");
    assert(OffsetOrErr && "offset not set for conditional tail call");

    auto BranchInfoOrErr = BranchData->getDirectCallBranch(*OffsetOrErr);
    if (!BranchInfoOrErr)
      continue;

    BC.MIA->addAnnotation(BC.Ctx.get(), *CTCInstr, "CTCTakenCount",
                          BranchInfoOrErr->Branches);
    BC.MIA->addAnnotation(BC.Ctx.get(), *CTCInstr, "CTCMispredCount",
                          BranchInfoOrErr->Mispreds);
  }
}

void BinaryFunction::mergeProfileDataInto(BinaryFunction &BF) const {
  // No reason to merge invalid or empty profiles into BF.
  if (!hasValidProfile())
    return;

  // Update function execution count.
  if (getExecutionCount() != BinaryFunction::COUNT_NO_PROFILE) {
    BF.setExecutionCount(BF.getKnownExecutionCount() + getExecutionCount());
  }

  // Since we are merging a valid profile, the new profile should be valid too.
  // It has either already been valid, or it has been cleaned up.
  BF.ProfileMatchRatio = 1.0f;

  // Update basic block and edge counts.
  auto BBMergeI = BF.begin();
  for (BinaryBasicBlock *BB : BasicBlocks) {
    BinaryBasicBlock *BBMerge = &*BBMergeI;
    assert(getIndex(BB) == BF.getIndex(BBMerge));

    // Update basic block count.
    if (BB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE) {
      BBMerge->setExecutionCount(
          BBMerge->getKnownExecutionCount() + BB->getExecutionCount());
    }

    // Update edge count for successors of this basic block.
    auto BBMergeSI = BBMerge->succ_begin();
    auto BIMergeI = BBMerge->branch_info_begin();
    auto BII = BB->branch_info_begin();
    for (const auto *BBSucc : BB->successors()) {
      (void)BBSucc;
      assert(getIndex(BBSucc) == BF.getIndex(*BBMergeSI));

      // At this point no branch count should be set to COUNT_NO_PROFILE.
      assert(BII->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");
      assert(BIMergeI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");

      BIMergeI->Count += BII->Count;

      // When we merge inferred and real fall-through branch data, the merged
      // data is considered inferred.
      if (BII->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED &&
          BIMergeI->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
        BIMergeI->MispredictedCount += BII->MispredictedCount;
      } else {
        BIMergeI->MispredictedCount = BinaryBasicBlock::COUNT_INFERRED;
      }

      ++BBMergeSI;
      ++BII;
      ++BIMergeI;
    }
    assert(BBMergeSI == BBMerge->succ_end());

    ++BBMergeI;
  }
  assert(BBMergeI == BF.end());
}

void BinaryFunction::readSampleData() {
  auto SampleDataOrErr = BC.DR.getFuncSampleData(getNames());

  if (!SampleDataOrErr)
    return;

  // Non-LBR mode territory
  // First step is to assign BB execution count based on samples from perf
  ProfileMatchRatio = 1.0f;
  removeTagsFromProfile();
  bool NormalizeByInsnCount =
      BC.DR.usesEvent("cycles") || BC.DR.usesEvent("instructions");
  bool NormalizeByCalls = BC.DR.usesEvent("branches");
  static bool NagUser{true};
  if (NagUser) {
    outs() << "BOLT-INFO: operating with non-LBR profiling data.\n";
    if (NormalizeByInsnCount) {
      outs() << "BOLT-INFO: normalizing samples by instruction count.\n";
    } else if (NormalizeByCalls) {
      outs() << "BOLT-INFO: normalizing samples by branches.\n";
    }
    NagUser = false;
  }
  uint64_t LastOffset = getSize();
  uint64_t TotalEntryCount{0};
  for (auto I = BasicBlockOffsets.rbegin(), E = BasicBlockOffsets.rend();
       I != E; ++I) {
    uint64_t CurOffset = I->first;
    // Always work with samples multiplied by 1000 to avoid losing them if we
    // later need to normalize numbers
    uint64_t NumSamples =
        SampleDataOrErr->getSamples(CurOffset, LastOffset) * 1000;
    if (NormalizeByInsnCount && I->second->getNumNonPseudos())
      NumSamples /= I->second->getNumNonPseudos();
    else if (NormalizeByCalls) {
      uint32_t NumCalls = I->second->getNumCalls();
      NumSamples /= NumCalls + 1;
    }
    I->second->setExecutionCount(NumSamples);
    if (I->second->isEntryPoint())
      TotalEntryCount += NumSamples;
    LastOffset = CurOffset;
  }
  ExecutionCount = TotalEntryCount;

  estimateEdgeCounts(BC, *this);

  if (opts::DoMCF != MCF_DISABLE)
    solveMCF(*this, opts::DoMCF);
}

void BinaryFunction::inferFallThroughCounts() {
  // Work on a basic block at a time, propagating frequency information
  // forwards.
  // It is important to walk in the layout order.
  for (auto *BB : BasicBlocks) {
    const uint64_t BBExecCount = BB->getExecutionCount();

    // Propagate this information to successors, filling in fall-through edges
    // with frequency information
    if (BB->succ_size() == 0)
      continue;

    // Calculate frequency of outgoing branches from this node according to
    // LBR data.
    uint64_t ReportedBranches = 0;
    for (const auto &SuccBI : BB->branch_info()) {
      if (SuccBI.Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        ReportedBranches += SuccBI.Count;
    }

    // Get taken count of conditional tail call if the block ends with one.
    uint64_t CTCTakenCount = 0;
    const auto CTCInstr = BB->getLastNonPseudoInstr();
    if (CTCInstr && BC.MIA->getConditionalTailCall(*CTCInstr)) {
      CTCTakenCount =
        BC.MIA->getAnnotationWithDefault<uint64_t>(*CTCInstr, "CTCTakenCount");
    }

    // Calculate frequency of throws from this node according to LBR data
    // for branching into associated landing pads. Since it is possible
    // for a landing pad to be associated with more than one basic blocks,
    // we may overestimate the frequency of throws for such blocks.
    uint64_t ReportedThrows = 0;
    for (const auto *LP: BB->landing_pads()) {
      ReportedThrows += LP->getExecutionCount();
    }

    const uint64_t TotalReportedJumps =
      ReportedBranches + CTCTakenCount + ReportedThrows;

    // Infer the frequency of the fall-through edge, representing not taking the
    // branch.
    uint64_t Inferred = 0;
    if (BBExecCount > TotalReportedJumps)
      Inferred = BBExecCount - TotalReportedJumps;

    DEBUG(
      if (BBExecCount < TotalReportedJumps)
        dbgs()
            << "Fall-through inference is slightly inconsistent. "
               "exec frequency is less than the outgoing edges frequency ("
            << BBExecCount << " < " << ReportedBranches
            << ") for  BB at offset 0x"
            << Twine::utohexstr(getAddress() + BB->getOffset()) << '\n';
    );

    if (BB->succ_size() <= 2) {
      // Skip if the last instruction is an unconditional jump.
      const auto *LastInstr = BB->getLastNonPseudoInstr();
      if (LastInstr &&
          (BC.MIA->isUnconditionalBranch(*LastInstr) ||
           BC.MIA->isIndirectBranch(*LastInstr)))
        continue;
      // If there is an FT it will be the last successor.
      auto &SuccBI = *BB->branch_info_rbegin();
      auto &Succ = *BB->succ_rbegin();
      if (SuccBI.Count == 0) {
        SuccBI.Count = Inferred;
        SuccBI.MispredictedCount = BinaryBasicBlock::COUNT_INFERRED;
        Succ->ExecutionCount += Inferred;
      }
    }
  }

  return;
}

bool BinaryFunction::fetchProfileForOtherEntryPoints() {
  if (!BranchData)
    return false;

  // Check if we are missing profiling data for secondary entry points
  bool First{true};
  bool Updated{false};
  for (auto BB : BasicBlocks) {
    if (First) {
      First = false;
      continue;
    }
    if (BB->isEntryPoint()) {
      uint64_t EntryAddress = BB->getOffset() + getAddress();
      // Look for branch data associated with this entry point
      std::vector<std::string> Names;
      std::multimap<uint64_t, std::string>::iterator I, E;
      for (std::tie(I, E) = BC.GlobalAddresses.equal_range(EntryAddress);
           I != E; ++I) {
        Names.push_back(I->second);
      }
      if (!Names.empty()) {
        if (FuncBranchData *Data = BC.DR.getFuncBranchData(Names)) {
          BranchData->appendFrom(*Data, BB->getOffset());
          Data->Used = true;
          Updated = true;
        }
      }
    }
  }
  return Updated;
}

void BinaryFunction::matchProfileMemData() {
  const auto AllMemData = BC.DR.getFuncMemDataRegex(getNames());
  for (auto *NewMemData : AllMemData) {
    // Prevent functions from sharing the same profile.
    if (NewMemData->Used)
      continue;

    if (MemData)
      MemData->Used = false;

    // Update function profile data with the new set.
    MemData = NewMemData;
    MemData->Used = true;
    break;
  }
}

void BinaryFunction::matchProfileData() {
  // This functionality is available for LBR-mode only
  // TODO: Implement evaluateProfileData() for samples, checking whether
  // sample addresses match instruction addresses in the function
  if (!BC.DR.hasLBR())
    return;

  if (BranchData) {
    ProfileMatchRatio = evaluateProfileData(*BranchData);
    if (ProfileMatchRatio == 1.0f) {
      if (fetchProfileForOtherEntryPoints()) {
        ProfileMatchRatio = evaluateProfileData(*BranchData);
        ExecutionCount = BranchData->ExecutionCount;
      }
      return;
    }
  }

  // Check if the function name can fluctuate between several compilations
  // possibly triggered by minor unrelated code changes in the source code
  // of the input binary.
  const auto HasVolatileName = [this]() {
    for (const auto Name : getNames()) {
      if (getLTOCommonName(Name))
        return true;
    }
    return false;
  }();
  if (!HasVolatileName)
    return;

  // Check for a profile that matches with 100% confidence.
  const auto AllBranchData = BC.DR.getFuncBranchDataRegex(getNames());
  for (auto *NewBranchData : AllBranchData) {
    // Prevent functions from sharing the same profile.
    if (NewBranchData->Used)
      continue;

    if (evaluateProfileData(*NewBranchData) != 1.0f)
      continue;

    if (BranchData)
      BranchData->Used = false;

    // Update function profile data with the new set.
    BranchData = NewBranchData;
    ExecutionCount = NewBranchData->ExecutionCount;
    ProfileMatchRatio = 1.0f;
    BranchData->Used = true;
    break;
  }
}

float BinaryFunction::evaluateProfileData(const FuncBranchData &BranchData) {
  // Until we define a minimal profile, we consider an empty branch data to be
  // a valid profile. It could happen to a function without branches when we
  // still have an EntryData for execution count.
  if (BranchData.Data.empty()) {
    return 1.0f;
  }

  uint64_t NumMatchedBranches = 0;
  for (const auto &BI : BranchData.Data) {
    bool IsValid = false;
    if (BI.From.Name == BI.To.Name) {
      // Try to record information with 0 count.
      IsValid = recordBranch(BI.From.Offset, BI.To.Offset, 0);
    } else {
      // The branch has to originate from this function.
      // Check for calls, tail calls, rets and indirect branches.
      // When matching profiling info, we did not reach the stage
      // when we identify tail calls, so they are still represented
      // by regular branch instructions and we need isBranch() here.
      auto *Instr = getInstructionAtOffset(BI.From.Offset);
      // If it's a prefix - skip it.
      if (Instr && BC.MIA->isPrefix(*Instr))
        Instr = getInstructionAtOffset(BI.From.Offset + 1);
      if (Instr &&
          (BC.MIA->isCall(*Instr) ||
           BC.MIA->isBranch(*Instr) ||
           BC.MIA->isReturn(*Instr))) {
        IsValid = true;
      }
    }

    if (IsValid) {
      ++NumMatchedBranches;
      continue;
    }

    DEBUG(dbgs()
        << "\tinvalid branch in " << *this << " : 0x"
        << Twine::utohexstr(BI.From.Offset) << " -> ";
        if (BI.From.Name == BI.To.Name)
          dbgs() << "0x" << Twine::utohexstr(BI.To.Offset) << '\n';
        else
          dbgs() << "<outbounds>\n";
    );
  }

  const auto MatchRatio = (float) NumMatchedBranches / BranchData.Data.size();
  if (opts::Verbosity >= 2 && NumMatchedBranches < BranchData.Data.size()) {
    errs() << "BOLT-WARNING: profile branches match only "
           << format("%.1f%%", MatchRatio * 100.0f) << " ("
           << NumMatchedBranches << '/' << BranchData.Data.size()
           << ") for function " << *this << '\n';
  }

  return MatchRatio;
}

void BinaryFunction::clearProfile() {
  // Keep function execution profile the same. Only clear basic block and edge
  // counts.
  for (auto *BB : BasicBlocks) {
    BB->ExecutionCount = 0;
    for (auto &BI : BB->branch_info()) {
      BI.Count = 0;
      BI.MispredictedCount = 0;
    }
  }
}

} // namespace bolt
} // namespace llvm
