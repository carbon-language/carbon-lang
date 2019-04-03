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
#include <numeric>

#define DEBUG_TYPE "ICP"
#define DEBUG_VERBOSE(Level, X) if (opts::Verbosity >= (Level)) { X; }

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
    clEnumValN(ICP_ALL, "all", "perform ICP on calls and jump tables")),
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

static cl::opt<unsigned>
IndirectCallPromotionCallsTopN(
    "indirect-call-promotion-calls-topn",
    cl::desc("number of targets to consider when doing indirect "
             "call promotion on calls"),
    cl::init(0),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static cl::opt<unsigned>
IndirectCallPromotionJumpTablesTopN(
    "indirect-call-promotion-jump-tables-topn",
    cl::desc("number of targets to consider when doing indirect "
             "call promotion on jump tables"),
    cl::init(0),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static cl::opt<bool>
EliminateLoads(
    "icp-eliminate-loads",
    cl::desc("enable load elimination using memory profiling data when "
             "performing ICP"),
    cl::init(true),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static cl::opt<unsigned>
ICPTopCallsites(
    "icp-top-callsites",
    cl::desc("only optimize calls that contribute to this percentage of all "
             "indirect calls"),
    cl::init(0),
    cl::Hidden,
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

namespace {

bool verifyProfile(std::map<uint64_t, BinaryFunction> &BFs) {
  bool IsValid = true;
  for (auto &BFI : BFs) {
    auto &BF = BFI.second;
    if (!BF.isSimple()) continue;
    for (auto BB : BF.layout()) {
      auto BI = BB->branch_info_begin();
      for (auto SuccBB : BB->successors()) {
        if (BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE && BI->Count > 0) {
          if (BB->getKnownExecutionCount() == 0 ||
              SuccBB->getKnownExecutionCount() == 0) {
            errs() << "BOLT-WARNING: profile verification failed after ICP for "
                      "function " << BF << '\n';
            IsValid = false;
          }
        }
        ++BI;
      }
    }
  }
  return IsValid;
}

}

IndirectCallPromotion::Callsite::Callsite(BinaryFunction &BF,
                                          const IndirectCallProfile &ICP)
  : From(BF.getSymbol()),
    To(ICP.Offset),
    Mispreds(ICP.Mispreds),
    Branches(ICP.Count) {
  if (ICP.IsFunction) {
    if (auto *BD = BF.getBinaryContext().getBinaryDataByName(ICP.Name)) {
      To.Sym = BD->getSymbol();
      To.Addr = 0;
    }
  }
}

// Get list of targets for a given call sorted by most frequently
// called first.
std::vector<IndirectCallPromotion::Callsite>
IndirectCallPromotion::getCallTargets(
  BinaryBasicBlock &BB,
  const MCInst &Inst
) const {
  auto &BF = *BB.getFunction();
  auto &BC = BF.getBinaryContext();
  std::vector<Callsite> Targets;

  if (const auto *JT = BF.getJumpTable(Inst)) {
    // Don't support PIC jump tables for now
    if (!opts::ICPJumpTablesByTarget && JT->Type == JumpTable::JTT_PIC)
      return Targets;
    const Location From(BF.getSymbol());
    const auto Range = JT->getEntriesForAddress(BC.MIB->getJumpTable(Inst));
    assert(JT->Counts.empty() || JT->Counts.size() >= Range.second);
    JumpTable::JumpInfo DefaultJI;
    const auto *JI = JT->Counts.empty() ? &DefaultJI : &JT->Counts[Range.first];
    const size_t JIAdj = JT->Counts.empty() ? 0 : 1;
    assert(JT->Type == JumpTable::JTT_PIC ||
           JT->EntrySize == BC.AsmInfo->getCodePointerSize());
    for (size_t I = Range.first; I < Range.second; ++I, JI += JIAdj) {
      auto *Entry = JT->Entries[I];
      assert(BF.getBasicBlockForLabel(Entry) ||
             Entry == BF.getFunctionEndLabel() ||
             Entry == BF.getFunctionColdEndLabel());
      if (Entry == BF.getFunctionEndLabel() ||
          Entry == BF.getFunctionColdEndLabel())
        continue;
      const Location To(Entry);
      const auto &BI = BB.getBranchInfo(Entry);
      Targets.emplace_back(
          From, To, BI.MispredictedCount, BI.Count, I - Range.first);
    }

    // Sort by symbol then addr.
    std::sort(Targets.begin(), Targets.end(),
              [](const Callsite &A, const Callsite &B) {
                if (A.To.Sym && B.To.Sym)
                  return A.To.Sym < B.To.Sym;
                else if (A.To.Sym && !B.To.Sym)
                  return true;
                else if (!A.To.Sym && B.To.Sym)
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
      if (A.To.Sym && B.To.Sym && A.To.Sym == B.To.Sym) {
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
    // Don't try to optimize PC relative indirect calls.
    if (Inst.getOperand(0).isReg() &&
        Inst.getOperand(0).getReg() == BC.MRI->getProgramCounter()) {
      return Targets;
    }
    auto ICSP =
      BC.MIB->tryGetAnnotationAs<IndirectCallSiteProfile>(Inst, "CallProfile");
    if (ICSP) {
      for (const auto &CSP : ICSP.get()) {
        Callsite Site(BF, CSP);
        if (Site.isValid())
          Targets.emplace_back(std::move(Site));
      }
    }
  }

  // Sort by most commonly called targets.
  std::stable_sort(Targets.begin(), Targets.end(),
                   [](const Callsite &A, const Callsite &B) {
                     return A.Branches > B.Branches;
                   });

  // Remove non-symbol targets
  auto Last = std::remove_if(Targets.begin(),
                             Targets.end(),
                             [](const Callsite &CS) {
                               return !CS.To.Sym;
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

IndirectCallPromotion::JumpTableInfoType
IndirectCallPromotion::maybeGetHotJumpTableTargets(
   BinaryContext &BC,
   BinaryFunction &Function,
   BinaryBasicBlock *BB,
   MCInst &CallInst,
   MCInst *&TargetFetchInst,
   const JumpTable *JT
) const {
  const auto *MemData = Function.getMemData();
  JumpTableInfoType HotTargets;

  assert(JT && "Can't get jump table addrs for non-jump tables.");

  if (!MemData || !opts::EliminateLoads)
    return JumpTableInfoType();

  MCInst *MemLocInstr;
  MCInst *PCRelBaseOut;
  unsigned BaseReg, IndexReg;
  int64_t DispValue;
  const MCExpr *DispExpr;
  MutableArrayRef<MCInst> Insts(&BB->front(), &CallInst);
  const auto Type = BC.MIB->analyzeIndirectBranch(
      CallInst, Insts.begin(), Insts.end(), BC.AsmInfo->getCodePointerSize(),
      MemLocInstr, BaseReg, IndexReg, DispValue, DispExpr, PCRelBaseOut);

  assert(MemLocInstr && "There should always be a load for jump tables");
  if (!MemLocInstr)
    return JumpTableInfoType();

  DEBUG({
      dbgs() << "BOLT-INFO: ICP attempting to find memory profiling data for "
             << "jump table in " << Function << " at @ "
             << (&CallInst - &BB->front()) << "\n"
             << "BOLT-INFO: ICP target fetch instructions:\n";
      BC.printInstruction(dbgs(), *MemLocInstr, 0, &Function);
      if (MemLocInstr != &CallInst) {
        BC.printInstruction(dbgs(), CallInst, 0, &Function);
      }
    });

  DEBUG_VERBOSE(1, {
      dbgs() << "Jmp info: Type = " << (unsigned)Type << ", "
             << "BaseReg = " << BC.MRI->getName(BaseReg) << ", "
             << "IndexReg = " << BC.MRI->getName(IndexReg) << ", "
             << "DispValue = " << Twine::utohexstr(DispValue) << ", "
             << "DispExpr = " << DispExpr << ", "
             << "MemLocInstr = ";
      BC.printInstruction(dbgs(), *MemLocInstr, 0, &Function);
      dbgs() << "\n";
    });

  ++TotalIndexBasedCandidates;

  // Try to get value profiling data for the method load instruction.
  auto DataOffset = BC.MIB->tryGetAnnotationAs<uint64_t>(*MemLocInstr,
                                                         "MemDataOffset");

  if (!DataOffset) {
    DEBUG_VERBOSE(1, dbgs() << "BOLT-INFO: ICP no memory profiling data found\n");
    return JumpTableInfoType();
  }

  uint64_t ArrayStart;
  if (DispExpr) {
    auto *BD = BC.getBinaryDataByName(DispExpr->getSymbol().getName());
    assert(BD && "global symbol needs a value");
    ArrayStart = BD->getAddress();
  } else {
    ArrayStart = static_cast<uint64_t>(DispValue);
  }

  if (BaseReg == BC.MRI->getProgramCounter()) {
    auto FunctionData = BC.getFunctionData(Function);
    const uint64_t Address = Function.getAddress() + DataOffset.get();
    MCInst OrigJmp;
    uint64_t Size;
    assert(FunctionData);
    auto Success = BC.DisAsm->getInstruction(OrigJmp,
                                             Size,
                                             *FunctionData,
                                             Address,
                                             nulls(),
                                             nulls());
    assert(Success && "Must be able to disassmble original jump instruction");
    ArrayStart += Address + Size;
  }

  // This is a map of [symbol] -> [count, index] and is used to combine indices
  // into the jump table since there may be multiple addresses that all have the
  // same entry.
  std::map<MCSymbol *, std::pair<uint64_t, uint64_t>> HotTargetMap;
  const auto Range = JT->getEntriesForAddress(ArrayStart);

  for (const auto &MI : MemData->getMemInfoRange(DataOffset.get())) {
    size_t Index;
    if (!MI.Addr.Offset) // mem data occasionally includes nulls, ignore them
      continue;

    if (MI.Addr.Offset % JT->EntrySize != 0) // ignore bogus data
      return JumpTableInfoType();

    if (MI.Addr.IsSymbol) {
      // Deal with bad/stale data
      if (!MI.Addr.Name.startswith("JUMP_TABLE/" + Function.getNames().front()))
        return JumpTableInfoType();
      Index = (MI.Addr.Offset - (ArrayStart - JT->getAddress())) / JT->EntrySize;
    } else {
      Index = (MI.Addr.Offset - ArrayStart) / JT->EntrySize;
    }

    // If Index is out of range it probably means the memory profiling data is
    // wrong for this instruction, bail out.
    if (Index >= Range.second) {
      DEBUG(dbgs() << "BOLT-INFO: Index out of range of " << Range.first
                   << ", " << Range.second << "\n");
      return JumpTableInfoType();
    }

    // Make sure the hot index points at a legal label corresponding to a BB,
    // e.g. not the end of function (unreachable) label.
    if (!Function.getBasicBlockForLabel(JT->Entries[Index + Range.first])) {
      DEBUG({
          dbgs() << "BOLT-INFO: hot index " << Index << " pointing at bogus "
                 << "label " << JT->Entries[Index + Range.first]->getName()
                 << " in jump table:\n";
          JT->print(dbgs());
          dbgs() << "HotTargetMap:\n";
          for (auto &HT : HotTargetMap) {
            dbgs() << "BOLT-INFO: " << HT.first->getName()
                   << " = (count=" << HT.first << ", index=" << HT.second
                   << ")\n";
          }
          dbgs() << "BOLT-INFO: MemData:\n";
          for (auto &MI : MemData->getMemInfoRange(DataOffset.get())) {
            dbgs() << "BOLT-INFO: " << MI << "\n";
          }
        });
      return JumpTableInfoType();
    }

    auto &HotTarget = HotTargetMap[JT->Entries[Index + Range.first]];
    HotTarget.first += MI.Count;
    HotTarget.second = Index;
  }

  std::transform(
     HotTargetMap.begin(),
     HotTargetMap.end(),
     std::back_inserter(HotTargets),
     [](const std::pair<MCSymbol *, std::pair<uint64_t, uint64_t>> &A) {
       return A.second;
     });

  // Sort with highest counts first.
  std::sort(HotTargets.rbegin(), HotTargets.rend());

  DEBUG({
      dbgs() << "BOLT-INFO: ICP jump table hot targets:\n";
      for (const auto &Target : HotTargets) {
        dbgs() << "BOLT-INFO:  Idx = " << Target.second << ", "
               << "Count = " << Target.first << "\n";
      }
    });

  BC.MIB->getOrCreateAnnotationAs<uint16_t>(CallInst, "JTIndexReg") = IndexReg;

  TargetFetchInst = MemLocInstr;

  return HotTargets;
}

IndirectCallPromotion::SymTargetsType
IndirectCallPromotion::findCallTargetSymbols(
  BinaryContext &BC,
  std::vector<Callsite> &Targets,
  const size_t N,
  BinaryFunction &Function,
  BinaryBasicBlock *BB,
  MCInst &CallInst,
  MCInst *&TargetFetchInst
) const {
  const auto *JT = Function.getJumpTable(CallInst);
  SymTargetsType SymTargets;

  if (JT) {
    auto HotTargets = maybeGetHotJumpTableTargets(BC,
                                                  Function,
                                                  BB,
                                                  CallInst,
                                                  TargetFetchInst,
                                                  JT);

    if (!HotTargets.empty()) {
      auto findTargetsIndex = [&](uint64_t JTIndex) {
        for (size_t I = 0; I < Targets.size(); ++I) {
          auto &JTIs = Targets[I].JTIndex;
          if (std::find(JTIs.begin(), JTIs.end(), JTIndex) != JTIs.end())
            return I;
        }
        DEBUG(dbgs() << "BOLT-ERROR: Unable to find target index for hot jump "
                     << " table entry in " << Function << "\n");
        llvm_unreachable("Hot indices must be referred to by at least one "
                         "callsite");
      };

      const auto MaxHotTargets = std::min(N, HotTargets.size());

      if (opts::Verbosity >= 1) {
        for (size_t I = 0; I < MaxHotTargets; ++I) {
          outs() << "BOLT-INFO: HotTarget[" << I << "] = ("
                 << HotTargets[I].first << ", " << HotTargets[I].second << ")\n";
        }
      }

      std::vector<Callsite> NewTargets;
      for (size_t I = 0; I < MaxHotTargets; ++I) {
        const auto JTIndex = HotTargets[I].second;
        const auto TargetIndex = findTargetsIndex(JTIndex);

        NewTargets.push_back(Targets[TargetIndex]);
        std::vector<uint64_t>({JTIndex}).swap(NewTargets.back().JTIndex);

        Targets.erase(Targets.begin() + TargetIndex);
      }
      std::copy(Targets.begin(), Targets.end(), std::back_inserter(NewTargets));
      assert(NewTargets.size() == Targets.size() + MaxHotTargets);
      std::swap(NewTargets, Targets);
    }

    for (size_t I = 0, TgtIdx = 0; I < N; ++TgtIdx) {
      auto &Target = Targets[TgtIdx];
      assert(Target.To.Sym && "All ICP targets must be to known symbols");
      assert(!Target.JTIndex.empty() && "Jump tables must have indices");
      for (auto Idx : Target.JTIndex) {
        SymTargets.push_back(std::make_pair(Target.To.Sym, Idx));
        ++I;
      }
    }
  } else {
    for (size_t I = 0; I < N; ++I) {
      assert(Targets[I].To.Sym &&
             "All ICP targets must be to known symbols");
      assert(Targets[I].JTIndex.empty() &&
             "Can't have jump table indices for non-jump tables");
      SymTargets.push_back(std::make_pair(Targets[I].To.Sym, 0));
    }
  }

  return SymTargets;
}

IndirectCallPromotion::MethodInfoType
IndirectCallPromotion::maybeGetVtableSyms(
   BinaryContext &BC,
   BinaryFunction &Function,
   BinaryBasicBlock *BB,
   MCInst &Inst,
   const SymTargetsType &SymTargets
) const {
  const auto *MemData = Function.getMemData();
  std::vector<std::pair<MCSymbol *, uint64_t>> VtableSyms;
  std::vector<MCInst *> MethodFetchInsns;
  unsigned VtableReg, MethodReg;
  uint64_t MethodOffset;

  assert(!Function.getJumpTable(Inst) &&
         "Can't get vtable addrs for jump tables.");

  if (!MemData || !opts::EliminateLoads)
    return MethodInfoType();

  MutableArrayRef<MCInst> Insts(&BB->front(), &Inst + 1);
  if (!BC.MIB->analyzeVirtualMethodCall(Insts.begin(),
                                        Insts.end(),
                                        MethodFetchInsns,
                                        VtableReg,
                                        MethodReg,
                                        MethodOffset)) {
    DEBUG_VERBOSE(1, dbgs() << "BOLT-INFO: ICP unable to analyze method call in "
                            << Function << " at @ " << (&Inst - &BB->front())
                            << "\n");
    return MethodInfoType();
  }

  ++TotalMethodLoadEliminationCandidates;

  DEBUG_VERBOSE(1,
    dbgs() << "BOLT-INFO: ICP found virtual method call in "
           << Function << " at @ " << (&Inst - &BB->front()) << "\n";
    dbgs() << "BOLT-INFO: ICP method fetch instructions:\n";
    for (auto *Inst : MethodFetchInsns) {
      BC.printInstruction(dbgs(), *Inst, 0, &Function);
    }
    if (MethodFetchInsns.back() != &Inst) {
      BC.printInstruction(dbgs(), Inst, 0, &Function);
    }
  );

  // Try to get value profiling data for the method load instruction.
  auto DataOffset = BC.MIB->tryGetAnnotationAs<uint64_t>(*MethodFetchInsns.back(),
                                                         "MemDataOffset");

  if (!DataOffset) {
    DEBUG_VERBOSE(1, dbgs() << "BOLT-INFO: ICP no memory profiling data found\n");
    return MethodInfoType();
  }

  // Find the vtable that each method belongs to.
  std::map<const MCSymbol *, uint64_t> MethodToVtable;

  for (auto &MI : MemData->getMemInfoRange(DataOffset.get())) {
    uint64_t Address;

    if (MI.Addr.IsSymbol) {
      auto *BD = BC.getBinaryDataByName(MI.Addr.Name);
      Address = BD ? BD->getAddress() + MI.Addr.Offset : 0;
    } else {
      Address = MI.Addr.Offset;
    }

    // Ignore bogus data.
    if (!Address)
      continue;

    const auto VtableBase = Address - MethodOffset;

    DEBUG_VERBOSE(1, dbgs() << "BOLT-INFO: ICP vtable = "
                            << Twine::utohexstr(VtableBase)
                            << "+" << MethodOffset << "/" << MI.Count
                            << "\n");

    if (auto MethodAddr = BC.extractPointerAtAddress(Address)) {
      auto *MethodBD = BC.getBinaryDataAtAddress(MethodAddr.get());
      if (!MethodBD)  // skip unknown methods
        continue;
      auto *MethodSym = MethodBD->getSymbol();
      MethodToVtable[MethodSym] = VtableBase;
      DEBUG_VERBOSE(1,
        const auto *Method = BC.getFunctionForSymbol(MethodSym);
        dbgs() << "BOLT-INFO: ICP found method = "
               << Twine::utohexstr(MethodAddr.get()) << "/"
               << (Method ? Method->getPrintName() : "") << "\n";
      );
    }
  }

  // Find the vtable for each target symbol.
  for (size_t I = 0; I < SymTargets.size(); ++I) {
    auto Itr = MethodToVtable.find(SymTargets[I].first);
    if (Itr != MethodToVtable.end()) {
      if (auto *BD = BC.getBinaryDataContainingAddress(Itr->second)) {
        const uint64_t Addend = Itr->second - BD->getAddress();
        VtableSyms.push_back(std::make_pair(BD->getSymbol(), Addend));
        continue;
      }
    }
    // Give up if we can't find the vtable for a method.
    DEBUG_VERBOSE(1, dbgs() << "BOLT-INFO: ICP can't find vtable for "
                            << SymTargets[I].first->getName() << "\n");
    return MethodInfoType();
  }

  // Make sure the vtable reg is not clobbered by the argument passing code
  if (VtableReg != MethodReg) {
    for (auto *CurInst = MethodFetchInsns.front(); CurInst < &Inst; ++CurInst) {
      const auto &InstrInfo = BC.MII->get(CurInst->getOpcode());
      if (InstrInfo.hasDefOfPhysReg(*CurInst, VtableReg, *BC.MRI)) {
        return MethodInfoType();
      }
    }
  }

  return MethodInfoType(VtableSyms, MethodFetchInsns);
}

std::vector<std::unique_ptr<BinaryBasicBlock>>
IndirectCallPromotion::rewriteCall(
   BinaryContext &BC,
   BinaryFunction &Function,
   BinaryBasicBlock *IndCallBlock,
   const MCInst &CallInst,
   MCPlusBuilder::ICPdata &&ICPcode,
   const std::vector<MCInst *> &MethodFetchInsns
) const {
  // Create new basic blocks with correct code in each one first.
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
  const bool IsTailCallOrJT = (BC.MIB->isTailCall(CallInst) ||
                               Function.getJumpTable(CallInst));

  // Move instructions from the tail of the original call block
  // to the merge block.

  // Remember any pseudo instructions following a tail call.  These
  // must be preserved and moved to the original block.
  std::vector<MCInst> TailInsts;
  const auto *TailInst = &CallInst;
  if (IsTailCallOrJT) {
    while (TailInst + 1 < &(*IndCallBlock->end()) &&
           BC.MII->get((TailInst + 1)->getOpcode()).isPseudo()) {
      TailInsts.push_back(*++TailInst);
    }
  }

  auto MovedInst = IndCallBlock->splitInstructions(&CallInst);

  IndCallBlock->eraseInstructions(MethodFetchInsns.begin(),
                                  MethodFetchInsns.end());
  if (IndCallBlock->empty() ||
      (!MethodFetchInsns.empty() && MethodFetchInsns.back() == &CallInst)) {
    IndCallBlock->addInstructions(ICPcode.front().second.begin(),
                                  ICPcode.front().second.end());
  } else {
    IndCallBlock->replaceInstruction(std::prev(IndCallBlock->end()),
                                     ICPcode.front().second);
  }
  IndCallBlock->addInstructions(TailInsts.begin(), TailInsts.end());

  for (auto Itr = ICPcode.begin() + 1; Itr != ICPcode.end(); ++Itr) {
    auto &Sym = Itr->first;
    auto &Insts = Itr->second;
    assert(Sym);
    auto TBB = Function.createBasicBlock(0, Sym);
    for (auto &Inst : Insts) { // sanitize new instructions.
      if (BC.MIB->isCall(Inst))
        BC.MIB->removeAnnotation(Inst, "CallProfile");
    }
    TBB->addInstructions(Insts.begin(), Insts.end());
    NewBBs.emplace_back(std::move(TBB));
  }

  // Move tail of instructions from after the original call to
  // the merge block.
  if (!IsTailCallOrJT) {
    NewBBs.back()->addInstructions(MovedInst.begin(), MovedInst.end());
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

  // Scale indirect call counts to the execution count of the original
  // basic block containing the indirect call.
  auto TotalCount = IndCallBlock->getKnownExecutionCount();
  uint64_t TotalIndirectBranches = 0;
  for (const auto &Target : Targets) {
    TotalIndirectBranches += Target.Branches;
  }
  std::vector<BinaryBranchInfo> BBI;
  std::vector<BinaryBranchInfo> ScaledBBI;
  for (const auto &Target : Targets) {
    const auto NumEntries = std::max(1UL, Target.JTIndex.size());
    for (size_t I = 0; I < NumEntries; ++I) {
      BBI.push_back(
          BinaryBranchInfo{(Target.Branches + NumEntries - 1) / NumEntries,
                           (Target.Mispreds + NumEntries - 1)  / NumEntries});
      ScaledBBI.push_back(BinaryBranchInfo{
          uint64_t(TotalCount * Target.Branches /
                                      (NumEntries * TotalIndirectBranches)),
          uint64_t(TotalCount * Target.Mispreds /
                                      (NumEntries * TotalIndirectBranches))});
    }
  }

  if (IsJumpTable) {
    auto *NewIndCallBlock = NewBBs.back().get();
    IndCallBlock->moveAllSuccessorsTo(NewIndCallBlock);

    std::vector<MCSymbol*> SymTargets;
    for (const auto &Target : Targets) {
      const auto NumEntries = std::max(1UL, Target.JTIndex.size());
      for (size_t I = 0; I < NumEntries; ++I) {
        SymTargets.push_back(Target.To.Sym);
      }
    }
    assert(SymTargets.size() > NewBBs.size() - 1 &&
           "There must be a target symbol associated with each new BB.");

    for (uint64_t I = 0; I < NewBBs.size(); ++I) {
      BinaryBasicBlock *SourceBB = I ? NewBBs[I - 1].get() : IndCallBlock;
      SourceBB->setExecutionCount(TotalCount);

      auto *TargetBB = Function.getBasicBlockForLabel(SymTargets[I]);
      SourceBB->addSuccessor(TargetBB, ScaledBBI[I]); // taken

      TotalCount -= ScaledBBI[I].Count;
      SourceBB->addSuccessor(NewBBs[I].get(), TotalCount); // fall-through

      // Update branch info for the indirect jump.
      auto &BranchInfo = NewIndCallBlock->getBranchInfo(*TargetBB);
      if (BranchInfo.Count > BBI[I].Count)
        BranchInfo.Count -= BBI[I].Count;
      else
        BranchInfo.Count = 0;

      if (BranchInfo.MispredictedCount > BBI[I].MispredictedCount)
        BranchInfo.MispredictedCount -= BBI[I].MispredictedCount;
      else
        BranchInfo.MispredictedCount = 0;
    }
  } else {
    assert(NewBBs.size() >= 2);
    assert(NewBBs.size() % 2 == 1 || IndCallBlock->succ_empty());
    assert(NewBBs.size() % 2 == 1 || IsTailCall);

    auto ScaledBI = ScaledBBI.begin();
    auto updateCurrentBranchInfo = [&]{
      assert(ScaledBI != ScaledBBI.end());
      TotalCount -= ScaledBI->Count;
      ++ScaledBI;
    };

    if (!IsTailCall) {
      MergeBlock = NewBBs.back().get();
      IndCallBlock->moveAllSuccessorsTo(MergeBlock);
    }

    // Fix up successors and execution counts.
    updateCurrentBranchInfo();
    IndCallBlock->addSuccessor(NewBBs[1].get(), TotalCount);
    IndCallBlock->addSuccessor(NewBBs[0].get(), ScaledBBI[0]);

    const size_t Adj = IsTailCall ? 1 : 2;
    for (size_t I = 0; I < NewBBs.size() - Adj; ++I) {
      assert(TotalCount <= IndCallBlock->getExecutionCount() ||
             TotalCount <= uint64_t(TotalIndirectBranches));
      auto ExecCount = ScaledBBI[(I+1)/2].Count;
      if (I % 2 == 0) {
        if (MergeBlock) {
          NewBBs[I]->addSuccessor(MergeBlock, ScaledBBI[(I+1)/2].Count);
        }
      } else {
        assert(I + 2 < NewBBs.size());
        updateCurrentBranchInfo();
        NewBBs[I]->addSuccessor(NewBBs[I+2].get(), TotalCount);
        NewBBs[I]->addSuccessor(NewBBs[I+1].get(), ScaledBBI[(I+1)/2]);
        ExecCount += TotalCount;
      }
      NewBBs[I]->setExecutionCount(ExecCount);
    }

    if (MergeBlock) {
      // Arrange for the MergeBlock to be the fallthrough for the first
      // promoted call block.
      std::unique_ptr<BinaryBasicBlock> MBPtr;
      std::swap(MBPtr, NewBBs.back());
      NewBBs.pop_back();
      NewBBs.emplace(NewBBs.begin() + 1, std::move(MBPtr));
      // TODO: is COUNT_FALLTHROUGH_EDGE the right thing here?
      NewBBs.back()->addSuccessor(MergeBlock, TotalCount); // uncond branch
    }
  }

  // Update the execution count.
  NewBBs.back()->setExecutionCount(TotalCount);

  // Update BB and BB layout.
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

  auto computeStats = [&](size_t N) {
    for (size_t I = 0; I < N; ++I) {
      if (!IsJumpTable)
        TotalNumFrequentCalls += Targets[I].Branches;
      else
        TotalNumFrequentJmps += Targets[I].Branches;
    }
  };

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

  size_t TopN = opts::IndirectCallPromotionTopN;
  if (IsJumpTable) {
    if (opts::IndirectCallPromotionJumpTablesTopN != 0)
      TopN = opts::IndirectCallPromotionJumpTablesTopN;
  } else if (opts::IndirectCallPromotionCallsTopN != 0) {
    TopN = opts::IndirectCallPromotionCallsTopN;
  }
  const auto TrialN = std::min(TopN, Targets.size());

  if (opts::ICPTopCallsites > 0) {
    auto &BC = BB->getFunction()->getBinaryContext();
    if (BC.MIB->hasAnnotation(Inst, "DoICP")) {
      computeStats(TrialN);
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
      Threshold -= (100.0 * Targets[I].Mispreds) / NumCalls;
      TotalMispredictsTopN += Targets[I].Mispreds;
    }
    computeStats(N);

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
    size_t MaxTargets = 0;

    // Count total number of calls for (at most) the top N targets.
    // We may choose a smaller N (TrialN vs. N) if the frequency threshold
    // is exceeded by fewer targets.
    double Threshold = double(opts::IndirectCallPromotionThreshold);
    for (size_t I = 0; I < TrialN && Threshold > 0; ++I, ++MaxTargets) {
      if (N + (Targets[I].JTIndex.empty() ? 1 : Targets[I].JTIndex.size()) >
          TrialN)
        break;
      TotalCallsTopN += Targets[I].Branches;
      TotalMispredictsTopN += Targets[I].Mispreds;
      Threshold -= (100.0 * Targets[I].Branches) / NumCalls;
      N += Targets[I].JTIndex.empty() ? 1 : Targets[I].JTIndex.size();
    }
    computeStats(MaxTargets);

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
    if (!IsJumpTable || opts::ICPJumpTablesByTarget) {
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
  }

  // Filter functions that can have ICP applied (for debugging)
  if (!opts::ICPFuncsList.empty()) {
    for (auto &Name : opts::ICPFuncsList) {
      if (BB->getFunction()->hasName(Name))
        return N;
    }
    return 0;
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
  const bool IsTailCall = BC.MIB->isTailCall(Inst);
  const bool IsJumpTable = BB->getFunction()->getJumpTable(Inst);
  const auto InstIdx = &Inst - &(*BB->begin());

  outs() << "BOLT-INFO: ICP candidate branch info: "
         << *BB->getFunction() << " @ " << InstIdx
         << " in " << BB->getName()
         << " -> calls = " << NumCalls
         << (IsTailCall ? " (tail)" : (IsJumpTable ? " (jump table)" : ""))
         << "\n";
  for (size_t I = 0; I < N; I++) {
    const auto Frequency = 100.0 * Targets[I].Branches / NumCalls;
    const auto MisFrequency = 100.0 * Targets[I].Mispreds / NumCalls;
    outs() << "BOLT-INFO:   ";
    if (Targets[I].To.Sym)
      outs() << Targets[I].To.Sym->getName();
    else
      outs() << Targets[I].To.Addr;
    outs() << ", calls = " << Targets[I].Branches
           << ", mispreds = " << Targets[I].Mispreds
           << ", taken freq = " << format("%.1f", Frequency) << "%"
           << ", mis. freq = " << format("%.1f", MisFrequency) << "%";
    bool First = true;
    for (auto JTIndex : Targets[I].JTIndex) {
      outs() << (First ? ", indices = " : ", ") << JTIndex;
      First = false;
    }
    outs() << "\n";
  }

  DEBUG({
    dbgs() << "BOLT-INFO: ICP original call instruction:";
    BC.printInstruction(dbgs(), Inst, Targets[0].From.Addr, nullptr, true);
  });
}

void IndirectCallPromotion::runOnFunctions(BinaryContext &BC) {
  if (opts::IndirectCallPromotion == ICP_NONE)
    return;

  auto &BFs = BC.getBinaryFunctions();

  const bool OptimizeCalls =
    (opts::IndirectCallPromotion == ICP_CALLS ||
     opts::IndirectCallPromotion == ICP_ALL);
  const bool OptimizeJumpTables =
    (opts::IndirectCallPromotion == ICP_JUMP_TABLES ||
     opts::IndirectCallPromotion == ICP_ALL);

  std::unique_ptr<RegAnalysis> RA;
  std::unique_ptr<BinaryFunctionCallGraph> CG;
  if (OptimizeJumpTables) {
    CG.reset(new BinaryFunctionCallGraph(buildCallGraph(BC)));
    RA.reset(new RegAnalysis(BC, &BFs, &*CG));
  }

  DEBUG_VERBOSE(2, {
      for (auto &BFIt : BFs) {
        auto &Function = BFIt.second;
        const auto *MemData = Function.getMemData();
        bool DidPrintFunc = false;
        uint64_t Offset = 0;

        if (!MemData || !Function.isSimple() || !opts::shouldProcess(Function))
          continue;

        for (auto &BB : Function) {
          bool PrintBB = false;
          for (auto &Inst : BB) {
            if (auto Mem =
                  BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, "MemDataOffset")) {
              for (auto &MI : MemData->getMemInfoRange(Mem.get())) {
                if (MI.Addr.IsSymbol) {
                  PrintBB = true;
                  break;
                }
                if (BC.getSectionForAddress(MI.Addr.Offset)) {
                  PrintBB = true;
                  break;
                }
              }
            }
          }
          if (PrintBB && !DidPrintFunc) {
            dbgs() << "\nNon-heap/stack memory data found in "
                   << Function << ":\n";
            DidPrintFunc = true;
          }
          Offset = BC.printInstructions(PrintBB ? dbgs() : nulls(),
                                        BB.begin(),
                                        BB.end(),
                                        Offset,
                                        &Function);
        }
      }
    });

  // If icp-top-callsites is enabled, compute the total number of indirect
  // calls and then optimize the hottest callsites that contribute to that
  // total.
  if (opts::ICPTopCallsites > 0) {
    using IndirectCallsite = std::pair<uint64_t, MCInst *>;
    std::vector<IndirectCallsite> IndirectCalls;
    size_t TotalIndirectCalls = 0;

    // Find all the indirect callsites.
    for (auto &BFIt : BFs) {
      auto &Function = BFIt.second;

      if (!Function.isSimple() ||
          !opts::shouldProcess(Function) ||
          !Function.hasProfile())
        continue;

      const bool HasLayout = !Function.layout_empty();

      for (auto &BB : Function) {
        if (HasLayout && Function.isSplit() && BB.isCold())
          continue;

        for (auto &Inst : BB) {
          const bool IsJumpTable = Function.getJumpTable(Inst);
          const bool HasIndirectCallProfile =
            BC.MIB->hasAnnotation(Inst, "CallProfile");
          const bool IsDirectCall = (BC.MIB->isCall(Inst) &&
                                     BC.MIB->getTargetSymbol(Inst, 0));

          if (!IsDirectCall &&
              ((HasIndirectCallProfile && !IsJumpTable && OptimizeCalls) ||
               (IsJumpTable && OptimizeJumpTables))) {
            uint64_t NumCalls = 0;
            for (const auto &BInfo : getCallTargets(BB, Inst)) {
              NumCalls += BInfo.Branches;
            }

            IndirectCalls.push_back(std::make_pair(NumCalls, &Inst));
            TotalIndirectCalls += NumCalls;
          }
        }
      }
    }

    // Sort callsites by execution count.
    std::sort(IndirectCalls.rbegin(), IndirectCalls.rend());

    // Find callsites that contribute to the top "opts::ICPTopCallsites"%
    // number of calls.
    const float TopPerc = opts::ICPTopCallsites / 100.0f;
    int64_t MaxCalls = TotalIndirectCalls * TopPerc;
    size_t Num = 0;
    for (auto &IC : IndirectCalls) {
      if (MaxCalls <= 0)
        break;
      MaxCalls -= IC.first;
      ++Num;
    }
    outs() << "BOLT-INFO: ICP Total indirect calls = " << TotalIndirectCalls
           << ", " << Num << " callsites cover " << opts::ICPTopCallsites
           << "% of all indirect calls\n";

    // Mark sites to optimize with "DoICP" annotation.
    for (size_t I = 0; I < Num; ++I) {
      auto *Inst = IndirectCalls[I].second;
      BC.MIB->addAnnotation(*Inst, "DoICP", true);
    }
  }

  for (auto &BFIt : BFs) {
    auto &Function = BFIt.second;

    if (!Function.isSimple() || !opts::shouldProcess(Function))
      continue;

    if (!Function.hasProfile())
      continue;

    const bool HasLayout = !Function.layout_empty();

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
        const bool IsTailCall = BC.MIB->isTailCall(Inst);
        const bool HasIndirectCallProfile =
          BC.MIB->hasAnnotation(Inst, "CallProfile");
        const bool IsJumpTable = Function.getJumpTable(Inst);

        if (BC.MIB->isCall(Inst)) {
          TotalCalls += BB->getKnownExecutionCount();
        }

        if (IsJumpTable && !OptimizeJumpTables)
          continue;

        if (!IsJumpTable && (!HasIndirectCallProfile || !OptimizeCalls))
          continue;

        // Ignore direct calls.
        if (BC.MIB->isCall(Inst) && BC.MIB->getTargetSymbol(Inst, 0))
          continue;

        assert((BC.MIB->isCall(Inst) || BC.MIB->isIndirectBranch(Inst))
               && "expected a call or an indirect jump instruction");

        if (IsJumpTable)
          ++TotalJumpTableCallsites;
        else
          ++TotalIndirectCallsites;

        auto Targets = getCallTargets(*BB, Inst);

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
        if (IsJumpTable) {
          auto State = Info.getLivenessAnalysis().getStateBefore(Inst);
          if (!State || (State && (*State)[BC.MIB->getFlagsReg()])) {
            if (opts::Verbosity >= 1) {
              outs() << "BOLT-INFO: ICP failed in " << Function << " @ "
                     << InstIdx << " in " << BB->getName()
                     << ", calls = " << NumCalls
                     << (State ? ", cannot clobber flags reg.\n"
                               : ", no liveness data available.\n");
            }
            continue;
          }
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
        MCInst *TargetFetchInst = nullptr;
        const auto SymTargets = findCallTargetSymbols(BC,
                                                      Targets,
                                                      N,
                                                      Function,
                                                      BB,
                                                      Inst,
                                                      TargetFetchInst);

        // If we can't resolve any of the target symbols, punt on this callsite.
        // TODO: can this ever happen?
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

        MethodInfoType MethodInfo;

        if (!IsJumpTable) {
          MethodInfo = maybeGetVtableSyms(BC,
                                          Function,
                                          BB,
                                          Inst,
                                          SymTargets);
          TotalMethodLoadsEliminated += MethodInfo.first.empty() ? 0 : 1;
          DEBUG(dbgs() << "BOLT-INFO: ICP "
                       << (!MethodInfo.first.empty() ? "found" : "did not find")
                       << " vtables for all methods.\n");
        } else if (TargetFetchInst) {
          ++TotalIndexBasedJumps;
          MethodInfo.second.push_back(TargetFetchInst);
        }

        // Generate new promoted call code for this callsite.
        auto ICPcode =
            (IsJumpTable && !opts::ICPJumpTablesByTarget)
                ? BC.MIB->jumpTablePromotion(Inst,
                                             SymTargets,
                                             MethodInfo.second,
                                             BC.Ctx.get())
                : BC.MIB->indirectCallPromotion(
                      Inst, SymTargets, MethodInfo.first, MethodInfo.second,
                      opts::ICPOldCodeSequence, BC.Ctx.get());

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
        auto NewBBs = rewriteCall(BC,
                                  Function,
                                  BB,
                                  Inst,
                                  std::move(ICPcode),
                                  MethodInfo.second);

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

  outs() << "BOLT-INFO: ICP total indirect callsites with profile = "
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
         << "BOLT-INFO: ICP percentage of indirect callsites that are "
            "optimized = "
         << format("%.1f", (100.0 * TotalOptimizedIndirectCallsites) /
                   std::max(TotalIndirectCallsites, 1ul))
         << "%\n"
         << "BOLT-INFO: ICP number of method load elimination candidates = "
         << TotalMethodLoadEliminationCandidates
         << "\n"
         << "BOLT-INFO: ICP percentage of method calls candidates that have "
            "loads eliminated = "
         << format("%.1f", (100.0 * TotalMethodLoadsEliminated) /
                   std::max(TotalMethodLoadEliminationCandidates, 1ul))
         << "%\n"
         << "BOLT-INFO: ICP percentage of indirect branches that are "
            "optimized = "
         << format("%.1f", (100.0 * TotalNumFrequentJmps) /
                   std::max(TotalIndirectJmps, 1ul))
         << "%\n"
         << "BOLT-INFO: ICP percentage of jump table callsites that are "
         << "optimized = "
         << format("%.1f", (100.0 * TotalOptimizedJumpTableCallsites) /
                   std::max(TotalJumpTableCallsites, 1ul))
         << "%\n"
         << "BOLT-INFO: ICP number of jump table callsites that can use hot "
         << "indices = " << TotalIndexBasedCandidates
         << "\n"
         << "BOLT-INFO: ICP percentage of jump table callsites that use hot "
            "indices = "
         << format("%.1f", (100.0 * TotalIndexBasedJumps) /
                   std::max(TotalIndexBasedCandidates, 1ul))
         << "%\n";

#ifndef NDEBUG
  verifyProfile(BFs);
#endif
}

} // namespace bolt
} // namespace llvm
