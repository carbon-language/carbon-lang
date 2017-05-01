//===--- Passes/FrameAnalysis.h -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
#include "FrameAnalysis.h"
#include <fstream>

#define DEBUG_TYPE "fa"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> Verbosity;
extern bool shouldProcess(const bolt::BinaryFunction &Function);

static cl::list<std::string>
    FrameOptFunctionNames("funcs-fop", cl::CommaSeparated,
                          cl::desc("list of functions to apply frame opts"),
                          cl::value_desc("func1,func2,func3,..."));

static cl::opt<std::string> FrameOptFunctionNamesFile(
    "funcs-file-fop",
    cl::desc("file with list of functions to frame optimize"));

bool shouldFrameOptimize(const llvm::bolt::BinaryFunction &Function) {
  if (!FrameOptFunctionNamesFile.empty()) {
    assert(!FrameOptFunctionNamesFile.empty() && "unexpected empty file name");
    std::ifstream FuncsFile(FrameOptFunctionNamesFile, std::ios::in);
    std::string FuncName;
    while (std::getline(FuncsFile, FuncName)) {
      FrameOptFunctionNames.push_back(FuncName);
    }
    FrameOptFunctionNamesFile = "";
  }

  bool IsValid = true;
  if (!FrameOptFunctionNames.empty()) {
    IsValid = false;
    for (auto &Name : FrameOptFunctionNames) {
      if (Function.hasName(Name)) {
        IsValid = true;
        break;
      }
    }
  }
  if (!IsValid)
    return false;

  return IsValid;
}
} // namespace opts

namespace llvm {
namespace bolt {

raw_ostream &operator<<(raw_ostream &OS, const FrameIndexEntry &FIE) {
  OS << "FrameIndexEntry<IsLoad: " << FIE.IsLoad << ", IsStore: " << FIE.IsStore
     << ", IsStoreFromReg: " << FIE.IsStoreFromReg
     << ", RegOrImm: " << FIE.RegOrImm << ", StackOffset: ";
  if (FIE.StackOffset < 0)
    OS << "-" << Twine::utohexstr(-FIE.StackOffset);
  else
    OS << "+" << Twine::utohexstr(FIE.StackOffset);
  OS << ", Size: " << FIE.Size << ", IsSimple: " << FIE.IsSimple << ">";
  return OS;
}

namespace {

/// This class should be used to iterate through basic blocks in layout order
/// to analyze instructions for frame accesses. The user should call
/// enterNewBB() whenever starting analyzing a new BB and doNext() for each
/// instruction. After doNext(), if isValidAccess() returns true, it means the
/// current instruction accesses the frame and getFIE() may be used to obtain
/// details about this access.
class FrameAccessAnalysis {
  /// We depend on Stack Pointer Tracking to figure out the current SP offset
  /// value at a given program point
  StackPointerTracking SPT;
  /// Context vars
  const BinaryContext &BC;
  const BinaryFunction &BF;
  // Vars used for storing useful CFI info to give us a hint about how the stack
  // is used in this function
  int SPOffset{0};
  int FPOffset{0};
  int64_t CfaOffset{-8};
  uint16_t CfaReg{7};
  std::stack<std::pair<int64_t, uint16_t>> CFIStack;
  /// Our pointer to access SPT info
  const MCInst *Prev{nullptr};
  /// Info about the last frame access
  bool IsValidAccess{false};
  FrameIndexEntry FIE;

  bool decodeFrameAccess(const MCInst &Inst) {
    int32_t SrcImm{0};
    MCPhysReg Reg{0};
    int64_t StackOffset{0};
    bool IsIndexed{false};
    if (!BC.MIA->isStackAccess(
            Inst, FIE.IsLoad, FIE.IsStore, FIE.IsStoreFromReg, Reg, SrcImm,
            FIE.StackPtrReg, StackOffset, FIE.Size, FIE.IsSimple, IsIndexed)) {
      return true;
    }

    if (IsIndexed) {
      DEBUG(dbgs() << "Giving up on indexed memory access in the frame\n");
      return false;
    }

    assert(FIE.Size != 0);

    FIE.RegOrImm = SrcImm;
    if (FIE.IsLoad || FIE.IsStoreFromReg)
      FIE.RegOrImm = Reg;

    if (FIE.StackPtrReg == BC.MIA->getStackPointer() && SPOffset != SPT.EMPTY &&
        SPOffset != SPT.SUPERPOSITION) {
      DEBUG(dbgs() << "Adding access via SP while CFA reg is another one\n");
      FIE.StackOffset = SPOffset + StackOffset;
    } else if (FIE.StackPtrReg == BC.MIA->getFramePointer() &&
               FPOffset != SPT.EMPTY && FPOffset != SPT.SUPERPOSITION) {
      DEBUG(dbgs() << "Adding access via FP while CFA reg is another one\n");
      FIE.StackOffset = FPOffset + StackOffset;
    } else if (FIE.StackPtrReg ==
               BC.MRI->getLLVMRegNum(CfaReg, /*isEH=*/false)) {
      FIE.StackOffset = CfaOffset + StackOffset;
    } else {
      DEBUG(dbgs() << "Found stack access with reg different than cfa reg.\n");
      DEBUG(dbgs() << "\tCurrent CFA reg: " << CfaReg
                   << "\n\tStack access reg: " << FIE.StackPtrReg << "\n");
      DEBUG(dbgs() << "Blame insn: ");
      DEBUG(Inst.dump());
      return false;
    }
    IsValidAccess = true;
    return true;
  }

public:
  FrameAccessAnalysis(const BinaryContext &BC, BinaryFunction &BF)
      : SPT(BC, BF), BC(BC), BF(BF) {
    {
      NamedRegionTimer T1("SPT", "Dataflow", true);
      SPT.run();
    }
  }

  void enterNewBB() { Prev = nullptr; }
  const FrameIndexEntry &getFIE() const { return FIE; }
  int getSPOffset() const { return SPOffset; }
  bool isValidAccess() const { return IsValidAccess; }

  bool doNext(const BinaryBasicBlock &BB, const MCInst &Inst) {
    IsValidAccess = false;
    std::tie(SPOffset, FPOffset) =
        Prev ? *SPT.getStateAt(*Prev) : *SPT.getStateAt(BB);
    Prev = &Inst;
    // Use CFI information to keep track of which register is being used to
    // access the frame
    if (BC.MIA->isCFI(Inst)) {
      const auto *CFI = BF.getCFIFor(Inst);
      switch (CFI->getOperation()) {
      case MCCFIInstruction::OpDefCfa:
        CfaOffset = CFI->getOffset();
      // Fall-through
      case MCCFIInstruction::OpDefCfaRegister:
        CfaReg = CFI->getRegister();
        break;
      case MCCFIInstruction::OpDefCfaOffset:
        CfaOffset = CFI->getOffset();
        break;
      case MCCFIInstruction::OpRememberState:
        CFIStack.push(std::make_pair(CfaOffset, CfaReg));
        break;
      case MCCFIInstruction::OpRestoreState: {
        if (CFIStack.empty()) {
          dbgs() << "Assertion is about to fail: " << BF.getPrintName() << "\n";
        }
        assert(!CFIStack.empty() && "Corrupt CFI stack");
        auto &Elem = CFIStack.top();
        CFIStack.pop();
        CfaOffset = Elem.first;
        CfaReg = Elem.second;
        break;
      }
      case MCCFIInstruction::OpAdjustCfaOffset:
        llvm_unreachable("Unhandled AdjustCfaOffset");
        break;
      default:
        break;
      }
      return true;
    }

    if (BC.MIA->leaksStackAddress(Inst, *BC.MRI, SPT.HasFramePointer)) {
      DEBUG(dbgs() << "Leaked stack address, giving up on this function.\n");
      DEBUG(dbgs() << "Blame insn: ");
      DEBUG(Inst.dump());
      return false;
    }

    return decodeFrameAccess(Inst);
  }
};

} // end anonymous namespace

void FrameAnalysis::addArgAccessesFor(const BinaryContext &BC, MCInst &Inst,
                                      ArgAccesses &&AA) {
  if (auto OldAA = getArgAccessesFor(BC, Inst)) {
    if (OldAA->AssumeEverything)
      return;
    *OldAA = std::move(AA);
    return;
  }
  if (AA.AssumeEverything) {
    // Index 0 in ArgAccessesVector represents an "assumeeverything" entry
    BC.MIA->addAnnotation(BC.Ctx.get(), Inst, "ArgAccessEntry", 0U);
    return;
  }
  BC.MIA->addAnnotation(BC.Ctx.get(), Inst, "ArgAccessEntry",
                        (unsigned)ArgAccessesVector.size());
  ArgAccessesVector.emplace_back(std::move(AA));
}

void FrameAnalysis::addArgInStackAccessFor(const BinaryContext &BC,
                                           MCInst &Inst,
                                           const ArgInStackAccess &Arg) {
  auto AA = getArgAccessesFor(BC, Inst);
  if (!AA) {
    addArgAccessesFor(BC, Inst, ArgAccesses(false));
    AA = getArgAccessesFor(BC, Inst);
    assert(AA && "Object setup failed");
  }
  auto &Set = AA->Set;
  assert(!AA->AssumeEverything && "Adding arg to AssumeEverything set");
  Set.emplace(Arg);
}

void FrameAnalysis::addFIEFor(const BinaryContext &BC, MCInst &Inst,
                              const FrameIndexEntry &FIE) {
  BC.MIA->addAnnotation(BC.Ctx.get(), Inst, "FrameAccessEntry",
                        (unsigned)FIEVector.size());
  FIEVector.emplace_back(FIE);
}

ErrorOr<ArgAccesses &>
FrameAnalysis::getArgAccessesFor(const BinaryContext &BC, const MCInst &Inst) {
  if (auto Idx = BC.MIA->tryGetAnnotationAs<unsigned>(Inst, "ArgAccessEntry")) {
    assert(ArgAccessesVector.size() > *Idx && "Out of bounds");
    return ArgAccessesVector[*Idx];
  }
  return make_error_code(errc::result_out_of_range);
}

ErrorOr<const ArgAccesses &>
FrameAnalysis::getArgAccessesFor(const BinaryContext &BC,
                                 const MCInst &Inst) const {
  if (auto Idx = BC.MIA->tryGetAnnotationAs<unsigned>(Inst, "ArgAccessEntry")) {
    assert(ArgAccessesVector.size() > *Idx && "Out of bounds");
    return ArgAccessesVector[*Idx];
  }
  return make_error_code(errc::result_out_of_range);
}

ErrorOr<const FrameIndexEntry &>
FrameAnalysis::getFIEFor(const BinaryContext &BC, const MCInst &Inst) const {
  if (auto Idx =
          BC.MIA->tryGetAnnotationAs<unsigned>(Inst, "FrameAccessEntry")) {
    assert(FIEVector.size() > *Idx && "Out of bounds");
    return FIEVector[*Idx];
  }
  return make_error_code(errc::result_out_of_range);
}

void FrameAnalysis::getInstClobberList(const BinaryContext &BC,
                                       const MCInst &Inst,
                                       BitVector &KillSet) const {
  if (!BC.MIA->isCall(Inst)) {
    BC.MIA->getClobberedRegs(Inst, KillSet, *BC.MRI);
    return;
  }

  const auto *TargetSymbol = BC.MIA->getTargetSymbol(Inst);
  // If indirect call, kill set should have all elements
  if (TargetSymbol == nullptr) {
    KillSet.set(0, KillSet.size());
    return;
  }

  const auto *Function = BC.getFunctionForSymbol(TargetSymbol);
  if (Function == nullptr) {
    // Call to a function without a BinaryFunction object.
    // This should be a call to a PLT entry, and since it is a trampoline to
    // a DSO, we can't really know the code in advance. Conservatively assume
    // everything is clobbered.
    KillSet.set(0, KillSet.size());
    return;
  }
  auto BV = RegsKilledMap.find(Function);
  if (BV != RegsKilledMap.end()) {
    KillSet |= BV->second;
    return;
  }
  // Ignore calls to function whose clobber list wasn't yet calculated. This
  // instruction will be evaluated again once we have info for the callee.
  return;
}

BitVector FrameAnalysis::getFunctionClobberList(const BinaryContext &BC,
                                                const BinaryFunction *Func) {
  BitVector RegsKilled = BitVector(BC.MRI->getNumRegs(), false);

  if (!Func->isSimple() || !Func->hasCFG()) {
    RegsKilled.set(0, RegsKilled.size());
    return RegsKilled;
  }

  for (const auto &BB : *Func) {
    for (const auto &Inst : BB) {
      getInstClobberList(BC, Inst, RegsKilled);
    }
  }

  return RegsKilled;
}

void FrameAnalysis::buildClobberMap(const BinaryContext &BC) {
  std::queue<BinaryFunction *> Queue;
  std::set<BinaryFunction *> InQueue;

  for (auto *Func : TopologicalCGOrder) {
    Queue.push(Func);
    InQueue.insert(Func);
  }

  while (!Queue.empty()) {
    auto *Func = Queue.front();
    Queue.pop();
    InQueue.erase(Func);

    BitVector RegsKilled = getFunctionClobberList(BC, Func);
    bool ArgsUpdated = ClobberAnalysisOnly ? false : computeArgsAccessed(BC, *Func);
    bool RegsUpdated = false;

    if (RegsKilledMap.find(Func) == RegsKilledMap.end()) {
      RegsKilledMap[Func] = std::move(RegsKilled);
    } else {
      RegsUpdated = RegsKilledMap[Func] != RegsKilled;
      if (RegsUpdated)
        RegsKilledMap[Func] = std::move(RegsKilled);
    }

    if (RegsUpdated || ArgsUpdated) {
      for (auto Caller : Cg.predecessors(Cg.getNodeId(Func))) {
        BinaryFunction *CallerFunc = Cg.nodeIdToFunc(Caller);
        if (!InQueue.count(CallerFunc)) {
          InQueue.insert(CallerFunc);
          Queue.push(CallerFunc);
        }
      }
    }
  }

  if (opts::Verbosity == 0) {
#ifndef NDEBUG
    if (!DebugFlag || !isCurrentDebugType("fa"))
      return;
#else
    return;
#endif
  }

  // This loop is for computing statistics only
  for (auto *Func : TopologicalCGOrder) {
    auto Iter = RegsKilledMap.find(Func);
    assert(Iter != RegsKilledMap.end() &&
           "Failed to compute all clobbers list");
    if (Iter->second.all()) {
      auto Count = Func->getExecutionCount();
      if (Count != BinaryFunction::COUNT_NO_PROFILE)
        CountFunctionsAllClobber += Count;
      ++NumFunctionsAllClobber;
    }
    DEBUG_WITH_TYPE("fa",
      dbgs() << "Killed regs set for func: " << Func->getPrintName() << "\n";
      const BitVector &RegsKilled = Iter->second;
      int RegIdx = RegsKilled.find_first();
      while (RegIdx != -1) {
        dbgs() << "\tREG" << RegIdx;
        RegIdx = RegsKilled.find_next(RegIdx);
      };
      dbgs() << "\n";
    );
  }
}

bool FrameAnalysis::updateArgsTouchedFor(const BinaryContext &BC,
                                         const BinaryFunction &BF, MCInst &Inst,
                                         int CurOffset) {
  if (!BC.MIA->isCall(Inst))
    return false;

  std::set<int64_t> Res;
  const auto *TargetSymbol = BC.MIA->getTargetSymbol(Inst);
  // If indirect call, we conservatively assume it accesses all stack positions
  if (TargetSymbol == nullptr) {
    addArgAccessesFor(BC, Inst, ArgAccesses(/*AssumeEverything=*/true));
    bool Updated{false};
    if (!FunctionsRequireAlignment.count(&BF)) {
      Updated = true;
      FunctionsRequireAlignment.insert(&BF);
    }
    return Updated;
  }

  const auto *Function = BC.getFunctionForSymbol(TargetSymbol);
  // Call to a function without a BinaryFunction object. Conservatively assume
  // it accesses all stack positions
  if (Function == nullptr) {
    addArgAccessesFor(BC, Inst, ArgAccesses(/*AssumeEverything=*/true));
    bool Updated{false};
    if (!FunctionsRequireAlignment.count(&BF)) {
      Updated = true;
      FunctionsRequireAlignment.insert(&BF);
    }
    return Updated;
  }

  auto Iter = ArgsTouchedMap.find(Function);
  if (Iter == ArgsTouchedMap.end())
    return false;

  bool Changed = false;
  if (BC.MIA->isTailCall(Inst)) {
    // Ignore checking CurOffset because we can't always reliably determine the
    // offset specially after an epilogue, where tailcalls happen. It should be
    // -8.
    for (auto Elem : Iter->second) {
      if (ArgsTouchedMap[&BF].find(Elem) == ArgsTouchedMap[&BF].end()) {
        ArgsTouchedMap[&BF].emplace(Elem);
        Changed = true;
      }
    }
  }
  if (FunctionsRequireAlignment.count(Function) &&
      !FunctionsRequireAlignment.count(&BF)) {
    Changed = true;
    FunctionsRequireAlignment.insert(&BF);
  }

  if (CurOffset == StackPointerTracking::EMPTY ||
      CurOffset == StackPointerTracking::SUPERPOSITION) {
    addArgAccessesFor(BC, Inst, ArgAccesses(/*AssumeEverything=*/true));
    return Changed;
  }

  for (auto Elem : Iter->second) {
    if (Elem.first == -1) {
      addArgAccessesFor(BC, Inst, ArgAccesses(/*AssumeEverything=*/true));
      break;
    }
    DEBUG(dbgs() << "Added arg in stack access annotation "
                 << CurOffset + Elem.first << "\n");
    addArgInStackAccessFor(
        BC, Inst,
        ArgInStackAccess{/*StackOffset=*/CurOffset + Elem.first,
                         /*Size=*/Elem.second});
  }
  return Changed;
}

bool FrameAnalysis::computeArgsAccessed(const BinaryContext &BC,
                                        BinaryFunction &BF) {
  if (!BF.isSimple() || !BF.hasCFG()) {
    DEBUG(dbgs() << "Treating " << BF.getPrintName() << " conservatively.\n");
    bool Updated = false;
    ArgsTouchedMap[&BF].emplace(std::make_pair(-1, 0));
    if (!FunctionsRequireAlignment.count(&BF)) {
      Updated = true;
      FunctionsRequireAlignment.insert(&BF);
    }
    return Updated;
  }

  bool UpdatedArgsTouched = false;
  FrameAccessAnalysis FAA(BC, BF);

  for (auto BB : BF.layout()) {
    FAA.enterNewBB();

    for (auto &Inst : *BB) {
      if (!FAA.doNext(*BB, Inst)) {
        ArgsTouchedMap[&BF].emplace(std::make_pair(-1, 0));
        break;
      }

      // Check for calls -- attach stack accessing info to them regarding their
      // target
      if (updateArgsTouchedFor(BC, BF, Inst, FAA.getSPOffset()))
        UpdatedArgsTouched = true;

      // Check for stack accesses that affect callers
      if (!FAA.isValidAccess())
        continue;

      const FrameIndexEntry &FIE = FAA.getFIE();
      if (FIE.StackOffset < 0)
        continue;
      if (ArgsTouchedMap[&BF].find(std::make_pair(FIE.StackOffset, FIE.Size)) !=
          ArgsTouchedMap[&BF].end())
        continue;

      // Record accesses to the previous stack frame
      ArgsTouchedMap[&BF].emplace(std::make_pair(FIE.StackOffset, FIE.Size));
      UpdatedArgsTouched = true;
      DEBUG({
        dbgs() << "Arg access offset " << FIE.StackOffset << " added to:\n";
        BC.printInstruction(dbgs(), Inst, 0, &BF, true);
      });
    }
  }
  if (FunctionsRequireAlignment.count(&BF))
    return UpdatedArgsTouched;

  bool UpdatedAlignedStatus = false;
  for (auto &BB : BF) {
    if (UpdatedAlignedStatus)
      break;
    for (auto &Inst : BB) {
      if (BC.MIA->requiresAlignedAddress(Inst)) {
        if (!FunctionsRequireAlignment.count(&BF)) {
          UpdatedAlignedStatus = true;
          FunctionsRequireAlignment.insert(&BF);
          break;
        }
      }
    }
  }
  return UpdatedArgsTouched || UpdatedAlignedStatus;
}

bool FrameAnalysis::restoreFrameIndex(const BinaryContext &BC,
                                      BinaryFunction &BF) {
  FrameAccessAnalysis FAA(BC, BF);

  DEBUG(dbgs() << "Restoring frame indices for \"" << BF.getPrintName()
               << "\"\n");
  for (auto BB : BF.layout()) {
    DEBUG(dbgs() << "\tNow at BB " << BB->getName() << "\n");
    FAA.enterNewBB();

    for (auto &Inst : *BB) {
      if (!FAA.doNext(*BB, Inst))
        return false;
      DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        dbgs() << "\t\t\tSP offset is " << FAA.getSPOffset() << "\n";
      });

      if (!FAA.isValidAccess())
        continue;

      const FrameIndexEntry &FIE = FAA.getFIE();

      addFIEFor(BC, Inst, FIE);
      DEBUG({
        dbgs() << "Frame index annotation " << FIE << " added to:\n";
        BC.printInstruction(dbgs(), Inst, 0, &BF, true);
      });
    }
  }
  return true;
}

void FrameAnalysis::cleanAnnotations(const BinaryContext &BC,
                                     std::map<uint64_t, BinaryFunction> &BFs) {
  for (auto &I : BFs) {
    for (auto &BB : I.second) {
      for (auto &Inst : BB) {
        BC.MIA->removeAnnotation(Inst, "ArgAccessEntry");
        BC.MIA->removeAnnotation(Inst, "FrameAccessEntry");
      }
    }
  }
}

void FrameAnalysis::runOnFunctions(BinaryContext &BC,
                                   std::map<uint64_t, BinaryFunction> &BFs,
                                   std::set<uint64_t> &) {
  {
    NamedRegionTimer T1("Callgraph construction", "FOP breakdown", true);
    Cg = buildCallGraph(BC, BFs);
  }
  {
    NamedRegionTimer T1("build cg traversal order", "FOP breakdown", true);
    TopologicalCGOrder = Cg.buildTraversalOrder();
  }
  {
    NamedRegionTimer T1("build clobber map", "FOP breakdown", true);
    buildClobberMap(BC);
  }

  if (ClobberAnalysisOnly)
    return;

  for (auto &I : BFs) {
    auto Count = I.second.getExecutionCount();
    if (Count != BinaryFunction::COUNT_NO_PROFILE)
      CountDenominator += Count;

    // "shouldOptimize" for passes that run after finalize
    if (!(I.second.isSimple() && I.second.hasCFG() &&
          opts::shouldProcess(I.second) && (I.second.getSize() > 0)) ||
        !opts::shouldFrameOptimize(I.second)) {
      ++NumFunctionsNotOptimized;
      if (Count != BinaryFunction::COUNT_NO_PROFILE)
        CountFunctionsNotOptimized += Count;
      continue;
    }

    {
      NamedRegionTimer T1("restore frame index", "FOP breakdown", true);
      if (!restoreFrameIndex(BC, I.second)) {
        ++NumFunctionsFailedRestoreFI;
        auto Count = I.second.getExecutionCount();
        if (Count != BinaryFunction::COUNT_NO_PROFILE)
          CountFunctionsFailedRestoreFI += Count;
        continue;
      }
    }
    AnalyzedFunctions.insert(&I.second);
  }
}

void FrameAnalysis::printStats() {
  outs() << "BOLT-INFO FRAME ANALYSIS: Number of functions conservatively "
            "treated as clobbering all registers: "
         << NumFunctionsAllClobber
         << format(" (%.1lf%% dyn cov)\n",
                   (100.0 * CountFunctionsAllClobber / CountDenominator))
         << "BOLT-INFO FRAME ANALYSIS: " << NumFunctionsNotOptimized
         << " function(s) "
         << format("(%.1lf%% dyn cov)",
                   (100.0 * CountFunctionsNotOptimized / CountDenominator))
         << " were not optimized.\n"
         << "BOLT-INFO FRAME ANALYSIS: " << NumFunctionsFailedRestoreFI
         << " function(s) "
         << format("(%.1lf%% dyn cov)",
                   (100.0 * CountFunctionsFailedRestoreFI / CountDenominator))
         << " could not have its frame indices restored.\n";
}

} // namespace bolt
} // namespace llvm
