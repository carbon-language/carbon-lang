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

#include "BinaryPasses.h"
#include "llvm/Support/Options.h"

#define DEBUG_TYPE "bolt"

namespace opts {

extern llvm::cl::opt<bool> PrintAll;
extern llvm::cl::opt<bool> PrintReordered;
extern llvm::cl::opt<bool> PrintEHRanges;
extern llvm::cl::opt<bool> PrintUCE;
extern llvm::cl::opt<llvm::bolt::BinaryFunction::SplittingType> SplitFunctions;
extern bool shouldProcess(const llvm::bolt::BinaryFunction &Function);

static llvm::cl::opt<int>
OptimizeIndirectBranchesThreshold(
    "optimize-indirect-branches-threshold",
    llvm::cl::desc("threshold for optimizing a frequently taken indirect call"),
    llvm::cl::init(90),
    llvm::cl::Optional);

static llvm::cl::opt<int>
OptimizeIndirectBranchesTopN(
    "optimize-indirect-branches-topn",
    llvm::cl::desc("number of targets to consider when doing indirect "
                   "branch optimization"),
    llvm::cl::init(2),
    llvm::cl::Optional);

static llvm::cl::opt<llvm::bolt::BinaryFunction::LayoutType>
ReorderBlocks(
    "reorder-blocks",
    llvm::cl::desc("change layout of basic blocks in a function"),
    llvm::cl::init(llvm::bolt::BinaryFunction::LT_NONE),
    llvm::cl::values(clEnumValN(llvm::bolt::BinaryFunction::LT_NONE,
                                "none",
                                "do not reorder basic blocks"),
                     clEnumValN(llvm::bolt::BinaryFunction::LT_REVERSE,
                                "reverse",
                                "layout blocks in reverse order"),
                     clEnumValN(llvm::bolt::BinaryFunction::LT_OPTIMIZE,
                                "normal",
                                "perform optimal layout based on profile"),
                     clEnumValN(llvm::bolt::BinaryFunction::LT_OPTIMIZE_BRANCH,
                                "branch-predictor",
                                "perform optimal layout prioritizing branch "
                                "predictions"),
                     clEnumValN(llvm::bolt::BinaryFunction::LT_OPTIMIZE_CACHE,
                                "cache",
                                "perform optimal layout prioritizing I-cache "
                                "behavior"),
                     clEnumValEnd));

} // namespace opts

namespace llvm {
namespace bolt {

void OptimizeBodylessFunctions::analyze(
    BinaryFunction &BF,
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs) {
  if (BF.size() != 1 || (*BF.begin()).size() == 0)
    return;

  auto &BB = *BF.begin();
  const auto &FirstInst = *BB.begin();

  if (!BC.MIA->isTailCall(FirstInst))
    return;

  auto &Op1 = FirstInst.getOperand(0);
  if (!Op1.isExpr())
    return;

  if (auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr())) {
    auto AddressIt = BC.GlobalSymbols.find(Expr->getSymbol().getName());
    if (AddressIt != BC.GlobalSymbols.end()) {
      auto CalleeIt = BFs.find(AddressIt->second);
      if (CalleeIt != BFs.end()) {
        assert(Expr->getSymbol().getName() == CalleeIt->second.getName());
        EquivalentCallTarget[BF.getName()] = &CalleeIt->second;
      }
    }
  }
}

void OptimizeBodylessFunctions::optimizeCalls(BinaryFunction &BF,
                                              BinaryContext &BC) {
  for (auto BBIt = BF.begin(), BBEnd = BF.end(); BBIt != BBEnd; ++BBIt) {
    for (auto InstIt = (*BBIt).begin(), InstEnd = (*BBIt).end();
        InstIt != InstEnd; ++InstIt) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst)) {
        auto &Op1 = Inst.getOperand(0);
        if (Op1.isExpr()) {
          if (auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr())) {
            auto OriginalTarget = Expr->getSymbol().getName();
            auto Target = OriginalTarget;
            // Iteratively update target since we could have f1() calling f2()
            // calling f3() calling f4() and we want to output f1() directly
            // calling f4().
            while (EquivalentCallTarget.count(Target)) {
              Target = EquivalentCallTarget.find(Target)->second->getName();
            }
            if (Target != OriginalTarget) {
              DEBUG(errs() << "BOLT-DEBUG: Optimizing " << BF.getName()
                           << ": replacing call to "
                           << OriginalTarget
                           << " by call to " << Target << "\n");
              Inst.clear();
              Inst.addOperand(MCOperand::createExpr(
                    MCSymbolRefExpr::create(
                      BC.Ctx->getOrCreateSymbol(Target), *BC.Ctx)));
            }
          }
        }
      }
    }
  }
}

void OptimizeBodylessFunctions::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (Function.isSimple() && opts::shouldProcess(Function)) {
      analyze(Function, BC, BFs);
    }
  }
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (Function.isSimple() && opts::shouldProcess(Function)) {
      optimizeCalls(Function, BC);
    }
  }
}

void InlineSmallFunctions::findInliningCandidates(
    BinaryContext &BC,
    const std::map<uint64_t, BinaryFunction> &BFs) {
  for (const auto &BFIt : BFs) {
    const auto &Function = BFIt.second;
    if (!Function.isSimple() ||
        !opts::shouldProcess(Function) ||
        Function.size() != 1)
      continue;
    auto &BB = *Function.begin();
    const auto &LastInstruction = *BB.rbegin();
    // Check if the function is small enough and doesn't do a tail call.
    // The size we use includes pseudo-instructions but here they shouldn't
    // matter. So some opportunities may be missed because of this.
    if (BB.size() > 0 &&
        BB.size() <= kMaxInstructions &&
        BC.MIA->isReturn(LastInstruction) &&
        !BC.MIA->isTailCall(LastInstruction)) {
      InliningCandidates.insert(Function.getName());
    }
  }

  DEBUG(errs() << "BOLT-DEBUG: " << InliningCandidates.size()
               << " inlineable functions.\n");
}

namespace {

/// Returns whether a function creates a stack frame for itself or not.
/// If so, we need to manipulate the stack pointer when calling this function.
/// Since we're only inlining very small functions, we return false for now, but
/// we could for instance check if the function starts with 'push ebp'.
/// TODO generalize this.
bool createsStackFrame(const BinaryBasicBlock &) {
  return false;
}

} // namespace

void InlineSmallFunctions::inlineCall(
    BinaryContext &BC,
    BinaryBasicBlock &BB,
    MCInst *CallInst,
    const BinaryBasicBlock &InlinedFunctionBB) {
  assert(BC.MIA->isCall(*CallInst) && "Can only inline a call.");
  assert(BC.MIA->isReturn(*InlinedFunctionBB.rbegin()) &&
         "Inlined function should end with a return.");

  std::vector<MCInst> InlinedInstance;

  bool ShouldAdjustStack = createsStackFrame(InlinedFunctionBB);

  // Move stack like 'call' would if needed.
  if (ShouldAdjustStack) {
    MCInst StackInc;
    BC.MIA->createStackPointerIncrement(StackInc);
    InlinedInstance.push_back(StackInc);
  }

  for (auto Instruction : InlinedFunctionBB) {
    if (BC.MIA->isReturn(Instruction)) {
      break;
    }
    if (!BC.MIA->isEHLabel(Instruction) &&
        !BC.MIA->isCFI(Instruction)) {
      InlinedInstance.push_back(Instruction);
    }
  }

  // Move stack pointer like 'ret' would.
  if (ShouldAdjustStack) {
    MCInst StackDec;
    BC.MIA->createStackPointerDecrement(StackDec);
    InlinedInstance.push_back(StackDec);
  }

  BB.replaceInstruction(CallInst, InlinedInstance);
}

void InlineSmallFunctions::inlineCallsInFunction(
    BinaryContext &BC,
    BinaryFunction &Function) {
  std::vector<BinaryBasicBlock *> Blocks(Function.layout().begin(),
                                         Function.layout().end());
  std::sort(Blocks.begin(), Blocks.end(),
      [](const BinaryBasicBlock *BB1, const BinaryBasicBlock *BB2) {
        return BB1->getExecutionCount() > BB2->getExecutionCount();
      });
  uint32_t ExtraSize = 0;

  for (auto BB : Blocks) {
    for (auto InstIt = BB->begin(), End = BB->end(); InstIt != End; ++InstIt) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst)) {
        totalDynamicCalls += BB->getExecutionCount();
      }
    }
  }

  for (auto BB : Blocks) {
    if (BB->isCold())
      continue;

    for (auto InstIt = BB->begin(), End = BB->end(); InstIt != End; ) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst) &&
          !BC.MIA->isTailCall(Inst) &&
          Inst.size() == 1 &&
          Inst.getOperand(0).isExpr()) {
        auto Target = dyn_cast<MCSymbolRefExpr>(
            Inst.getOperand(0).getExpr());
        assert(Target && "Not MCSymbolRefExpr");
        auto FunctionIt = FunctionByName.find(Target->getSymbol().getName());
        if (FunctionIt != FunctionByName.end()) {
          auto &TargetFunction = *FunctionIt->second;
          bool CallToInlineableFunction =
            InliningCandidates.count(TargetFunction.getName());

          totalInlineableCalls +=
            CallToInlineableFunction * BB->getExecutionCount();

          if (CallToInlineableFunction &&
              TargetFunction.getSize() + ExtraSize
              + Function.estimateHotSize() < Function.getMaxSize()) {
            auto NextInstIt = std::next(InstIt);
            inlineCall(BC, *BB, &Inst, *TargetFunction.begin());
            DEBUG(errs() << "BOLT-DEBUG: Inlining call to "
                         << TargetFunction.getName() << " in "
                         << Function.getName() << "\n");
            InstIt = NextInstIt;
            ExtraSize += TargetFunction.getSize();
            inlinedDynamicCalls += BB->getExecutionCount();
            continue;
          }
        }
      }

      ++InstIt;
    }
  }
}

void InlineSmallFunctions::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &) {
  for (auto &It : BFs) {
    FunctionByName[It.second.getName()] = &It.second;
  }
  findInliningCandidates(BC, BFs);
  uint32_t ConsideredFunctions = 0;
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (!Function.isSimple() || !opts::shouldProcess(Function))
      continue;
    if (ConsideredFunctions == kMaxFunctions)
      break;
    inlineCallsInFunction(BC, Function);
    ++ConsideredFunctions;
  }
  DEBUG(errs() << "BOLT-DEBUG: Inlined " << inlinedDynamicCalls << " of "
               << totalDynamicCalls << " function calls in the profile.\n");
  DEBUG(errs() << "BOLT-DEBUG: Inlined calls represent "
               << (100.0 * inlinedDynamicCalls / totalInlineableCalls)
               << "% of all inlineable calls in the profile.\n");
}

void EliminateUnreachableBlocks::runOnFunction(BinaryFunction& Function) {
  if (!Function.isSimple() || !opts::shouldProcess(Function)) return;

  // FIXME: this wouldn't work with C++ exceptions until we implement
  //        support for those as there will be "invisible" edges
  //        in the graph.
  if (Function.layout_size() > 0) {
    if (NagUser) {
      outs()
        << "BOLT-WARNING: Using -eliminate-unreachable is experimental and "
        "unsafe for exceptions\n";
      NagUser = false;
    }

    if (Function.hasEHRanges()) return;

    std::stack<BinaryBasicBlock*> Stack;
    std::map<BinaryBasicBlock *, bool> Reachable;
    BinaryBasicBlock *Entry = *Function.layout_begin();
    Stack.push(Entry);
    Reachable[Entry] = true;
    // Determine reachable BBs from the entry point
    while (!Stack.empty()) {
      auto BB = Stack.top();
      Stack.pop();
      for (auto Succ : BB->successors()) {
        if (Reachable[Succ])
          continue;
        Reachable[Succ] = true;
        Stack.push(Succ);
      }
    }

    auto Count = Function.eraseDeadBBs(Reachable);
    if (Count) {
      DEBUG(dbgs() << "BOLT: Removed " << Count
            << " dead basic block(s) in function "
            << Function.getName() << '\n');
    }

    if (opts::PrintAll || opts::PrintUCE)
      Function.print(errs(), "after unreachable code elimination", true);
  }
}

void EliminateUnreachableBlocks::runOnFunctions(
  BinaryContext&,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
    runOnFunction(It.second);
  }
}

void ReorderBasicBlocks::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &LargeFunctions
) {
  for (auto &It : BFs) {
    auto &Function = It.second;

    if (!Function.isSimple())
      continue;

    if (!opts::shouldProcess(Function))
      continue;

    if (opts::ReorderBlocks != BinaryFunction::LT_NONE) {
      bool ShouldSplit =
        (opts::SplitFunctions == BinaryFunction::ST_ALL) ||
        (opts::SplitFunctions == BinaryFunction::ST_EH &&
         Function.hasEHRanges()) ||
        (LargeFunctions.find(It.first) != LargeFunctions.end());
      Function.modifyLayout(opts::ReorderBlocks, ShouldSplit);
      if (opts::PrintAll || opts::PrintReordered)
        Function.print(errs(), "after reordering blocks", true);
    }
  }
}

void FixupFunctions::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
    auto &Function = It.second;

    if (!Function.isSimple())
      continue;

    if (!opts::shouldProcess(Function))
      continue;

    // Fix the CFI state.
    if (!Function.fixCFIState()) {
      errs() << "BOLT-WARNING: unable to fix CFI state for function "
             << Function.getName() << ". Skipping.\n";
      Function.setSimple(false);
      continue;
    }

    // Update exception handling information.
    Function.updateEHRanges();
    if (opts::PrintAll || opts::PrintEHRanges)
      Function.print(errs(), "after updating EH ranges", true);
  }
}

bool SimplifyConditionalTailCalls::fixTailCalls(BinaryContext &BC,
                                                BinaryFunction &BF) {
  if (BF.layout_size() == 0)
    return false;

  auto &MIA = BC.MIA;
  uint64_t NumLocalTailCalls = 0;
  uint64_t NumLocalPatchedTailCalls = 0;

  for (auto* BB : BF.layout()) {
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;

    // Determine the control flow at the end of each basic block
    if (!BB->analyzeBranch(*MIA, TBB, FBB, CondBranch, UncondBranch)) {
      continue;
    }

    // TODO: do we need to test for other branch patterns?

    // For this particular case, the first basic block ends with
    // a conditional branch and has two successors, one fall-through
    // and one for when the condition is true.
    // The target of the conditional is a basic block with a single
    // unconditional branch (i.e. tail call) to another function.
    // We don't care about the contents of the fall-through block.
    // Note: this code makes the assumption that the fall-through
    // block is the last successor.
    if (CondBranch && !UncondBranch && BB->succ_size() == 2) {
      // Find conditional branch target assuming the fall-through is
      // always the last successor.
      auto *CondTargetBB = *BB->succ_begin();

      // Does the BB contain a single instruction?
      if (CondTargetBB->size() - CondTargetBB->getNumPseudos() == 1) {
        // Check to see if the sole instruction is a tail call.
        auto const &Instr = *CondTargetBB->begin();

        if (MIA->isTailCall(Instr)) {
          ++NumTailCallCandidates;
          ++NumLocalTailCalls;

          auto const &TailTargetSymExpr =
            cast<MCSymbolRefExpr>(Instr.getOperand(0).getExpr());
          auto const &TailTarget = TailTargetSymExpr->getSymbol();

          // Lookup the address for the current function and
          // the tail call target.
          auto const FnAddress = BC.GlobalSymbols.find(BF.getName());
          auto const TailAddress = BC.GlobalSymbols.find(TailTarget.getName());
          if (FnAddress == BC.GlobalSymbols.end() ||
              TailAddress == BC.GlobalSymbols.end()) {
            continue;
          }

          // Check to make sure we would be doing a forward jump.
          // This assumes the address range of the current BB and the
          // tail call target address don't overlap.
          if (FnAddress->second < TailAddress->second) {
            ++NumTailCallsPatched;
            ++NumLocalPatchedTailCalls;

            // Is the original jump forward or backward?
            const bool isForward =
              TailAddress->second > FnAddress->second + BB->getOffset();

            if (isForward) ++NumOrigForwardBranches;

            // Patch the new target address into the conditional branch.
            CondBranch->getOperand(0).setExpr(TailTargetSymExpr);
            // Remove the unused successor which may be eliminated later
            // if there are no other users.
            BB->removeSuccessor(CondTargetBB);
            DEBUG(dbgs() << "patched " << (isForward ? "(fwd)" : "(back)")
                  << " tail call in " << BF.getName() << ".\n";);
          }
        }
      }
    }
  }

  DEBUG(dbgs() << "BOLT: patched " << NumLocalPatchedTailCalls
        << " tail calls (" << NumOrigForwardBranches << " forward)"
        << " from a total of " << NumLocalTailCalls
        << " in function " << BF.getName() << "\n";);

  return NumLocalPatchedTailCalls > 0;
}

void SimplifyConditionalTailCalls::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
    auto &Function = It.second;

    if (!Function.isSimple())
      continue;

    // Fix tail calls to reduce branch mispredictions.
    if (fixTailCalls(BC, Function)) {
      if (opts::PrintAll || opts::PrintReordered) {
        Function.print(errs(), "after tail call patching", true);
      }
    }
  }

  outs() << "BOLT: patched " << NumTailCallsPatched
         << " tail calls (" << NumOrigForwardBranches << " forward)"
         << " from a total of " << NumTailCallCandidates << "\n";
}

namespace {

template <typename S>
void printInstruction(S& OS, BinaryContext& BC, const MCInst &Instruction, bool printMCInst = false) {
  if (!BC.MIA->isUnsupported(Instruction)) {
    BC.InstPrinter->printInst(&Instruction, OS, "", *BC.STI);
  } else {
    OS << "unsupported (probably jmpr)";
  }
  OS << "\n";
  if (printMCInst) {
    Instruction.dump_pretty(OS, BC.InstPrinter.get());
    OS << "\n";
  }
}

template <typename Itr>
uint64_t computeCodeSize(BinaryContext& BC, Itr beg, Itr end) {
  uint64_t size = 0;
  while (beg != end) {
    // Calculate the size of the instruction.
    // Note: this is imprecise since happening prior to relaxation.
    SmallString<256> Code;
    SmallVector<MCFixup, 4> Fixups;
    raw_svector_ostream VecOS(Code);
    printInstruction(dbgs(), BC, *beg, false);
    BC.MCE->encodeInstruction(*beg++, VecOS, Fixups, *BC.STI);
    size += Code.size();
  }
  return size;
}

}

void OptimizeIndirectBranches::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &LargeFunctions
) {
  uint64_t TotalBranches = 0;
  uint64_t TotalIndirectCalls = 0;
  uint64_t TotalIndirectCallsites = 0;
  uint64_t TotalIndirectCandidateCalls = 0;
  for (auto &BFIt : BFs) {
    auto &Function = BFIt.second;

    if (!Function.isSimple() || !opts::shouldProcess(Function))
      continue;

    auto BranchDataOrErr = BC.DR.getFuncBranchData(Function.getName());
    if (std::error_code EC = BranchDataOrErr.getError()) {
      DEBUG(dbgs() << "no branch data found for \""
                   << Function.getName() << "\"\n");
      continue;
    }
    const FuncBranchData &BranchData = BranchDataOrErr.get();

    // Note: this is not just counting calls.
    TotalBranches += BranchData.ExecutionCount;

    uint64_t Total = 0;
    for (auto &nlib : Function.nlibs()) {
      auto Branches = BranchData.getBranchRange(nlib);
      for (auto &BInfo : Branches) {
        Total += BInfo.Branches;
      }
      std::vector<BranchInfo> targets;
      for (auto &BInfo : Branches) {
        targets.push_back(BInfo);
      }

      std::sort(targets.begin(), targets.end(),
                [](const BranchInfo& a, const BranchInfo& b) {
                  return a.Branches > b.Branches;
                });

      if (!targets.empty()) {
        uint64_t TopNBranches = 0;

        const int NumTargets = std::distance(targets.begin(), targets.end());
        const int N = std::min(int(opts::OptimizeIndirectBranchesTopN),
                               NumTargets);

        for (int i = 0; i < N; ++i) {
          TopNBranches += targets[i].Branches;
        }

        const double TopNFrequency = 100.0 * TopNBranches / Total;

        if (TopNFrequency >= opts::OptimizeIndirectBranchesThreshold) {
          double Threshold = double(opts::OptimizeIndirectBranchesThreshold);
          bool Separator = false;

          dbgs() << "BOLT: candidate branch info: "
                 << Function.getName() << " @ " << nlib
                 << " -> ";
          
          for (int i = 0; i < N && Threshold > 0; i++) {
            const auto Frequency = 100.0 * targets[i].Branches / Total;
            if (Separator) {
              dbgs() << " | ";
            }
            Separator = true;
            dbgs() << targets[i].To.Name
                   << ", count = " << targets[i].Branches
                   << ", mispreds = " << targets[i].Mispreds
                   << ", freq = " << (int)Frequency << "%";
            TotalIndirectCandidateCalls += targets[i].Branches;
            Threshold -= Frequency;
          }
          dbgs() << "\n";

          //assert(!targets[0].From.IsSymbol);
          auto IndCallBlock =
            Function.getBasicBlockContainingOffset(targets[0].From.Offset);

#if 0
          // scan insts for jump (use analyze?)
          const MCSymbol *TBB = nullptr;
          const MCSymbol *FBB = nullptr;
          MCInst *CondBranch = nullptr;
          MCInst *UncondBranch = nullptr;
          bool Found = MIA->analyzeBranch(IndCallBlock->Instructions,
                                          TBB,
                                          FBB,
                                          CondBranch,
                                          UncondBranch);
          assert(Found);
          // how to assert that UncondBranch is the one we want?
          assert(UncondBranch != nullptr);
#else
          MCInst* CallInst = nullptr;
          uint64_t InstOffset{RoundUpToAlignment(IndCallBlock->getOffset(),
                                                 IndCallBlock->getAlignment())};

          size_t CallInstIdx = 0;
          for (auto &Instr : *IndCallBlock) {
            // Calculate the size of the instruction.
            // Note: this is imprecise since happening prior to relaxation.
            SmallString<256> Code;
            SmallVector<MCFixup, 4> Fixups;
            raw_svector_ostream VecOS(Code);
            BC.MCE->encodeInstruction(Instr, VecOS, Fixups, *BC.STI);
            if (InstOffset == targets[0].From.Offset) {
              CallInst = &Instr;
            }
            ++CallInstIdx;
            InstOffset += Code.size();
          }
          assert(CallInst);
#endif

          std::vector<MCSymbol*> Targets;
          for (int i = 0; i < N; ++i) {
            assert(targets[i].To.IsSymbol);
            // Is this right?  lookupSym doesn't always return a result
            auto Symbol = BC.Ctx->getOrCreateSymbol(targets[i].To.Name);
            assert(Symbol);
            Targets.push_back(Symbol);
          }

          MCInst* SourceInst = CallInst; // for now
#if 0
          for (auto &Instr : *IndCallBlock) {
            if (&Instr == CallInst) break;
            if (Instr.getNumOperands() > 0) {
              printInstruction(dbgs(), BC, Instr, true);
              for (unsigned int i = 0; i < Instr.getNumOperands(); ++i) {
                auto &Operand = Instr.getOperand(i);
                dbgs() << "isreg("<< i << ") = " << Operand.isReg() << "\n";
                dbgs() << "isexpr(" << i << ") = " << Operand.isExpr() << "\n";
                SourceInst = &Instr; // WRONG
              }
            }
            if (&Instr == CallInst) break;
          }
          dbgs() << "-----------\n";
          assert(SourceInst);
#endif

          auto ICPcode = BC.MIA->indirectCallPromotion(
            *SourceInst,  // == CallInst for now
            *CallInst,
            Targets,
            BC.Ctx.get());

          if (!ICPcode.empty()) {
            for (auto &entry : ICPcode) {
              auto &Sym = entry.first;
              auto &Insts = entry.second;
              if (Sym) dbgs() << Sym->getName() << ":\n";
              for (auto &Instr : Insts) {
                printInstruction(dbgs(), BC, Instr, false);
              }
            }

            // create new bbs with correct code in each one
            // first
            auto oldSuccRange = IndCallBlock->successors();
            std::vector<BinaryBasicBlock*> oldSucc(oldSuccRange.begin(), oldSuccRange.end());
            BinaryBasicBlock* LastBlock = IndCallBlock;
            BinaryBasicBlock* MergeBlock = nullptr;
            std::vector<BinaryBasicBlock*> newBBs;

            assert(!BC.MIA->isTailCall(*CallInst) || oldSucc.empty());

            // Remove all successors from block doing the indirect call.
            for (auto succ : oldSucc) {
              IndCallBlock->removeSuccessor(succ);
            }
            assert(IndCallBlock->succ_empty());

            dbgs() << "IndCallBlock = " << IndCallBlock << "\n";

            if (ICPcode.back().second.empty()) { // merge block
              // Create BB for merge block following old call

              uint64_t total = 0;
              for (auto &entry : ICPcode) {
                total += computeCodeSize(BC, entry.second.begin(), entry.second.end());
              }

              // adjust all other blocks by total
              for (auto &BB : Function) {
                if (BB.getOffset() > IndCallBlock->getOffset()) {
                  BB.setOffset(BB.getOffset() + total);
                }
              }

              //dbgs() << "total = " << total << "\n";
              //dbgs() << "InstOffset = " << InstOffset << "\n";
              MergeBlock = Function.addBasicBlock(total + InstOffset, ICPcode.back().first);
              newBBs.push_back(MergeBlock);
              for (auto succ : oldSucc) {
                MergeBlock->addSuccessor(succ);
              }
              dbgs() << "MergeBlock = " << MergeBlock << "\n";

              // Move instructions from the tail of the original call block
              // to the merge block.
              std::vector<MCInst> MovedInst;

              while(&IndCallBlock->back() != CallInst) {
                auto &lastInst = IndCallBlock->back();
                MovedInst.push_back(lastInst);
                IndCallBlock->eraseInstruction(&lastInst);
              }
              IndCallBlock->eraseInstruction(CallInst);

              for (auto itr = MovedInst.rbegin(); itr != MovedInst.rend(); ++itr) {
                MergeBlock->addInstruction(*itr);
              }

              ICPcode.pop_back();  // remove merge block
            }

            for (auto &entry : ICPcode) {
              auto &Sym = entry.first;
              auto &Insts = entry.second;
              if (Sym) {
                auto TBB = Function.addBasicBlock(InstOffset, Sym);
                newBBs.push_back(TBB);
                LastBlock->addSuccessor(TBB);
                LastBlock = TBB;
                InstOffset += computeCodeSize(BC, Insts.begin(), Insts.end());
                dbgs() << "TBB = " << TBB << "\n";
              }
              for (auto &Inst : Insts) {
                LastBlock->addInstruction(Inst);
              }
              if (MergeBlock) LastBlock->addSuccessor(MergeBlock);
            }

            // update BBlayout in Function, XXX is this right?
            Function.updateLayout(IndCallBlock, newBBs);
          }
        }
      }

      ++TotalIndirectCallsites;
    }
    TotalIndirectCalls += Total;
  }

  dbgs() << "BOLT: total indirect callsites/candidate calls/calls/branches = "
         << TotalIndirectCallsites << "/"
         << TotalIndirectCandidateCalls << "/"
         << TotalIndirectCalls << "/"
         << TotalBranches << "\n";
}

} // namespace bolt
} // namespace llvm
