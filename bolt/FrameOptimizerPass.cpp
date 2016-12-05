//===--- FrameOptimizerPass.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "FrameOptimizerPass.h"
#include <queue>
#include <unordered_map>

#define DEBUG_TYPE "fop"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> Verbosity;
}

namespace llvm {
namespace bolt {

void FrameOptimizerPass::buildCallGraph(
    BinaryContext &BC, std::map<uint64_t, BinaryFunction> &BFs) {
  for (auto &I : BFs) {
    BinaryFunction &Caller = I.second;

    Functions.emplace(&Caller);

    for (BinaryBasicBlock &BB : Caller) {
      for (MCInst &Inst : BB) {
        if (!BC.MIA->isCall(Inst))
          continue;

        const auto *TargetSymbol = BC.MIA->getTargetSymbol(Inst);
        if (!TargetSymbol) {
          // This is an indirect call, we cannot record a target.
          continue;
        }

        const auto *Function = BC.getFunctionForSymbol(TargetSymbol);
        if (!Function) {
          // Call to a function without a BinaryFunction object.
          continue;
        }
        // Create a new edge in the call graph
        CallGraphEdges[&Caller].emplace_back(Function);
        ReverseCallGraphEdges[Function].emplace_back(&Caller);
      }
    }
  }
}

void FrameOptimizerPass::buildCGTraversalOrder() {
  enum NodeStatus { NEW, VISITING, VISITED };
  std::unordered_map<const BinaryFunction *, NodeStatus> NodeStatus;
  std::stack<const BinaryFunction *> Worklist;

  for (auto *Func : Functions) {
    Worklist.push(Func);
    NodeStatus[Func] = NEW;
  }

  while (!Worklist.empty()) {
    const auto *Func = Worklist.top();
    Worklist.pop();

    if (NodeStatus[Func] == VISITED)
      continue;

    if (NodeStatus[Func] == VISITING) {
      TopologicalCGOrder.push_back(Func);
      NodeStatus[Func] = VISITED;
      continue;
    }

    assert(NodeStatus[Func] == NEW);
    NodeStatus[Func] = VISITING;
    Worklist.push(Func);
    for (const auto *Callee : CallGraphEdges[Func]) {
      if (NodeStatus[Callee] == VISITING || NodeStatus[Callee] == VISITED)
        continue;
      Worklist.push(Callee);
    }
  }
}

void FrameOptimizerPass::getInstClobberList(BinaryContext &BC,
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

BitVector
FrameOptimizerPass::getFunctionClobberList(BinaryContext &BC,
                                           const BinaryFunction *Func) {
  BitVector RegsKilled = BitVector(BC.MRI->getNumRegs(), false);

  if (!Func->isSimple() || !shouldOptimize(*Func)) {
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

void FrameOptimizerPass::buildClobberMap(BinaryContext &BC) {
  std::queue<const BinaryFunction *> Queue;

  for (auto *Func : TopologicalCGOrder) {
    Queue.push(Func);
  }

  while (!Queue.empty()) {
    auto *Func = Queue.front();
    Queue.pop();

    BitVector RegsKilled = getFunctionClobberList(BC, Func);

    if (RegsKilledMap.find(Func) == RegsKilledMap.end()) {
      RegsKilledMap[Func] = std::move(RegsKilled);
      continue;
    }

    if (RegsKilledMap[Func] != RegsKilled) {
      for (auto Caller : ReverseCallGraphEdges[Func]) {
        Queue.push(Caller);
      }
    }
    RegsKilledMap[Func] = std::move(RegsKilled);
  }

  if (opts::Verbosity == 0 && (!DebugFlag || !isCurrentDebugType("fop")))
    return;

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
    if (!DebugFlag || !isCurrentDebugType("fop"))
      continue;
    // DEBUG only
    dbgs() << "Killed regs set for func: " << Func->getPrintName() << "\n";
    const BitVector &RegsKilled = Iter->second;
    int RegIdx = RegsKilled.find_first();
    while (RegIdx != -1) {
      dbgs() << "\tREG" << RegIdx;
      RegIdx = RegsKilled.find_next(RegIdx);
    };
    dbgs() << "\n";
  }
}

bool FrameOptimizerPass::restoreFrameIndex(BinaryContext &BC,
                                           BinaryFunction *BF) {
  // Vars used for storing useful CFI info to give us a hint about how the stack
  // is used in this function
  int64_t CfaOffset{8};
  uint16_t CfaReg{0};
  bool CfaRegLocked{false};
  uint16_t CfaRegLockedVal{0};
  std::stack<std::pair<int64_t, uint16_t>> CFIStack;

  DEBUG(dbgs() << "Restoring frame indices for \"" << BF->getPrintName()
               << "\"\n");

  // TODO: Implement SP tracking and improve this analysis
  for (auto &BB : *BF) {
    for (const auto &Inst : BB) {
      // Use CFI information to keep track of which register is being used to
      // access the frame
      if (BC.MIA->isCFI(Inst)) {
        const auto *CFI = BF->getCFIFor(Inst);
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
        continue;
      }

      if (BC.MIA->leaksStackAddress(Inst, *BC.MRI)) {
        DEBUG(dbgs() << "Leaked stack address, giving up on this function.\n");
        DEBUG(dbgs() << "Blame insn: ");
        DEBUG(Inst.dump());
        return false;
      }

      bool IsLoad = false;
      bool IsStoreFromReg = false;
      bool IsSimple = false;
      int32_t SrcImm{0};
      MCPhysReg Reg{0};
      MCPhysReg StackPtrReg{0};
      int64_t StackOffset{0};
      uint8_t Size{0};
      if (BC.MIA->isStackAccess(Inst, IsLoad, IsStoreFromReg, Reg, SrcImm,
                                StackPtrReg, StackOffset, Size, IsSimple)) {
        assert(Size != 0);
        if (CfaRegLocked && CfaRegLockedVal != CfaReg) {
          DEBUG(dbgs() << "CFA reg changed, giving up on this function.\n");
          return false;
        }
        if (StackPtrReg != BC.MRI->getLLVMRegNum(CfaReg, /*isEH=*/false)) {
          DEBUG(dbgs()
                << "Found stack access with reg different than cfa reg.\n");
          DEBUG(dbgs() << "\tCurrent CFA reg: " << CfaReg
                       << "\n\tStack access reg: " << StackPtrReg << "\n");
          DEBUG(dbgs() << "Blame insn: ");
          DEBUG(Inst.dump());
          return false;
        }
        CfaRegLocked = true;
        CfaRegLockedVal = CfaReg;
        if (IsStoreFromReg || IsLoad)
          SrcImm = Reg;
        FrameIndexMap.emplace(
            &Inst, FrameIndexEntry{IsLoad, IsStoreFromReg, SrcImm,
                                   CfaOffset + StackOffset, Size, IsSimple});

        if (!DebugFlag || !isCurrentDebugType("fop"))
          continue;
        // DEBUG only
        dbgs() << "Frame index annotation added to:\n";
        BC.printInstruction(dbgs(), Inst, 0, BF, true);
        dbgs() << " FrameIndexEntry <IsLoad:" << IsLoad << " StackOffset:";
        if (FrameIndexMap[&Inst].StackOffset < 0)
          dbgs() << "-" << Twine::utohexstr(-FrameIndexMap[&Inst].StackOffset);
        else
          dbgs() << "+" << Twine::utohexstr(FrameIndexMap[&Inst].StackOffset);
        dbgs() << " IsStoreFromReg:" << FrameIndexMap[&Inst].IsStoreFromReg
               << " RegOrImm:" << FrameIndexMap[&Inst].RegOrImm << ">\n";
      }
    }
  }
  return true;
}

void FrameOptimizerPass::removeUnnecessarySpills(BinaryContext &BC,
                                                 BinaryFunction *BF) {
  DEBUG(dbgs() << "Optimizing redundant loads on \"" << BF->getPrintName()
               << "\"\n");

  // Used to size the set of expressions/definitions being tracked by the
  // dataflow analysis
  uint64_t NumInstrs{0};
  // We put every MCInst we want to track (which one representing an
  // expression/def) into a vector because we need to associate them with
  // small numbers. They will be tracked via BitVectors throughout the dataflow
  // analysis.
  std::vector<const MCInst *> Expressions;
  // Maps expressions defs (MCInsts) to its index in the Expressions vector
  std::unordered_map<MCInst *, uint64_t> ExprToIdx;
  // Populate our universe of tracked expressions. We are interested in tracking
  // available stores to frame position at any given point of the program.
  for (auto &BB : *BF) {
    for (auto &Inst : BB) {
      if (FrameIndexMap.count(&Inst) == 1 &&
          FrameIndexMap[&Inst].IsLoad == false &&
          FrameIndexMap[&Inst].IsSimple == true) {
        Expressions.push_back(&Inst);
        ExprToIdx[&Inst] = NumInstrs++;
      }
    }
  }

  // Tracks the set of available exprs at the end of each MCInst in this
  // function
  std::unordered_map<const MCInst *, BitVector> SetAtPoint;
  // Tracks the set of available exprs at basic block start
  std::unordered_map<const BinaryBasicBlock *, BitVector> SetAtBBEntry;

  // Define the function computing the kill set -- whether expression Y, a
  // tracked expression, will be considered to be dead after executing X.
  auto doesXKillsY = [&](const MCInst *X, const MCInst *Y) -> bool {
    // if both are stores, and both store to the same stack location, return
    // true
    if (FrameIndexMap.count(X) == 1 && FrameIndexMap.count(Y) == 1) {
      const FrameIndexEntry &FIEX = FrameIndexMap[X];
      const FrameIndexEntry &FIEY = FrameIndexMap[Y];
      if (FIEX.IsLoad == 0 && FIEY.IsLoad == 0 &&
          FIEX.StackOffset + FIEX.Size > FIEY.StackOffset &&
          FIEX.StackOffset < FIEY.StackOffset + FIEY.Size)
        return true;
    }
    // getClobberedRegs for X and Y. If they intersect, return true
    BitVector XClobbers = BitVector(BC.MRI->getNumRegs(), false);
    BitVector YClobbers = BitVector(BC.MRI->getNumRegs(), false);
    getInstClobberList(BC, *X, XClobbers);
    // If Y is a store to stack, its clobber list is its source reg. This is
    // different than the rest because we want to check if the store source
    // reaches its corresponding load untouched.
    if (FrameIndexMap.count(Y) == 1 && FrameIndexMap[Y].IsLoad == 0 &&
        FrameIndexMap[Y].IsStoreFromReg) {
      YClobbers.set(FrameIndexMap[Y].RegOrImm);
    } else {
      getInstClobberList(BC, *Y, YClobbers);
    }
    XClobbers &= YClobbers;
    return XClobbers.any();
  };

  // Initialize state for all points of the function
  for (auto &BB : *BF) {
    // Entry points start with empty set (Function entry and landing pads). All
    // others start with the full set.
    if (BB.pred_size() == 0)
      SetAtBBEntry[&BB] = BitVector(NumInstrs, false);
    else
      SetAtBBEntry[&BB] = BitVector(NumInstrs, true);
    for (auto &Inst : BB) {
      SetAtPoint[&Inst] = BitVector(NumInstrs, true);
    }
  }
  assert(BF->begin() != BF->end() && "Unexpected empty function");

  std::queue<BinaryBasicBlock *> Worklist;
  // TODO: Pushing this in a DFS ordering will greatly speed up the dataflow
  // performance.
  for (auto &BB : *BF) {
    Worklist.push(&BB);
  }

  // Main dataflow loop
  while (!Worklist.empty()) {
    auto *BB = Worklist.front();
    Worklist.pop();

    DEBUG(dbgs() <<"\tNow at BB " << BB->getName() << "\n");

    // Calculate state at the entry of first instruction in BB
    BitVector &SetAtEntry = SetAtBBEntry[BB];
    for (auto I = BB->pred_begin(), E = BB->pred_end(); I != E; ++I) {
      auto Last = (*I)->rbegin();
      if (Last != (*I)->rend()) {
        SetAtEntry &= SetAtPoint[&*Last];
      } else {
        SetAtEntry &= SetAtBBEntry[*I];
      }
    }
    DEBUG({
      int ExprIdx = SetAtEntry.find_first();
      while (ExprIdx != -1) {
        dbgs() << "\t\tReached by ";
        Expressions[ExprIdx]->dump();
        ExprIdx = SetAtEntry.find_next(ExprIdx);
      }
    });

    // Skip empty
    if (BB->begin() == BB->end())
      continue;

    // Propagate information from first instruction down to the last one
    bool Changed = false;
    BitVector *PrevState = &SetAtEntry;
    const MCInst *LAST = &*BB->rbegin();
    for (auto &Inst : *BB) {
      BitVector CurState = *PrevState;
      DEBUG(dbgs() << "\t\tNow at ");
      DEBUG(Inst.dump());
      // Kill
      int ExprIdx = CurState.find_first();
      while (ExprIdx != -1) {
        assert(Expressions[ExprIdx] != nullptr && "Lost pointers");
        DEBUG(dbgs() << "\t\t\tDoes it kill ");
        DEBUG(Expressions[ExprIdx]->dump());
        if (doesXKillsY(&Inst, Expressions[ExprIdx])) {
          DEBUG(dbgs() << "\t\t\t\tYes\n");
          CurState.reset(ExprIdx);
        }
        ExprIdx = CurState.find_next(ExprIdx);
      };
      // Gen
      if (FrameIndexMap.count(&Inst) == 1 &&
          FrameIndexMap[&Inst].IsLoad == false &&
          FrameIndexMap[&Inst].IsSimple == true)
        CurState.set(ExprToIdx[&Inst]);

      if (SetAtPoint[&Inst] != CurState) {
        SetAtPoint[&Inst] = CurState;
        if (&Inst == LAST)
          Changed = true;
      }
      PrevState = &SetAtPoint[&Inst];
    }

    if (Changed) {
      for (auto I = BB->succ_begin(), E = BB->succ_end(); I != E; ++I) {
        Worklist.push(*I);
      }
    }
  }

  DEBUG(dbgs() << "Performing frame optimization\n");
  std::vector<std::pair<BinaryBasicBlock *, MCInst *>> ToErase;
  for (auto &BB : *BF) {
    DEBUG(dbgs() <<"\tNow at BB " << BB.getName() << "\n");
    BitVector *PrevState = &SetAtBBEntry[&BB];
    for (auto &Inst : BB) {
      DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        int ExprIdx = PrevState->find_first();
        while (ExprIdx != -1) {
          dbgs() << "\t\t\tReached by: ";
          Expressions[ExprIdx]->dump();
          ExprIdx = PrevState->find_next(ExprIdx);
        }
      });
      // if Inst is a load from stack and the current available expressions show
      // this value is available in a register or immediate, replace this load
      // with move from register or from immediate.
      const auto Iter = FrameIndexMap.find(&Inst);
      if (Iter == FrameIndexMap.end()) {
        PrevState = &SetAtPoint[&Inst];
        continue;
      }
      const FrameIndexEntry &FIEX = Iter->second;
      // FIXME: Change to remove IsSimple == 0. We're being conservative here,
      // but once replaceMemOperandWithReg is ready, we should feed it with all
      // sorts of complex instructions.
      if (FIEX.IsLoad == 0 || FIEX.IsSimple == 0) {
        PrevState = &SetAtPoint[&Inst];
        continue;
      }

      int ExprIdx = PrevState->find_first();
      while (ExprIdx != -1) {
        const MCInst *AvailableInst = Expressions[ExprIdx];
        const auto Iter = FrameIndexMap.find(AvailableInst);
        if (Iter == FrameIndexMap.end()) {
          ExprIdx = PrevState->find_next(ExprIdx);
          continue;
        }
        const FrameIndexEntry &FIEY = Iter->second;
        assert(FIEY.IsLoad == 0 && FIEY.IsSimple != 0);
        if (FIEX.StackOffset != FIEY.StackOffset || FIEX.Size != FIEY.Size) {
          ExprIdx = PrevState->find_next(ExprIdx);
          continue;
        }
        ++NumRedundantLoads;
        DEBUG(dbgs() << "Redundant load instruction: ");
        DEBUG(Inst.dump());
        DEBUG(dbgs() << "Related store instruction: ");
        DEBUG(AvailableInst->dump());
        DEBUG(dbgs() << "@BB: " << BB.getName() << "\n");
        ExprIdx = PrevState->find_next(ExprIdx);
        // Replace load
        if (FIEY.IsStoreFromReg) {
          if (!BC.MIA->replaceMemOperandWithReg(Inst, FIEY.RegOrImm)) {
            DEBUG(dbgs() << "FAILED to change operand to a reg\n");
            break;
          }
          ++NumLoadsChangedToReg;
          DEBUG(dbgs() << "Changed operand to a reg\n");
          if (BC.MIA->isRedundantMove(Inst)) {
            ++NumLoadsDeleted;
            DEBUG(dbgs() << "Created a redundant move\n");
            // Delete it!
            ToErase.emplace_back(&BB, &Inst);
          }
        } else {
          char Buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          support::ulittle64_t::ref(Buf + 0) = FIEY.RegOrImm;
          DEBUG(dbgs() << "Changing operand to an imm... ");
          if (!BC.MIA->replaceMemOperandWithImm(Inst, StringRef(Buf, 8), 0)) {
            DEBUG(dbgs() << "FAILED\n");
          } else {
            ++NumLoadsChangedToImm;
            DEBUG(dbgs() << "Ok\n");
          }
        }
        DEBUG(dbgs() << "Changed to: ");
        DEBUG(Inst.dump());
        break;
      }
      PrevState = &SetAtPoint[&Inst];
    }
  }
  for (auto I : ToErase) {
    I.first->eraseInstruction(I.second);
  }
}

void FrameOptimizerPass::runOnFunctions(BinaryContext &BC,
                                        std::map<uint64_t, BinaryFunction> &BFs,
                                        std::set<uint64_t> &) {
  uint64_t NumFunctionsNotOptimized{0};
  uint64_t NumFunctionsFailedRestoreFI{0};
  uint64_t CountFunctionsNotOptimized{0};
  uint64_t CountFunctionsFailedRestoreFI{0};
  uint64_t CountDenominator{0};
  buildCallGraph(BC, BFs);
  buildCGTraversalOrder();
  buildClobberMap(BC);
  for (auto &I : BFs) {
    auto Count = I.second.getExecutionCount();
    if (Count != BinaryFunction::COUNT_NO_PROFILE)
      CountDenominator += Count;
    if (!shouldOptimize(I.second)) {
      ++NumFunctionsNotOptimized;
      if (Count != BinaryFunction::COUNT_NO_PROFILE)
        CountFunctionsNotOptimized += Count;
      continue;
    }
    if (!restoreFrameIndex(BC, &I.second)) {
      ++NumFunctionsFailedRestoreFI;
      auto Count = I.second.getExecutionCount();
      if (Count != BinaryFunction::COUNT_NO_PROFILE)
        CountFunctionsFailedRestoreFI += Count;
      continue;
    }
    removeUnnecessarySpills(BC, &I.second);
  }

  if (opts::Verbosity == 0 && (!DebugFlag || !isCurrentDebugType("fop")))
    return;

  outs() << "BOLT-INFO: FOP found " << NumRedundantLoads
         << " redundant load(s).\n"
         << "BOLT-INFO: FOP changed " << NumLoadsChangedToReg
         << " load(s) to use a register instead of a stack access, and "
         << NumLoadsChangedToImm << " to use an immediate.\n"
         << "BOLT-INFO: FOP deleted " << NumLoadsDeleted << " load(s).\n"
         << "BOLT-INFO: FOP: Number of functions conservatively treated as "
            "clobbering all registers: "
         << NumFunctionsAllClobber
         << format(" (%.1lf%% dyn cov)\n",
                   (100.0 * CountFunctionsAllClobber / CountDenominator))
         << "BOLT-INFO: FOP: " << NumFunctionsNotOptimized << " function(s) "
         << format("(%.1lf%% dyn cov)",
                   (100.0 * CountFunctionsNotOptimized / CountDenominator))
         << " were not optimized.\n"
         << "BOLT-INFO: FOP: " << NumFunctionsFailedRestoreFI << " function(s) "
         << format("(%.1lf%% dyn cov)",
                   (100.0 * CountFunctionsFailedRestoreFI / CountDenominator))
         << " could not have its frame indices restored.\n";
}

} // namespace bolt
} // namespace llvm
