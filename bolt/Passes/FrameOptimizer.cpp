//===--- Passes/FrameOptimizer.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "FrameOptimizer.h"
#include <queue>
#include <unordered_map>

#define DEBUG_TYPE "fop"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> Verbosity;
}

namespace llvm {
namespace bolt {

void FrameOptimizerPass::getInstClobberList(const BinaryContext &BC,
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
FrameOptimizerPass::getFunctionClobberList(const BinaryContext &BC,
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

void FrameOptimizerPass::buildClobberMap(const BinaryContext &BC) {
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
      for (auto Caller : Cg.Nodes[Cg.FuncToNodeId.at(Func)].Preds) {
        Queue.push(Cg.Funcs[Caller]);
      }
    }
    RegsKilledMap[Func] = std::move(RegsKilled);
  }

  if (opts::Verbosity == 0) {
#ifndef NDEBUG
    if (!DebugFlag || !isCurrentDebugType("fop"))
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
    DEBUG_WITH_TYPE("fop",
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

namespace {

template <typename StateTy>
class ForwardDataflow {
protected:
  /// Reference to the function being analysed
  const BinaryFunction &Func;

  /// Tracks the set of available exprs at the end of each MCInst in this
  /// function
  std::unordered_map<const MCInst *, StateTy> StateAtPoint;
  /// Tracks the set of available exprs at basic block start
  std::unordered_map<const BinaryBasicBlock *, StateTy> StateAtBBEntry;

  virtual void preflight() = 0;

  virtual StateTy getStartingStateAtBB(const BinaryBasicBlock &BB) = 0;

  virtual StateTy getStartingStateAtPoint(const MCInst &Point) = 0;

  virtual void doConfluence(StateTy &StateOut, const StateTy &StateIn) = 0;

  virtual StateTy computeNext(const MCInst &Point, const StateTy &Cur) = 0;

public:
  ForwardDataflow(const BinaryFunction &BF) : Func(BF) {}
  virtual ~ForwardDataflow() {}

  ErrorOr<const StateTy &>getStateAt(const BinaryBasicBlock &BB) const {
    auto Iter = StateAtBBEntry.find(&BB);
    if (Iter == StateAtBBEntry.end())
      return make_error_code(errc::result_out_of_range);
    return Iter->second;
  }

  ErrorOr<const StateTy &>getStateAt(const MCInst &Point) const {
    auto Iter = StateAtPoint.find(&Point);
    if (Iter == StateAtPoint.end())
      return make_error_code(errc::result_out_of_range);
    return Iter->second;
  }

  void run() {
    preflight();

    // Initialize state for all points of the function
    for (auto &BB : Func) {
      StateAtBBEntry[&BB] = getStartingStateAtBB(BB);
      for (auto &Inst : BB) {
        StateAtPoint[&Inst] = getStartingStateAtPoint(Inst);
      }
    }
    assert(Func.begin() != Func.end() && "Unexpected empty function");

    std::queue<const BinaryBasicBlock *> Worklist;
    // TODO: Pushing this in a DFS ordering will greatly speed up the dataflow
    // performance.
    for (auto &BB : Func) {
      Worklist.push(&BB);
    }

    // Main dataflow loop
    while (!Worklist.empty()) {
      auto *BB = Worklist.front();
      Worklist.pop();

      DEBUG(dbgs() << "\tNow at BB " << BB->getName() << "\n");

      // Calculate state at the entry of first instruction in BB
      StateTy &StateAtEntry = StateAtBBEntry[BB];
      for (auto I = BB->pred_begin(), E = BB->pred_end(); I != E; ++I) {
        auto Last = (*I)->rbegin();
        if (Last != (*I)->rend()) {
          doConfluence(StateAtEntry, StateAtPoint[&*Last]);
        } else {
          doConfluence(StateAtEntry, StateAtBBEntry[*I]);
        }
      }
      // Skip empty
      if (BB->begin() == BB->end())
        continue;

      // Propagate information from first instruction down to the last one
      bool Changed = false;
      StateTy *PrevState = &StateAtEntry;
      const MCInst *LAST = &*BB->rbegin();
      for (auto &Inst : *BB) {
        DEBUG(dbgs() << "\t\tNow at ");
        DEBUG(Inst.dump());

        StateTy CurState = computeNext(Inst, *PrevState);

        if (StateAtPoint[&Inst] != CurState) {
          StateAtPoint[&Inst] = CurState;
          if (&Inst == LAST)
            Changed = true;
        }
        PrevState = &StateAtPoint[&Inst];
      }

      if (Changed) {
        for (auto I = BB->succ_begin(), E = BB->succ_end(); I != E; ++I) {
          Worklist.push(*I);
        }
      }
    }
  }
};

class StackAvailableExpressions : public ForwardDataflow<BitVector> {
public:
  StackAvailableExpressions(const FrameOptimizerPass &FOP,
                            const BinaryContext &BC, const BinaryFunction &BF)
      : ForwardDataflow(BF), FOP(FOP), FrameIndexMap(FOP.FrameIndexMap),
        BC(BC) {}
  virtual ~StackAvailableExpressions() {}

  /// Define an iterator for navigating the expressions calculated by the
  /// dataflow at each program point
  class ExprIterator
      : public std::iterator<std::forward_iterator_tag, const MCInst *> {
  public:
    ExprIterator &operator++() {
      assert(Idx != -1 && "Iterator already at the end");
      Idx = BV->find_next(Idx);
      return *this;
    }
    ExprIterator operator++(int) {
      assert(Idx != -1 && "Iterator already at the end");
      ExprIterator Ret = *this;
      ++(*this);
      return Ret;
    }
    bool operator==(ExprIterator Other) const { return Idx == Other.Idx; }
    bool operator!=(ExprIterator Other) const { return Idx != Other.Idx; }
    const MCInst *operator*() {
      assert(Idx != -1 && "Invalid access to end iterator");
      return Expressions[Idx];
    }
    ExprIterator(const BitVector *BV, const std::vector<const MCInst *> &Exprs)
        : BV(BV), Expressions(Exprs) {
      Idx = BV->find_first();
    }
    ExprIterator(const BitVector *BV, const std::vector<const MCInst *> &Exprs,
                 int Idx)
        : BV(BV), Expressions(Exprs), Idx(Idx) {}

  private:
    const BitVector *BV;
    const std::vector<const MCInst *> &Expressions;
  public:
    int Idx;
  };
  ExprIterator expr_begin(const BitVector &BV) const {
    return ExprIterator(&BV, Expressions);
  }
  ExprIterator expr_begin(const MCInst &Point) const {
    auto Iter = StateAtPoint.find(&Point);
    if (Iter == StateAtPoint.end())
      return expr_end();
    return ExprIterator(&Iter->second, Expressions);
  }
  ExprIterator expr_begin(const BinaryBasicBlock &BB) const {
    auto Iter = StateAtBBEntry.find(&BB);
    if (Iter == StateAtBBEntry.end())
      return expr_end();
    return ExprIterator(&Iter->second, Expressions);
  }
  ExprIterator expr_end() const {
    return ExprIterator(nullptr, Expressions, -1);
  }

private:
  /// Reference to the result of stack frame analysis
  const FrameOptimizerPass &FOP;
  const FrameOptimizerPass::FrameIndexMapTy &FrameIndexMap;
  const BinaryContext &BC;

  /// Used to size the set of expressions/definitions being tracked by the
  /// dataflow analysis
  uint64_t NumInstrs{0};
  /// We put every MCInst we want to track (which one representing an
  /// expression/def) into a vector because we need to associate them with
  /// small numbers. They will be tracked via BitVectors throughout the
  /// dataflow analysis.
  std::vector<const MCInst *> Expressions;
  /// Maps expressions defs (MCInsts) to its index in the Expressions vector
  std::unordered_map<const MCInst *, uint64_t> ExprToIdx;

  void preflight() override {
    DEBUG(dbgs() << "Starting StackAvailableExpressions on \""
                 << Func.getPrintName() << "\"\n");

    // Populate our universe of tracked expressions. We are interested in
    // tracking available stores to frame position at any given point of the
    // program.
    for (auto &BB : Func) {
      for (auto &Inst : BB) {
        auto FIEIter = FrameIndexMap.find(&Inst);
        if (FIEIter == FrameIndexMap.end())
          continue;
        const auto &FIE = FIEIter->second;
        if (FIE.IsLoad == false && FIE.IsSimple == true) {
          Expressions.push_back(&Inst);
          ExprToIdx[&Inst] = NumInstrs++;
        }
      }
    }
  }

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) override {
    // Entry points start with empty set (Function entry and landing pads).
    // All others start with the full set.
    if (BB.pred_size() == 0)
      return BitVector(NumInstrs, false);
    return BitVector(NumInstrs, true);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) override {
    return BitVector(NumInstrs, true);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) override {
    StateOut &= StateIn;
  }

  /// Define the function computing the kill set -- whether expression Y, a
  /// tracked expression, will be considered to be dead after executing X.
  bool doesXKillsY(const MCInst *X, const MCInst *Y) {
    // if both are stores, and both store to the same stack location, return
    // true
    auto FIEIterX = FrameIndexMap.find(X);
    auto FIEIterY = FrameIndexMap.find(Y);
    if (FIEIterX != FrameIndexMap.end() && FIEIterY != FrameIndexMap.end()) {
      const FrameOptimizerPass::FrameIndexEntry &FIEX = FIEIterX->second;
      const FrameOptimizerPass::FrameIndexEntry &FIEY = FIEIterY->second;;
      if (FIEX.IsLoad == 0 && FIEY.IsLoad == 0 &&
          FIEX.StackOffset + FIEX.Size > FIEY.StackOffset &&
          FIEX.StackOffset < FIEY.StackOffset + FIEY.Size)
        return true;
    }
    // getClobberedRegs for X and Y. If they intersect, return true
    BitVector XClobbers = BitVector(BC.MRI->getNumRegs(), false);
    BitVector YClobbers = BitVector(BC.MRI->getNumRegs(), false);
    FOP.getInstClobberList(BC, *X, XClobbers);
    // If Y is a store to stack, its clobber list is its source reg. This is
    // different than the rest because we want to check if the store source
    // reaches its corresponding load untouched.
    if (FIEIterY != FrameIndexMap.end() && FIEIterY->second.IsLoad == 0 &&
        FIEIterY->second.IsStoreFromReg) {
      YClobbers.set(FIEIterY->second.RegOrImm);
    } else {
      FOP.getInstClobberList(BC, *Y, YClobbers);
    }
    XClobbers &= YClobbers;
    return XClobbers.any();
  }

  BitVector computeNext(const MCInst &Point, const BitVector &Cur) override {
    BitVector Next = Cur;
    // Kill
    for (auto I = expr_begin(Next), E = expr_end(); I != E; ++I) {
      assert(*I != nullptr && "Lost pointers");
      DEBUG(dbgs() << "\t\t\tDoes it kill ");
      DEBUG((*I)->dump());
      if (doesXKillsY(&Point, *I)) {
        DEBUG(dbgs() << "\t\t\t\tYes\n");
        Next.reset(I.Idx);
      }
     };
    // Gen
    auto FIEIter = FrameIndexMap.find(&Point);
    if (FIEIter != FrameIndexMap.end() &&
        FIEIter->second.IsLoad == false &&
        FIEIter->second.IsSimple == true)
      Next.set(ExprToIdx[&Point]);
    return Next;
  }
};

class StackPointerTracking : public ForwardDataflow<int> {
  const BinaryContext &BC;

  void preflight() override {
    DEBUG(dbgs() << "Starting StackPointerTracking on \""
                 << Func.getPrintName() << "\"\n");
  }

  int getStartingStateAtBB(const BinaryBasicBlock &BB) override {
    // Entry BB start with offset 8 from CFA.
    // All others start with EMPTY (meaning we don't know anything).
    if (BB.isEntryPoint())
      return -8;
    return EMPTY;
  }

  int getStartingStateAtPoint(const MCInst &Point) override {
    return EMPTY;
  }

  void doConfluence(int &StateOut, const int &StateIn) override {
    if (StateOut == EMPTY) {
      StateOut = StateIn;
      return;
    }
    if (StateIn == EMPTY || StateIn == StateOut)
      return;

    // We can't agree on a specific value from this point on
    StateOut = SUPERPOSITION;
  }

  int computeNext(const MCInst &Point, const int &Cur) override {
    const auto &MIA = BC.MIA;

    if (Cur == EMPTY || Cur == SUPERPOSITION)
      return Cur;

    if (int Sz = MIA->getPushSize(Point))
      return Cur - Sz;

    if (int Sz = MIA->getPopSize(Point))
      return Cur + Sz;

    if (BC.MII->get(Point.getOpcode())
            .hasDefOfPhysReg(Point, MIA->getStackPointer(), *BC.MRI)) {
      int64_t Offset = Cur;
      if (!MIA->evaluateSimple(Point, Offset, std::make_pair(0, 0),
                               std::make_pair(0, 0)))
        return SUPERPOSITION;

      return static_cast<int>(Offset);
    }

    return Cur;
  }
public:
  StackPointerTracking(const BinaryContext &BC, const BinaryFunction &BF)
      : ForwardDataflow(BF), BC(BC) {}
  virtual ~StackPointerTracking() {}

  static constexpr int SUPERPOSITION = std::numeric_limits<int>::max();
  static constexpr int EMPTY = std::numeric_limits<int>::min();
};

} // anonymous namespace

bool FrameOptimizerPass::restoreFrameIndex(const BinaryContext &BC,
                                           const BinaryFunction &BF) {
  StackPointerTracking SPT(BC, BF);

  SPT.run();

  // Vars used for storing useful CFI info to give us a hint about how the stack
  // is used in this function
  int64_t CfaOffset{-8};
  uint16_t CfaReg{7};
  bool CfaRegLocked{false};
  uint16_t CfaRegLockedVal{0};
  std::stack<std::pair<int64_t, uint16_t>> CFIStack;

  DEBUG(dbgs() << "Restoring frame indices for \"" << BF.getPrintName()
               << "\"\n");

  // TODO: Implement SP tracking and improve this analysis
  for (auto &BB : BF) {
    DEBUG(dbgs() <<"\tNow at BB " << BB.getName() << "\n");

    const MCInst *Prev = nullptr;
    for (const auto &Inst : BB) {
      int SPOffset = (Prev ? *SPT.getStateAt(*Prev) : *SPT.getStateAt(BB));
      DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        dbgs() << "\t\t\tSP offset is " << SPOffset << "\n";
      });
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

      if (BC.MIA->leaksStackAddress(Inst, *BC.MRI, false)) {
        DEBUG(dbgs() << "Leaked stack address, giving up on this function.\n");
        DEBUG(dbgs() << "Blame insn: ");
        DEBUG(Inst.dump());
        return false;
      }

      bool IsLoad = false;
      bool IsStore = false;
      bool IsStoreFromReg = false;
      bool IsSimple = false;
      int32_t SrcImm{0};
      MCPhysReg Reg{0};
      MCPhysReg StackPtrReg{0};
      int64_t StackOffset{0};
      uint8_t Size{0};
      bool IsIndexed = false;
      if (BC.MIA->isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, Reg,
                                SrcImm, StackPtrReg, StackOffset, Size,
                                IsSimple, IsIndexed)) {
        assert(Size != 0);
        if (CfaRegLocked && CfaRegLockedVal != CfaReg) {
          DEBUG(dbgs() << "CFA reg changed, giving up on this function.\n");
          return false;
        }
        if (StackPtrReg != BC.MRI->getLLVMRegNum(CfaReg, /*isEH=*/false)) {
          if (StackPtrReg != BC.MIA->getStackPointer() ||
              SPOffset == SPT.EMPTY || SPOffset == SPT.SUPERPOSITION) {
            DEBUG(dbgs()
                  << "Found stack access with reg different than cfa reg.\n");
            DEBUG(dbgs() << "\tCurrent CFA reg: " << CfaReg
                         << "\n\tStack access reg: " << StackPtrReg << "\n");
            DEBUG(dbgs() << "Blame insn: ");
            DEBUG(Inst.dump());
            return false;
          }
          DEBUG(dbgs() << "Adding access via SP while CFA reg is another one\n");
          if (IsStoreFromReg || IsLoad)
            SrcImm = Reg;
          // Ignore accesses to the previous stack frame
          if (SPOffset + StackOffset >= 0)
            continue;
          FrameIndexMap.emplace(
              &Inst, FrameIndexEntry{IsLoad, IsStoreFromReg, SrcImm,
                                     SPOffset + StackOffset, Size, IsSimple});
        } else {
          CfaRegLocked = true;
          CfaRegLockedVal = CfaReg;
          if (IsStoreFromReg || IsLoad)
            SrcImm = Reg;
          // Ignore accesses to the previous stack frame
          if (CfaOffset + StackOffset >= 0)
            continue;
          FrameIndexMap.emplace(
              &Inst, FrameIndexEntry{IsLoad, IsStoreFromReg, SrcImm,
                                     CfaOffset + StackOffset, Size, IsSimple});
        }

        DEBUG_WITH_TYPE("fop",
          dbgs() << "Frame index annotation added to:\n";
          BC.printInstruction(dbgs(), Inst, 0, &BF, true);
          dbgs() << " FrameIndexEntry <IsLoad:" << IsLoad << " StackOffset:";
          if (FrameIndexMap[&Inst].StackOffset < 0)
            dbgs() << "-" << Twine::utohexstr(-FrameIndexMap[&Inst].StackOffset);
          else
            dbgs() << "+" << Twine::utohexstr(FrameIndexMap[&Inst].StackOffset);
          dbgs() << " IsStoreFromReg:" << FrameIndexMap[&Inst].IsStoreFromReg
                 << " RegOrImm:" << FrameIndexMap[&Inst].RegOrImm << ">\n";
        );
      }
    }
  }
  return true;
}

void FrameOptimizerPass::removeUnnecessarySpills(const BinaryContext &BC,
                                                 BinaryFunction &BF) {
  StackAvailableExpressions SAE(*this, BC, BF);

  SAE.run();

  DEBUG(dbgs() << "Performing frame optimization\n");
  std::deque<std::pair<BinaryBasicBlock *, MCInst *>> ToErase;
  bool Changed = false;
  const auto ExprEnd = SAE.expr_end();
  for (auto &BB : BF) {
    DEBUG(dbgs() <<"\tNow at BB " << BB.getName() << "\n");
    const MCInst *Prev = nullptr;
    for (auto &Inst : BB) {
      DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        for (auto I = Prev ? SAE.expr_begin(*Prev) : SAE.expr_begin(BB);
             I != ExprEnd; ++I) {
          dbgs() << "\t\t\tReached by: ";
          (*I)->dump();
        }
      });
      // if Inst is a load from stack and the current available expressions show
      // this value is available in a register or immediate, replace this load
      // with move from register or from immediate.
      const auto Iter = FrameIndexMap.find(&Inst);
      if (Iter == FrameIndexMap.end()) {
        Prev = &Inst;
        continue;
      }
      const FrameIndexEntry &FIEX = Iter->second;
      // FIXME: Change to remove IsSimple == 0. We're being conservative here,
      // but once replaceMemOperandWithReg is ready, we should feed it with all
      // sorts of complex instructions.
      if (FIEX.IsLoad == 0 || FIEX.IsSimple == 0) {
        Prev = &Inst;
        continue;
      }

      for (auto I = Prev ? SAE.expr_begin(*Prev) : SAE.expr_begin(BB);
           I != ExprEnd; ++I) {
        const MCInst *AvailableInst = *I;
        const auto Iter = FrameIndexMap.find(AvailableInst);
        if (Iter == FrameIndexMap.end())
          continue;

        const FrameIndexEntry &FIEY = Iter->second;
        assert(FIEY.IsLoad == 0 && FIEY.IsSimple != 0);
        if (FIEX.StackOffset != FIEY.StackOffset || FIEX.Size != FIEY.Size)
          continue;

        ++NumRedundantLoads;
        Changed = true;
        DEBUG(dbgs() << "Redundant load instruction: ");
        DEBUG(Inst.dump());
        DEBUG(dbgs() << "Related store instruction: ");
        DEBUG(AvailableInst->dump());
        DEBUG(dbgs() << "@BB: " << BB.getName() << "\n");
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
            ToErase.push_front(std::make_pair(&BB, &Inst));
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
      Prev = &Inst;
    }
  }
  if (Changed) {
    DEBUG(dbgs() << "FOP modified \"" << BF.getPrintName() << "\"\n");
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
  Cg = buildCallGraph(BC, BFs);
  TopologicalCGOrder = Cg.buildTraversalOrder();
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
    if (!restoreFrameIndex(BC, I.second)) {
      ++NumFunctionsFailedRestoreFI;
      auto Count = I.second.getExecutionCount();
      if (Count != BinaryFunction::COUNT_NO_PROFILE)
        CountFunctionsFailedRestoreFI += Count;
      continue;
    }
    removeUnnecessarySpills(BC, I.second);
  }

  outs() << "BOLT-INFO: FOP optimized " << NumRedundantLoads
         << " redundant load(s).\n";

  if (opts::Verbosity == 0) {
#ifndef NDEBUG
    if (!DebugFlag || !isCurrentDebugType("fop"))
      return;
#else
    return;
#endif
  }

  outs() << "BOLT-INFO: FOP changed " << NumLoadsChangedToReg
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
