#include "Passes/BranchPredictionInfo.h"
#include "BinaryBasicBlock.h"

namespace llvm {
namespace bolt {

void BranchPredictionInfo::findLoopEdgesInfo(const BinaryLoopInfo &LoopsInfo) {
  // Traverse discovered loops
  std::stack<BinaryLoop *> Loops;
  for (BinaryLoop *BL : LoopsInfo)
    Loops.push(BL);

  while (!Loops.empty()) {
    BinaryLoop *Loop = Loops.top();
    Loops.pop();
    BinaryBasicBlock *LoopHeader = Loop->getHeader();
    LoopHeaders.insert(LoopHeader);

    // Add nested loops in the stack.
    for (BinaryLoop::iterator I = Loop->begin(), E = Loop->end(); I != E; ++I) {
      Loops.push(*I);
    }

    SmallVector<BinaryBasicBlock *, 1> Latches;
    Loop->getLoopLatches(Latches);

    // Find back edges.
    for (BinaryBasicBlock *Latch : Latches) {
      for (BinaryBasicBlock *Succ : Latch->successors()) {
        if (Succ == LoopHeader) {
          Edge CFGEdge = std::make_pair(Latch->getLabel(), Succ->getLabel());
          BackEdges.insert(CFGEdge);
        }
      }
    }

    // Find exit edges.
    SmallVector<BinaryLoop::Edge, 1> AuxExitEdges;
    Loop->getExitEdges(AuxExitEdges);
    for (BinaryLoop::Edge &Exit : AuxExitEdges) {
      ExitEdges.insert(Exit);
    }
  }
}

void BranchPredictionInfo::findBasicBlockInfo(const BinaryFunction &Function,
                                              BinaryContext &BC) {
  for (auto &BB : Function) {
    for (auto &Inst : BB) {
      if (BC.MIB->isCall(Inst))
        CallSet.insert(&BB);
      else if (BC.MIB->isStore(Inst))
        StoreSet.insert(&BB);
    }
  }
}

bool BranchPredictionInfo::isBackEdge(const Edge &CFGEdge) const {
  return BackEdges.count(CFGEdge);
}

bool BranchPredictionInfo::isBackEdge(const BinaryBasicBlock *SrcBB,
                                      const BinaryBasicBlock *DstBB) const {
  const Edge CFGEdge = std::make_pair(SrcBB->getLabel(), DstBB->getLabel());
  return isBackEdge(CFGEdge);
}

bool BranchPredictionInfo::isExitEdge(const BinaryLoop::Edge &CFGEdge) const {
  return ExitEdges.count(CFGEdge);
}

bool BranchPredictionInfo::isExitEdge(const BinaryBasicBlock *SrcBB,
                                      const BinaryBasicBlock *DstBB) const {
  const BinaryLoop::Edge CFGEdge = std::make_pair(SrcBB, DstBB);
  return isExitEdge(CFGEdge);
}

bool BranchPredictionInfo::isLoopHeader(const BinaryBasicBlock *BB) const {
  return LoopHeaders.count(BB);
}

bool BranchPredictionInfo::hasCallInst(const BinaryBasicBlock *BB) const {
  return CallSet.count(BB);
}

bool BranchPredictionInfo::hasStoreInst(const BinaryBasicBlock *BB) const {
  return StoreSet.count(BB);
}

bool BranchPredictionInfo::callToExit(BinaryBasicBlock *BB,
                                      BinaryContext &BC) const {
  auto &currBB = *BB;
  for (auto &Inst : currBB) {
    if (BC.MIB->isCall(Inst)) {
      if (const auto *CalleeSymbol = BC.MIB->getTargetSymbol(Inst)) {
        StringRef CalleeName = CalleeSymbol->getName();
        if (CalleeName == "__cxa_throw@PLT" ||
            CalleeName == "_Unwind_Resume@PLT" ||
            CalleeName == "__cxa_rethrow@PLT" || CalleeName == "exit@PLT" ||
            CalleeName == "abort@PLT")
          return true;
      }
    }
  }

  return false;
}

unsigned BranchPredictionInfo::countBackEdges(BinaryBasicBlock *BB) const {
  unsigned CountEdges = 0;

  for (BinaryBasicBlock *SuccBB : BB->successors()) {
    const Edge CFGEdge = std::make_pair(BB->getLabel(), SuccBB->getLabel());
    if (BackEdges.count(CFGEdge))
      ++CountEdges;
  }

  return CountEdges;
}

unsigned BranchPredictionInfo::countExitEdges(BinaryBasicBlock *BB) const {
  unsigned CountEdges = 0;

  for (BinaryBasicBlock *SuccBB : BB->successors()) {
    const BinaryLoop::Edge CFGEdge = std::make_pair(BB, SuccBB);
    if (ExitEdges.count(CFGEdge))
      ++CountEdges;
  }

  return CountEdges;
}

void BranchPredictionInfo::clear() {
  LoopHeaders.clear();
  BackEdges.clear();
  ExitEdges.clear();
  CallSet.clear();
  StoreSet.clear();
}

} // namespace bolt
} // namespace llvm
