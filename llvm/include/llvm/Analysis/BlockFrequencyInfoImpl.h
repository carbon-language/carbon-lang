//==- BlockFrequencyInfoImpl.h - Block Frequency Implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Shared implementation of BlockFrequencyInfo for IR and Machine Instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace llvm {


class BlockFrequencyInfo;
class MachineBlockFrequencyInfo;

/// BlockFrequencyInfoImpl implements block frequency algorithm for IR and
/// Machine Instructions. Algorithm starts with value ENTRY_FREQ
/// for the entry block and then propagates frequencies using branch weights
/// from (Machine)BranchProbabilityInfo. LoopInfo is not required because
/// algorithm can find "backedges" by itself.
template<class BlockT, class FunctionT, class BlockProbInfoT>
class BlockFrequencyInfoImpl {

  DenseMap<const BlockT *, BlockFrequency> Freqs;

  BlockProbInfoT *BPI;

  FunctionT *Fn;

  typedef GraphTraits< Inverse<BlockT *> > GT;

  static const uint64_t EntryFreq = 1 << 14;

  std::string getBlockName(BasicBlock *BB) const {
    return BB->getName().str();
  }

  std::string getBlockName(MachineBasicBlock *MBB) const {
    std::string str;
    raw_string_ostream ss(str);
    ss << "BB#" << MBB->getNumber();

    if (const BasicBlock *BB = MBB->getBasicBlock())
      ss << " derived from LLVM BB " << BB->getName();

    return ss.str();
  }

  void setBlockFreq(BlockT *BB, BlockFrequency Freq) {
    Freqs[BB] = Freq;
    DEBUG(dbgs() << "Frequency(" << getBlockName(BB) << ") = ";
          printBlockFreq(dbgs(), Freq) << "\n");
  }

  /// getEdgeFreq - Return edge frequency based on SRC frequency and Src -> Dst
  /// edge probability.
  BlockFrequency getEdgeFreq(BlockT *Src, BlockT *Dst) const {
    BranchProbability Prob = BPI->getEdgeProbability(Src, Dst);
    return getBlockFreq(Src) * Prob;
  }

  /// incBlockFreq - Increase BB block frequency by FREQ.
  ///
  void incBlockFreq(BlockT *BB, BlockFrequency Freq) {
    Freqs[BB] += Freq;
    DEBUG(dbgs() << "Frequency(" << getBlockName(BB) << ") += ";
          printBlockFreq(dbgs(), Freq) << " --> ";
          printBlockFreq(dbgs(), Freqs[BB]) << "\n");
  }

  // All blocks in postorder.
  std::vector<BlockT *> POT;

  // Map Block -> Position in reverse-postorder list.
  DenseMap<BlockT *, unsigned> RPO;

  // For each loop header, record the per-iteration probability of exiting the
  // loop. This is the reciprocal of the expected number of loop iterations.
  typedef DenseMap<BlockT*, BranchProbability> LoopExitProbMap;
  LoopExitProbMap LoopExitProb;

  // (reverse-)postorder traversal iterators.
  typedef typename std::vector<BlockT *>::iterator pot_iterator;
  typedef typename std::vector<BlockT *>::reverse_iterator rpot_iterator;

  pot_iterator pot_begin() { return POT.begin(); }
  pot_iterator pot_end() { return POT.end(); }

  rpot_iterator rpot_begin() { return POT.rbegin(); }
  rpot_iterator rpot_end() { return POT.rend(); }

  rpot_iterator rpot_at(BlockT *BB) {
    rpot_iterator I = rpot_begin();
    unsigned idx = RPO.lookup(BB);
    assert(idx);
    std::advance(I, idx - 1);

    assert(*I == BB);
    return I;
  }

  /// isBackedge - Return if edge Src -> Dst is a reachable backedge.
  ///
  bool isBackedge(BlockT *Src, BlockT *Dst) const {
    unsigned a = RPO.lookup(Src);
    if (!a)
      return false;
    unsigned b = RPO.lookup(Dst);
    assert(b && "Destination block should be reachable");
    return a >= b;
  }

  /// getSingleBlockPred - return single BB block predecessor or NULL if
  /// BB has none or more predecessors.
  BlockT *getSingleBlockPred(BlockT *BB) {
    typename GT::ChildIteratorType
      PI = GraphTraits< Inverse<BlockT *> >::child_begin(BB),
      PE = GraphTraits< Inverse<BlockT *> >::child_end(BB);

    if (PI == PE)
      return 0;

    BlockT *Pred = *PI;

    ++PI;
    if (PI != PE)
      return 0;

    return Pred;
  }

  void doBlock(BlockT *BB, BlockT *LoopHead,
               SmallPtrSet<BlockT *, 8> &BlocksInLoop) {

    DEBUG(dbgs() << "doBlock(" << getBlockName(BB) << ")\n");
    setBlockFreq(BB, 0);

    if (BB == LoopHead) {
      setBlockFreq(BB, EntryFreq);
      return;
    }

    if (BlockT *Pred = getSingleBlockPred(BB)) {
      if (BlocksInLoop.count(Pred))
        setBlockFreq(BB, getEdgeFreq(Pred, BB));
      // TODO: else? irreducible, ignore it for now.
      return;
    }

    bool isInLoop = false;
    bool isLoopHead = false;

    for (typename GT::ChildIteratorType
         PI = GraphTraits< Inverse<BlockT *> >::child_begin(BB),
         PE = GraphTraits< Inverse<BlockT *> >::child_end(BB);
         PI != PE; ++PI) {
      BlockT *Pred = *PI;

      if (isBackedge(Pred, BB)) {
        isLoopHead = true;
      } else if (BlocksInLoop.count(Pred)) {
        incBlockFreq(BB, getEdgeFreq(Pred, BB));
        isInLoop = true;
      }
      // TODO: else? irreducible.
    }

    if (!isInLoop)
      return;

    if (!isLoopHead)
      return;

    // This block is a loop header, so boost its frequency by the expected
    // number of loop iterations. The loop blocks will be revisited so they all
    // get this boost.
    typename LoopExitProbMap::const_iterator I = LoopExitProb.find(BB);
    assert(I != LoopExitProb.end() && "Loop header missing from table");
    Freqs[BB] /= I->second;
    DEBUG(dbgs() << "Loop header scaled to ";
          printBlockFreq(dbgs(), Freqs[BB]) << ".\n");
  }

  /// doLoop - Propagate block frequency down through the loop.
  void doLoop(BlockT *Head, BlockT *Tail) {
    DEBUG(dbgs() << "doLoop(" << getBlockName(Head) << ", "
                 << getBlockName(Tail) << ")\n");

    SmallPtrSet<BlockT *, 8> BlocksInLoop;

    for (rpot_iterator I = rpot_at(Head), E = rpot_at(Tail); ; ++I) {
      BlockT *BB = *I;
      doBlock(BB, Head, BlocksInLoop);

      BlocksInLoop.insert(BB);
      if (I == E)
        break;
    }

    // Compute loop's cyclic probability using backedges probabilities.
    BlockFrequency BackFreq;
    for (typename GT::ChildIteratorType
         PI = GraphTraits< Inverse<BlockT *> >::child_begin(Head),
         PE = GraphTraits< Inverse<BlockT *> >::child_end(Head);
         PI != PE; ++PI) {
      BlockT *Pred = *PI;
      assert(Pred);
      if (isBackedge(Pred, Head))
        BackFreq += getEdgeFreq(Pred, Head);
    }

    // The cyclic probability is freq(BackEdges) / freq(Head), where freq(Head)
    // only counts edges entering the loop, not the loop backedges.
    // The probability of leaving the loop on each iteration is:
    //
    //   ExitProb = 1 - CyclicProb
    //
    // The Expected number of loop iterations is:
    //
    //   Iterations = 1 / ExitProb
    //
    uint64_t D = std::max(getBlockFreq(Head).getFrequency(), UINT64_C(1));
    uint64_t N = std::max(BackFreq.getFrequency(), UINT64_C(1));
    if (N < D)
      N = D - N;
    else
      // We'd expect N < D, but rounding and saturation means that can't be
      // guaranteed.
      N = 1;

    // Now ExitProb = N / D, make sure it fits in an i32/i32 fraction.
    assert(N <= D);
    if (D > UINT32_MAX) {
      unsigned Shift = 32 - countLeadingZeros(D);
      D >>= Shift;
      N >>= Shift;
      if (N == 0)
        N = 1;
    }
    BranchProbability LEP = BranchProbability(N, D);
    LoopExitProb.insert(std::make_pair(Head, LEP));
    DEBUG(dbgs() << "LoopExitProb[" << getBlockName(Head) << "] = " << LEP
          << " from 1 - ";
          printBlockFreq(dbgs(), BackFreq) << " / ";
          printBlockFreq(dbgs(), getBlockFreq(Head)) << ".\n");
  }

  friend class BlockFrequencyInfo;
  friend class MachineBlockFrequencyInfo;

  BlockFrequencyInfoImpl() { }

  void doFunction(FunctionT *fn, BlockProbInfoT *bpi) {
    Fn = fn;
    BPI = bpi;

    // Clear everything.
    RPO.clear();
    POT.clear();
    LoopExitProb.clear();
    Freqs.clear();

    BlockT *EntryBlock = fn->begin();

    std::copy(po_begin(EntryBlock), po_end(EntryBlock), std::back_inserter(POT));

    unsigned RPOidx = 0;
    for (rpot_iterator I = rpot_begin(), E = rpot_end(); I != E; ++I) {
      BlockT *BB = *I;
      RPO[BB] = ++RPOidx;
      DEBUG(dbgs() << "RPO[" << getBlockName(BB) << "] = " << RPO[BB] << "\n");
    }

    // Travel over all blocks in postorder.
    for (pot_iterator I = pot_begin(), E = pot_end(); I != E; ++I) {
      BlockT *BB = *I;
      BlockT *LastTail = 0;
      DEBUG(dbgs() << "POT: " << getBlockName(BB) << "\n");

      for (typename GT::ChildIteratorType
           PI = GraphTraits< Inverse<BlockT *> >::child_begin(BB),
           PE = GraphTraits< Inverse<BlockT *> >::child_end(BB);
           PI != PE; ++PI) {

        BlockT *Pred = *PI;
        if (isBackedge(Pred, BB) && (!LastTail || RPO[Pred] > RPO[LastTail]))
          LastTail = Pred;
      }

      if (LastTail)
        doLoop(BB, LastTail);
    }

    // At the end assume the whole function as a loop, and travel over it once
    // again.
    doLoop(*(rpot_begin()), *(pot_begin()));
  }

public:

  uint64_t getEntryFreq() { return EntryFreq; }

  /// getBlockFreq - Return block frequency. Return 0 if we don't have it.
  BlockFrequency getBlockFreq(const BlockT *BB) const {
    typename DenseMap<const BlockT *, BlockFrequency>::const_iterator
      I = Freqs.find(BB);
    if (I != Freqs.end())
      return I->second;
    return 0;
  }

  void print(raw_ostream &OS) const {
    OS << "\n\n---- Block Freqs ----\n";
    for (typename FunctionT::iterator I = Fn->begin(), E = Fn->end(); I != E;) {
      BlockT *BB = I++;
      OS << " " << getBlockName(BB) << " = ";
      printBlockFreq(OS, getBlockFreq(BB)) << "\n";

      for (typename GraphTraits<BlockT *>::ChildIteratorType
           SI = GraphTraits<BlockT *>::child_begin(BB),
           SE = GraphTraits<BlockT *>::child_end(BB); SI != SE; ++SI) {
        BlockT *Succ = *SI;
        OS << "  " << getBlockName(BB) << " -> " << getBlockName(Succ)
           << " = "; printBlockFreq(OS, getEdgeFreq(BB, Succ)) << "\n";
      }
    }
  }

  void dump() const {
    print(dbgs());
  }

  // Utility method that looks up the block frequency associated with BB and
  // prints it to OS.
  raw_ostream &printBlockFreq(raw_ostream &OS,
                              const BlockT *BB) {
    return printBlockFreq(OS, getBlockFreq(BB));
  }

  raw_ostream &printBlockFreq(raw_ostream &OS,
                              const BlockFrequency &Freq) const {
    // Convert fixed-point number to decimal.
    uint64_t Frequency = Freq.getFrequency();
    OS << Frequency / EntryFreq << ".";
    uint64_t Rem = Frequency % EntryFreq;
    uint64_t Eps = 1;
    do {
      Rem *= 10;
      Eps *= 10;
      OS << Rem / EntryFreq;
      Rem = Rem % EntryFreq;
    } while (Rem >= Eps/2);
    return OS;
  }

};

}

#endif
