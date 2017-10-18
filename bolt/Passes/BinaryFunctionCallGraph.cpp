//===--- Passes/BinaryFunctionCallGraph.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryFunctionCallGraph.h"
#include "BinaryFunction.h"
#include "BinaryContext.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/Timer.h"

#define DEBUG_TYPE "callgraph"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
extern llvm::cl::opt<unsigned> Verbosity;
}

namespace llvm {
namespace bolt {

CallGraph::NodeId BinaryFunctionCallGraph::addNode(BinaryFunction *BF,
                                                   uint32_t Size,
                                                   uint64_t Samples) {
  auto Id = CallGraph::addNode(Size, Samples);
  assert(size_t(Id) == Funcs.size());
  Funcs.push_back(BF);
  FuncToNodeId[BF] = Id;
  assert(Funcs[Id] == BF);
  return Id;
}

std::deque<BinaryFunction *> BinaryFunctionCallGraph::buildTraversalOrder() {
  NamedRegionTimer T1("Build cg traversal order", "CG breakdown",
                      opts::TimeOpts);
  std::deque<BinaryFunction *> TopologicalOrder;
  enum NodeStatus { NEW, VISITING, VISITED };
  std::vector<NodeStatus> NodeStatus(Funcs.size());
  std::stack<NodeId> Worklist;

  for (auto *Func : Funcs) {
    const auto Id = FuncToNodeId.at(Func);
    Worklist.push(Id);
    NodeStatus[Id] = NEW;
  }

  while (!Worklist.empty()) {
    const auto FuncId = Worklist.top();
    Worklist.pop();

    if (NodeStatus[FuncId] == VISITED)
      continue;

    if (NodeStatus[FuncId] == VISITING) {
      TopologicalOrder.push_back(Funcs[FuncId]);
      NodeStatus[FuncId] = VISITED;
      continue;
    }

    assert(NodeStatus[FuncId] == NEW);
    NodeStatus[FuncId] = VISITING;
    Worklist.push(FuncId);
    for (const auto Callee : successors(FuncId)) {
      if (NodeStatus[Callee] == VISITING || NodeStatus[Callee] == VISITED)
        continue;
      Worklist.push(Callee);
    }
  }

  return TopologicalOrder;
}

BinaryFunctionCallGraph buildCallGraph(BinaryContext &BC,
                                       std::map<uint64_t, BinaryFunction> &BFs,
                                       CgFilterFunction Filter,
                                       bool CgFromPerfData,
                                       bool IncludeColdCalls,
                                       bool UseFunctionHotSize,
                                       bool UseSplitHotSize,
                                       bool UseEdgeCounts,
                                       bool IgnoreRecursiveCalls) {
  NamedRegionTimer T1("Callgraph construction", "CG breakdown", opts::TimeOpts);
  BinaryFunctionCallGraph Cg;
  static constexpr auto COUNT_NO_PROFILE = BinaryBasicBlock::COUNT_NO_PROFILE;

  // Compute function size
  auto functionSize = [&](const BinaryFunction *Function) {
    return UseFunctionHotSize && Function->isSplit()
      ? Function->estimateHotSize(UseSplitHotSize)
      : Function->estimateSize();
  };

  // Add call graph nodes.
  auto lookupNode = [&](BinaryFunction *Function) {
    const auto Id = Cg.maybeGetNodeId(Function);
    if (Id == CallGraph::InvalidId) {
      // It's ok to use the hot size here when the function is split.  This is
      // because emitFunctions will emit the hot part first in the order that is
      // computed by ReorderFunctions.  The cold part will be emitted with the
      // rest of the cold functions and code.
      const auto Size = functionSize(Function);
      // NOTE: for functions without a profile, we set the number of samples
      // to zero.  This will keep these functions from appearing in the hot
      // section.  This is a little weird because we wouldn't be trying to
      // create a node for a function unless it was the target of a call from
      // a hot block.  The alternative would be to set the count to one or
      // accumulate the number of calls from the callsite into the function
      // samples.  Results from perfomance testing seem to favor the zero
      // count though, so I'm leaving it this way for now.
      const auto Samples =
        Function->hasProfile() ? Function->getExecutionCount() : 0;
      return Cg.addNode(Function, Size, Samples);
    } else {
      return Id;
    }
  };

  // Add call graph edges.
  uint64_t NotProcessed = 0;
  uint64_t TotalCallsites = 0;
  uint64_t NoProfileCallsites = 0;
  uint64_t NumFallbacks = 0;
  uint64_t RecursiveCallsites = 0;
  for (auto &It : BFs) {
    auto *Function = &It.second;

    if (Filter(*Function)) {
      continue;
    }

    const auto *BranchData = Function->getBranchData();
    const auto SrcId = lookupNode(Function);
    // Offset of the current basic block from the beginning of the function
    uint64_t Offset = 0;

    auto recordCall = [&](const MCSymbol *DestSymbol, const uint64_t Count) {
      if (auto *DstFunc =
          DestSymbol ? BC.getFunctionForSymbol(DestSymbol) : nullptr) {
        if (DstFunc == Function) {
          DEBUG(dbgs() << "BOLT-INFO: recursive call detected in "
                       << *DstFunc << "\n");
          ++RecursiveCallsites;
          if (IgnoreRecursiveCalls)
            return false;
        }
        const auto DstId = lookupNode(DstFunc);
        const bool IsValidCount = Count != COUNT_NO_PROFILE;
        const auto AdjCount = UseEdgeCounts && IsValidCount ? Count : 1;
        if (!IsValidCount)
          ++NoProfileCallsites;
        Cg.incArcWeight(SrcId, DstId, AdjCount, Offset);
        DEBUG(
          if (opts::Verbosity > 1) {
            dbgs() << "BOLT-DEBUG: buildCallGraph: call " << *Function
                   << " -> " << *DstFunc << " @ " << Offset << "\n";
          });
        return true;
      }
      
      return false;
    };

    auto getCallInfoFromBranchData = [&](const BranchInfo &BI, bool IsStale) {
      MCSymbol *DstSym = nullptr;
      uint64_t Count;
      if (BI.To.IsSymbol && (DstSym = BC.getGlobalSymbolByName(BI.To.Name))) {
        Count = BI.Branches;
      } else {
        Count = COUNT_NO_PROFILE;
      }
      // If we are using the perf data for a stale function we need to filter
      // out data which comes from branches.  We'll assume that the To offset
      // is non-zero for branches.
      if (IsStale && BI.To.Offset != 0 &&
          (!DstSym || Function == BC.getFunctionForSymbol(DstSym))) {
        DstSym = nullptr;
        Count = COUNT_NO_PROFILE;
      }
      return std::make_pair(DstSym, Count);
    };

    // Get pairs of (symbol, count) for each target at this callsite.
    // If the call is to an unknown function the symbol will be nullptr.
    // If there is no profiling data the count will be COUNT_NO_PROFILE.
    auto getCallInfo = [&](const BinaryBasicBlock *BB, const MCInst &Inst) {
      std::vector<std::pair<const MCSymbol *, uint64_t>> Counts;
      const auto *DstSym = BC.MIA->getTargetSymbol(Inst);

      // If this is an indirect call use perf data directly.
      if (!DstSym && BranchData &&
          BC.MIA->hasAnnotation(Inst, "Offset")) {
        const auto InstrOffset =
          BC.MIA->getAnnotationAs<uint64_t>(Inst, "Offset");
        for (const auto &BI : BranchData->getBranchRange(InstrOffset)) {
          Counts.push_back(getCallInfoFromBranchData(BI, false));
        }
      } else {
        const auto Count = BB->getExecutionCount();
        Counts.push_back(std::make_pair(DstSym, Count));
      }

      return Counts;
    };

    // If the function has an invalid profile, try to use the perf data
    // directly (if requested).  If there is no perf data for this function,
    // fall back to the CFG walker which attempts to handle missing data.
    if (!Function->hasValidProfile() && CgFromPerfData && BranchData) {
      DEBUG(dbgs() << "BOLT-DEBUG: buildCallGraph: Falling back to perf data"
                   << " for " << *Function << "\n");
      ++NumFallbacks;
      const auto Size = functionSize(Function);
      for (const auto &BI : BranchData->Data) {
        Offset = BI.From.Offset;
        // The computed offset may exceed the hot part of the function; hence,
        // bound it the size
        if (Offset > Size)
          Offset = Size;

        const auto CI = getCallInfoFromBranchData(BI, true);
        if (!CI.first && CI.second == COUNT_NO_PROFILE) // probably a branch
          continue;
        ++TotalCallsites;
        if (!recordCall(CI.first, CI.second)) {
          ++NotProcessed;
        }
      }
    } else {
      for (auto *BB : Function->layout()) {
        // Don't count calls from cold blocks unless requested.
        if (BB->isCold() && !IncludeColdCalls)
          continue;

        // Determine whether the block is included in Function's (hot) size
        // See BinaryFunction::estimateHotSize
        bool BBIncludedInFunctionSize = false;
        if (UseFunctionHotSize && Function->isSplit()) {
          if (UseSplitHotSize)
            BBIncludedInFunctionSize = !BB->isCold();
          else
            BBIncludedInFunctionSize = BB->getKnownExecutionCount() != 0;
        } else {
          BBIncludedInFunctionSize = true;
        }

        for (auto &Inst : *BB) {
          // Find call instructions and extract target symbols from each one.
          if (BC.MIA->isCall(Inst)) {
            const auto CallInfo = getCallInfo(BB, Inst);

            if (!CallInfo.empty()) {
              for (const auto &CI : CallInfo) {
                ++TotalCallsites;
                if (!recordCall(CI.first, CI.second))
                  ++NotProcessed;
              }
            } else {
              ++TotalCallsites;
              ++NotProcessed;
            }
          }
          // Increase Offset if needed
          if (BBIncludedInFunctionSize) {
            Offset += BC.computeCodeSize(&Inst, &Inst + 1);
          }
        }
      }
    }
  }

#ifndef NDEBUG
  bool PrintInfo = DebugFlag && isCurrentDebugType("callgraph");
#else
  bool PrintInfo = false;
#endif
  if (PrintInfo || opts::Verbosity > 0) {
    outs() << format("BOLT-INFO: buildCallGraph: %u nodes, %u callsites "
                     "(%u recursive), density = %.6lf, %u callsites not "
                     "processed, %u callsites with invalid profile, "
                     "used perf data for %u stale functions.\n",
                     Cg.numNodes(), TotalCallsites, RecursiveCallsites,
                     Cg.density(), NotProcessed, NoProfileCallsites,
                     NumFallbacks);
  }

  return Cg;
}

}
}
