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
                                       bool IncludeColdCalls,
                                       bool UseFunctionHotSize,
                                       bool UseEdgeCounts) {
  NamedRegionTimer T1("Callgraph construction", "CG breakdown", opts::TimeOpts);
  BinaryFunctionCallGraph Cg;

  // Add call graph nodes.
  auto lookupNode = [&](BinaryFunction *Function) {
    const auto Id = Cg.maybeGetNodeId(Function);
    if (Id == CallGraph::InvalidId) {
      // It's ok to use the hot size here when the function is split.  This is
      // because emitFunctions will emit the hot part first in the order that is
      // computed by ReorderFunctions.  The cold part will be emitted with the
      // rest of the cold functions and code.
      const auto Size = UseFunctionHotSize && Function->isSplit()
        ? Function->estimateHotSize()
        : Function->estimateSize();
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
  uint64_t TotalCalls = 0;
  for (auto &It : BFs) {
    auto *Function = &It.second;

    if(Filter(*Function)) {
      continue;
    }

    auto BranchDataOrErr = BC.DR.getFuncBranchData(Function->getNames());
    const auto SrcId = lookupNode(Function);
    uint64_t Offset = Function->getAddress();

    auto recordCall = [&](const MCSymbol *DestSymbol, const uint64_t Count) {
      if (auto *DstFunc = BC.getFunctionForSymbol(DestSymbol)) {
        const auto DstId = lookupNode(DstFunc);
        const auto AvgDelta = !UseEdgeCounts ? Offset - DstFunc->getAddress() : 0;
        Cg.incArcWeight(SrcId, DstId, Count, AvgDelta);
        DEBUG(
          if (opts::Verbosity > 1) {
            dbgs() << "BOLT-DEBUG: buildCallGraph: call " << *Function
                   << " -> " << *DstFunc << " @ " << Offset << "\n";
          });
        return true;
      }
      return false;
    };

    for (auto *BB : Function->layout()) {
      // Don't count calls from cold blocks
      if (BB->isCold() && !IncludeColdCalls)
        continue;

      for (auto &Inst : *BB) {
        // Find call instructions and extract target symbols from each one.
        if (!BC.MIA->isCall(Inst))
          continue;

        ++TotalCalls;
        if (const auto *DstSym = BC.MIA->getTargetSymbol(Inst)) {
          // For direct calls, just use the BB execution count.
          const auto Count = UseEdgeCounts && BB->hasProfile()
                           ? BB->getExecutionCount() : 1;
          if (!recordCall(DstSym, Count))
            ++NotProcessed;
        } else if (BC.MIA->hasAnnotation(Inst, "EdgeCountData")) {
          // For indirect calls and jump tables, use branch data.
          if (!BranchDataOrErr) {
            ++NotProcessed;
            continue;
          }
          const FuncBranchData &BranchData = BranchDataOrErr.get();
          const auto DataOffset =
            BC.MIA->getAnnotationAs<uint64_t>(Inst, "EdgeCountData");

          for (const auto &BI : BranchData.getBranchRange(DataOffset)) {
            // Count each target as a separate call.
            ++TotalCalls;

            if (!BI.To.IsSymbol) {
              ++NotProcessed;
              continue;
            }

            auto Itr = BC.GlobalSymbols.find(BI.To.Name);
            if (Itr == BC.GlobalSymbols.end()) {
              ++NotProcessed;
              continue;
            }

            const auto *DstSym =
              BC.getOrCreateGlobalSymbol(Itr->second, "FUNCat");

            if (!recordCall(DstSym, UseEdgeCounts ? BI.Branches : 1))
              ++NotProcessed;
          }
        }

        if (!UseEdgeCounts) {
          Offset += BC.computeCodeSize(&Inst, &Inst + 1);
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
    outs() << format("BOLT-INFO: buildCallGraph: %u nodes, density = %.6lf, "
                     "%u callsites not processed out of %u.\n",
                     Cg.numNodes(), Cg.density(), NotProcessed, TotalCalls);
  }

  return Cg;
}

}
}
