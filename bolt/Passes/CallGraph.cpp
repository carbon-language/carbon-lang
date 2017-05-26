//===--- Passes/CallGraph.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "CallGraph.h"
#include "BinaryFunction.h"
#include "BinaryContext.h"

#define DEBUG_TYPE "callgraph"

#if defined(__x86_64__) && !defined(_MSC_VER)
#  if (!defined USE_SSECRC)
#    define USE_SSECRC
#  endif
#else
#  undef USE_SSECRC
#endif

namespace {

inline size_t hash_int64_fallback(int64_t key) {
  // "64 bit Mix Functions", from Thomas Wang's "Integer Hash Function."
  // http://www.concentric.net/~ttwang/tech/inthash.htm
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ ((unsigned long long)key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ ((unsigned long long)key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ ((unsigned long long)key >> 28);
  return static_cast<size_t>(static_cast<uint32_t>(key));
}

inline size_t hash_int64(int64_t k) {
#if defined(USE_SSECRC) && defined(__SSE4_2__)
  size_t h = 0;
  __asm("crc32q %1, %0\n" : "+r"(h) : "rm"(k));
  return h;
#else
  return hash_int64_fallback(k);
#endif
}
  
inline size_t hash_int64_pair(int64_t k1, int64_t k2) {
#if defined(USE_SSECRC) && defined(__SSE4_2__)
  // crc32 is commutative, so we need to perturb k1 so that (k1, k2) hashes
  // differently from (k2, k1).
  k1 += k1;
  __asm("crc32q %1, %0\n" : "+r" (k1) : "rm"(k2));
  return k1;
#else
  return (hash_int64(k1) << 1) ^ hash_int64(k2);
#endif
}
  
}

namespace llvm {
namespace bolt {

int64_t CallGraph::Arc::Hash::operator()(const Arc &Arc) const {
#ifdef USE_STD_HASH
  std::hash<int64_t> Hasher;
  return hashCombine(Hasher(Arc.Src), Arc.Dst);
#else
  return hash_int64_pair(int64_t(Arc.Src), int64_t(Arc.Dst));
#endif
}

CallGraph buildCallGraph(BinaryContext &BC,
                         std::map<uint64_t, BinaryFunction> &BFs,
                         std::function<bool (const BinaryFunction &BF)> Filter,
                         bool IncludeColdCalls,
                         bool UseFunctionHotSize,
                         bool UseEdgeCounts) {
  CallGraph Cg;

  // Add call graph nodes.
  auto lookupNode = [&](BinaryFunction *Function) {
    auto It = Cg.FuncToNodeId.find(Function);
    if (It == Cg.FuncToNodeId.end()) {
      // It's ok to use the hot size here when the function is split.  This is
      // because emitFunctions will emit the hot part first in the order that is
      // computed by ReorderFunctions.  The cold part will be emitted with the
      // rest of the cold functions and code.
      const auto Size = UseFunctionHotSize && Function->isSplit()
        ? Function->estimateHotSize()
        : Function->estimateSize();
      const auto Id = Cg.addNode(Size);
      assert(size_t(Id) == Cg.Funcs.size());
      Cg.Funcs.push_back(Function);
      Cg.FuncToNodeId[Function] = Id;
      // NOTE: for functions without a profile, we set the number of samples
      // to zero.  This will keep these functions from appearing in the hot
      // section.  This is a little weird because we wouldn't be trying to
      // create a node for a function unless it was the target of a call from
      // a hot block.  The alternative would be to set the count to one or
      // accumulate the number of calls from the callsite into the function
      // samples.  Results from perfomance testing seem to favor the zero
      // count though, so I'm leaving it this way for now.
      Cg.Nodes[Id].Samples = Function->hasProfile() ? Function->getExecutionCount() : 0;
      assert(Cg.Funcs[Id] == Function);
      return Id;
    } else {
      return It->second;
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
        auto &A = Cg.incArcWeight(SrcId, DstId, Count);
        if (!UseEdgeCounts) {
          A.AvgCallOffset += (Offset - DstFunc->getAddress());
        }
        DEBUG(dbgs() << "BOLT-DEBUG: buildCallGraph: call " << *Function
              << " -> " << *DstFunc << " @ " << Offset << "\n");
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
          if(!BranchDataOrErr) {
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

  outs() << "BOLT-WARNING: buildCallGraph: " << NotProcessed
         << " callsites not processed out of " << TotalCalls << "\n";

  return Cg;
}

CallGraph::NodeId CallGraph::addNode(uint32_t Size, uint32_t Samples) {
  auto Id = Nodes.size();
  Nodes.emplace_back(Size, Samples);
  return Id;
}

const CallGraph::Arc &CallGraph::incArcWeight(NodeId Src, NodeId Dst, double W) {
  auto Res = Arcs.emplace(Src, Dst, W);
  if (!Res.second) {
    Res.first->Weight += W;
    return *Res.first;
  }
  Nodes[Src].Succs.push_back(Dst);
  Nodes[Dst].Preds.push_back(Src);
  return *Res.first;
}

std::deque<BinaryFunction *> CallGraph::buildTraversalOrder() {
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
    for (const auto Callee : Nodes[FuncId].Succs) {
      if (NodeStatus[Callee] == VISITING || NodeStatus[Callee] == VISITED)
        continue;
      Worklist.push(Callee);
    }
  }

  return TopologicalOrder;
}

}
}
