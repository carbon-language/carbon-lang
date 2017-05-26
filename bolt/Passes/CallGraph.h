//===--- Passes/CallGraph.h -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_CALLGRAPH_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_CALLGRAPH_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <functional>
#include <map>
#include <deque>

namespace llvm {
namespace bolt {

class BinaryFunction;
class BinaryContext;

// TODO: find better place for this
inline int64_t hashCombine(const int64_t Seed, const int64_t Val) {
  std::hash<int64_t> Hasher;
  return Seed ^ (Hasher(Val) + 0x9e3779b9 + (Seed << 6) + (Seed >> 2));
}

/// A call graph class.
class CallGraph {
public:
  using NodeId = size_t;
  static constexpr NodeId InvalidId = -1;

  class Arc {
  public:
    struct Hash {
      int64_t operator()(const Arc &Arc) const;
    };

    Arc(NodeId S, NodeId D, double W = 0)
      : Src(S)
      , Dst(D)
      , Weight(W)
    {}
    Arc(const Arc&) = delete;

    friend bool operator==(const Arc &Lhs, const Arc &Rhs) {
      return Lhs.Src == Rhs.Src && Lhs.Dst == Rhs.Dst;
    }

    const NodeId Src;
    const NodeId Dst;
    mutable double Weight;
    mutable double NormalizedWeight{0};
    mutable double AvgCallOffset{0};
  };

  class Node {
  public:
    explicit Node(uint32_t Size, uint32_t Samples = 0)
      : Size(Size), Samples(Samples)
    {}

    uint32_t Size;
    uint32_t Samples;

    // preds and succs contain no duplicate elements and self arcs are not allowed
    std::vector<NodeId> Preds;
    std::vector<NodeId> Succs;
  };

  NodeId addNode(uint32_t Size, uint32_t Samples = 0);
  const Arc &incArcWeight(NodeId Src, NodeId Dst, double W = 1.0);

  /// Compute a DFS traversal of the call graph.
  std::deque<BinaryFunction *> buildTraversalOrder();

  std::vector<Node> Nodes;
  std::unordered_set<Arc, Arc::Hash> Arcs;
  std::vector<BinaryFunction *> Funcs;
  std::unordered_map<const BinaryFunction *, NodeId> FuncToNodeId;
};

inline bool NoFilter(const BinaryFunction &) { return false; }

/// Builds a call graph from the map of BinaryFunctions provided in BFs.
/// The arguments control how the graph is constructed.
/// Filter is called on each function, any function that it returns true for
/// is omitted from the graph.
/// If IncludeColdCalls is true, then calls from cold BBs are considered for the
/// graph, otherwise they are ignored.
/// UseFunctionHotSize controls whether the hot size of a function is used when
/// filling in the Size attribute of new Nodes.
/// UseEdgeCounts is used to control if the AvgCallOffset attribute on Arcs is
/// computed using the offsets of call instructions.
CallGraph buildCallGraph(BinaryContext &BC,
                         std::map<uint64_t, BinaryFunction> &BFs,
                         std::function<bool (const BinaryFunction &BF)> Filter = NoFilter,
                         bool IncludeColdCalls = true,
                         bool UseFunctionHotSize = false,
                         bool UseEdgeCounts = false);

} // namespace bolt
} // namespace llvm

#endif
