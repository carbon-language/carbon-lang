//===- ScheduleDAGILP.h - ILP metric for ScheduleDAGInstrs ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of an ILP metric for machine level instruction scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAGILP_H
#define LLVM_CODEGEN_SCHEDULEDAGILP_H

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class raw_ostream;
class IntEqClasses;
class ScheduleDAGInstrs;
class SUnit;

/// \brief Represent the ILP of the subDAG rooted at a DAG node.
///
/// When computed using bottom-up DFS, this metric assumes that the DAG is a
/// forest of trees with roots at the bottom of the schedule branching upward.
struct ILPValue {
  unsigned InstrCount;
  /// Length may either correspond to depth or height, depending on direction,
  /// and cycles or nodes depending on context.
  unsigned Length;

  ILPValue(unsigned count, unsigned length):
    InstrCount(count), Length(length) {}

  // Order by the ILP metric's value.
  bool operator<(ILPValue RHS) const {
    return (uint64_t)InstrCount * RHS.Length
      < (uint64_t)Length * RHS.InstrCount;
  }
  bool operator>(ILPValue RHS) const {
    return RHS < *this;
  }
  bool operator<=(ILPValue RHS) const {
    return (uint64_t)InstrCount * RHS.Length
      <= (uint64_t)Length * RHS.InstrCount;
  }
  bool operator>=(ILPValue RHS) const {
    return RHS <= *this;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &OS) const;

  void dump() const;
#endif
};

/// \brief Compute the values of each DAG node for various metrics during DFS.
///
/// ILPValues summarize the DAG subtree rooted at each node up to
/// SubtreeLimit. ILPValues are also valid for interior nodes of a subtree, not
/// just the root.
class SchedDFSResult {
  friend class SchedDFSImpl;

  /// \brief Per-SUnit data computed during DFS for various metrics.
  struct NodeData {
    unsigned InstrCount;
    unsigned SubtreeID;

    NodeData(): InstrCount(0), SubtreeID(0) {}
  };

  /// \brief Record a connection between subtrees and the connection level.
  struct Connection {
    unsigned TreeID;
    unsigned Level;

    Connection(unsigned tree, unsigned level): TreeID(tree), Level(level) {}
  };

  bool IsBottomUp;
  unsigned SubtreeLimit;
  /// DFS results for each SUnit in this DAG.
  std::vector<NodeData> DFSData;

  // For each subtree discovered during DFS, record its connections to other
  // subtrees.
  std::vector<SmallVector<Connection, 4> > SubtreeConnections;

  /// Cache the current connection level of each subtree.
  /// This mutable array is updated during scheduling.
  std::vector<unsigned> SubtreeConnectLevels;

public:
  SchedDFSResult(bool IsBU, unsigned lim)
    : IsBottomUp(IsBU), SubtreeLimit(lim) {}

  /// \brief Clear the results.
  void clear() {
    DFSData.clear();
    SubtreeConnections.clear();
    SubtreeConnectLevels.clear();
  }

  /// \brief Initialize the result data with the size of the DAG.
  void resize(unsigned NumSUnits) {
    DFSData.resize(NumSUnits);
  }

  /// \brief Compute various metrics for the DAG with given roots.
  void compute(ArrayRef<SUnit *> Roots);

  /// \brief Get the ILP value for a DAG node.
  ///
  /// A leaf node has an ILP of 1/1.
  ILPValue getILP(const SUnit *SU) {
    return ILPValue(DFSData[SU->NodeNum].InstrCount, 1 + SU->getDepth());
  }

  /// \brief The number of subtrees detected in this DAG.
  unsigned getNumSubtrees() const { return SubtreeConnectLevels.size(); }

  /// \brief Get the ID of the subtree the given DAG node belongs to.
  unsigned getSubtreeID(const SUnit *SU) {
    return DFSData[SU->NodeNum].SubtreeID;
  }

  /// \brief Get the connection level of a subtree.
  ///
  /// For bottom-up trees, the connection level is the latency depth (in cycles)
  /// of the deepest connection to another subtree.
  unsigned getSubtreeLevel(unsigned SubtreeID) {
    return SubtreeConnectLevels[SubtreeID];
  }

  /// \brief Scheduler callback to update SubtreeConnectLevels when a tree is
  /// initially scheduled.
  void scheduleTree(unsigned SubtreeID);
};

raw_ostream &operator<<(raw_ostream &OS, const ILPValue &Val);

} // namespace llvm

#endif
