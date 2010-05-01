//===---- ScheduleDAGSDNodes.h - SDNode Scheduling --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAGSDNodes class, which implements
// scheduling for an SDNode-based dependency graph.
//
//===----------------------------------------------------------------------===//

#ifndef SCHEDULEDAGSDNODES_H
#define SCHEDULEDAGSDNODES_H

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {
  /// ScheduleDAGSDNodes - A ScheduleDAG for scheduling SDNode-based DAGs.
  /// 
  /// Edges between SUnits are initially based on edges in the SelectionDAG,
  /// and additional edges can be added by the schedulers as heuristics.
  /// SDNodes such as Constants, Registers, and a few others that are not
  /// interesting to schedulers are not allocated SUnits.
  ///
  /// SDNodes with MVT::Flag operands are grouped along with the flagged
  /// nodes into a single SUnit so that they are scheduled together.
  ///
  /// SDNode-based scheduling graphs do not use SDep::Anti or SDep::Output
  /// edges.  Physical register dependence information is not carried in
  /// the DAG and must be handled explicitly by schedulers.
  ///
  class ScheduleDAGSDNodes : public ScheduleDAG {
  public:
    SelectionDAG *DAG;                    // DAG of the current basic block

    explicit ScheduleDAGSDNodes(MachineFunction &mf);

    virtual ~ScheduleDAGSDNodes() {}

    /// Run - perform scheduling.
    ///
    void Run(SelectionDAG *dag, MachineBasicBlock *bb,
             MachineBasicBlock::iterator insertPos);

    /// isPassiveNode - Return true if the node is a non-scheduled leaf.
    ///
    static bool isPassiveNode(SDNode *Node) {
      if (isa<ConstantSDNode>(Node))       return true;
      if (isa<ConstantFPSDNode>(Node))     return true;
      if (isa<RegisterSDNode>(Node))       return true;
      if (isa<GlobalAddressSDNode>(Node))  return true;
      if (isa<BasicBlockSDNode>(Node))     return true;
      if (isa<FrameIndexSDNode>(Node))     return true;
      if (isa<ConstantPoolSDNode>(Node))   return true;
      if (isa<JumpTableSDNode>(Node))      return true;
      if (isa<ExternalSymbolSDNode>(Node)) return true;
      if (isa<BlockAddressSDNode>(Node))   return true;
      if (Node->getOpcode() == ISD::EntryToken ||
          isa<MDNodeSDNode>(Node)) return true;
      return false;
    }

    /// NewSUnit - Creates a new SUnit and return a ptr to it.
    ///
    SUnit *NewSUnit(SDNode *N) {
#ifndef NDEBUG
      const SUnit *Addr = 0;
      if (!SUnits.empty())
        Addr = &SUnits[0];
#endif
      SUnits.push_back(SUnit(N, (unsigned)SUnits.size()));
      assert((Addr == 0 || Addr == &SUnits[0]) &&
             "SUnits std::vector reallocated on the fly!");
      SUnits.back().OrigNode = &SUnits.back();
      return &SUnits.back();
    }

    /// Clone - Creates a clone of the specified SUnit. It does not copy the
    /// predecessors / successors info nor the temporary scheduling states.
    ///
    SUnit *Clone(SUnit *N);
    
    /// BuildSchedGraph - Build the SUnit graph from the selection dag that we
    /// are input.  This SUnit graph is similar to the SelectionDAG, but
    /// excludes nodes that aren't interesting to scheduling, and represents
    /// flagged together nodes with a single SUnit.
    virtual void BuildSchedGraph(AliasAnalysis *AA);

    /// ComputeLatency - Compute node latency.
    ///
    virtual void ComputeLatency(SUnit *SU);

    virtual MachineBasicBlock *EmitSchedule();

    /// Schedule - Order nodes according to selected style, filling
    /// in the Sequence member.
    ///
    virtual void Schedule() = 0;

    virtual void dumpNode(const SUnit *SU) const;

    virtual std::string getGraphNodeLabel(const SUnit *SU) const;

    virtual void getCustomGraphFeatures(GraphWriter<ScheduleDAG*> &GW) const;

  private:
    /// ClusterNeighboringLoads - Cluster loads from "near" addresses into
    /// combined SUnits.
    void ClusterNeighboringLoads();

    /// BuildSchedUnits, AddSchedEdges - Helper functions for BuildSchedGraph.
    void BuildSchedUnits();
    void AddSchedEdges();
  };
}

#endif
