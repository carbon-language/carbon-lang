//===------- llvm/CodeGen/ScheduleDAG.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAG class, which is used as the common
// base class for SelectionDAG-based instruction scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAG_H
#define LLVM_CODEGEN_SCHEDULEDAG_H

#include "llvm/CodeGen/SelectionDAG.h"

#include <set>

namespace llvm {
  struct InstrStage;
  class MachineConstantPool;
  class MachineDebugInfo;
  class MachineInstr;
  class MRegisterInfo;
  class SelectionDAG;
  class SelectionDAGISel;
  class SSARegMap;
  class TargetInstrInfo;
  class TargetInstrDescriptor;
  class TargetMachine;

  /// HazardRecognizer - This determines whether or not an instruction can be
  /// issued this cycle, and whether or not a noop needs to be inserted to handle
  /// the hazard.
  class HazardRecognizer {
  public:
    virtual ~HazardRecognizer();
    
    enum HazardType {
      NoHazard,      // This instruction can be emitted at this cycle.
      Hazard,        // This instruction can't be emitted at this cycle.
      NoopHazard     // This instruction can't be emitted, and needs noops.
    };
    
    /// getHazardType - Return the hazard type of emitting this node.  There are
    /// three possible results.  Either:
    ///  * NoHazard: it is legal to issue this instruction on this cycle.
    ///  * Hazard: issuing this instruction would stall the machine.  If some
    ///     other instruction is available, issue it first.
    ///  * NoopHazard: issuing this instruction would break the program.  If
    ///     some other instruction can be issued, do so, otherwise issue a noop.
    virtual HazardType getHazardType(SDNode *Node) {
      return NoHazard;
    }
    
    /// EmitInstruction - This callback is invoked when an instruction is
    /// emitted, to advance the hazard state.
    virtual void EmitInstruction(SDNode *Node) {
    }
    
    /// AdvanceCycle - This callback is invoked when no instructions can be
    /// issued on this cycle without a hazard.  This should increment the
    /// internal state of the hazard recognizer so that previously "Hazard"
    /// instructions will now not be hazards.
    virtual void AdvanceCycle() {
    }
    
    /// EmitNoop - This callback is invoked when a noop was added to the
    /// instruction stream.
    virtual void EmitNoop() {
    }
  };
  
  /// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
  /// a group of nodes flagged together.
  struct SUnit {
    SDNode *Node;                       // Representative node.
    std::vector<SDNode*> FlaggedNodes;  // All nodes flagged to Node.
    
    // Preds/Succs - The SUnits before/after us in the graph.  The boolean value
    // is true if the edge is a token chain edge, false if it is a value edge. 
    std::set<std::pair<SUnit*,bool> > Preds;  // All sunit predecessors.
    std::set<std::pair<SUnit*,bool> > Succs;  // All sunit successors.

    short NumPreds;                     // # of preds.
    short NumSuccs;                     // # of sucss.
    short NumPredsLeft;                 // # of preds not scheduled.
    short NumSuccsLeft;                 // # of succs not scheduled.
    short NumChainPredsLeft;            // # of chain preds not scheduled.
    short NumChainSuccsLeft;            // # of chain succs not scheduled.
    bool isTwoAddress     : 1;          // Is a two-address instruction.
    bool isCommutable     : 1;          // Is a commutable instruction.
    bool isPending        : 1;          // True once pending.
    bool isAvailable      : 1;          // True once available.
    bool isScheduled      : 1;          // True once scheduled.
    unsigned short Latency;             // Node latency.
    unsigned CycleBound;                // Upper/lower cycle to be scheduled at.
    unsigned Cycle;                     // Once scheduled, the cycle of the op.
    unsigned Depth;                     // Node depth;
    unsigned Height;                    // Node height;
    unsigned NodeNum;                   // Entry # of node in the node vector.
    
    SUnit(SDNode *node, unsigned nodenum)
      : Node(node), NumPreds(0), NumSuccs(0), NumPredsLeft(0), NumSuccsLeft(0),
        NumChainPredsLeft(0), NumChainSuccsLeft(0),
        isTwoAddress(false), isCommutable(false),
        isPending(false), isAvailable(false), isScheduled(false),
        Latency(0), CycleBound(0), Cycle(0), Depth(0), Height(0),
        NodeNum(nodenum) {}
    
    void dump(const SelectionDAG *G) const;
    void dumpAll(const SelectionDAG *G) const;
  };

  //===--------------------------------------------------------------------===//
  /// SchedulingPriorityQueue - This interface is used to plug different
  /// priorities computation algorithms into the list scheduler. It implements
  /// the interface of a standard priority queue, where nodes are inserted in 
  /// arbitrary order and returned in priority order.  The computation of the
  /// priority and the representation of the queue are totally up to the
  /// implementation to decide.
  /// 
  class SchedulingPriorityQueue {
  public:
    virtual ~SchedulingPriorityQueue() {}
  
    virtual void initNodes(const std::vector<SUnit> &SUnits) = 0;
    virtual void releaseState() = 0;
  
    virtual bool empty() const = 0;
    virtual void push(SUnit *U) = 0;
  
    virtual void push_all(const std::vector<SUnit *> &Nodes) = 0;
    virtual SUnit *pop() = 0;

    /// ScheduledNode - As each node is scheduled, this method is invoked.  This
    /// allows the priority function to adjust the priority of node that have
    /// already been emitted.
    virtual void ScheduledNode(SUnit *Node) {}
  };

  class ScheduleDAG {
  public:
    SelectionDAG &DAG;                    // DAG of the current basic block
    MachineBasicBlock *BB;                // Current basic block
    const TargetMachine &TM;              // Target processor
    const TargetInstrInfo *TII;           // Target instruction information
    const MRegisterInfo *MRI;             // Target processor register info
    SSARegMap *RegMap;                    // Virtual/real register map
    MachineConstantPool *ConstPool;       // Target constant pool
    std::vector<SUnit*> Sequence;         // The schedule. Null SUnit*'s
                                          // represent noop instructions.
    std::map<SDNode*, SUnit*> SUnitMap;   // SDNode to SUnit mapping (n -> 1).
    std::vector<SUnit> SUnits;            // The scheduling units.
    std::set<SDNode*> CommuteSet;         // Nodes the should be commuted.

    ScheduleDAG(SelectionDAG &dag, MachineBasicBlock *bb,
                const TargetMachine &tm)
      : DAG(dag), BB(bb), TM(tm) {}

    virtual ~ScheduleDAG() {}

    /// Run - perform scheduling.
    ///
    MachineBasicBlock *Run();

    /// isPassiveNode - Return true if the node is a non-scheduled leaf.
    ///
    static bool isPassiveNode(SDNode *Node) {
      if (isa<ConstantSDNode>(Node))       return true;
      if (isa<RegisterSDNode>(Node))       return true;
      if (isa<GlobalAddressSDNode>(Node))  return true;
      if (isa<BasicBlockSDNode>(Node))     return true;
      if (isa<FrameIndexSDNode>(Node))     return true;
      if (isa<ConstantPoolSDNode>(Node))   return true;
      if (isa<JumpTableSDNode>(Node))      return true;
      if (isa<ExternalSymbolSDNode>(Node)) return true;
      return false;
    }

    /// NewSUnit - Creates a new SUnit and return a ptr to it.
    ///
    SUnit *NewSUnit(SDNode *N) {
      SUnits.push_back(SUnit(N, SUnits.size()));
      return &SUnits.back();
    }

    /// BuildSchedUnits - Build SUnits from the selection dag that we are input.
    /// This SUnit graph is similar to the SelectionDAG, but represents flagged
    /// together nodes with a single SUnit.
    void BuildSchedUnits();

    /// CalculateDepths, CalculateHeights - Calculate node depth / height.
    ///
    void CalculateDepths();
    void CalculateHeights();

    /// EmitNode - Generate machine code for an node and needed dependencies.
    /// VRBaseMap contains, for each already emitted node, the first virtual
    /// register number for the results of the node.
    ///
    void EmitNode(SDNode *Node, std::map<SDNode*, unsigned> &VRBaseMap);
    
    /// EmitNoop - Emit a noop instruction.
    ///
    void EmitNoop();
    
    void EmitSchedule();

    void dumpSchedule() const;

    /// Schedule - Order nodes according to selected style.
    ///
    virtual void Schedule() {}

  private:
    void AddOperand(MachineInstr *MI, SDOperand Op, unsigned IIOpNum,
                    const TargetInstrDescriptor *II,
                    std::map<SDNode*, unsigned> &VRBaseMap);
  };

  /// createBFS_DAGScheduler - This creates a simple breadth first instruction
  /// scheduler.
  ScheduleDAG *createBFS_DAGScheduler(SelectionDAGISel *IS,
                                      SelectionDAG *DAG,
                                      MachineBasicBlock *BB);
  
  /// createSimpleDAGScheduler - This creates a simple two pass instruction
  /// scheduler using instruction itinerary.
  ScheduleDAG* createSimpleDAGScheduler(SelectionDAGISel *IS,
                                        SelectionDAG *DAG,
                                        MachineBasicBlock *BB);

  /// createNoItinsDAGScheduler - This creates a simple two pass instruction
  /// scheduler without using instruction itinerary.
  ScheduleDAG* createNoItinsDAGScheduler(SelectionDAGISel *IS,
                                         SelectionDAG *DAG,
                                         MachineBasicBlock *BB);

  /// createBURRListDAGScheduler - This creates a bottom up register usage
  /// reduction list scheduler.
  ScheduleDAG* createBURRListDAGScheduler(SelectionDAGISel *IS,
                                          SelectionDAG *DAG,
                                          MachineBasicBlock *BB);
  
  /// createTDRRListDAGScheduler - This creates a top down register usage
  /// reduction list scheduler.
  ScheduleDAG* createTDRRListDAGScheduler(SelectionDAGISel *IS,
                                          SelectionDAG *DAG,
                                          MachineBasicBlock *BB);
  
  /// createTDListDAGScheduler - This creates a top-down list scheduler with
  /// a hazard recognizer.
  ScheduleDAG* createTDListDAGScheduler(SelectionDAGISel *IS,
                                        SelectionDAG *DAG,
                                        MachineBasicBlock *BB);
                                        
}

#endif
