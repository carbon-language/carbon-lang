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

namespace llvm {
  struct InstrStage;
  class MachineConstantPool;
  class MachineDebugInfo;
  class MachineInstr;
  class MRegisterInfo;
  class SelectionDAG;
  class SSARegMap;
  class TargetInstrInfo;
  class TargetInstrDescriptor;
  class TargetMachine;

  class NodeInfo;
  typedef NodeInfo *NodeInfoPtr;
  typedef std::vector<NodeInfoPtr>           NIVector;
  typedef std::vector<NodeInfoPtr>::iterator NIIterator;

  /// HazardRecognizer - This determines whether or not an instruction can be
  /// issued this cycle, and whether or not a noop needs to be inserted to handle
  /// the hazard.
  class HazardRecognizer {
  public:
    virtual ~HazardRecognizer();
    
    enum HazardType {
      NoHazard,      // This instruction can be emitted at this cycle.
      Hazard,        // This instruction can't be emitted at this cycle.
      NoopHazard,    // This instruction can't be emitted, and needs noops.
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

  //===--------------------------------------------------------------------===//
  ///
  /// Node group -  This struct is used to manage flagged node groups.
  ///
  class NodeGroup {
  public:
    NodeGroup     *Next;
  private:
    NIVector      Members;                // Group member nodes
    NodeInfo      *Dominator;             // Node with highest latency
    unsigned      Latency;                // Total latency of the group
    int           Pending;                // Number of visits pending before
                                          // adding to order  

  public:
    // Ctor.
    NodeGroup() : Next(NULL), Dominator(NULL), Pending(0) {}
  
    // Accessors
    inline void setDominator(NodeInfo *D) { Dominator = D; }
    inline NodeInfo *getTop() { return Members.front(); }
    inline NodeInfo *getBottom() { return Members.back(); }
    inline NodeInfo *getDominator() { return Dominator; }
    inline void setLatency(unsigned L) { Latency = L; }
    inline unsigned getLatency() { return Latency; }
    inline int getPending() const { return Pending; }
    inline void setPending(int P)  { Pending = P; }
    inline int addPending(int I)  { return Pending += I; }
  
    // Pass thru
    inline bool group_empty() { return Members.empty(); }
    inline NIIterator group_begin() { return Members.begin(); }
    inline NIIterator group_end() { return Members.end(); }
    inline void group_push_back(const NodeInfoPtr &NI) {
      Members.push_back(NI);
    }
    inline NIIterator group_insert(NIIterator Pos, const NodeInfoPtr &NI) {
      return Members.insert(Pos, NI);
    }
    inline void group_insert(NIIterator Pos, NIIterator First,
                             NIIterator Last) {
      Members.insert(Pos, First, Last);
    }

    static void Add(NodeInfo *D, NodeInfo *U);
  };

  //===--------------------------------------------------------------------===//
  ///
  /// NodeInfo - This struct tracks information used to schedule the a node.
  ///
  class NodeInfo {
  private:
    int           Pending;                // Number of visits pending before
                                          // adding to order
  public:
    SDNode        *Node;                  // DAG node
    InstrStage    *StageBegin;            // First stage in itinerary
    InstrStage    *StageEnd;              // Last+1 stage in itinerary
    unsigned      Latency;                // Total cycles to complete instr
    bool          IsCall : 1;             // Is function call
    bool          IsLoad : 1;             // Is memory load
    bool          IsStore : 1;            // Is memory store
    unsigned      Slot;                   // Node's time slot
    NodeGroup     *Group;                 // Grouping information
#ifndef NDEBUG
    unsigned      Preorder;               // Index before scheduling
#endif

    // Ctor.
    NodeInfo(SDNode *N = NULL)
      : Pending(0)
      , Node(N)
      , StageBegin(NULL)
      , StageEnd(NULL)
      , Latency(0)
      , IsCall(false)
      , Slot(0)
      , Group(NULL)
#ifndef NDEBUG
      , Preorder(0)
#endif
    {}
  
    // Accessors
    inline bool isInGroup() const {
      assert(!Group || !Group->group_empty() && "Group with no members");
      return Group != NULL;
    }
    inline bool isGroupDominator() const {
      return isInGroup() && Group->getDominator() == this;
    }
    inline int getPending() const {
      return Group ? Group->getPending() : Pending;
    }
    inline void setPending(int P) {
      if (Group) Group->setPending(P);
      else       Pending = P;
    }
    inline int addPending(int I) {
      if (Group) return Group->addPending(I);
      else       return Pending += I;
    }
  };

  //===--------------------------------------------------------------------===//
  ///
  /// NodeGroupIterator - Iterates over all the nodes indicated by the node
  /// info. If the node is in a group then iterate over the members of the
  /// group, otherwise just the node info.
  ///
  class NodeGroupIterator {
  private:
    NodeInfo   *NI;                       // Node info
    NIIterator NGI;                       // Node group iterator
    NIIterator NGE;                       // Node group iterator end
  
  public:
    // Ctor.
    NodeGroupIterator(NodeInfo *N) : NI(N) {
      // If the node is in a group then set up the group iterator.  Otherwise
      // the group iterators will trip first time out.
      if (N->isInGroup()) {
        // get Group
        NodeGroup *Group = NI->Group;
        NGI = Group->group_begin();
        NGE = Group->group_end();
        // Prevent this node from being used (will be in members list
        NI = NULL;
      }
    }
  
    /// next - Return the next node info, otherwise NULL.
    ///
    NodeInfo *next() {
      // If members list
      if (NGI != NGE) return *NGI++;
      // Use node as the result (may be NULL)
      NodeInfo *Result = NI;
      // Only use once
      NI = NULL;
      // Return node or NULL
      return Result;
    }
  };
  //===--------------------------------------------------------------------===//


  //===--------------------------------------------------------------------===//
  ///
  /// NodeGroupOpIterator - Iterates over all the operands of a node.  If the
  /// node is a member of a group, this iterates over all the operands of all
  /// the members of the group.
  ///
  class NodeGroupOpIterator {
  private:
    NodeInfo            *NI;              // Node containing operands
    NodeGroupIterator   GI;               // Node group iterator
    SDNode::op_iterator OI;               // Operand iterator
    SDNode::op_iterator OE;               // Operand iterator end
  
    /// CheckNode - Test if node has more operands.  If not get the next node
    /// skipping over nodes that have no operands.
    void CheckNode() {
      // Only if operands are exhausted first
      while (OI == OE) {
        // Get next node info
        NodeInfo *NI = GI.next();
        // Exit if nodes are exhausted
        if (!NI) return;
        // Get node itself
        SDNode *Node = NI->Node;
        // Set up the operand iterators
        OI = Node->op_begin();
        OE = Node->op_end();
      }
    }
  
  public:
    // Ctor.
    NodeGroupOpIterator(NodeInfo *N)
      : NI(N), GI(N), OI(SDNode::op_iterator()), OE(SDNode::op_iterator()) {}
  
    /// isEnd - Returns true when not more operands are available.
    ///
    inline bool isEnd() { CheckNode(); return OI == OE; }
  
    /// next - Returns the next available operand.
    ///
    inline SDOperand next() {
      assert(OI != OE &&
             "Not checking for end of NodeGroupOpIterator correctly");
      return *OI++;
    }
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
      if (isa<ExternalSymbolSDNode>(Node)) return true;
      return false;
    }

    /// EmitNode - Generate machine code for an node and needed dependencies.
    /// VRBaseMap contains, for each already emitted node, the first virtual
    /// register number for the results of the node.
    ///
    void EmitNode(SDNode *Node, std::map<SDNode*, unsigned> &VRBaseMap);
    
    /// EmitNoop - Emit a noop instruction.
    ///
    void EmitNoop();
    

    /// Schedule - Order nodes according to selected style.
    ///
    virtual void Schedule() {}

  private:
    void AddOperand(MachineInstr *MI, SDOperand Op, unsigned IIOpNum,
                    const TargetInstrDescriptor *II,
                    std::map<SDNode*, unsigned> &VRBaseMap);
  };

  ScheduleDAG *createBFS_DAGScheduler(SelectionDAG &DAG, MachineBasicBlock *BB);
  
  /// createSimpleDAGScheduler - This creates a simple two pass instruction
  /// scheduler.
  ScheduleDAG* createSimpleDAGScheduler(bool NoItins, SelectionDAG &DAG,
                                        MachineBasicBlock *BB);

  /// createBURRListDAGScheduler - This creates a bottom up register usage
  /// reduction list scheduler.
  ScheduleDAG* createBURRListDAGScheduler(SelectionDAG &DAG,
                                          MachineBasicBlock *BB);
  
  /// createTDListDAGScheduler - This creates a top-down list scheduler with
  /// the specified hazard recognizer.  This takes ownership of the hazard
  /// recognizer and deletes it when done.
  ScheduleDAG* createTDListDAGScheduler(SelectionDAG &DAG,
                                        MachineBasicBlock *BB,
                                        HazardRecognizer *HR);
}

#endif
