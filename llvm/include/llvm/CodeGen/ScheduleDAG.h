//===------- llvm/CodeGen/ScheduleDAG.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAG class, which is used as the common
// base class for instruction schedulers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAG_H
#define LLVM_CODEGEN_SCHEDULEDAG_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
  struct SUnit;
  class MachineConstantPool;
  class MachineFunction;
  class MachineModuleInfo;
  class MachineRegisterInfo;
  class MachineInstr;
  class TargetRegisterInfo;
  class ScheduleDAG;
  class SelectionDAG;
  class SDNode;
  class TargetInstrInfo;
  class TargetInstrDesc;
  class TargetLowering;
  class TargetMachine;
  class TargetRegisterClass;
  template<class Graph> class GraphWriter;

  /// SDep - Scheduling dependency. It keeps track of dependent nodes,
  /// cost of the depdenency, etc.
  struct SDep {
    SUnit    *Dep;           // Dependent - either a predecessor or a successor.
    unsigned  Reg;           // If non-zero, this dep is a phy register dependency.
    int       Cost;          // Cost of the dependency.
    bool      isCtrl    : 1; // True iff it's a control dependency.
    bool      isSpecial : 1; // True iff it's a special ctrl dep added during sched.
    SDep(SUnit *d, unsigned r, int t, bool c, bool s)
      : Dep(d), Reg(r), Cost(t), isCtrl(c), isSpecial(s) {}
  };

  /// SUnit - Scheduling unit. This is a node in the scheduling DAG.
  struct SUnit {
  private:
    SDNode *Node;                       // Representative node.
    MachineInstr *Instr;                // Alternatively, a MachineInstr.
  public:
    SUnit *OrigNode;                    // If not this, the node from which
                                        // this node was cloned.
    
    // Preds/Succs - The SUnits before/after us in the graph.  The boolean value
    // is true if the edge is a token chain edge, false if it is a value edge. 
    SmallVector<SDep, 4> Preds;  // All sunit predecessors.
    SmallVector<SDep, 4> Succs;  // All sunit successors.

    typedef SmallVector<SDep, 4>::iterator pred_iterator;
    typedef SmallVector<SDep, 4>::iterator succ_iterator;
    typedef SmallVector<SDep, 4>::const_iterator const_pred_iterator;
    typedef SmallVector<SDep, 4>::const_iterator const_succ_iterator;
    
    unsigned NodeNum;                   // Entry # of node in the node vector.
    unsigned NodeQueueId;               // Queue id of node.
    unsigned short Latency;             // Node latency.
    short NumPreds;                     // # of non-control preds.
    short NumSuccs;                     // # of non-control sucss.
    short NumPredsLeft;                 // # of preds not scheduled.
    short NumSuccsLeft;                 // # of succs not scheduled.
    bool isTwoAddress     : 1;          // Is a two-address instruction.
    bool isCommutable     : 1;          // Is a commutable instruction.
    bool hasPhysRegDefs   : 1;          // Has physreg defs that are being used.
    bool isPending        : 1;          // True once pending.
    bool isAvailable      : 1;          // True once available.
    bool isScheduled      : 1;          // True once scheduled.
    unsigned CycleBound;                // Upper/lower cycle to be scheduled at.
    unsigned Cycle;                     // Once scheduled, the cycle of the op.
    unsigned Depth;                     // Node depth;
    unsigned Height;                    // Node height;
    const TargetRegisterClass *CopyDstRC; // Is a special copy node if not null.
    const TargetRegisterClass *CopySrcRC;
    
    /// SUnit - Construct an SUnit for pre-regalloc scheduling to represent
    /// an SDNode and any nodes flagged to it.
    SUnit(SDNode *node, unsigned nodenum)
      : Node(node), Instr(0), OrigNode(0), NodeNum(nodenum), NodeQueueId(0),
        Latency(0), NumPreds(0), NumSuccs(0), NumPredsLeft(0), NumSuccsLeft(0),
        isTwoAddress(false), isCommutable(false), hasPhysRegDefs(false),
        isPending(false), isAvailable(false), isScheduled(false),
        CycleBound(0), Cycle(~0u), Depth(0), Height(0),
        CopyDstRC(NULL), CopySrcRC(NULL) {}

    /// SUnit - Construct an SUnit for post-regalloc scheduling to represent
    /// a MachineInstr.
    SUnit(MachineInstr *instr, unsigned nodenum)
      : Node(0), Instr(instr), OrigNode(0), NodeNum(nodenum), NodeQueueId(0),
        Latency(0), NumPreds(0), NumSuccs(0), NumPredsLeft(0), NumSuccsLeft(0),
        isTwoAddress(false), isCommutable(false), hasPhysRegDefs(false),
        isPending(false), isAvailable(false), isScheduled(false),
        CycleBound(0), Cycle(~0u), Depth(0), Height(0),
        CopyDstRC(NULL), CopySrcRC(NULL) {}

    /// setNode - Assign the representative SDNode for this SUnit.
    /// This may be used during pre-regalloc scheduling.
    void setNode(SDNode *N) {
      assert(!Instr && "Setting SDNode of SUnit with MachineInstr!");
      Node = N;
    }

    /// getNode - Return the representative SDNode for this SUnit.
    /// This may be used during pre-regalloc scheduling.
    SDNode *getNode() const {
      assert(!Instr && "Reading SDNode of SUnit with MachineInstr!");
      return Node;
    }

    /// setInstr - Assign the instruction for the SUnit.
    /// This may be used during post-regalloc scheduling.
    void setInstr(MachineInstr *MI) {
      assert(!Node && "Setting MachineInstr of SUnit with SDNode!");
      Instr = MI;
    }

    /// getInstr - Return the representative MachineInstr for this SUnit.
    /// This may be used during post-regalloc scheduling.
    MachineInstr *getInstr() const {
      assert(!Node && "Reading MachineInstr of SUnit with SDNode!");
      return Instr;
    }

    /// addPred - This adds the specified node as a pred of the current node if
    /// not already.  This returns true if this is a new pred.
    bool addPred(SUnit *N, bool isCtrl, bool isSpecial,
                 unsigned PhyReg = 0, int Cost = 1) {
      for (unsigned i = 0, e = (unsigned)Preds.size(); i != e; ++i)
        if (Preds[i].Dep == N &&
            Preds[i].isCtrl == isCtrl && Preds[i].isSpecial == isSpecial)
          return false;
      Preds.push_back(SDep(N, PhyReg, Cost, isCtrl, isSpecial));
      N->Succs.push_back(SDep(this, PhyReg, Cost, isCtrl, isSpecial));
      if (!isCtrl) {
        ++NumPreds;
        ++N->NumSuccs;
      }
      if (!N->isScheduled)
        ++NumPredsLeft;
      if (!isScheduled)
        ++N->NumSuccsLeft;
      return true;
    }

    bool removePred(SUnit *N, bool isCtrl, bool isSpecial) {
      for (SmallVector<SDep, 4>::iterator I = Preds.begin(), E = Preds.end();
           I != E; ++I)
        if (I->Dep == N && I->isCtrl == isCtrl && I->isSpecial == isSpecial) {
          bool FoundSucc = false;
          for (SmallVector<SDep, 4>::iterator II = N->Succs.begin(),
                 EE = N->Succs.end(); II != EE; ++II)
            if (II->Dep == this &&
                II->isCtrl == isCtrl && II->isSpecial == isSpecial) {
              FoundSucc = true;
              N->Succs.erase(II);
              break;
            }
          assert(FoundSucc && "Mismatching preds / succs lists!");
          Preds.erase(I);
          if (!isCtrl) {
            --NumPreds;
            --N->NumSuccs;
          }
          if (!N->isScheduled)
            --NumPredsLeft;
          if (!isScheduled)
            --N->NumSuccsLeft;
          return true;
        }
      return false;
    }

    bool isPred(SUnit *N) {
      for (unsigned i = 0, e = (unsigned)Preds.size(); i != e; ++i)
        if (Preds[i].Dep == N)
          return true;
      return false;
    }
    
    bool isSucc(SUnit *N) {
      for (unsigned i = 0, e = (unsigned)Succs.size(); i != e; ++i)
        if (Succs[i].Dep == N)
          return true;
      return false;
    }
    
    void dump(const ScheduleDAG *G) const;
    void dumpAll(const ScheduleDAG *G) const;
    void print(raw_ostream &O, const ScheduleDAG *G) const;
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
  
    virtual void initNodes(std::vector<SUnit> &SUnits) = 0;
    virtual void addNode(const SUnit *SU) = 0;
    virtual void updateNode(const SUnit *SU) = 0;
    virtual void releaseState() = 0;

    virtual unsigned size() const = 0;
    virtual bool empty() const = 0;
    virtual void push(SUnit *U) = 0;
  
    virtual void push_all(const std::vector<SUnit *> &Nodes) = 0;
    virtual SUnit *pop() = 0;

    virtual void remove(SUnit *SU) = 0;

    /// ScheduledNode - As each node is scheduled, this method is invoked.  This
    /// allows the priority function to adjust the priority of related
    /// unscheduled nodes, for example.
    ///
    virtual void ScheduledNode(SUnit *) {}

    virtual void UnscheduledNode(SUnit *) {}
  };

  class ScheduleDAG {
  public:
    SelectionDAG *DAG;                    // DAG of the current basic block
    MachineBasicBlock *BB;                // Current basic block
    const TargetMachine &TM;              // Target processor
    const TargetInstrInfo *TII;           // Target instruction information
    const TargetRegisterInfo *TRI;        // Target processor register info
    TargetLowering *TLI;                  // Target lowering info
    MachineFunction *MF;                  // Machine function
    MachineRegisterInfo &MRI;             // Virtual/real register map
    MachineConstantPool *ConstPool;       // Target constant pool
    std::vector<SUnit*> Sequence;         // The schedule. Null SUnit*'s
                                          // represent noop instructions.
    std::vector<SUnit> SUnits;            // The scheduling units.

    ScheduleDAG(SelectionDAG *dag, MachineBasicBlock *bb,
                const TargetMachine &tm);

    virtual ~ScheduleDAG();

    /// viewGraph - Pop up a GraphViz/gv window with the ScheduleDAG rendered
    /// using 'dot'.
    ///
    void viewGraph();
  
    /// Run - perform scheduling.
    ///
    void Run();

    /// BuildSchedUnits - Build SUnits and set up their Preds and Succs
    /// to form the scheduling dependency graph.
    ///
    virtual void BuildSchedUnits() = 0;

    /// ComputeLatency - Compute node latency.
    ///
    virtual void ComputeLatency(SUnit *SU) { SU->Latency = 1; }

    /// CalculateDepths, CalculateHeights - Calculate node depth / height.
    ///
    void CalculateDepths();
    void CalculateHeights();

  protected:
    /// EmitNoop - Emit a noop instruction.
    ///
    void EmitNoop();

  public:
    virtual MachineBasicBlock *EmitSchedule() = 0;

    void dumpSchedule() const;

    /// Schedule - Order nodes according to selected style, filling
    /// in the Sequence member.
    ///
    virtual void Schedule() = 0;

    virtual void dumpNode(const SUnit *SU) const = 0;

    /// getGraphNodeLabel - Return a label for an SUnit node in a visualization
    /// of the ScheduleDAG.
    virtual std::string getGraphNodeLabel(const SUnit *SU) const = 0;

    /// addCustomGraphFeatures - Add custom features for a visualization of
    /// the ScheduleDAG.
    virtual void addCustomGraphFeatures(GraphWriter<ScheduleDAG*> &GW) const {}

#ifndef NDEBUG
    /// VerifySchedule - Verify that all SUnits were scheduled and that
    /// their state is consistent.
    void VerifySchedule(bool isBottomUp);
#endif

  protected:
    void AddMemOperand(MachineInstr *MI, const MachineMemOperand &MO);

    void EmitCrossRCCopy(SUnit *SU, DenseMap<SUnit*, unsigned> &VRBaseMap);

  private:
    /// EmitLiveInCopy - Emit a copy for a live in physical register. If the
    /// physical register has only a single copy use, then coalesced the copy
    /// if possible.
    void EmitLiveInCopy(MachineBasicBlock *MBB,
                        MachineBasicBlock::iterator &InsertPos,
                        unsigned VirtReg, unsigned PhysReg,
                        const TargetRegisterClass *RC,
                        DenseMap<MachineInstr*, unsigned> &CopyRegMap);

    /// EmitLiveInCopies - If this is the first basic block in the function,
    /// and if it has live ins that need to be copied into vregs, emit the
    /// copies into the top of the block.
    void EmitLiveInCopies(MachineBasicBlock *MBB);
  };

  class SUnitIterator : public forward_iterator<SUnit, ptrdiff_t> {
    SUnit *Node;
    unsigned Operand;

    SUnitIterator(SUnit *N, unsigned Op) : Node(N), Operand(Op) {}
  public:
    bool operator==(const SUnitIterator& x) const {
      return Operand == x.Operand;
    }
    bool operator!=(const SUnitIterator& x) const { return !operator==(x); }

    const SUnitIterator &operator=(const SUnitIterator &I) {
      assert(I.Node == Node && "Cannot assign iterators to two different nodes!");
      Operand = I.Operand;
      return *this;
    }

    pointer operator*() const {
      return Node->Preds[Operand].Dep;
    }
    pointer operator->() const { return operator*(); }

    SUnitIterator& operator++() {                // Preincrement
      ++Operand;
      return *this;
    }
    SUnitIterator operator++(int) { // Postincrement
      SUnitIterator tmp = *this; ++*this; return tmp;
    }

    static SUnitIterator begin(SUnit *N) { return SUnitIterator(N, 0); }
    static SUnitIterator end  (SUnit *N) {
      return SUnitIterator(N, (unsigned)N->Preds.size());
    }

    unsigned getOperand() const { return Operand; }
    const SUnit *getNode() const { return Node; }
    bool isCtrlDep() const { return Node->Preds[Operand].isCtrl; }
    bool isSpecialDep() const { return Node->Preds[Operand].isSpecial; }
  };

  template <> struct GraphTraits<SUnit*> {
    typedef SUnit NodeType;
    typedef SUnitIterator ChildIteratorType;
    static inline NodeType *getEntryNode(SUnit *N) { return N; }
    static inline ChildIteratorType child_begin(NodeType *N) {
      return SUnitIterator::begin(N);
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return SUnitIterator::end(N);
    }
  };

  template <> struct GraphTraits<ScheduleDAG*> : public GraphTraits<SUnit*> {
    typedef std::vector<SUnit>::iterator nodes_iterator;
    static nodes_iterator nodes_begin(ScheduleDAG *G) {
      return G->SUnits.begin();
    }
    static nodes_iterator nodes_end(ScheduleDAG *G) {
      return G->SUnits.end();
    }
  };
}

#endif
