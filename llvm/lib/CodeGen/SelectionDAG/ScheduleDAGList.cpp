//===---- ScheduleDAGList.cpp - Implement a list scheduler for isel DAG ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements bottom-up and top-down list schedulers, using standard
// algorithms.  The basic approach uses a priority queue of available nodes to
// schedule.  One at a time, nodes are taken from the priority queue (thus in
// priority order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <climits>
#include <iostream>
#include <queue>
#include <set>
#include <vector>
#include "llvm/Support/CommandLine.h"
using namespace llvm;

namespace {
  Statistic<> NumNoops ("scheduler", "Number of noops inserted");
  Statistic<> NumStalls("scheduler", "Number of pipeline stalls");

  /// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
  /// a group of nodes flagged together.
  struct SUnit {
    SDNode *Node;                       // Representative node.
    std::vector<SDNode*> FlaggedNodes;  // All nodes flagged to Node.
    
    // Preds/Succs - The SUnits before/after us in the graph.  The boolean value
    // is true if the edge is a token chain edge, false if it is a value edge. 
    std::set<std::pair<SUnit*,bool> > Preds;  // All sunit predecessors.
    std::set<std::pair<SUnit*,bool> > Succs;  // All sunit successors.

    short NumPredsLeft;                 // # of preds not scheduled.
    short NumSuccsLeft;                 // # of succs not scheduled.
    short NumChainPredsLeft;            // # of chain preds not scheduled.
    short NumChainSuccsLeft;            // # of chain succs not scheduled.
    bool isTwoAddress     : 1;          // Is a two-address instruction.
    bool isDefNUseOperand : 1;          // Is a def&use operand.
    bool isPending        : 1;          // True once pending.
    bool isAvailable      : 1;          // True once available.
    bool isScheduled      : 1;          // True once scheduled.
    unsigned short Latency;             // Node latency.
    unsigned CycleBound;                // Upper/lower cycle to be scheduled at.
    unsigned Cycle;                     // Once scheduled, the cycle of the op.
    unsigned NodeNum;                   // Entry # of node in the node vector.
    
    SUnit(SDNode *node, unsigned nodenum)
      : Node(node), NumPredsLeft(0), NumSuccsLeft(0),
      NumChainPredsLeft(0), NumChainSuccsLeft(0),
      isTwoAddress(false), isDefNUseOperand(false),
      isPending(false), isAvailable(false), isScheduled(false), 
      Latency(0), CycleBound(0), Cycle(0), NodeNum(nodenum) {}
    
    void dump(const SelectionDAG *G) const;
    void dumpAll(const SelectionDAG *G) const;
  };
}

void SUnit::dump(const SelectionDAG *G) const {
  std::cerr << "SU: ";
  Node->dump(G);
  std::cerr << "\n";
  if (FlaggedNodes.size() != 0) {
    for (unsigned i = 0, e = FlaggedNodes.size(); i != e; i++) {
      std::cerr << "    ";
      FlaggedNodes[i]->dump(G);
      std::cerr << "\n";
    }
  }
}

void SUnit::dumpAll(const SelectionDAG *G) const {
  dump(G);

  std::cerr << "  # preds left       : " << NumPredsLeft << "\n";
  std::cerr << "  # succs left       : " << NumSuccsLeft << "\n";
  std::cerr << "  # chain preds left : " << NumChainPredsLeft << "\n";
  std::cerr << "  # chain succs left : " << NumChainSuccsLeft << "\n";
  std::cerr << "  Latency            : " << Latency << "\n";

  if (Preds.size() != 0) {
    std::cerr << "  Predecessors:\n";
    for (std::set<std::pair<SUnit*,bool> >::const_iterator I = Preds.begin(),
           E = Preds.end(); I != E; ++I) {
      if (I->second)
        std::cerr << "   ch  ";
      else
        std::cerr << "   val ";
      I->first->dump(G);
    }
  }
  if (Succs.size() != 0) {
    std::cerr << "  Successors:\n";
    for (std::set<std::pair<SUnit*, bool> >::const_iterator I = Succs.begin(),
           E = Succs.end(); I != E; ++I) {
      if (I->second)
        std::cerr << "   ch  ";
      else
        std::cerr << "   val ";
      I->first->dump(G);
    }
  }
  std::cerr << "\n";
}

//===----------------------------------------------------------------------===//
/// SchedulingPriorityQueue - This interface is used to plug different
/// priorities computation algorithms into the list scheduler. It implements the
/// interface of a standard priority queue, where nodes are inserted in 
/// arbitrary order and returned in priority order.  The computation of the
/// priority and the representation of the queue are totally up to the
/// implementation to decide.
/// 
namespace {
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
}



namespace {
//===----------------------------------------------------------------------===//
/// ScheduleDAGList - The actual list scheduler implementation.  This supports
/// both top-down and bottom-up scheduling.
///
class ScheduleDAGList : public ScheduleDAG {
private:
  // SDNode to SUnit mapping (many to one).
  std::map<SDNode*, SUnit*> SUnitMap;
  // The schedule.  Null SUnit*'s represent noop instructions.
  std::vector<SUnit*> Sequence;
  
  // The scheduling units.
  std::vector<SUnit> SUnits;

  /// isBottomUp - This is true if the scheduling problem is bottom-up, false if
  /// it is top-down.
  bool isBottomUp;
  
  /// AvailableQueue - The priority queue to use for the available SUnits.
  ///
  SchedulingPriorityQueue *AvailableQueue;
  
  /// PendingQueue - This contains all of the instructions whose operands have
  /// been issued, but their results are not ready yet (due to the latency of
  /// the operation).  Once the operands becomes available, the instruction is
  /// added to the AvailableQueue.  This keeps track of each SUnit and the
  /// number of cycles left to execute before the operation is available.
  std::vector<std::pair<unsigned, SUnit*> > PendingQueue;
  
  /// HazardRec - The hazard recognizer to use.
  HazardRecognizer *HazardRec;
  
public:
  ScheduleDAGList(SelectionDAG &dag, MachineBasicBlock *bb,
                  const TargetMachine &tm, bool isbottomup,
                  SchedulingPriorityQueue *availqueue,
                  HazardRecognizer *HR)
    : ScheduleDAG(dag, bb, tm), isBottomUp(isbottomup), 
      AvailableQueue(availqueue), HazardRec(HR) {
    }

  ~ScheduleDAGList() {
    delete HazardRec;
    delete AvailableQueue;
  }

  void Schedule();

  void dumpSchedule() const;

private:
  SUnit *NewSUnit(SDNode *N);
  void ReleasePred(SUnit *PredSU, bool isChain, unsigned CurCycle);
  void ReleaseSucc(SUnit *SuccSU, bool isChain);
  void ScheduleNodeBottomUp(SUnit *SU, unsigned CurCycle);
  void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
  void ListScheduleTopDown();
  void ListScheduleBottomUp();
  void BuildSchedUnits();
  void EmitSchedule();
};
}  // end anonymous namespace

HazardRecognizer::~HazardRecognizer() {}


/// NewSUnit - Creates a new SUnit and return a ptr to it.
SUnit *ScheduleDAGList::NewSUnit(SDNode *N) {
  SUnits.push_back(SUnit(N, SUnits.size()));
  return &SUnits.back();
}

/// BuildSchedUnits - Build SUnits from the selection dag that we are input.
/// This SUnit graph is similar to the SelectionDAG, but represents flagged
/// together nodes with a single SUnit.
void ScheduleDAGList::BuildSchedUnits() {
  // Reserve entries in the vector for each of the SUnits we are creating.  This
  // ensure that reallocation of the vector won't happen, so SUnit*'s won't get
  // invalidated.
  SUnits.reserve(std::distance(DAG.allnodes_begin(), DAG.allnodes_end()));
  
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();
  
  for (SelectionDAG::allnodes_iterator NI = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); NI != E; ++NI) {
    if (isPassiveNode(NI))  // Leaf node, e.g. a TargetImmediate.
      continue;
    
    // If this node has already been processed, stop now.
    if (SUnitMap[NI]) continue;
    
    SUnit *NodeSUnit = NewSUnit(NI);
    
    // See if anything is flagged to this node, if so, add them to flagged
    // nodes.  Nodes can have at most one flag input and one flag output.  Flags
    // are required the be the last operand and result of a node.
    
    // Scan up, adding flagged preds to FlaggedNodes.
    SDNode *N = NI;
    while (N->getNumOperands() &&
           N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Flag) {
      N = N->getOperand(N->getNumOperands()-1).Val;
      NodeSUnit->FlaggedNodes.push_back(N);
      SUnitMap[N] = NodeSUnit;
    }
    
    // Scan down, adding this node and any flagged succs to FlaggedNodes if they
    // have a user of the flag operand.
    N = NI;
    while (N->getValueType(N->getNumValues()-1) == MVT::Flag) {
      SDOperand FlagVal(N, N->getNumValues()-1);
      
      // There are either zero or one users of the Flag result.
      bool HasFlagUse = false;
      for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end(); 
           UI != E; ++UI)
        if (FlagVal.isOperand(*UI)) {
          HasFlagUse = true;
          NodeSUnit->FlaggedNodes.push_back(N);
          SUnitMap[N] = NodeSUnit;
          N = *UI;
          break;
        }
          if (!HasFlagUse) break;
    }
    
    // Now all flagged nodes are in FlaggedNodes and N is the bottom-most node.
    // Update the SUnit
    NodeSUnit->Node = N;
    SUnitMap[N] = NodeSUnit;
    
    // Compute the latency for the node.  We use the sum of the latencies for
    // all nodes flagged together into this SUnit.
    if (InstrItins.isEmpty()) {
      // No latency information.
      NodeSUnit->Latency = 1;
    } else {
      NodeSUnit->Latency = 0;
      if (N->isTargetOpcode()) {
        unsigned SchedClass = TII->getSchedClass(N->getTargetOpcode());
        InstrStage *S = InstrItins.begin(SchedClass);
        InstrStage *E = InstrItins.end(SchedClass);
        for (; S != E; ++S)
          NodeSUnit->Latency += S->Cycles;
      }
      for (unsigned i = 0, e = NodeSUnit->FlaggedNodes.size(); i != e; ++i) {
        SDNode *FNode = NodeSUnit->FlaggedNodes[i];
        if (FNode->isTargetOpcode()) {
          unsigned SchedClass = TII->getSchedClass(FNode->getTargetOpcode());
          InstrStage *S = InstrItins.begin(SchedClass);
          InstrStage *E = InstrItins.end(SchedClass);
          for (; S != E; ++S)
            NodeSUnit->Latency += S->Cycles;
        }
      }
    }
  }
  
  // Pass 2: add the preds, succs, etc.
  for (unsigned su = 0, e = SUnits.size(); su != e; ++su) {
    SUnit *SU = &SUnits[su];
    SDNode *MainNode = SU->Node;
    
    if (MainNode->isTargetOpcode() &&
        TII->isTwoAddrInstr(MainNode->getTargetOpcode()))
      SU->isTwoAddress = true;
    
    // Find all predecessors and successors of the group.
    // Temporarily add N to make code simpler.
    SU->FlaggedNodes.push_back(MainNode);
    
    for (unsigned n = 0, e = SU->FlaggedNodes.size(); n != e; ++n) {
      SDNode *N = SU->FlaggedNodes[n];
      
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
        SDNode *OpN = N->getOperand(i).Val;
        if (isPassiveNode(OpN)) continue;   // Not scheduled.
        SUnit *OpSU = SUnitMap[OpN];
        assert(OpSU && "Node has no SUnit!");
        if (OpSU == SU) continue;           // In the same group.
        
        MVT::ValueType OpVT = N->getOperand(i).getValueType();
        assert(OpVT != MVT::Flag && "Flagged nodes should be in same sunit!");
        bool isChain = OpVT == MVT::Other;
        
        if (SU->Preds.insert(std::make_pair(OpSU, isChain)).second) {
          if (!isChain) {
            SU->NumPredsLeft++;
          } else {
            SU->NumChainPredsLeft++;
          }
        }
        if (OpSU->Succs.insert(std::make_pair(SU, isChain)).second) {
          if (!isChain) {
            OpSU->NumSuccsLeft++;
          } else {
            OpSU->NumChainSuccsLeft++;
          }
        }
      }
    }
    
    // Remove MainNode from FlaggedNodes again.
    SU->FlaggedNodes.pop_back();
  }
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
        SUnits[su].dumpAll(&DAG));
}

/// EmitSchedule - Emit the machine code in scheduled order.
void ScheduleDAGList::EmitSchedule() {
  std::map<SDNode*, unsigned> VRBaseMap;
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i]) {
      for (unsigned j = 0, ee = SU->FlaggedNodes.size(); j != ee; j++)
        EmitNode(SU->FlaggedNodes[j], VRBaseMap);
      EmitNode(SU->Node, VRBaseMap);
    } else {
      // Null SUnit* is a noop.
      EmitNoop();
    }
  }
}

/// dump - dump the schedule.
void ScheduleDAGList::dumpSchedule() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      SU->dump(&DAG);
    else
      std::cerr << "**** NOOP ****\n";
  }
}

/// Schedule - Schedule the DAG using list scheduling.
/// FIXME: Right now it only supports the burr (bottom up register reducing)
/// heuristic.
void ScheduleDAGList::Schedule() {
  DEBUG(std::cerr << "********** List Scheduling **********\n");
  
  // Build scheduling units.
  BuildSchedUnits();
  
  AvailableQueue->initNodes(SUnits);
  
  // Execute the actual scheduling loop Top-Down or Bottom-Up as appropriate.
  if (isBottomUp)
    ListScheduleBottomUp();
  else
    ListScheduleTopDown();
  
  AvailableQueue->releaseState();
  
  DEBUG(std::cerr << "*** Final schedule ***\n");
  DEBUG(dumpSchedule());
  DEBUG(std::cerr << "\n");
  
  // Emit in scheduled order
  EmitSchedule();
}

//===----------------------------------------------------------------------===//
//  Bottom-Up Scheduling
//===----------------------------------------------------------------------===//

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the Available queue is the count reaches zero. Also update its cycle bound.
void ScheduleDAGList::ReleasePred(SUnit *PredSU, bool isChain, 
                                  unsigned CurCycle) {
  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  PredSU->CycleBound = std::max(PredSU->CycleBound, CurCycle + PredSU->Latency);

  if (!isChain)
    PredSU->NumSuccsLeft--;
  else
    PredSU->NumChainSuccsLeft--;
  
#ifndef NDEBUG
  if (PredSU->NumSuccsLeft < 0 || PredSU->NumChainSuccsLeft < 0) {
    std::cerr << "*** List scheduling failed! ***\n";
    PredSU->dump(&DAG);
    std::cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if ((PredSU->NumSuccsLeft + PredSU->NumChainSuccsLeft) == 0) {
    // EntryToken has to go last!  Special case it here.
    if (PredSU->Node->getOpcode() != ISD::EntryToken) {
      PredSU->isAvailable = true;
      AvailableQueue->push(PredSU);
    }
  }
}
/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGList::ScheduleNodeBottomUp(SUnit *SU, unsigned CurCycle) {
  DEBUG(std::cerr << "*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(&DAG));
  SU->Cycle = CurCycle;

  Sequence.push_back(SU);

  // Bottom up: release predecessors
  for (std::set<std::pair<SUnit*, bool> >::iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
    ReleasePred(I->first, I->second, CurCycle);
    // FIXME: This is something used by the priority function that it should
    // calculate directly.
    if (!I->second)
      SU->NumPredsLeft--;
  }
}

/// isReady - True if node's lower cycle bound is less or equal to the current
/// scheduling cycle. Always true if all nodes have uniform latency 1.
static inline bool isReady(SUnit *SU, unsigned CurrCycle) {
  return SU->CycleBound <= CurrCycle;
}

/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGList::ListScheduleBottomUp() {
  unsigned CurrCycle = 0;
  // Add root to Available queue.
  AvailableQueue->push(SUnitMap[DAG.getRoot().Val]);

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back. Schedule the node.
  std::vector<SUnit*> NotReady;
  while (!AvailableQueue->empty()) {
    SUnit *CurrNode = AvailableQueue->pop();

    while (!isReady(CurrNode, CurrCycle)) {
      NotReady.push_back(CurrNode);
      CurrNode = AvailableQueue->pop();
    }
    
    // Add the nodes that aren't ready back onto the available list.
    AvailableQueue->push_all(NotReady);
    NotReady.clear();

    ScheduleNodeBottomUp(CurrNode, CurrCycle);
    CurrCycle++;
    CurrNode->isScheduled = true;
    AvailableQueue->ScheduledNode(CurrNode);
  }

  // Add entry node last
  if (DAG.getEntryNode().Val != DAG.getRoot().Val) {
    SUnit *Entry = SUnitMap[DAG.getEntryNode().Val];
    Sequence.push_back(Entry);
  }

  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());
  
  
#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (SUnits[i].NumSuccsLeft != 0 || SUnits[i].NumChainSuccsLeft != 0) {
      if (!AnyNotSched)
        std::cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(&DAG);
      std::cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero.
void ScheduleDAGList::ReleaseSucc(SUnit *SuccSU, bool isChain) {
  if (!isChain)
    SuccSU->NumPredsLeft--;
  else
    SuccSU->NumChainPredsLeft--;
  
  assert(SuccSU->NumPredsLeft >= 0 && SuccSU->NumChainPredsLeft >= 0 &&
         "List scheduling internal error");
  
  if ((SuccSU->NumPredsLeft + SuccSU->NumChainPredsLeft) == 0) {
    // Compute how many cycles it will be before this actually becomes
    // available.  This is the max of the start time of all predecessors plus
    // their latencies.
    unsigned AvailableCycle = 0;
    for (std::set<std::pair<SUnit*, bool> >::iterator I = SuccSU->Preds.begin(),
         E = SuccSU->Preds.end(); I != E; ++I) {
      AvailableCycle = std::max(AvailableCycle, 
                                I->first->Cycle + I->first->Latency);
    }
    
    PendingQueue.push_back(std::make_pair(AvailableCycle, SuccSU));
    SuccSU->isPending = true;
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DEBUG(std::cerr << "*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(&DAG));
  
  Sequence.push_back(SU);
  SU->Cycle = CurCycle;
  
  // Bottom up: release successors.
  for (std::set<std::pair<SUnit*, bool> >::iterator I = SU->Succs.begin(),
       E = SU->Succs.end(); I != E; ++I)
    ReleaseSucc(I->first, I->second);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGList::ListScheduleTopDown() {
  unsigned CurCycle = 0;
  SUnit *Entry = SUnitMap[DAG.getEntryNode().Val];

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.size() == 0 && &SUnits[i] != Entry) {
      AvailableQueue->push(&SUnits[i]);
      SUnits[i].isAvailable = SUnits[i].isPending = true;
    }
  }
  
  // Emit the entry node first.
  ScheduleNodeTopDown(Entry, CurCycle);
  HazardRec->EmitInstruction(Entry->Node);
  
  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  while (!AvailableQueue->empty() || !PendingQueue.empty()) {
    // Check to see if any of the pending instructions are ready to issue.  If
    // so, add them to the available queue.
    for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i)
      if (PendingQueue[i].first == CurCycle) {
        AvailableQueue->push(PendingQueue[i].second);
        PendingQueue[i].second->isAvailable = true;
        PendingQueue[i] = PendingQueue.back();
        PendingQueue.pop_back();
        --i; --e;
      } else {
        assert(PendingQueue[i].first > CurCycle && "Negative latency?");
      }
    
    SUnit *FoundNode = 0;

    bool HasNoopHazards = false;
    while (!AvailableQueue->empty()) {
      SUnit *CurNode = AvailableQueue->pop();
      
      // Get the node represented by this SUnit.
      SDNode *N = CurNode->Node;
      // If this is a pseudo op, like copyfromreg, look to see if there is a
      // real target node flagged to it.  If so, use the target node.
      for (unsigned i = 0, e = CurNode->FlaggedNodes.size(); 
           N->getOpcode() < ISD::BUILTIN_OP_END && i != e; ++i)
        N = CurNode->FlaggedNodes[i];
      
      HazardRecognizer::HazardType HT = HazardRec->getHazardType(N);
      if (HT == HazardRecognizer::NoHazard) {
        FoundNode = CurNode;
        break;
      }
      
      // Remember if this is a noop hazard.
      HasNoopHazards |= HT == HazardRecognizer::NoopHazard;
      
      NotReady.push_back(CurNode);
    }
    
    // Add the nodes that aren't ready back onto the available list.
    AvailableQueue->push_all(NotReady);
    NotReady.clear();

    // If we found a node to schedule, do it now.
    if (FoundNode) {
      ScheduleNodeTopDown(FoundNode, CurCycle);
      HazardRec->EmitInstruction(FoundNode->Node);
      FoundNode->isScheduled = true;
      AvailableQueue->ScheduledNode(FoundNode);

      // If this is a pseudo-op node, we don't want to increment the current
      // cycle.
      if (FoundNode->Latency == 0)
        continue;   // Don't increment for pseudo-ops!
    } else if (!HasNoopHazards) {
      // Otherwise, we have a pipeline stall, but no other problem, just advance
      // the current cycle and try again.
      DEBUG(std::cerr << "*** Advancing cycle, no work to do\n");
      HazardRec->AdvanceCycle();
      ++NumStalls;
    } else {
      // Otherwise, we have no instructions to issue and we have instructions
      // that will fault if we don't do this right.  This is the case for
      // processors without pipeline interlocks and other cases.
      DEBUG(std::cerr << "*** Emitting noop\n");
      HazardRec->EmitNoop();
      Sequence.push_back(0);   // NULL SUnit* -> noop
      ++NumNoops;
    }
    ++CurCycle;
  }

#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (SUnits[i].NumPredsLeft != 0 || SUnits[i].NumChainPredsLeft != 0) {
      if (!AnyNotSched)
        std::cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(&DAG);
      std::cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}

//===----------------------------------------------------------------------===//
//                RegReductionPriorityQueue Implementation
//===----------------------------------------------------------------------===//
//
// This is a SchedulingPriorityQueue that schedules using Sethi Ullman numbers
// to reduce register pressure.
// 
namespace {
  class RegReductionPriorityQueue;
  
  /// Sorting functions for the Available queue.
  struct ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    RegReductionPriorityQueue *SPQ;
    ls_rr_sort(RegReductionPriorityQueue *spq) : SPQ(spq) {}
    ls_rr_sort(const ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };
}  // end anonymous namespace

namespace {
  class RegReductionPriorityQueue : public SchedulingPriorityQueue {
    // SUnits - The SUnits for the current graph.
    const std::vector<SUnit> *SUnits;
    
    // SethiUllmanNumbers - The SethiUllman number for each node.
    std::vector<int> SethiUllmanNumbers;
    
    std::priority_queue<SUnit*, std::vector<SUnit*>, ls_rr_sort> Queue;
  public:
    RegReductionPriorityQueue() : Queue(ls_rr_sort(this)) {
    }
    
    void initNodes(const std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      // Calculate node priorities.
      CalculatePriorities();
    }
    void releaseState() {
      SUnits = 0;
      SethiUllmanNumbers.clear();
    }
    
    unsigned getSethiUllmanNumber(unsigned NodeNum) const {
      assert(NodeNum < SethiUllmanNumbers.size());
      return SethiUllmanNumbers[NodeNum];
    }
    
    bool empty() const { return Queue.empty(); }
    
    void push(SUnit *U) {
      Queue.push(U);
    }
    void push_all(const std::vector<SUnit *> &Nodes) {
      for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
        Queue.push(Nodes[i]);
    }
    
    SUnit *pop() {
      SUnit *V = Queue.top();
      Queue.pop();
      return V;
    }
  private:
    void CalculatePriorities();
    int CalcNodePriority(const SUnit *SU);
  };
}

bool ls_rr_sort::operator()(const SUnit *left, const SUnit *right) const {
  unsigned LeftNum  = left->NodeNum;
  unsigned RightNum = right->NodeNum;
  
  int LBonus = (int)left ->isDefNUseOperand;
  int RBonus = (int)right->isDefNUseOperand;
  
  // Special tie breaker: if two nodes share a operand, the one that
  // use it as a def&use operand is preferred.
  if (left->isTwoAddress && !right->isTwoAddress) {
    SDNode *DUNode = left->Node->getOperand(0).Val;
    if (DUNode->isOperand(right->Node))
      LBonus++;
  }
  if (!left->isTwoAddress && right->isTwoAddress) {
    SDNode *DUNode = right->Node->getOperand(0).Val;
    if (DUNode->isOperand(left->Node))
      RBonus++;
  }
  
  // Priority1 is just the number of live range genned.
  int LPriority1 = left ->NumPredsLeft - LBonus;
  int RPriority1 = right->NumPredsLeft - RBonus;
  int LPriority2 = SPQ->getSethiUllmanNumber(LeftNum) + LBonus;
  int RPriority2 = SPQ->getSethiUllmanNumber(RightNum) + RBonus;
  
  if (LPriority1 > RPriority1)
    return true;
  else if (LPriority1 == RPriority1)
    if (LPriority2 < RPriority2)
      return true;
    else if (LPriority2 == RPriority2)
      if (left->CycleBound > right->CycleBound) 
        return true;
  
  return false;
}


/// CalcNodePriority - Priority is the Sethi Ullman number. 
/// Smaller number is the higher priority.
int RegReductionPriorityQueue::CalcNodePriority(const SUnit *SU) {
  int &SethiUllmanNumber = SethiUllmanNumbers[SU->NodeNum];
  if (SethiUllmanNumber != INT_MIN)
    return SethiUllmanNumber;
  
  if (SU->Preds.size() == 0) {
    SethiUllmanNumber = 1;
  } else {
    int Extra = 0;
    for (std::set<std::pair<SUnit*, bool> >::const_iterator
         I = SU->Preds.begin(), E = SU->Preds.end(); I != E; ++I) {
      if (I->second) continue;  // ignore chain preds.
      SUnit *PredSU = I->first;
      int PredSethiUllman = CalcNodePriority(PredSU);
      if (PredSethiUllman > SethiUllmanNumber) {
        SethiUllmanNumber = PredSethiUllman;
        Extra = 0;
      } else if (PredSethiUllman == SethiUllmanNumber)
        Extra++;
    }
    
    if (SU->Node->getOpcode() != ISD::TokenFactor)
      SethiUllmanNumber += Extra;
    else
      SethiUllmanNumber = (Extra == 1) ? 0 : Extra-1;
  }
  
  return SethiUllmanNumber;
}

/// CalculatePriorities - Calculate priorities of all scheduling units.
void RegReductionPriorityQueue::CalculatePriorities() {
  SethiUllmanNumbers.assign(SUnits->size(), INT_MIN);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodePriority(&(*SUnits)[i]);
}

//===----------------------------------------------------------------------===//
//                    LatencyPriorityQueue Implementation
//===----------------------------------------------------------------------===//
//
// This is a SchedulingPriorityQueue that schedules using latency information to
// reduce the length of the critical path through the basic block.
// 
namespace {
  class LatencyPriorityQueue;
  
  /// Sorting functions for the Available queue.
  struct latency_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    LatencyPriorityQueue *PQ;
    latency_sort(LatencyPriorityQueue *pq) : PQ(pq) {}
    latency_sort(const latency_sort &RHS) : PQ(RHS.PQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };
}  // end anonymous namespace

namespace {
  class LatencyPriorityQueue : public SchedulingPriorityQueue {
    // SUnits - The SUnits for the current graph.
    const std::vector<SUnit> *SUnits;
    
    // Latencies - The latency (max of latency from this node to the bb exit)
    // for each node.
    std::vector<int> Latencies;

    /// NumNodesSolelyBlocking - This vector contains, for every node in the
    /// Queue, the number of nodes that the node is the sole unscheduled
    /// predecessor for.  This is used as a tie-breaker heuristic for better
    /// mobility.
    std::vector<unsigned> NumNodesSolelyBlocking;

    std::priority_queue<SUnit*, std::vector<SUnit*>, latency_sort> Queue;
public:
    LatencyPriorityQueue() : Queue(latency_sort(this)) {
    }
    
    void initNodes(const std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      // Calculate node priorities.
      CalculatePriorities();
    }
    void releaseState() {
      SUnits = 0;
      Latencies.clear();
    }
    
    unsigned getLatency(unsigned NodeNum) const {
      assert(NodeNum < Latencies.size());
      return Latencies[NodeNum];
    }
    
    unsigned getNumSolelyBlockNodes(unsigned NodeNum) const {
      assert(NodeNum < NumNodesSolelyBlocking.size());
      return NumNodesSolelyBlocking[NodeNum];
    }
    
    bool empty() const { return Queue.empty(); }
    
    virtual void push(SUnit *U) {
      push_impl(U);
    }
    void push_impl(SUnit *U);
    
    void push_all(const std::vector<SUnit *> &Nodes) {
      for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
        push_impl(Nodes[i]);
    }
    
    SUnit *pop() {
      SUnit *V = Queue.top();
      Queue.pop();
      return V;
    }
    
    // ScheduledNode - As nodes are scheduled, we look to see if there are any
    // successor nodes that have a single unscheduled predecessor.  If so, that
    // single predecessor has a higher priority, since scheduling it will make
    // the node available.
    void ScheduledNode(SUnit *Node);
    
private:
    void CalculatePriorities();
    int CalcLatency(const SUnit &SU);
    void AdjustPriorityOfUnscheduledPreds(SUnit *SU);
    
    /// RemoveFromPriorityQueue - This is a really inefficient way to remove a
    /// node from a priority queue.  We should roll our own heap to make this
    /// better or something.
    void RemoveFromPriorityQueue(SUnit *SU) {
      std::vector<SUnit*> Temp;
      
      assert(!Queue.empty() && "Not in queue!");
      while (Queue.top() != SU) {
        Temp.push_back(Queue.top());
        Queue.pop();
        assert(!Queue.empty() && "Not in queue!");
      }

      // Remove the node from the PQ.
      Queue.pop();
      
      // Add all the other nodes back.
      for (unsigned i = 0, e = Temp.size(); i != e; ++i)
        Queue.push(Temp[i]);
    }
  };
}

bool latency_sort::operator()(const SUnit *LHS, const SUnit *RHS) const {
  unsigned LHSNum = LHS->NodeNum;
  unsigned RHSNum = RHS->NodeNum;

  // The most important heuristic is scheduling the critical path.
  unsigned LHSLatency = PQ->getLatency(LHSNum);
  unsigned RHSLatency = PQ->getLatency(RHSNum);
  if (LHSLatency < RHSLatency) return true;
  if (LHSLatency > RHSLatency) return false;
  
  // After that, if two nodes have identical latencies, look to see if one will
  // unblock more other nodes than the other.
  unsigned LHSBlocked = PQ->getNumSolelyBlockNodes(LHSNum);
  unsigned RHSBlocked = PQ->getNumSolelyBlockNodes(RHSNum);
  if (LHSBlocked < RHSBlocked) return true;
  if (LHSBlocked > RHSBlocked) return false;
  
  // Finally, just to provide a stable ordering, use the node number as a
  // deciding factor.
  return LHSNum < RHSNum;
}


/// CalcNodePriority - Calculate the maximal path from the node to the exit.
///
int LatencyPriorityQueue::CalcLatency(const SUnit &SU) {
  int &Latency = Latencies[SU.NodeNum];
  if (Latency != -1)
    return Latency;
  
  int MaxSuccLatency = 0;
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU.Succs.begin(),
       E = SU.Succs.end(); I != E; ++I)
    MaxSuccLatency = std::max(MaxSuccLatency, CalcLatency(*I->first));

  return Latency = MaxSuccLatency + SU.Latency;
}

/// CalculatePriorities - Calculate priorities of all scheduling units.
void LatencyPriorityQueue::CalculatePriorities() {
  Latencies.assign(SUnits->size(), -1);
  NumNodesSolelyBlocking.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcLatency((*SUnits)[i]);
}

/// getSingleUnscheduledPred - If there is exactly one unscheduled predecessor
/// of SU, return it, otherwise return null.
static SUnit *getSingleUnscheduledPred(SUnit *SU) {
  SUnit *OnlyAvailablePred = 0;
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU->Preds.begin(),
       E = SU->Preds.end(); I != E; ++I)
    if (!I->first->isScheduled) {
      // We found an available, but not scheduled, predecessor.  If it's the
      // only one we have found, keep track of it... otherwise give up.
      if (OnlyAvailablePred && OnlyAvailablePred != I->first)
        return 0;
      OnlyAvailablePred = I->first;
    }
      
  return OnlyAvailablePred;
}

void LatencyPriorityQueue::push_impl(SUnit *SU) {
  // Look at all of the successors of this node.  Count the number of nodes that
  // this node is the sole unscheduled node for.
  unsigned NumNodesBlocking = 0;
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU->Succs.begin(),
       E = SU->Succs.end(); I != E; ++I)
    if (getSingleUnscheduledPred(I->first) == SU)
      ++NumNodesBlocking;
  NumNodesSolelyBlocking[SU->NodeNum] = NumNodesBlocking;
  
  Queue.push(SU);
}


// ScheduledNode - As nodes are scheduled, we look to see if there are any
// successor nodes that have a single unscheduled predecessor.  If so, that
// single predecessor has a higher priority, since scheduling it will make
// the node available.
void LatencyPriorityQueue::ScheduledNode(SUnit *SU) {
  for (std::set<std::pair<SUnit*, bool> >::const_iterator I = SU->Succs.begin(),
       E = SU->Succs.end(); I != E; ++I)
    AdjustPriorityOfUnscheduledPreds(I->first);
}

/// AdjustPriorityOfUnscheduledPreds - One of the predecessors of SU was just
/// scheduled.  If SU is not itself available, then there is at least one
/// predecessor node that has not been scheduled yet.  If SU has exactly ONE
/// unscheduled predecessor, we want to increase its priority: it getting
/// scheduled will make this node available, so it is better than some other
/// node of the same priority that will not make a node available.
void LatencyPriorityQueue::AdjustPriorityOfUnscheduledPreds(SUnit *SU) {
  if (SU->isPending) return;  // All preds scheduled.
  
  SUnit *OnlyAvailablePred = getSingleUnscheduledPred(SU);
  if (OnlyAvailablePred == 0 || !OnlyAvailablePred->isAvailable) return;
  
  // Okay, we found a single predecessor that is available, but not scheduled.
  // Since it is available, it must be in the priority queue.  First remove it.
  RemoveFromPriorityQueue(OnlyAvailablePred);

  // Reinsert the node into the priority queue, which recomputes its
  // NumNodesSolelyBlocking value.
  push(OnlyAvailablePred);
}


//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

llvm::ScheduleDAG* llvm::createBURRListDAGScheduler(SelectionDAG &DAG,
                                                    MachineBasicBlock *BB) {
  return new ScheduleDAGList(DAG, BB, DAG.getTarget(), true, 
                             new RegReductionPriorityQueue(),
                             new HazardRecognizer());
}

/// createTDListDAGScheduler - This creates a top-down list scheduler with the
/// specified hazard recognizer.
ScheduleDAG* llvm::createTDListDAGScheduler(SelectionDAG &DAG,
                                            MachineBasicBlock *BB,
                                            HazardRecognizer *HR) {
  return new ScheduleDAGList(DAG, BB, DAG.getTarget(), false,
                             new LatencyPriorityQueue(),
                             HR);
}
