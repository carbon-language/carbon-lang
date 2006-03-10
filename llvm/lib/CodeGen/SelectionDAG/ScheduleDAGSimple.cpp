//===-- ScheduleDAGSimple.cpp - Implement a trivial DAG scheduler ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a simple two pass scheduler.  The first pass attempts to push
// backward any lengthy instructions and critical paths.  The second pass packs
// instructions into semi-optimal time slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
using namespace llvm;

namespace {
//===----------------------------------------------------------------------===//
///
/// BitsIterator - Provides iteration through individual bits in a bit vector.
///
template<class T>
class BitsIterator {
private:
  T Bits;                               // Bits left to iterate through

public:
  /// Ctor.
  BitsIterator(T Initial) : Bits(Initial) {}
  
  /// Next - Returns the next bit set or zero if exhausted.
  inline T Next() {
    // Get the rightmost bit set
    T Result = Bits & -Bits;
    // Remove from rest
    Bits &= ~Result;
    // Return single bit or zero
    return Result;
  }
};
  
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// ResourceTally - Manages the use of resources over time intervals.  Each
/// item (slot) in the tally vector represents the resources used at a given
/// moment.  A bit set to 1 indicates that a resource is in use, otherwise
/// available.  An assumption is made that the tally is large enough to schedule 
/// all current instructions (asserts otherwise.)
///
template<class T>
class ResourceTally {
private:
  std::vector<T> Tally;                 // Resources used per slot
  typedef typename std::vector<T>::iterator Iter;
                                        // Tally iterator 
  
  /// SlotsAvailable - Returns true if all units are available.
	///
  bool SlotsAvailable(Iter Begin, unsigned N, unsigned ResourceSet,
                                              unsigned &Resource) {
    assert(N && "Must check availability with N != 0");
    // Determine end of interval
    Iter End = Begin + N;
    assert(End <= Tally.end() && "Tally is not large enough for schedule");
    
    // Iterate thru each resource
    BitsIterator<T> Resources(ResourceSet & ~*Begin);
    while (unsigned Res = Resources.Next()) {
      // Check if resource is available for next N slots
      Iter Interval = End;
      do {
        Interval--;
        if (*Interval & Res) break;
      } while (Interval != Begin);
      
      // If available for N
      if (Interval == Begin) {
        // Success
        Resource = Res;
        return true;
      }
    }
    
    // No luck
    Resource = 0;
    return false;
  }
	
	/// RetrySlot - Finds a good candidate slot to retry search.
  Iter RetrySlot(Iter Begin, unsigned N, unsigned ResourceSet) {
    assert(N && "Must check availability with N != 0");
    // Determine end of interval
    Iter End = Begin + N;
    assert(End <= Tally.end() && "Tally is not large enough for schedule");
		
		while (Begin != End--) {
			// Clear units in use
			ResourceSet &= ~*End;
			// If no units left then we should go no further 
			if (!ResourceSet) return End + 1;
		}
		// Made it all the way through
		return Begin;
	}
  
  /// FindAndReserveStages - Return true if the stages can be completed. If
  /// so mark as busy.
  bool FindAndReserveStages(Iter Begin,
                            InstrStage *Stage, InstrStage *StageEnd) {
    // If at last stage then we're done
    if (Stage == StageEnd) return true;
    // Get number of cycles for current stage
    unsigned N = Stage->Cycles;
    // Check to see if N slots are available, if not fail
    unsigned Resource;
    if (!SlotsAvailable(Begin, N, Stage->Units, Resource)) return false;
    // Check to see if remaining stages are available, if not fail
    if (!FindAndReserveStages(Begin + N, Stage + 1, StageEnd)) return false;
    // Reserve resource
    Reserve(Begin, N, Resource);
    // Success
    return true;
  }

  /// Reserve - Mark busy (set) the specified N slots.
  void Reserve(Iter Begin, unsigned N, unsigned Resource) {
    // Determine end of interval
    Iter End = Begin + N;
    assert(End <= Tally.end() && "Tally is not large enough for schedule");
 
    // Set resource bit in each slot
    for (; Begin < End; Begin++)
      *Begin |= Resource;
  }

  /// FindSlots - Starting from Begin, locate consecutive slots where all stages
  /// can be completed.  Returns the address of first slot.
  Iter FindSlots(Iter Begin, InstrStage *StageBegin, InstrStage *StageEnd) {
    // Track position      
    Iter Cursor = Begin;
    
    // Try all possible slots forward
    while (true) {
      // Try at cursor, if successful return position.
      if (FindAndReserveStages(Cursor, StageBegin, StageEnd)) return Cursor;
      // Locate a better position
			Cursor = RetrySlot(Cursor + 1, StageBegin->Cycles, StageBegin->Units);
    }
  }
  
public:
  /// Initialize - Resize and zero the tally to the specified number of time
  /// slots.
  inline void Initialize(unsigned N) {
    Tally.assign(N, 0);   // Initialize tally to all zeros.
  }

  // FindAndReserve - Locate an ideal slot for the specified stages and mark
  // as busy.
  unsigned FindAndReserve(unsigned Slot, InstrStage *StageBegin,
                                         InstrStage *StageEnd) {
		// Where to begin 
		Iter Begin = Tally.begin() + Slot;
		// Find a free slot
		Iter Where = FindSlots(Begin, StageBegin, StageEnd);
		// Distance is slot number
		unsigned Final = Where - Tally.begin();
    return Final;
  }

};

//===----------------------------------------------------------------------===//
///
/// ScheduleDAGSimple - Simple two pass scheduler.
///
class ScheduleDAGSimple : public ScheduleDAG {
private:
  ResourceTally<unsigned> Tally;        // Resource usage tally
  unsigned NSlots;                      // Total latency
  static const unsigned NotFound = ~0U; // Search marker
  
public:

  // Ctor.
  ScheduleDAGSimple(SchedHeuristics hstc, SelectionDAG &dag,
                    MachineBasicBlock *bb, const TargetMachine &tm)
    : ScheduleDAG(hstc, dag, bb, tm), Tally(), NSlots(0) {
    assert(&TII && "Target doesn't provide instr info?");
    assert(&MRI && "Target doesn't provide register info?");
  }

  virtual ~ScheduleDAGSimple() {};

  void Schedule();

private:
  static bool isDefiner(NodeInfo *A, NodeInfo *B);
  void IncludeNode(NodeInfo *NI);
  void VisitAll();
  void GatherSchedulingInfo();
  void FakeGroupDominators(); 
  bool isStrongDependency(NodeInfo *A, NodeInfo *B);
  bool isWeakDependency(NodeInfo *A, NodeInfo *B);
  void ScheduleBackward();
  void ScheduleForward();
};

//===----------------------------------------------------------------------===//
/// Special case itineraries.
///
enum {
  CallLatency = 40,          // To push calls back in time

  RSInteger   = 0xC0000000,  // Two integer units
  RSFloat     = 0x30000000,  // Two float units
  RSLoadStore = 0x0C000000,  // Two load store units
  RSBranch    = 0x02000000   // One branch unit
};
static InstrStage CallStage  = { CallLatency, RSBranch };
static InstrStage LoadStage  = { 5, RSLoadStore };
static InstrStage StoreStage = { 2, RSLoadStore };
static InstrStage IntStage   = { 2, RSInteger };
static InstrStage FloatStage = { 3, RSFloat };
//===----------------------------------------------------------------------===//

} // namespace

//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// isDefiner - Return true if node A is a definer for B.
///
bool ScheduleDAGSimple::isDefiner(NodeInfo *A, NodeInfo *B) {
  // While there are A nodes
  NodeGroupIterator NII(A);
  while (NodeInfo *NI = NII.next()) {
    // Extract node
    SDNode *Node = NI->Node;
    // While there operands in nodes of B
    NodeGroupOpIterator NGOI(B);
    while (!NGOI.isEnd()) {
      SDOperand Op = NGOI.next();
      // If node from A defines a node in B
      if (Node == Op.Val) return true;
    }
  }
  return false;
}

/// IncludeNode - Add node to NodeInfo vector.
///
void ScheduleDAGSimple::IncludeNode(NodeInfo *NI) {
  // Get node
  SDNode *Node = NI->Node;
  // Ignore entry node
  if (Node->getOpcode() == ISD::EntryToken) return;
  // Check current count for node
  int Count = NI->getPending();
  // If the node is already in list
  if (Count < 0) return;
  // Decrement count to indicate a visit
  Count--;
  // If count has gone to zero then add node to list
  if (!Count) {
    // Add node
    if (NI->isInGroup()) {
      Ordering.push_back(NI->Group->getDominator());
    } else {
      Ordering.push_back(NI);
    }
    // indicate node has been added
    Count--;
  }
  // Mark as visited with new count 
  NI->setPending(Count);
}

/// GatherSchedulingInfo - Get latency and resource information about each node.
///
void ScheduleDAGSimple::GatherSchedulingInfo() {
  // Get instruction itineraries for the target
  const InstrItineraryData &InstrItins = TM.getInstrItineraryData();
  
  // For each node
  for (unsigned i = 0, N = NodeCount; i < N; i++) {
    // Get node info
    NodeInfo* NI = &Info[i];
    SDNode *Node = NI->Node;
    
    // If there are itineraries and it is a machine instruction
    if (InstrItins.isEmpty() || Heuristic == simpleNoItinScheduling) {
      // If machine opcode
      if (Node->isTargetOpcode()) {
        // Get return type to guess which processing unit 
        MVT::ValueType VT = Node->getValueType(0);
        // Get machine opcode
        MachineOpCode TOpc = Node->getTargetOpcode();
        NI->IsCall = TII->isCall(TOpc);
        NI->IsLoad = TII->isLoad(TOpc);
        NI->IsStore = TII->isStore(TOpc);

        if (TII->isLoad(TOpc))             NI->StageBegin = &LoadStage;
        else if (TII->isStore(TOpc))       NI->StageBegin = &StoreStage;
        else if (MVT::isInteger(VT))       NI->StageBegin = &IntStage;
        else if (MVT::isFloatingPoint(VT)) NI->StageBegin = &FloatStage;
        if (NI->StageBegin) NI->StageEnd = NI->StageBegin + 1;
      }
    } else if (Node->isTargetOpcode()) {
      // get machine opcode
      MachineOpCode TOpc = Node->getTargetOpcode();
      // Check to see if it is a call
      NI->IsCall = TII->isCall(TOpc);
      // Get itinerary stages for instruction
      unsigned II = TII->getSchedClass(TOpc);
      NI->StageBegin = InstrItins.begin(II);
      NI->StageEnd = InstrItins.end(II);
    }
    
    // One slot for the instruction itself
    NI->Latency = 1;
    
    // Add long latency for a call to push it back in time
    if (NI->IsCall) NI->Latency += CallLatency;
    
    // Sum up all the latencies
    for (InstrStage *Stage = NI->StageBegin, *E = NI->StageEnd;
        Stage != E; Stage++) {
      NI->Latency += Stage->Cycles;
    }
    
    // Sum up all the latencies for max tally size
    NSlots += NI->Latency;
  }
  
  // Unify metrics if in a group
  if (HasGroups) {
    for (unsigned i = 0, N = NodeCount; i < N; i++) {
      NodeInfo* NI = &Info[i];
      
      if (NI->isInGroup()) {
        NodeGroup *Group = NI->Group;
        
        if (!Group->getDominator()) {
          NIIterator NGI = Group->group_begin(), NGE = Group->group_end();
          NodeInfo *Dominator = *NGI;
          unsigned Latency = 0;
          
          for (NGI++; NGI != NGE; NGI++) {
            NodeInfo* NGNI = *NGI;
            Latency += NGNI->Latency;
            if (Dominator->Latency < NGNI->Latency) Dominator = NGNI;
          }
          
          Dominator->Latency = Latency;
          Group->setDominator(Dominator);
        }
      }
    }
  }
}

/// VisitAll - Visit each node breadth-wise to produce an initial ordering.
/// Note that the ordering in the Nodes vector is reversed.
void ScheduleDAGSimple::VisitAll() {
  // Add first element to list
  NodeInfo *NI = getNI(DAG.getRoot().Val);
  if (NI->isInGroup()) {
    Ordering.push_back(NI->Group->getDominator());
  } else {
    Ordering.push_back(NI);
  }

  // Iterate through all nodes that have been added
  for (unsigned i = 0; i < Ordering.size(); i++) { // note: size() varies
    // Visit all operands
    NodeGroupOpIterator NGI(Ordering[i]);
    while (!NGI.isEnd()) {
      // Get next operand
      SDOperand Op = NGI.next();
      // Get node
      SDNode *Node = Op.Val;
      // Ignore passive nodes
      if (isPassiveNode(Node)) continue;
      // Check out node
      IncludeNode(getNI(Node));
    }
  }

  // Add entry node last (IncludeNode filters entry nodes)
  if (DAG.getEntryNode().Val != DAG.getRoot().Val)
    Ordering.push_back(getNI(DAG.getEntryNode().Val));
    
  // Reverse the order
  std::reverse(Ordering.begin(), Ordering.end());
}

/// FakeGroupDominators - Set dominators for non-scheduling.
/// 
void ScheduleDAGSimple::FakeGroupDominators() {
  for (unsigned i = 0, N = NodeCount; i < N; i++) {
    NodeInfo* NI = &Info[i];
    
    if (NI->isInGroup()) {
      NodeGroup *Group = NI->Group;
      
      if (!Group->getDominator()) {
        Group->setDominator(NI);
      }
    }
  }
}

/// isStrongDependency - Return true if node A has results used by node B. 
/// I.E., B must wait for latency of A.
bool ScheduleDAGSimple::isStrongDependency(NodeInfo *A, NodeInfo *B) {
  // If A defines for B then it's a strong dependency or
  // if a load follows a store (may be dependent but why take a chance.)
  return isDefiner(A, B) || (A->IsStore && B->IsLoad);
}

/// isWeakDependency Return true if node A produces a result that will
/// conflict with operands of B.  It is assumed that we have called
/// isStrongDependency prior.
bool ScheduleDAGSimple::isWeakDependency(NodeInfo *A, NodeInfo *B) {
  // TODO check for conflicting real registers and aliases
#if 0 // FIXME - Since we are in SSA form and not checking register aliasing
  return A->Node->getOpcode() == ISD::EntryToken || isStrongDependency(B, A);
#else
  return A->Node->getOpcode() == ISD::EntryToken;
#endif
}

/// ScheduleBackward - Schedule instructions so that any long latency
/// instructions and the critical path get pushed back in time. Time is run in
/// reverse to allow code reuse of the Tally and eliminate the overhead of
/// biasing every slot indices against NSlots.
void ScheduleDAGSimple::ScheduleBackward() {
  // Size and clear the resource tally
  Tally.Initialize(NSlots);
  // Get number of nodes to schedule
  unsigned N = Ordering.size();
  
  // For each node being scheduled
  for (unsigned i = N; 0 < i--;) {
    NodeInfo *NI = Ordering[i];
    // Track insertion
    unsigned Slot = NotFound;
    
    // Compare against those previously scheduled nodes
    unsigned j = i + 1;
    for (; j < N; j++) {
      // Get following instruction
      NodeInfo *Other = Ordering[j];
      
      // Check dependency against previously inserted nodes
      if (isStrongDependency(NI, Other)) {
        Slot = Other->Slot + Other->Latency;
        break;
      } else if (isWeakDependency(NI, Other)) {
        Slot = Other->Slot;
        break;
      }
    }
    
    // If independent of others (or first entry)
    if (Slot == NotFound) Slot = 0;
    
#if 0 // FIXME - measure later
    // Find a slot where the needed resources are available
    if (NI->StageBegin != NI->StageEnd)
      Slot = Tally.FindAndReserve(Slot, NI->StageBegin, NI->StageEnd);
#endif
      
    // Set node slot
    NI->Slot = Slot;
    
    // Insert sort based on slot
    j = i + 1;
    for (; j < N; j++) {
      // Get following instruction
      NodeInfo *Other = Ordering[j];
      // Should we look further (remember slots are in reverse time)
      if (Slot >= Other->Slot) break;
      // Shuffle other into ordering
      Ordering[j - 1] = Other;
    }
    // Insert node in proper slot
    if (j != i + 1) Ordering[j - 1] = NI;
  }
}

/// ScheduleForward - Schedule instructions to maximize packing.
///
void ScheduleDAGSimple::ScheduleForward() {
  // Size and clear the resource tally
  Tally.Initialize(NSlots);
  // Get number of nodes to schedule
  unsigned N = Ordering.size();
  
  // For each node being scheduled
  for (unsigned i = 0; i < N; i++) {
    NodeInfo *NI = Ordering[i];
    // Track insertion
    unsigned Slot = NotFound;
    
    // Compare against those previously scheduled nodes
    unsigned j = i;
    for (; 0 < j--;) {
      // Get following instruction
      NodeInfo *Other = Ordering[j];
      
      // Check dependency against previously inserted nodes
      if (isStrongDependency(Other, NI)) {
        Slot = Other->Slot + Other->Latency;
        break;
      } else if (Other->IsCall || isWeakDependency(Other, NI)) {
        Slot = Other->Slot;
        break;
      }
    }
    
    // If independent of others (or first entry)
    if (Slot == NotFound) Slot = 0;
    
    // Find a slot where the needed resources are available
    if (NI->StageBegin != NI->StageEnd)
      Slot = Tally.FindAndReserve(Slot, NI->StageBegin, NI->StageEnd);
      
    // Set node slot
    NI->Slot = Slot;
    
    // Insert sort based on slot
    j = i;
    for (; 0 < j--;) {
      // Get prior instruction
      NodeInfo *Other = Ordering[j];
      // Should we look further
      if (Slot >= Other->Slot) break;
      // Shuffle other into ordering
      Ordering[j + 1] = Other;
    }
    // Insert node in proper slot
    if (j != i) Ordering[j + 1] = NI;
  }
}

/// Schedule - Order nodes according to selected style.
///
void ScheduleDAGSimple::Schedule() {
  // Set up minimum info for scheduling
  PrepareNodeInfo();
  // Construct node groups for flagged nodes
  IdentifyGroups();
  
  // Test to see if scheduling should occur
  bool ShouldSchedule = NodeCount > 3 && Heuristic != noScheduling;
  // Don't waste time if is only entry and return
  if (ShouldSchedule) {
    // Get latency and resource requirements
    GatherSchedulingInfo();
  } else if (HasGroups) {
    // Make sure all the groups have dominators
    FakeGroupDominators();
  }

  // Breadth first walk of DAG
  VisitAll();

#ifndef NDEBUG
  static unsigned Count = 0;
  Count++;
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    NodeInfo *NI = Ordering[i];
    NI->Preorder = i;
  }
#endif  
  
  // Don't waste time if is only entry and return
  if (ShouldSchedule) {
    // Push back long instructions and critical path
    ScheduleBackward();
    
    // Pack instructions to maximize resource utilization
    ScheduleForward();
  }
  
  DEBUG(printChanges(Count));
  
  // Emit in scheduled order
  EmitAll();
}


/// createSimpleDAGScheduler - This creates a simple two pass instruction
/// scheduler.
llvm::ScheduleDAG* llvm::createSimpleDAGScheduler(SchedHeuristics Heuristic,
                                                  SelectionDAG &DAG,
                                                  MachineBasicBlock *BB) {
  return new ScheduleDAGSimple(Heuristic, DAG, BB,  DAG.getTarget());
}
