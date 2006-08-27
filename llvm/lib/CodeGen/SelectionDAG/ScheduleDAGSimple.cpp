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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <iostream>
using namespace llvm;


namespace {

static RegisterScheduler
  bfsDAGScheduler("none", "  No scheduling: breadth first sequencing",
                  createBFS_DAGScheduler);
static RegisterScheduler
  simpleDAGScheduler("simple",
                     "  Simple two pass scheduling: minimize critical path "
                     "and maximize processor utilization",
                      createSimpleDAGScheduler);
static RegisterScheduler
  noitinDAGScheduler("simple-noitin",
                     "  Simple two pass scheduling: Same as simple "
                     "except using generic latency",
                     createNoItinsDAGScheduler);
                     
class NodeInfo;
typedef NodeInfo *NodeInfoPtr;
typedef std::vector<NodeInfoPtr>           NIVector;
typedef std::vector<NodeInfoPtr>::iterator NIIterator;

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
class VISIBILITY_HIDDEN ScheduleDAGSimple : public ScheduleDAG {
private:
  bool NoSched;                         // Just do a BFS schedule, nothing fancy
  bool NoItins;                         // Don't use itineraries?
  ResourceTally<unsigned> Tally;        // Resource usage tally
  unsigned NSlots;                      // Total latency
  static const unsigned NotFound = ~0U; // Search marker

  unsigned NodeCount;                   // Number of nodes in DAG
  std::map<SDNode *, NodeInfo *> Map;   // Map nodes to info
  bool HasGroups;                       // True if there are any groups
  NodeInfo *Info;                       // Info for nodes being scheduled
  NIVector Ordering;                    // Emit ordering of nodes
  NodeGroup *HeadNG, *TailNG;           // Keep track of allocated NodeGroups
  
public:

  // Ctor.
  ScheduleDAGSimple(bool noSched, bool noItins, SelectionDAG &dag,
                    MachineBasicBlock *bb, const TargetMachine &tm)
    : ScheduleDAG(dag, bb, tm), NoSched(noSched), NoItins(noItins), NSlots(0),
    NodeCount(0), HasGroups(false), Info(NULL), HeadNG(NULL), TailNG(NULL) {
    assert(&TII && "Target doesn't provide instr info?");
    assert(&MRI && "Target doesn't provide register info?");
  }

  virtual ~ScheduleDAGSimple() {
    if (Info)
      delete[] Info;
    
    NodeGroup *NG = HeadNG;
    while (NG) {
      NodeGroup *NextSU = NG->Next;
      delete NG;
      NG = NextSU;
    }
  }

  void Schedule();

  /// getNI - Returns the node info for the specified node.
  ///
  NodeInfo *getNI(SDNode *Node) { return Map[Node]; }
  
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
  
  void AddToGroup(NodeInfo *D, NodeInfo *U);
  /// PrepareNodeInfo - Set up the basic minimum node info for scheduling.
  /// 
  void PrepareNodeInfo();
  
  /// IdentifyGroups - Put flagged nodes into groups.
  ///
  void IdentifyGroups();
  
  /// print - Print ordering to specified output stream.
  ///
  void print(std::ostream &O) const;
  
  void dump(const char *tag) const;
  
  virtual void dump() const;
  
  /// EmitAll - Emit all nodes in schedule sorted order.
  ///
  void EmitAll();

  /// printNI - Print node info.
  ///
  void printNI(std::ostream &O, NodeInfo *NI) const;
  
  /// printChanges - Hilight changes in order caused by scheduling.
  ///
  void printChanges(unsigned Index) const;
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

/// PrepareNodeInfo - Set up the basic minimum node info for scheduling.
/// 
void ScheduleDAGSimple::PrepareNodeInfo() {
  // Allocate node information
  Info = new NodeInfo[NodeCount];
  
  unsigned i = 0;
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I, ++i) {
    // Fast reference to node schedule info
    NodeInfo* NI = &Info[i];
    // Set up map
    Map[I] = NI;
    // Set node
    NI->Node = I;
    // Set pending visit count
    NI->setPending(I->use_size());
  }
}

/// IdentifyGroups - Put flagged nodes into groups.
///
void ScheduleDAGSimple::IdentifyGroups() {
  for (unsigned i = 0, N = NodeCount; i < N; i++) {
    NodeInfo* NI = &Info[i];
    SDNode *Node = NI->Node;
    
    // For each operand (in reverse to only look at flags)
    for (unsigned N = Node->getNumOperands(); 0 < N--;) {
      // Get operand
      SDOperand Op = Node->getOperand(N);
      // No more flags to walk
      if (Op.getValueType() != MVT::Flag) break;
      // Add to node group
      AddToGroup(getNI(Op.Val), NI);
      // Let everyone else know
      HasGroups = true;
    }
  }
}

/// CountInternalUses - Returns the number of edges between the two nodes.
///
static unsigned CountInternalUses(NodeInfo *D, NodeInfo *U) {
  unsigned N = 0;
  for (unsigned M = U->Node->getNumOperands(); 0 < M--;) {
    SDOperand Op = U->Node->getOperand(M);
    if (Op.Val == D->Node) N++;
  }
  
  return N;
}

//===----------------------------------------------------------------------===//
/// Add - Adds a definer and user pair to a node group.
///
void ScheduleDAGSimple::AddToGroup(NodeInfo *D, NodeInfo *U) {
  // Get current groups
  NodeGroup *DGroup = D->Group;
  NodeGroup *UGroup = U->Group;
  // If both are members of groups
  if (DGroup && UGroup) {
    // There may have been another edge connecting 
    if (DGroup == UGroup) return;
    // Add the pending users count
    DGroup->addPending(UGroup->getPending());
    // For each member of the users group
    NodeGroupIterator UNGI(U);
    while (NodeInfo *UNI = UNGI.next() ) {
      // Change the group
      UNI->Group = DGroup;
      // For each member of the definers group
      NodeGroupIterator DNGI(D);
      while (NodeInfo *DNI = DNGI.next() ) {
        // Remove internal edges
        DGroup->addPending(-CountInternalUses(DNI, UNI));
      }
    }
    // Merge the two lists
    DGroup->group_insert(DGroup->group_end(),
                         UGroup->group_begin(), UGroup->group_end());
  } else if (DGroup) {
    // Make user member of definers group
    U->Group = DGroup;
    // Add users uses to definers group pending
    DGroup->addPending(U->Node->use_size());
    // For each member of the definers group
    NodeGroupIterator DNGI(D);
    while (NodeInfo *DNI = DNGI.next() ) {
      // Remove internal edges
      DGroup->addPending(-CountInternalUses(DNI, U));
    }
    DGroup->group_push_back(U);
  } else if (UGroup) {
    // Make definer member of users group
    D->Group = UGroup;
    // Add definers uses to users group pending
    UGroup->addPending(D->Node->use_size());
    // For each member of the users group
    NodeGroupIterator UNGI(U);
    while (NodeInfo *UNI = UNGI.next() ) {
      // Remove internal edges
      UGroup->addPending(-CountInternalUses(D, UNI));
    }
    UGroup->group_insert(UGroup->group_begin(), D);
  } else {
    D->Group = U->Group = DGroup = new NodeGroup();
    DGroup->addPending(D->Node->use_size() + U->Node->use_size() -
                       CountInternalUses(D, U));
    DGroup->group_push_back(D);
    DGroup->group_push_back(U);
    
    if (HeadNG == NULL)
      HeadNG = DGroup;
    if (TailNG != NULL)
      TailNG->Next = DGroup;
    TailNG = DGroup;
  }
}


/// print - Print ordering to specified output stream.
///
void ScheduleDAGSimple::print(std::ostream &O) const {
#ifndef NDEBUG
  O << "Ordering\n";
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    NodeInfo *NI = Ordering[i];
    printNI(O, NI);
    O << "\n";
    if (NI->isGroupDominator()) {
      NodeGroup *Group = NI->Group;
      for (NIIterator NII = Group->group_begin(), E = Group->group_end();
           NII != E; NII++) {
        O << "    ";
        printNI(O, *NII);
        O << "\n";
      }
    }
  }
#endif
}

void ScheduleDAGSimple::dump(const char *tag) const {
  std::cerr << tag; dump();
}

void ScheduleDAGSimple::dump() const {
  print(std::cerr);
}


/// EmitAll - Emit all nodes in schedule sorted order.
///
void ScheduleDAGSimple::EmitAll() {
  // If this is the first basic block in the function, and if it has live ins
  // that need to be copied into vregs, emit the copies into the top of the
  // block before emitting the code for the block.
  MachineFunction &MF = DAG.getMachineFunction();
  if (&MF.front() == BB && MF.livein_begin() != MF.livein_end()) {
    for (MachineFunction::livein_iterator LI = MF.livein_begin(),
         E = MF.livein_end(); LI != E; ++LI)
      if (LI->second)
        MRI->copyRegToReg(*MF.begin(), MF.begin()->end(), LI->second,
                          LI->first, RegMap->getRegClass(LI->second));
  }
  
  std::map<SDNode*, unsigned> VRBaseMap;
  
  // For each node in the ordering
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    // Get the scheduling info
    NodeInfo *NI = Ordering[i];
    if (NI->isInGroup()) {
      NodeGroupIterator NGI(Ordering[i]);
      while (NodeInfo *NI = NGI.next()) EmitNode(NI->Node, VRBaseMap);
    } else {
      EmitNode(NI->Node, VRBaseMap);
    }
  }
}

/// isFlagDefiner - Returns true if the node defines a flag result.
static bool isFlagDefiner(SDNode *A) {
  unsigned N = A->getNumValues();
  return N && A->getValueType(N - 1) == MVT::Flag;
}

/// isFlagUser - Returns true if the node uses a flag result.
///
static bool isFlagUser(SDNode *A) {
  unsigned N = A->getNumOperands();
  return N && A->getOperand(N - 1).getValueType() == MVT::Flag;
}

/// printNI - Print node info.
///
void ScheduleDAGSimple::printNI(std::ostream &O, NodeInfo *NI) const {
#ifndef NDEBUG
  SDNode *Node = NI->Node;
  O << " "
    << std::hex << Node << std::dec
    << ", Lat=" << NI->Latency
    << ", Slot=" << NI->Slot
    << ", ARITY=(" << Node->getNumOperands() << ","
    << Node->getNumValues() << ")"
    << " " << Node->getOperationName(&DAG);
  if (isFlagDefiner(Node)) O << "<#";
  if (isFlagUser(Node)) O << ">#";
#endif
}

/// printChanges - Hilight changes in order caused by scheduling.
///
void ScheduleDAGSimple::printChanges(unsigned Index) const {
#ifndef NDEBUG
  // Get the ordered node count
  unsigned N = Ordering.size();
  // Determine if any changes
  unsigned i = 0;
  for (; i < N; i++) {
    NodeInfo *NI = Ordering[i];
    if (NI->Preorder != i) break;
  }
  
  if (i < N) {
    std::cerr << Index << ". New Ordering\n";
    
    for (i = 0; i < N; i++) {
      NodeInfo *NI = Ordering[i];
      std::cerr << "  " << NI->Preorder << ". ";
      printNI(std::cerr, NI);
      std::cerr << "\n";
      if (NI->isGroupDominator()) {
        NodeGroup *Group = NI->Group;
        for (NIIterator NII = Group->group_begin(), E = Group->group_end();
             NII != E; NII++) {
          std::cerr << "          ";
          printNI(std::cerr, *NII);
          std::cerr << "\n";
        }
      }
    }
  } else {
    std::cerr << Index << ". No Changes\n";
  }
#endif
}

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
    if (InstrItins.isEmpty() || NoItins) {
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
  // Number the nodes
  NodeCount = std::distance(DAG.allnodes_begin(), DAG.allnodes_end());

  // Set up minimum info for scheduling
  PrepareNodeInfo();
  // Construct node groups for flagged nodes
  IdentifyGroups();
  
  // Test to see if scheduling should occur
  bool ShouldSchedule = NodeCount > 3 && !NoSched;
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
/// scheduler using instruction itinerary.
llvm::ScheduleDAG* llvm::createSimpleDAGScheduler(SelectionDAGISel *IS,
                                                  SelectionDAG *DAG,
                                                  MachineBasicBlock *BB) {
  return new ScheduleDAGSimple(false, false, *DAG, BB, DAG->getTarget());
}

/// createNoItinsDAGScheduler - This creates a simple two pass instruction
/// scheduler without using instruction itinerary.
llvm::ScheduleDAG* llvm::createNoItinsDAGScheduler(SelectionDAGISel *IS,
                                                   SelectionDAG *DAG,
                                                   MachineBasicBlock *BB) {
  return new ScheduleDAGSimple(false, true, *DAG, BB, DAG->getTarget());
}

/// createBFS_DAGScheduler - This creates a simple breadth first instruction
/// scheduler.
llvm::ScheduleDAG* llvm::createBFS_DAGScheduler(SelectionDAGISel *IS,
                                                SelectionDAG *DAG,
                                                MachineBasicBlock *BB) {
  return new ScheduleDAGSimple(true, false, *DAG, BB,  DAG->getTarget());
}
