//===-- ScheduleDAG.cpp - Implement a trivial DAG scheduler ---------------===//
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
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetInstrItineraries.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <algorithm>
using namespace llvm;

namespace {
  // Style of scheduling to use.
  enum ScheduleChoices {
    noScheduling,
    simpleScheduling,
    simpleNoItinScheduling
  };
} // namespace

cl::opt<ScheduleChoices> ScheduleStyle("sched",
  cl::desc("Choose scheduling style"),
  cl::init(noScheduling),
  cl::values(
    clEnumValN(noScheduling, "none",
              "Trivial emission with no analysis"),
    clEnumValN(simpleScheduling, "simple",
              "Minimize critical path and maximize processor utilization"),
    clEnumValN(simpleNoItinScheduling, "simple-noitin",
              "Same as simple except using generic latency"),
   clEnumValEnd));


#ifndef NDEBUG
static cl::opt<bool>
ViewDAGs("view-sched-dags", cl::Hidden,
         cl::desc("Pop up a window to show sched dags as they are processed"));
#else
static const bool ViewDAGs = 0;
#endif

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

// Forward
class NodeInfo;
typedef NodeInfo *NodeInfoPtr;
typedef std::vector<NodeInfoPtr>           NIVector;
typedef std::vector<NodeInfoPtr>::iterator NIIterator;

//===----------------------------------------------------------------------===//
///
/// Node group -  This struct is used to manage flagged node groups.
///
class NodeGroup {
private:
  NIVector      Members;                // Group member nodes
  NodeInfo      *Dominator;             // Node with highest latency
  unsigned      Latency;                // Total latency of the group
  int           Pending;                // Number of visits pending before
                                        //    adding to order  

public:
  // Ctor.
  NodeGroup() : Dominator(NULL), Pending(0) {}
  
  // Accessors
  inline void setDominator(NodeInfo *D) { Dominator = D; }
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
  inline void group_push_back(const NodeInfoPtr &NI) { Members.push_back(NI); }
  inline NIIterator group_insert(NIIterator Pos, const NodeInfoPtr &NI) {
    return Members.insert(Pos, NI);
  }
  inline void group_insert(NIIterator Pos, NIIterator First, NIIterator Last) {
    Members.insert(Pos, First, Last);
  }

  static void Add(NodeInfo *D, NodeInfo *U);
  static unsigned CountInternalUses(NodeInfo *D, NodeInfo *U);
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// NodeInfo - This struct tracks information used to schedule the a node.
///
class NodeInfo {
private:
  int           Pending;                // Number of visits pending before
                                        //    adding to order
public:
  SDNode        *Node;                  // DAG node
  InstrStage    *StageBegin;            // First stage in itinerary
  InstrStage    *StageEnd;              // Last+1 stage in itinerary
  unsigned      Latency;                // Total cycles to complete instruction
  bool          IsCall : 1;             // Is function call
  bool          IsLoad : 1;             // Is memory load
  bool          IsStore : 1;            // Is memory store
  unsigned      Slot;                   // Node's time slot
  NodeGroup     *Group;                 // Grouping information
  unsigned      VRBase;                 // Virtual register base
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
  , VRBase(0)
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
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// NodeGroupIterator - Iterates over all the nodes indicated by the node info.
/// If the node is in a group then iterate over the members of the group,
/// otherwise just the node info.
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
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// NodeGroupOpIterator - Iterates over all the operands of a node.  If the node
/// is a member of a group, this iterates over all the operands of all the
/// members of the group.
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
    assert(OI != OE && "Not checking for end of NodeGroupOpIterator correctly");
    return *OI++;
  }
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// SimpleSched - Simple two pass scheduler.
///
class SimpleSched {
private:
  MachineBasicBlock *BB;                // Current basic block
  SelectionDAG &DAG;                    // DAG of the current basic block
  const TargetMachine &TM;              // Target processor
  const TargetInstrInfo &TII;           // Target instruction information
  const MRegisterInfo &MRI;             // Target processor register information
  SSARegMap *RegMap;                    // Virtual/real register map
  MachineConstantPool *ConstPool;       // Target constant pool
  unsigned NodeCount;                   // Number of nodes in DAG
  bool HasGroups;                       // True if there are any groups
  NodeInfo *Info;                       // Info for nodes being scheduled
  std::map<SDNode *, NodeInfo *> Map;   // Map nodes to info
  NIVector Ordering;                    // Emit ordering of nodes
  ResourceTally<unsigned> Tally;        // Resource usage tally
  unsigned NSlots;                      // Total latency
  static const unsigned NotFound = ~0U; // Search marker
  
public:

  // Ctor.
  SimpleSched(SelectionDAG &D, MachineBasicBlock *bb)
    : BB(bb), DAG(D), TM(D.getTarget()), TII(*TM.getInstrInfo()),
      MRI(*TM.getRegisterInfo()), RegMap(BB->getParent()->getSSARegMap()),
      ConstPool(BB->getParent()->getConstantPool()),
      NodeCount(0), HasGroups(false), Info(NULL), Map(), Tally(), NSlots(0) {
    assert(&TII && "Target doesn't provide instr info?");
    assert(&MRI && "Target doesn't provide register info?");
  }
  
  // Run - perform scheduling.
  MachineBasicBlock *Run() {
    Schedule();
    return BB;
  }
  
private:
  /// getNI - Returns the node info for the specified node.
  ///
  inline NodeInfo *getNI(SDNode *Node) { return Map[Node]; }
  
  /// getVR - Returns the virtual register number of the node.
  ///
  inline unsigned getVR(SDOperand Op) {
    NodeInfo *NI = getNI(Op.Val);
    assert(NI->VRBase != 0 && "Node emitted out of order - late");
    return NI->VRBase + Op.ResNo;
  }

  static bool isFlagDefiner(SDNode *A);
  static bool isFlagUser(SDNode *A);
  static bool isDefiner(NodeInfo *A, NodeInfo *B);
  static bool isPassiveNode(SDNode *Node);
  void IncludeNode(NodeInfo *NI);
  void VisitAll();
  void Schedule();
  void IdentifyGroups();
  void GatherSchedulingInfo();
  void FakeGroupDominators(); 
  void PrepareNodeInfo();
  bool isStrongDependency(NodeInfo *A, NodeInfo *B);
  bool isWeakDependency(NodeInfo *A, NodeInfo *B);
  void ScheduleBackward();
  void ScheduleForward();
  void EmitAll();
  void EmitNode(NodeInfo *NI);
  static unsigned CountResults(SDNode *Node);
  static unsigned CountOperands(SDNode *Node);
  unsigned CreateVirtualRegisters(MachineInstr *MI,
                                  unsigned NumResults,
                                  const TargetInstrDescriptor &II);

  void printChanges(unsigned Index);
  void printSI(std::ostream &O, NodeInfo *NI) const;
  void print(std::ostream &O) const;
  inline void dump(const char *tag) const { std::cerr << tag; dump(); }
  void dump() const;
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


//===----------------------------------------------------------------------===//

} // namespace

//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Add - Adds a definer and user pair to a node group.
///
void NodeGroup::Add(NodeInfo *D, NodeInfo *U) {
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
  }
}

/// CountInternalUses - Returns the number of edges between the two nodes.
///
unsigned NodeGroup::CountInternalUses(NodeInfo *D, NodeInfo *U) {
  unsigned N = 0;
  for (unsigned M = U->Node->getNumOperands(); 0 < M--;) {
    SDOperand Op = U->Node->getOperand(M);
    if (Op.Val == D->Node) N++;
  }

  return N;
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// isFlagDefiner - Returns true if the node defines a flag result.
bool SimpleSched::isFlagDefiner(SDNode *A) {
  unsigned N = A->getNumValues();
  return N && A->getValueType(N - 1) == MVT::Flag;
}

/// isFlagUser - Returns true if the node uses a flag result.
///
bool SimpleSched::isFlagUser(SDNode *A) {
  unsigned N = A->getNumOperands();
  return N && A->getOperand(N - 1).getValueType() == MVT::Flag;
}

/// isDefiner - Return true if node A is a definer for B.
///
bool SimpleSched::isDefiner(NodeInfo *A, NodeInfo *B) {
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

/// isPassiveNode - Return true if the node is a non-scheduled leaf.
///
bool SimpleSched::isPassiveNode(SDNode *Node) {
  if (isa<ConstantSDNode>(Node))       return true;
  if (isa<RegisterSDNode>(Node))       return true;
  if (isa<GlobalAddressSDNode>(Node))  return true;
  if (isa<BasicBlockSDNode>(Node))     return true;
  if (isa<FrameIndexSDNode>(Node))     return true;
  if (isa<ConstantPoolSDNode>(Node))   return true;
  if (isa<ExternalSymbolSDNode>(Node)) return true;
  return false;
}

/// IncludeNode - Add node to NodeInfo vector.
///
void SimpleSched::IncludeNode(NodeInfo *NI) {
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

/// VisitAll - Visit each node breadth-wise to produce an initial ordering.
/// Note that the ordering in the Nodes vector is reversed.
void SimpleSched::VisitAll() {
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

/// IdentifyGroups - Put flagged nodes into groups.
///
void SimpleSched::IdentifyGroups() {
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
      NodeGroup::Add(getNI(Op.Val), NI);
      // Let evryone else know
      HasGroups = true;
    }
  }
}

/// GatherSchedulingInfo - Get latency and resource information about each node.
///
void SimpleSched::GatherSchedulingInfo() {
  // Get instruction itineraries for the target
  const InstrItineraryData InstrItins = TM.getInstrItineraryData();
  
  // For each node
  for (unsigned i = 0, N = NodeCount; i < N; i++) {
    // Get node info
    NodeInfo* NI = &Info[i];
    SDNode *Node = NI->Node;
    
    // If there are itineraries and it is a machine instruction
    if (InstrItins.isEmpty() || ScheduleStyle == simpleNoItinScheduling) {
      // If machine opcode
      if (Node->isTargetOpcode()) {
        // Get return type to guess which processing unit 
        MVT::ValueType VT = Node->getValueType(0);
        // Get machine opcode
        MachineOpCode TOpc = Node->getTargetOpcode();
        NI->IsCall = TII.isCall(TOpc);
        NI->IsLoad = TII.isLoad(TOpc);
        NI->IsStore = TII.isStore(TOpc);

        if (TII.isLoad(TOpc))              NI->StageBegin = &LoadStage;
        else if (TII.isStore(TOpc))        NI->StageBegin = &StoreStage;
        else if (MVT::isInteger(VT))       NI->StageBegin = &IntStage;
        else if (MVT::isFloatingPoint(VT)) NI->StageBegin = &FloatStage;
        if (NI->StageBegin) NI->StageEnd = NI->StageBegin + 1;
      }
    } else if (Node->isTargetOpcode()) {
      // get machine opcode
      MachineOpCode TOpc = Node->getTargetOpcode();
      // Check to see if it is a call
      NI->IsCall = TII.isCall(TOpc);
      // Get itinerary stages for instruction
      unsigned II = TII.getSchedClass(TOpc);
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

/// FakeGroupDominators - Set dominators for non-scheduling.
/// 
void SimpleSched::FakeGroupDominators() {
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

/// PrepareNodeInfo - Set up the basic minimum node info for scheduling.
/// 
void SimpleSched::PrepareNodeInfo() {
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

/// isStrongDependency - Return true if node A has results used by node B. 
/// I.E., B must wait for latency of A.
bool SimpleSched::isStrongDependency(NodeInfo *A, NodeInfo *B) {
  // If A defines for B then it's a strong dependency or
  // if a load follows a store (may be dependent but why take a chance.)
  return isDefiner(A, B) || (A->IsStore && B->IsLoad);
}

/// isWeakDependency Return true if node A produces a result that will
/// conflict with operands of B.  It is assumed that we have called
/// isStrongDependency prior.
bool SimpleSched::isWeakDependency(NodeInfo *A, NodeInfo *B) {
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
void SimpleSched::ScheduleBackward() {
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
void SimpleSched::ScheduleForward() {
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

/// EmitAll - Emit all nodes in schedule sorted order.
///
void SimpleSched::EmitAll() {
  // For each node in the ordering
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    // Get the scheduling info
    NodeInfo *NI = Ordering[i];
    if (NI->isInGroup()) {
      NodeGroupIterator NGI(Ordering[i]);
      while (NodeInfo *NI = NGI.next()) EmitNode(NI);
    } else {
      EmitNode(NI);
    }
  }
}

/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the machine instrs.)
unsigned SimpleSched::CountResults(SDNode *Node) {
  unsigned N = Node->getNumValues();
  while (N && Node->getValueType(N - 1) == MVT::Flag)
    --N;
  if (N && Node->getValueType(N - 1) == MVT::Other)
    --N;    // Skip over chain result.
  return N;
}

/// CountOperands  The inputs to target nodes have any actual inputs first,
/// followed by an optional chain operand, then flag operands.  Compute the
/// number of actual operands that  will go into the machine instr.
unsigned SimpleSched::CountOperands(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Node->getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}

/// CreateVirtualRegisters - Add result register values for things that are
/// defined by this instruction.
unsigned SimpleSched::CreateVirtualRegisters(MachineInstr *MI,
                                             unsigned NumResults,
                                             const TargetInstrDescriptor &II) {
  // Create the result registers for this node and add the result regs to
  // the machine instruction.
  const TargetOperandInfo *OpInfo = II.OpInfo;
  unsigned ResultReg = RegMap->createVirtualRegister(OpInfo[0].RegClass);
  MI->addRegOperand(ResultReg, MachineOperand::Def);
  for (unsigned i = 1; i != NumResults; ++i) {
    assert(OpInfo[i].RegClass && "Isn't a register operand!");
    MI->addRegOperand(RegMap->createVirtualRegister(OpInfo[i].RegClass),
                      MachineOperand::Def);
  }
  return ResultReg;
}

/// EmitNode - Generate machine code for an node and needed dependencies.
///
void SimpleSched::EmitNode(NodeInfo *NI) {
  unsigned VRBase = 0;                 // First virtual register for node
  SDNode *Node = NI->Node;
  
  // If machine instruction
  if (Node->isTargetOpcode()) {
    unsigned Opc = Node->getTargetOpcode();
    const TargetInstrDescriptor &II = TII.get(Opc);

    unsigned NumResults = CountResults(Node);
    unsigned NodeOperands = CountOperands(Node);
    unsigned NumMIOperands = NodeOperands + NumResults;
#ifndef NDEBUG
    assert((unsigned(II.numOperands) == NumMIOperands || II.numOperands == -1)&&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, NumMIOperands, true, true);
    
    // Add result register values for things that are defined by this
    // instruction.
    
    // If the node is only used by a CopyToReg and the dest reg is a vreg, use
    // the CopyToReg'd destination register instead of creating a new vreg.
    if (NumResults == 1) {
      for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
           UI != E; ++UI) {
        SDNode *Use = *UI;
        if (Use->getOpcode() == ISD::CopyToReg && 
            Use->getOperand(2).Val == Node) {
          unsigned Reg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
          if (MRegisterInfo::isVirtualRegister(Reg)) {
            VRBase = Reg;
            MI->addRegOperand(Reg, MachineOperand::Def);
            break;
          }
        }
      }
    }
    
    // Otherwise, create new virtual registers.
    if (NumResults && VRBase == 0)
      VRBase = CreateVirtualRegisters(MI, NumResults, II);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i) {
      if (Node->getOperand(i).isTargetOpcode()) {
        // Note that this case is redundant with the final else block, but we
        // include it because it is the most common and it makes the logic
        // simpler here.
        assert(Node->getOperand(i).getValueType() != MVT::Other &&
               Node->getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");

        // Get/emit the operand.
        unsigned VReg = getVR(Node->getOperand(i));
        MI->addRegOperand(VReg, MachineOperand::Use);
        
        // Verify that it is right.
        assert(MRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
        assert(II.OpInfo[i+NumResults].RegClass &&
               "Don't have operand info for this instruction!");
        assert(RegMap->getRegClass(VReg) == II.OpInfo[i+NumResults].RegClass &&
               "Register class of operand and regclass of use don't agree!");
      } else if (ConstantSDNode *C =
                 dyn_cast<ConstantSDNode>(Node->getOperand(i))) {
        MI->addZeroExtImm64Operand(C->getValue());
      } else if (RegisterSDNode*R =
                 dyn_cast<RegisterSDNode>(Node->getOperand(i))) {
        MI->addRegOperand(R->getReg(), MachineOperand::Use);
      } else if (GlobalAddressSDNode *TGA =
                       dyn_cast<GlobalAddressSDNode>(Node->getOperand(i))) {
        MI->addGlobalAddressOperand(TGA->getGlobal(), false, TGA->getOffset());
      } else if (BasicBlockSDNode *BB =
                       dyn_cast<BasicBlockSDNode>(Node->getOperand(i))) {
        MI->addMachineBasicBlockOperand(BB->getBasicBlock());
      } else if (FrameIndexSDNode *FI =
                       dyn_cast<FrameIndexSDNode>(Node->getOperand(i))) {
        MI->addFrameIndexOperand(FI->getIndex());
      } else if (ConstantPoolSDNode *CP = 
                    dyn_cast<ConstantPoolSDNode>(Node->getOperand(i))) {
        unsigned Idx = ConstPool->getConstantPoolIndex(CP->get());
        MI->addConstantPoolIndexOperand(Idx);
      } else if (ExternalSymbolSDNode *ES = 
                 dyn_cast<ExternalSymbolSDNode>(Node->getOperand(i))) {
        MI->addExternalSymbolOperand(ES->getSymbol(), false);
      } else {
        assert(Node->getOperand(i).getValueType() != MVT::Other &&
               Node->getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        unsigned VReg = getVR(Node->getOperand(i));
        MI->addRegOperand(VReg, MachineOperand::Use);
        
        // Verify that it is right.
        assert(MRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
        assert(II.OpInfo[i+NumResults].RegClass &&
               "Don't have operand info for this instruction!");
        assert(RegMap->getRegClass(VReg) == II.OpInfo[i+NumResults].RegClass &&
               "Register class of operand and regclass of use don't agree!");
      }
    }
    
    // Now that we have emitted all operands, emit this instruction itself.
    if ((II.Flags & M_USES_CUSTOM_DAG_SCHED_INSERTION) == 0) {
      BB->insert(BB->end(), MI);
    } else {
      // Insert this instruction into the end of the basic block, potentially
      // taking some custom action.
      BB = DAG.getTargetLoweringInfo().InsertAtEndOfBasicBlock(MI, BB);
    }
  } else {
    switch (Node->getOpcode()) {
    default:
      Node->dump(); 
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: // fall thru
    case ISD::TokenFactor:
      break;
    case ISD::CopyToReg: {
      unsigned InReg = getVR(Node->getOperand(2));
      unsigned DestReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      if (InReg != DestReg)   // Coallesced away the copy?
        MRI.copyRegToReg(*BB, BB->end(), DestReg, InReg,
                         RegMap->getRegClass(InReg));
      break;
    }
    case ISD::CopyFromReg: {
      unsigned SrcReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      if (MRegisterInfo::isVirtualRegister(SrcReg)) {
        VRBase = SrcReg;  // Just use the input register directly!
        break;
      }

      // If the node is only used by a CopyToReg and the dest reg is a vreg, use
      // the CopyToReg'd destination register instead of creating a new vreg.
      for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
           UI != E; ++UI) {
        SDNode *Use = *UI;
        if (Use->getOpcode() == ISD::CopyToReg && 
            Use->getOperand(2).Val == Node) {
          unsigned DestReg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
          if (MRegisterInfo::isVirtualRegister(DestReg)) {
            VRBase = DestReg;
            break;
          }
        }
      }

      // Figure out the register class to create for the destreg.
      const TargetRegisterClass *TRC = 0;
      if (VRBase) {
        TRC = RegMap->getRegClass(VRBase);
      } else {

        // Pick the register class of the right type that contains this physreg.
        for (MRegisterInfo::regclass_iterator I = MRI.regclass_begin(),
             E = MRI.regclass_end(); I != E; ++I)
          if ((*I)->hasType(Node->getValueType(0)) &&
              (*I)->contains(SrcReg)) {
            TRC = *I;
            break;
          }
        assert(TRC && "Couldn't find register class for reg copy!");
      
        // Create the reg, emit the copy.
        VRBase = RegMap->createVirtualRegister(TRC);
      }
      MRI.copyRegToReg(*BB, BB->end(), VRBase, SrcReg, TRC);
      break;
    }
    }
  }

  assert(NI->VRBase == 0 && "Node emitted out of order - early");
  NI->VRBase = VRBase;
}

/// Schedule - Order nodes according to selected style.
///
void SimpleSched::Schedule() {
  // Number the nodes
  NodeCount = std::distance(DAG.allnodes_begin(), DAG.allnodes_end());
  // Test to see if scheduling should occur
  bool ShouldSchedule = NodeCount > 3 && ScheduleStyle != noScheduling;
  // Set up minimum info for scheduling
  PrepareNodeInfo();
  // Construct node groups for flagged nodes
  IdentifyGroups();

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

/// printChanges - Hilight changes in order caused by scheduling.
///
void SimpleSched::printChanges(unsigned Index) {
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
      printSI(std::cerr, NI);
      std::cerr << "\n";
      if (NI->isGroupDominator()) {
        NodeGroup *Group = NI->Group;
        for (NIIterator NII = Group->group_begin(), E = Group->group_end();
             NII != E; NII++) {
          std::cerr << "          ";
          printSI(std::cerr, *NII);
          std::cerr << "\n";
        }
      }
    }
  } else {
    std::cerr << Index << ". No Changes\n";
  }
#endif
}

/// printSI - Print schedule info.
///
void SimpleSched::printSI(std::ostream &O, NodeInfo *NI) const {
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

/// print - Print ordering to specified output stream.
///
void SimpleSched::print(std::ostream &O) const {
#ifndef NDEBUG
  using namespace std;
  O << "Ordering\n";
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    NodeInfo *NI = Ordering[i];
    printSI(O, NI);
    O << "\n";
    if (NI->isGroupDominator()) {
      NodeGroup *Group = NI->Group;
      for (NIIterator NII = Group->group_begin(), E = Group->group_end();
           NII != E; NII++) {
        O << "    ";
        printSI(O, *NII);
        O << "\n";
      }
    }
  }
#endif
}

/// dump - Print ordering to std::cerr.
///
void SimpleSched::dump() const {
  print(std::cerr);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// ScheduleAndEmitDAG - Pick a safe ordering and emit instructions for each
/// target node in the graph.
void SelectionDAGISel::ScheduleAndEmitDAG(SelectionDAG &SD) {
  if (ViewDAGs) SD.viewGraph();
  BB = SimpleSched(SD, BB).Run();  
}
