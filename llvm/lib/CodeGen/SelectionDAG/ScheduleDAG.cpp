//===-- ScheduleDAG.cpp - Implement a trivial DAG scheduler ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
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
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;

namespace {
  // Style of scheduling to use.
  enum ScheduleChoices {
    noScheduling,
    simpleScheduling,
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
  
  /// AllInUse - Test to see if all of the resources in the slot are busy (set.)
  inline bool AllInUse(Iter Cursor, unsigned ResourceSet) {
    return (*Cursor & ResourceSet) == ResourceSet;
  }

  /// Skip - Skip over slots that use all of the specified resource (all are
  /// set.)
  Iter Skip(Iter Cursor, unsigned ResourceSet) {
    assert(ResourceSet && "At least one resource bit needs to bet set");
    
    // Continue to the end
    while (true) {
      // Break out if one of the resource bits is not set
      if (!AllInUse(Cursor, ResourceSet)) return Cursor;
      // Try next slot
      Cursor++;
      assert(Cursor < Tally.end() && "Tally is not large enough for schedule");
    }
  }
  
  /// FindSlots - Starting from Begin, locate N consecutive slots where at least 
  /// one of the resource bits is available.  Returns the address of first slot.
  Iter FindSlots(Iter Begin, unsigned N, unsigned ResourceSet,
                                         unsigned &Resource) {
    // Track position      
    Iter Cursor = Begin;
    
    // Try all possible slots forward
    while (true) {
      // Skip full slots
      Cursor = Skip(Cursor, ResourceSet);
      // Determine end of interval
      Iter End = Cursor + N;
      assert(End <= Tally.end() && "Tally is not large enough for schedule");
      
      // Iterate thru each resource
      BitsIterator<T> Resources(ResourceSet & ~*Cursor);
      while (unsigned Res = Resources.Next()) {
        // Check if resource is available for next N slots
        // Break out if resource is busy
        Iter Interval = Cursor;
        for (; Interval < End && !(*Interval & Res); Interval++) {}
        
        // If available for interval, return where and which resource
        if (Interval == End) {
          Resource = Res;
          return Cursor;
        }
        // Otherwise, check if worth checking other resources
        if (AllInUse(Interval, ResourceSet)) {
          // Start looking beyond interval
          Cursor = Interval;
          break;
        }
      }
      Cursor++;
    }
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

public:
  /// Initialize - Resize and zero the tally to the specified number of time
  /// slots.
  inline void Initialize(unsigned N) {
    Tally.assign(N, 0);   // Initialize tally to all zeros.
  }
  
  // FindAndReserve - Locate and mark busy (set) N bits started at slot I, using
  // ResourceSet for choices.
  unsigned FindAndReserve(unsigned I, unsigned N, unsigned ResourceSet) {
    // Which resource used
    unsigned Resource;
    // Find slots for instruction.
    Iter Where = FindSlots(Tally.begin() + I, N, ResourceSet, Resource);
    // Reserve the slots
    Reserve(Where, N, Resource);
    // Return time slot (index)
    return Where - Tally.begin();
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// This struct tracks information used to schedule the a node.
struct ScheduleInfo {
  SDOperand     Op;                     // Operand information
  unsigned      Latency;                // Cycles to complete instruction
  unsigned      ResourceSet;            // Bit vector of usable resources
  unsigned      Slot;                   // Operand's time slot
  
  // Ctor.
  ScheduleInfo(SDOperand op)
  : Op(op)
  , Latency(0)
  , ResourceSet(0)
  , Slot(0)
  {}
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
class SimpleSched {
private:
  // TODO - get ResourceSet from TII
  enum {
    RSInteger = 0x3,                    // Two integer units
    RSFloat = 0xC,                      // Two float units
    RSLoadStore = 0x30,                 // Two load store units
    RSOther = 0                         // Processing unit independent
  };
  
  MachineBasicBlock *BB;                // Current basic block
  SelectionDAG &DAG;                    // DAG of the current basic block
  const TargetMachine &TM;              // Target processor
  const TargetInstrInfo &TII;           // Target instruction information
  const MRegisterInfo &MRI;             // Target processor register information
  SSARegMap *RegMap;                    // Virtual/real register map
  MachineConstantPool *ConstPool;       // Target constant pool
  std::vector<ScheduleInfo> Operands;   // All operands to be scheduled
  std::vector<ScheduleInfo*> Ordering;  // Emit ordering of operands
  std::map<SDNode *, int> Visited;      // Operands that have been visited
  ResourceTally<unsigned> Tally;        // Resource usage tally
  unsigned NSlots;                      // Total latency
  std::map<SDNode *, unsigned>VRMap;    // Operand to VR map
  static const unsigned NotFound = ~0U; // Search marker
  
public:

  // Ctor.
  SimpleSched(SelectionDAG &D, MachineBasicBlock *bb)
    : BB(bb), DAG(D), TM(D.getTarget()), TII(*TM.getInstrInfo()),
      MRI(*TM.getRegisterInfo()), RegMap(BB->getParent()->getSSARegMap()),
      ConstPool(BB->getParent()->getConstantPool()),
      NSlots(0) {
    assert(&TII && "Target doesn't provide instr info?");
    assert(&MRI && "Target doesn't provide register info?");
  }
  
  // Run - perform scheduling.
  MachineBasicBlock *Run() {
    Schedule();
    return BB;
  }
  
private:
  static bool isFlagDefiner(SDOperand Op) { return isFlagDefiner(Op.Val); }
  static bool isFlagUser(SDOperand Op) { return isFlagUser(Op.Val); }
  static bool isFlagDefiner(SDNode *A);
  static bool isFlagUser(SDNode *A);
  static bool isDefiner(SDNode *A, SDNode *B);
  static bool isPassiveOperand(SDOperand Op);
  void IncludeOperand(SDOperand Op);
  void VisitAll();
  void Schedule();
  void GatherOperandInfo();
  bool isStrongDependency(SDOperand A, SDOperand B) {
    return isStrongDependency(A.Val, B.Val);
  }
  bool isWeakDependency(SDOperand A, SDOperand B) {
    return isWeakDependency(A.Val, B.Val);
  }
  static bool isStrongDependency(SDNode *A, SDNode *B);
  static bool isWeakDependency(SDNode *A, SDNode *B);
  void ScheduleBackward();
  void ScheduleForward();
  void EmitAll();
  void EmitFlagUsers(SDOperand Op);
  static unsigned CountResults(SDOperand Op);
  static unsigned CountOperands(SDOperand Op);
  unsigned CreateVirtualRegisters(SDOperand Op, MachineInstr *MI,
                                  unsigned NumResults,
                                  const TargetInstrDescriptor &II);
  unsigned Emit(SDOperand A);

  void printSI(std::ostream &O, ScheduleInfo *SI) const ;
  void print(std::ostream &O) const ;
  inline void dump(const char *tag) const { std::cerr << tag; dump(); }
  void dump() const;
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
class FlagUserIterator {
private:
  SDNode               *Definer;        // Node defining flag
  SDNode::use_iterator UI;              // User node iterator
  SDNode::use_iterator E;               // End of user nodes
  unsigned             MinRes;          // Minimum flag result

public:
  // Ctor.
  FlagUserIterator(SDNode *D)
  : Definer(D)
  , UI(D->use_begin())
  , E(D->use_end())
  , MinRes(D->getNumValues()) {
    // Find minimum flag result.
    while (MinRes && D->getValueType(MinRes - 1) == MVT::Flag) --MinRes;
  }
  
  /// isFlagUser - Return true if  node uses definer's flag.
  bool isFlagUser(SDNode *U) {
    // For each operand (in reverse to only look at flags)
    for (unsigned N = U->getNumOperands(); 0 < N--;) {
      // Get operand
      SDOperand Op = U->getOperand(N);
      // Not user if there are no flags
      if (Op.getValueType() != MVT::Flag) return false;
      // Return true if it is one of the flag results 
      if (Op.Val == Definer && Op.ResNo >= MinRes) return true;
    }
    // Not a flag user
    return false;
  }
  
  SDNode *next() {
    // Continue to next user
    while (UI != E) {
      // Next user node
      SDNode *User = *UI++;
      // Return true if is a flag user
      if (isFlagUser(User)) return User;
    }
    
    // No more user nodes
    return NULL;
  }
};

} // namespace


//===----------------------------------------------------------------------===//
/// isFlagDefiner - Returns true if the operand defines a flag result.
bool SimpleSched::isFlagDefiner(SDNode *A) {
  unsigned N = A->getNumValues();
  return N && A->getValueType(N - 1) == MVT::Flag;
}

/// isFlagUser - Returns true if the operand uses a flag result.
///
bool SimpleSched::isFlagUser(SDNode *A) {
  unsigned N = A->getNumOperands();
  return N && A->getOperand(N - 1).getValueType() == MVT::Flag;
}

/// isDefiner - Return true if Node A is a definder for B.
///
bool SimpleSched::isDefiner(SDNode *A, SDNode *B) {
  for (unsigned i = 0, N = B->getNumOperands(); i < N; i++) {
    if (B->getOperand(i).Val == A) return true;
  }
  return false;
}

/// isPassiveOperand - Return true if the operand is a non-scheduled leaf
/// operand.
bool SimpleSched::isPassiveOperand(SDOperand Op) {
  if (isa<ConstantSDNode>(Op))       return true;
  if (isa<RegisterSDNode>(Op))       return true;
  if (isa<GlobalAddressSDNode>(Op))  return true;
  if (isa<BasicBlockSDNode>(Op))     return true;
  if (isa<FrameIndexSDNode>(Op))     return true;
  if (isa<ConstantPoolSDNode>(Op))   return true;
  if (isa<ExternalSymbolSDNode>(Op)) return true;
  return false;
}

/// IncludeOperand - Add operand to ScheduleInfo vector.
///
void SimpleSched::IncludeOperand(SDOperand Op) {
  // Ignore entry node
  if (Op.getOpcode() == ISD::EntryToken) return;
  // Check current count for operand
  int Count = Visited[Op.Val];
  // If the operand is already in list
  if (Count < 0) return;
  // If this the first time then get count 
  if (!Count) Count = Op.Val->use_size();
  // Decrement count to indicate a visit
  Count--;
  // If count has gone to zero then add operand to list
  if (!Count) {
    // Add operand
    Operands.push_back(ScheduleInfo(Op));
    // indicate operand has been added
    Count--;
  }
  // Mark as visited with new count 
  Visited[Op.Val] = Count;
}

/// VisitAll - Visit each operand breadth-wise to produce an initial ordering.
/// Note that the ordering in the Operands vector is reversed.
void SimpleSched::VisitAll() {
  // Add first element to list
  Operands.push_back(DAG.getRoot());
  for (unsigned i = 0; i < Operands.size(); i++) { // note: size() varies
    // Get next operand. Need copy because Operands vector is growing and
    // addresses can be ScheduleInfo changing.
    SDOperand Op = Operands[i].Op;
    // Get the number of real operands
    unsigned NodeOperands = CountOperands(Op);
    // Get the total number of operands
    unsigned NumOperands = Op.getNumOperands();

    // Visit all operands skipping the Other operand if present
    for (unsigned i = NumOperands; 0 < i--;) {
      SDOperand OpI = Op.getOperand(i);
      // Ignore passive operands
      if (isPassiveOperand(OpI)) continue;
      // Check out operand
      IncludeOperand(OpI);
    }
  }

  // Add entry node last (IncludeOperand filters entry nodes)
  if (DAG.getEntryNode().Val != DAG.getRoot().Val)
    Operands.push_back(DAG.getEntryNode());
}

/// GatherOperandInfo - Get latency and resource information about each operand.
///
void SimpleSched::GatherOperandInfo() {
  // Add addresses of operand info to ordering vector
  // Get number of operands
  unsigned N = Operands.size();
  // FIXME: This is an ugly (but temporary!) hack to test the scheduler before
  // we have real target info.
  
  // For each operand being scheduled
  for (unsigned i = 0; i < N; i++) {
    ScheduleInfo* SI = &Operands[N - i - 1];
    SDOperand Op = SI->Op;
    MVT::ValueType VT = Op.Val->getValueType(0);
    if (Op.isTargetOpcode()) {
      MachineOpCode TOpc = Op.getTargetOpcode();
      // FIXME SI->Latency = std::max(1, TII.maxLatency(TOpc));
      // FIXME SI->ResourceSet = TII.resources(TOpc);
      if (TII.isCall(TOpc)) {
        SI->ResourceSet = RSInteger;
        SI->Latency = 40;
      } else if (TII.isLoad(TOpc)) {
        SI->ResourceSet = RSLoadStore;
        SI->Latency = 5;
      } else if (TII.isStore(TOpc)) {
        SI->ResourceSet = RSLoadStore;
        SI->Latency = 2;
      } else if (MVT::isInteger(VT)) {
        SI->ResourceSet = RSInteger;
        SI->Latency = 2;
      } else if (MVT::isFloatingPoint(VT)) {
        SI->ResourceSet = RSFloat;
        SI->Latency = 3;
      } else {
        SI->ResourceSet = RSOther;
        SI->Latency = 0;
      }
    } else {
      if (MVT::isInteger(VT)) {
        SI->ResourceSet = RSInteger;
        SI->Latency = 2;
      } else if (MVT::isFloatingPoint(VT)) {
        SI->ResourceSet = RSFloat;
        SI->Latency = 3;
      } else {
        SI->ResourceSet = RSOther;
        SI->Latency = 0;
      }
    }
    
    // Add one slot for the instruction itself
    SI->Latency++;
    
    // Sum up all the latencies for max tally size
    NSlots += SI->Latency;
    
    // Place in initial sorted order
    // FIXME - PUNT - ignore flag users 
    if (!isFlagUser(Op)) Ordering.push_back(SI);
  }
}

/// isStrongDependency - Return true if operand A has results used by operand B. 
/// I.E., B must wait for latency of A.
bool SimpleSched::isStrongDependency(SDNode *A, SDNode *B) {
  // If A defines for B then it's a strong dependency
  if (isDefiner(A, B)) return true;
  // If A defines a flag then it's users are part of the dependency
  if (isFlagDefiner(A)) {
    // Check each flag user
    FlagUserIterator FI(A);
    while (SDNode *User = FI.next()) {
      // If flag user has strong dependency so does B
      if (isStrongDependency(User, B)) return true;
    }
  }
  // If B defines a flag then it's users are part of the dependency
  if (isFlagDefiner(B)) {
    // Check each flag user
    FlagUserIterator FI(B);
    while (SDNode *User = FI.next()) {
      // If flag user has strong dependency so does B
      if (isStrongDependency(A, User)) return true;
    }
  }
  return false;
}

/// isWeakDependency Return true if operand A produces a result that will
/// conflict with operands of B.
bool SimpleSched::isWeakDependency(SDNode *A, SDNode *B) {
  // TODO check for conflicting real registers and aliases
#if 0 // Since we are in SSA form and not checking register aliasing
  return A->getOpcode() == ISD::EntryToken || isStrongDependency(B, A);
#else
  return A->getOpcode() == ISD::EntryToken;
#endif
}

/// ScheduleBackward - Schedule instructions so that any long latency
/// instructions and the critical path get pushed back in time. Time is run in
/// reverse to allow code reuse of the Tally and eliminate the overhead of
/// biasing every slot indices against NSlots.
void SimpleSched::ScheduleBackward() {
  // Size and clear the resource tally
  Tally.Initialize(NSlots);
  // Get number of operands to schedule
  unsigned N = Ordering.size();
  
  // For each operand being scheduled
  for (unsigned i = N; 0 < i--;) {
    ScheduleInfo *SI = Ordering[i];
    // Track insertion
    unsigned Slot = NotFound;
    
    // Compare against those previously scheduled operands
    for (unsigned j = i + 1; j < N; j++) {
      // Get following instruction
      ScheduleInfo *Other = Ordering[j];
      
      // Check dependency against previously inserted operands
      if (isStrongDependency(SI->Op, Other->Op)) {
        Slot = Other->Slot + Other->Latency;
        break;
      } else if (isWeakDependency(SI->Op, Other->Op)) {
        Slot = Other->Slot;
        break;
      }
    }
    
    // If independent of others (or first entry)
    if (Slot == NotFound) Slot = 0;
    
    // Find a slot where the needed resources are available
    if (SI->ResourceSet)
      Slot = Tally.FindAndReserve(Slot, SI->Latency, SI->ResourceSet);
      
    // Set operand slot
    SI->Slot = Slot;
    
    // Insert sort based on slot
    unsigned j = i + 1;
    for (; j < N; j++) {
      // Get following instruction
      ScheduleInfo *Other = Ordering[j];
      // Should we look further
      if (Slot >= Other->Slot) break;
      // Shuffle other into ordering
      Ordering[j - 1] = Other;
    }
    // Insert operand in proper slot
    if (j != i + 1) Ordering[j - 1] = SI;
  }
}

/// ScheduleForward - Schedule instructions to maximize packing.
///
void SimpleSched::ScheduleForward() {
  // Size and clear the resource tally
  Tally.Initialize(NSlots);
  // Get number of operands to schedule
  unsigned N = Ordering.size();
  
  // For each operand being scheduled
  for (unsigned i = 0; i < N; i++) {
    ScheduleInfo *SI = Ordering[i];
    // Track insertion
    unsigned Slot = NotFound;
    
    // Compare against those previously scheduled operands
    for (unsigned j = i; 0 < j--;) {
      // Get following instruction
      ScheduleInfo *Other = Ordering[j];
      
      // Check dependency against previously inserted operands
      if (isStrongDependency(Other->Op, SI->Op)) {
        Slot = Other->Slot + Other->Latency;
        break;
      } else if (isWeakDependency(Other->Op, SI->Op)) {
        Slot = Other->Slot;
        break;
      }
    }
    
    // If independent of others (or first entry)
    if (Slot == NotFound) Slot = 0;
    
    // Find a slot where the needed resources are available
    if (SI->ResourceSet)
      Slot = Tally.FindAndReserve(Slot, SI->Latency, SI->ResourceSet);
      
    // Set operand slot
    SI->Slot = Slot;
    
    // Insert sort based on slot
    unsigned j = i;
    for (; 0 < j--;) {
      // Get following instruction
      ScheduleInfo *Other = Ordering[j];
      // Should we look further
      if (Slot >= Other->Slot) break;
      // Shuffle other into ordering
      Ordering[j + 1] = Other;
    }
    // Insert operand in proper slot
    if (j != i) Ordering[j + 1] = SI;
  }
}

/// EmitAll - Emit all operands in schedule sorted order.
///
void SimpleSched::EmitAll() {
  // For each operand in the ordering
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    // Get the scheduling info
    ScheduleInfo *SI = Ordering[i];
    // Get the operand
    SDOperand Op = SI->Op;
    // Emit the operand
    Emit(Op);
    // FIXME - PUNT - If Op defines a flag then it's users need to be emitted now
    if (isFlagDefiner(Op)) EmitFlagUsers(Op);
  }
}

/// EmitFlagUsers - Emit users of operands flag.
///
void SimpleSched::EmitFlagUsers(SDOperand Op) {
  // Check each flag user
  FlagUserIterator FI(Op.Val);
  while (SDNode *User = FI.next()) {
    // Construct user node as operand
    SDOperand OpU(User, 0);
    // Emit  user node
    Emit(OpU);
    // If user defines a flag then it's users need to be emitted now
    if (isFlagDefiner(User)) EmitFlagUsers(OpU);
  }
}

/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the machine instrs.)
unsigned SimpleSched::CountResults(SDOperand Op) {
  unsigned N = Op.Val->getNumValues();
  while (N && Op.Val->getValueType(N - 1) == MVT::Flag)
    --N;
  if (N && Op.Val->getValueType(N - 1) == MVT::Other)
    --N;    // Skip over chain result.
  return N;
}

/// CountOperands  The inputs to target nodes have any actual inputs first,
/// followed by an optional chain operand, then flag operands.  Compute the
/// number of actual operands that  will go into the machine instr.
unsigned SimpleSched::CountOperands(SDOperand Op) {
  unsigned N = Op.getNumOperands();
  while (N && Op.getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Op.getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}

/// CreateVirtualRegisters - Add result register values for things that are
/// defined by this instruction.
unsigned SimpleSched::CreateVirtualRegisters(SDOperand Op, MachineInstr *MI,
                                             unsigned NumResults,
                                             const TargetInstrDescriptor &II) {
  // Create the result registers for this node and add the result regs to
  // the machine instruction.
  const TargetOperandInfo *OpInfo = II.OpInfo;
  unsigned ResultReg = RegMap->createVirtualRegister(OpInfo[0].RegClass);
  MI->addRegOperand(ResultReg, MachineOperand::Def);
  for (unsigned i = 1; i != NumResults; ++i) {
    assert(OpInfo[i].RegClass && "Isn't a register operand!");
    MI->addRegOperand(RegMap->createVirtualRegister(OpInfo[0].RegClass),
                      MachineOperand::Def);
  }
  return ResultReg;
}

/// Emit - Generate machine code for an operand and needed dependencies.
///
unsigned SimpleSched::Emit(SDOperand Op) {
  std::map<SDNode *, unsigned>::iterator OpI = VRMap.lower_bound(Op.Val);
  if (OpI != VRMap.end() && OpI->first == Op.Val)
    return OpI->second + Op.ResNo;
  unsigned &OpSlot = VRMap.insert(OpI, std::make_pair(Op.Val, 0))->second;
  
  unsigned ResultReg = 0;
  if (Op.isTargetOpcode()) {
    unsigned Opc = Op.getTargetOpcode();
    const TargetInstrDescriptor &II = TII.get(Opc);

    unsigned NumResults = CountResults(Op);
    unsigned NodeOperands = CountOperands(Op);
    unsigned NumMIOperands = NodeOperands + NumResults;
#ifndef NDEBUG
    assert((unsigned(II.numOperands) == NumMIOperands || II.numOperands == -1)&&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, NumMIOperands, true, true);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults) ResultReg = CreateVirtualRegisters(Op, MI, NumResults, II);
    
    // If there is a token chain operand, emit it first, as a hack to get avoid
    // really bad cases.
    if (Op.getNumOperands() > NodeOperands &&
        Op.getOperand(NodeOperands).getValueType() == MVT::Other) {
      Emit(Op.getOperand(NodeOperands));
    }
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i) {
      if (Op.getOperand(i).isTargetOpcode()) {
        // Note that this case is redundant with the final else block, but we
        // include it because it is the most common and it makes the logic
        // simpler here.
        assert(Op.getOperand(i).getValueType() != MVT::Other &&
               Op.getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        
        MI->addRegOperand(Emit(Op.getOperand(i)), MachineOperand::Use);
      } else if (ConstantSDNode *C =
                                   dyn_cast<ConstantSDNode>(Op.getOperand(i))) {
        MI->addZeroExtImm64Operand(C->getValue());
      } else if (RegisterSDNode*R =dyn_cast<RegisterSDNode>(Op.getOperand(i))) {
        MI->addRegOperand(R->getReg(), MachineOperand::Use);
      } else if (GlobalAddressSDNode *TGA =
                       dyn_cast<GlobalAddressSDNode>(Op.getOperand(i))) {
        MI->addGlobalAddressOperand(TGA->getGlobal(), false, 0);
      } else if (BasicBlockSDNode *BB =
                       dyn_cast<BasicBlockSDNode>(Op.getOperand(i))) {
        MI->addMachineBasicBlockOperand(BB->getBasicBlock());
      } else if (FrameIndexSDNode *FI =
                       dyn_cast<FrameIndexSDNode>(Op.getOperand(i))) {
        MI->addFrameIndexOperand(FI->getIndex());
      } else if (ConstantPoolSDNode *CP = 
                    dyn_cast<ConstantPoolSDNode>(Op.getOperand(i))) {
        unsigned Idx = ConstPool->getConstantPoolIndex(CP->get());
        MI->addConstantPoolIndexOperand(Idx);
      } else if (ExternalSymbolSDNode *ES = 
                 dyn_cast<ExternalSymbolSDNode>(Op.getOperand(i))) {
        MI->addExternalSymbolOperand(ES->getSymbol(), false);
      } else {
        assert(Op.getOperand(i).getValueType() != MVT::Other &&
               Op.getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        MI->addRegOperand(Emit(Op.getOperand(i)), MachineOperand::Use);
      }
    }

    // Finally, if this node has any flag operands, we *must* emit them last, to
    // avoid emitting operations that might clobber the flags.
    if (Op.getNumOperands() > NodeOperands) {
      unsigned i = NodeOperands;
      if (Op.getOperand(i).getValueType() == MVT::Other)
        ++i;  // the chain is already selected.
      for (unsigned N = Op.getNumOperands(); i < N; i++) {
        assert(Op.getOperand(i).getValueType() == MVT::Flag &&
               "Must be flag operands!");
        Emit(Op.getOperand(i));
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
    switch (Op.getOpcode()) {
    default:
      Op.Val->dump(); 
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: break;
    case ISD::TokenFactor:
      for (unsigned i = 0, N = Op.getNumOperands(); i < N; i++) {
        Emit(Op.getOperand(i));
      }
      break;
    case ISD::CopyToReg: {
      SDOperand FlagOp;
      if (Op.getNumOperands() == 4) {
        FlagOp = Op.getOperand(3);
      }
      if (Op.getOperand(0).Val != FlagOp.Val) {
        Emit(Op.getOperand(0));   // Emit the chain.
      }
      unsigned Val = Emit(Op.getOperand(2));
      if (FlagOp.Val) {
        Emit(FlagOp);
      }
      MRI.copyRegToReg(*BB, BB->end(),
                       cast<RegisterSDNode>(Op.getOperand(1))->getReg(), Val,
                       RegMap->getRegClass(Val));
      break;
    }
    case ISD::CopyFromReg: {
      Emit(Op.getOperand(0));   // Emit the chain.
      unsigned SrcReg = cast<RegisterSDNode>(Op.getOperand(1))->getReg();
      
      // Figure out the register class to create for the destreg.
      const TargetRegisterClass *TRC = 0;
      if (MRegisterInfo::isVirtualRegister(SrcReg)) {
        TRC = RegMap->getRegClass(SrcReg);
      } else {
        // FIXME: we don't know what register class to generate this for.  Do
        // a brute force search and pick the first match. :(
        for (MRegisterInfo::regclass_iterator I = MRI.regclass_begin(),
               E = MRI.regclass_end(); I != E; ++I)
          if ((*I)->contains(SrcReg)) {
            TRC = *I;
            break;
          }
        assert(TRC && "Couldn't find register class for reg copy!");
      }
      
      // Create the reg, emit the copy.
      ResultReg = RegMap->createVirtualRegister(TRC);
      MRI.copyRegToReg(*BB, BB->end(), ResultReg, SrcReg, TRC);
      break;
    }
    }
  }

  OpSlot = ResultReg;
  return ResultReg+Op.ResNo;
}

/// Schedule - Order operands according to selected style.
///
void SimpleSched::Schedule() {
  switch (ScheduleStyle) {
  case simpleScheduling:
    // Breadth first walk of DAG
    VisitAll();
    // Get latency and resource requirements
    GatherOperandInfo();
    // Don't waste time if is only entry and return
    if (Operands.size() > 2) {
      DEBUG(dump("Pre-"));
      // Push back long instructions and critical path
      ScheduleBackward();
      DEBUG(dump("Mid-"));
      // Pack instructions to maximize resource utilization
      ScheduleForward();
      DEBUG(dump("Post-"));
      // Emit in scheduled order
      EmitAll();
      break;
    } // fall thru
  case noScheduling:
    // Emit instructions in using a DFS from the exit root
    Emit(DAG.getRoot());
    break;
  }
}

/// printSI - Print schedule info.
///
void SimpleSched::printSI(std::ostream &O, ScheduleInfo *SI) const {
#ifndef NDEBUG
  using namespace std;
  SDOperand Op = SI->Op;
  O << " "
    << hex << Op.Val
    << ", RS=" << SI->ResourceSet
    << ", Lat=" << SI->Latency
    << ", Slot=" << SI->Slot
    << ", ARITY=(" << Op.getNumOperands() << ","
                   << Op.Val->getNumValues() << ")"
    << " " << Op.Val->getOperationName(&DAG);
  if (isFlagDefiner(Op)) O << "<#";
  if (isFlagUser(Op)) O << ">#";
#endif
}

/// print - Print ordering to specified output stream.
///
void SimpleSched::print(std::ostream &O) const {
#ifndef NDEBUG
  using namespace std;
  O << "Ordering\n";
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    printSI(O, Ordering[i]);
    O << "\n";
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
