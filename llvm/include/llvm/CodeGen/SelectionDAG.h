//===-- llvm/CodeGen/SelectionDAG.h - InstSelection DAG Rep. ----*- C++ -*-===//
// 
// This file declares the SelectionDAG class, which is used to represent an LLVM
// function in a low-level representation suitable for instruction selection.
// This DAG is constructed as the first step of instruction selection in order
// to allow implementation of machine specific optimizations and code
// simplifications.
//
// The representation used by the SelectionDAG is a target-independent
// representation, which is loosly modeled after the GCC RTL representation, but
// is significantly simpler.
//   
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAG_H
#define LLVM_CODEGEN_SELECTIONDAG_H

#include "llvm/CodeGen/ValueTypes.h"
#include "Support/DataTypes.h"
#include <map>
#include <vector>
class Value;
class Type;
class Instruction;
class BasicBlock;
class MachineBasicBlock;
class MachineFunction;
class TargetMachine;
class SelectionDAGNode;
class SelectionDAGBlock;
class SelectionDAGBuilder;
class SelectionDAGTargetBuilder;

/// ISD namespace - This namespace contains an enum which represents all of the
/// SelectionDAG node types and value types.
///
namespace ISD {
  enum NodeType {
    // ChainNode nodes are used to sequence operations within a basic block
    // which cannot be reordered (such as loads, stores, calls, etc).
    // BlockChainNodes are used to connect the DAG's for different basic blocks
    // into one big DAG.
    ChainNode, BlockChainNode,

    // ProtoNodes are nodes that are only half way constructed.
    ProtoNode,

    // Leaf nodes.
    Constant, FrameIndex,

    // Simple binary arithmetic operators
    Plus, Minus, Times, SDiv, UDiv, SRem, URem,

    // Bitwise operators
    And, Or, Xor,

    // Control flow instructions
    Br, Switch, Ret, RetVoid,

    // Other operators
    Load, Store, PHI, Call,
  };
}

class SelectionDAG {
  friend class SelectionDAGBuilder;
  MachineFunction &F;
  const TargetMachine &TM;
  MVT::ValueType PointerType;    // The ValueType the target uses for pointers

  // ValueMap - The SelectionDAGNode for each LLVM value in the function.
  std::map<const Value*, SelectionDAGNode*> ValueMap;

  // BlockMap - The MachineBasicBlock created for each LLVM BasicBlock
  std::map<const BasicBlock*, MachineBasicBlock*> BlockMap;

  // Root - The root of the entire DAG
  SelectionDAGNode *Root;

  // AllNodes - All of the nodes in the DAG
  std::vector<SelectionDAGNode*> AllNodes;
public:
  /// SelectionDAG constructor - Build a SelectionDAG for the specified
  /// function.  Implemented in DAGBuilder.cpp
  ///
  SelectionDAG(MachineFunction &F, const TargetMachine &TM,
               SelectionDAGTargetBuilder &SDTB);
  ~SelectionDAG();

  /// getValueType - Return the ValueType for the specified LLVM type.  This
  /// method works on all scalar LLVM types.
  ///
  MVT::ValueType getValueType(const Type *Ty) const;

  /// getRoot - Return the root of the current SelectionDAG.
  ///
  SelectionDAGNode *getRoot() const { return Root; }

  /// getMachineFunction - Return the MachineFunction object that this
  /// SelectionDAG corresponds to.
  ///
  MachineFunction &getMachineFunction() const { return F; }

  //===--------------------------------------------------------------------===//
  // Addition and updating methods
  //

  /// addNode - Add the specified node to the SelectionDAG so that it will be
  /// deleted when the DAG is...
  ///
  SelectionDAGNode *addNode(SelectionDAGNode *N) {
    AllNodes.push_back(N);
    return N;
  }

  /// addNodeForValue - Add the specified node to the SelectionDAG so that it
  /// will be deleted when the DAG is... and update the value map to indicate
  /// that the specified DAG node computes the value.  Note that it is an error
  /// to specify multiple DAG nodes that compute the same value.
  ///
  SelectionDAGNode *addNodeForValue(SelectionDAGNode *N, const Value *V) {
    assert(ValueMap.count(V) == 0 && "Value already has a DAG node!");
    return addNode(ValueMap[V] = N);
  }

  void dump() const;
private:
  void addInstructionToDAG(const Instruction &I, const BasicBlock &BB);
};


/// SelectionDAGReducedValue - During the reducer pass we need the ability to
/// add an arbitrary (but usually 1 or 0) number of arbitrarily sized values to
/// the selection DAG.  Because of this, we represent these values as a singly
/// linked list of values attached to the DAGNode.  We end up putting the
/// arbitrary state for the value in subclasses of this node.
///
/// Note that this class does not have a virtual dtor, this is because we know
/// that the subclasses will not hold state that needs to be destroyed.
///
class SelectionDAGReducedValue {
  unsigned Code;
  SelectionDAGReducedValue *Next;
public:
  SelectionDAGReducedValue(unsigned C) : Code(C), Next(0) {}

  /// getValueCode - Return the code for this reducer value...
  ///
  unsigned getValueCode() const { return Code; }
  
  /// getNext - Return the next value in the list
  ///
  const SelectionDAGReducedValue *getNext() const { return Next; }
  void setNext(SelectionDAGReducedValue *N) { Next = N; }

  SelectionDAGReducedValue *getNext() { return Next; }
};



/// SelectionDAGNode - Represents one node in the selection DAG.
///
class SelectionDAGNode {
  std::vector<SelectionDAGNode*> Uses;
  ISD::NodeType  NodeType;
  MVT::ValueType ValueType;
  MachineBasicBlock *BB;
  SelectionDAGReducedValue *ValList;

  /// Costs - Each pair of elements of 'Costs' contains the cost of producing
  /// the value with the target specific slot number and the production number
  /// to use to produce it.  A zero value for the production number indicates
  /// that the cost has not yet been computed.
  unsigned *Costs;
public:
  SelectionDAGNode(ISD::NodeType NT, MVT::ValueType VT,
                   MachineBasicBlock *bb = 0) 
    : NodeType(NT), ValueType(VT), BB(bb), ValList(0), Costs(0) {}

  SelectionDAGNode(ISD::NodeType NT, MVT::ValueType VT, MachineBasicBlock *bb,
                   SelectionDAGNode *N)
    : NodeType(NT), ValueType(VT), BB(bb), ValList(0), Costs(0) {
    assert(NT != ISD::ProtoNode && "Cannot specify uses for a protonode!");
    Uses.reserve(1); Uses.push_back(N);
  }
  SelectionDAGNode(ISD::NodeType NT, MVT::ValueType VT, MachineBasicBlock *bb,
                   SelectionDAGNode *N1, SelectionDAGNode *N2)
    : NodeType(NT), ValueType(VT), BB(bb), ValList(0), Costs(0) {
    assert(NT != ISD::ProtoNode && "Cannot specify uses for a protonode!");
    Uses.reserve(2); Uses.push_back(N1); Uses.push_back(N2);
  }

  ~SelectionDAGNode() { delete [] Costs; delete ValList; }

  void setNode(ISD::NodeType NT, MachineBasicBlock *bb) {
    assert(NodeType == ISD::ProtoNode && NT != ISD::ProtoNode);
    NodeType = NT; BB = bb;
  }
  void setNode(ISD::NodeType NT, MachineBasicBlock *bb, SelectionDAGNode *N) {
    assert(NodeType == ISD::ProtoNode && NT != ISD::ProtoNode);
    NodeType = NT; BB = bb; Uses.reserve(1); Uses.push_back(N);
  }
  void setNode(ISD::NodeType NT, MachineBasicBlock *bb, 
               SelectionDAGNode *N1, SelectionDAGNode *N2) {
    assert(NodeType == ISD::ProtoNode && NT != ISD::ProtoNode);
    NodeType = NT; BB = bb;
    Uses.reserve(1); Uses.push_back(N1); Uses.push_back(N2);
  }

  //===--------------------------------------------------------------------===//
  //  Accessors
  //
  ISD::NodeType  getNodeType()  const { return NodeType; }
  MVT::ValueType getValueType() const { return ValueType; }
  MachineBasicBlock *getBB() const { return BB; }

  SelectionDAGNode *getUse(unsigned Num) {
    assert(Num < Uses.size() && "Invalid child # of SelectionDAGNode!");
    return Uses[Num];
  }

  template<class Type>
  Type *getValue(unsigned Code) const {
    SelectionDAGReducedValue *Vals = ValList;
    while (1) {
      assert(Vals && "Code does not exist in this list!");
      if (Vals->getValueCode() == Code)
        return (Type*)Vals;
      Vals = Vals->getNext();
    }
  }

  template<class Type>
  Type *hasValue(unsigned Code) const {
    SelectionDAGReducedValue *Vals = ValList;
    while (Vals) {
      if (Vals->getValueCode() == Code)
        return (Type*)Vals;
      Vals = Vals->getNext();
    }
    return false;
  }

  void addValue(SelectionDAGReducedValue *New) {
    assert(New->getNext() == 0);
    New->setNext(ValList);
    ValList = New;
  }

  //===--------------------------------------------------------------------===//
  // Utility methods used by the pattern matching instruction selector
  //

  /// getPatternFor - Return the pattern selected to compute the specified slot,
  /// or zero if there is no pattern yet.
  ///
  unsigned getPatternFor(unsigned Slot) const {
    return Costs ? Costs[Slot*2] : 0;
  }

  /// getCostFor - Return the cost to compute the value corresponding to Slot.
  ///
  unsigned getCostFor(unsigned Slot) const {
    return Costs ? Costs[Slot*2+1] : 0;
  }

  /// setPatternCostFor - Sets the pattern and the cost for the specified slot
  /// to the specified values.  This allocates the Costs vector if necessary, so
  /// you must specify the maximum number of slots that may be used.
  ///
  void setPatternCostFor(unsigned Slot, unsigned Pattern, unsigned Cost,
                         unsigned NumSlots) {
    if (Costs == 0) {
      Costs = new unsigned[NumSlots*2];
      for (unsigned i = 0; i != NumSlots*2; ++i) Costs[i] = 0;
    }
    Costs[Slot*2] = Pattern;
    Costs[Slot*2+1] = Cost;
  }

  void dump() const;
private:
  void printit(unsigned Offset, unsigned &LastID,
               std::map<const SelectionDAGNode*, unsigned> &NodeIDs) const;
};


/// SelectionDAGTargetBuilder - This class must be implemented by the target, to
/// indicate how to perform the extremely target-specific tasks of building DAG
/// nodes to represent the calling convention used by the target.
///
struct SelectionDAGTargetBuilder {
  /// expandArguments - This method is called once by the SelectionDAG
  /// construction mechanisms to add DAG nodes for each formal argument to the
  /// current function.  If any of the incoming arguments lives on the stack,
  /// this method should also create the stack slots for the arguments as
  /// necessary.
  virtual void expandArguments(SelectionDAG &SD, MachineFunction &MF) = 0;
};

namespace ISD {
  enum {   // Builtin Slot numbers
    Constant_i1_Slot,
    Constant_i8_Slot,
    Constant_i16_Slot,
    Constant_i32_Slot,
    Constant_i64_Slot,
    Constant_f32_Slot,
    Constant_f64_Slot,

    FrameIndex_i32_Slot,
    FrameIndex_i64_Slot,
    NumBuiltinSlots
  };
}

template<typename ValType, unsigned NodeCode>
struct ReducedValue : public SelectionDAGReducedValue {
  ReducedValue(const ValType &V) : SelectionDAGReducedValue(NodeCode), Val(V) {}
  ValType Val;
};

typedef ReducedValue<int, ISD::FrameIndex_i32_Slot > ReducedValue_FrameIndex_i32;
typedef ReducedValue<int, ISD::FrameIndex_i64_Slot > ReducedValue_FrameIndex_i64;

typedef ReducedValue<bool          , ISD::Constant_i1_Slot > ReducedValue_Constant_i1;
typedef ReducedValue<unsigned char , ISD::Constant_i8_Slot > ReducedValue_Constant_i8;
typedef ReducedValue<unsigned short, ISD::Constant_i16_Slot> ReducedValue_Constant_i16;
typedef ReducedValue<unsigned      , ISD::Constant_i32_Slot> ReducedValue_Constant_i32;
typedef ReducedValue<uint64_t      , ISD::Constant_i64_Slot> ReducedValue_Constant_i64;
typedef ReducedValue<float         , ISD::Constant_f32_Slot> ReducedValue_Constant_f32;
typedef ReducedValue<double        , ISD::Constant_f64_Slot> ReducedValue_Constant_f64;

#endif
