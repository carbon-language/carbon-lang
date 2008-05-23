//===-- SelectionDAG.cpp - Implement the SelectionDAG data structures -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG class.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Constants.h"
#include "llvm/GlobalAlias.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

/// makeVTList - Return an instance of the SDVTList struct initialized with the
/// specified members.
static SDVTList makeVTList(const MVT::ValueType *VTs, unsigned NumVTs) {
  SDVTList Res = {VTs, NumVTs};
  return Res;
}

static const fltSemantics *MVTToAPFloatSemantics(MVT::ValueType VT) {
  switch (VT) {
  default: assert(0 && "Unknown FP format");
  case MVT::f32:     return &APFloat::IEEEsingle;
  case MVT::f64:     return &APFloat::IEEEdouble;
  case MVT::f80:     return &APFloat::x87DoubleExtended;
  case MVT::f128:    return &APFloat::IEEEquad;
  case MVT::ppcf128: return &APFloat::PPCDoubleDouble;
  }
}

SelectionDAG::DAGUpdateListener::~DAGUpdateListener() {}

//===----------------------------------------------------------------------===//
//                              ConstantFPSDNode Class
//===----------------------------------------------------------------------===//

/// isExactlyValue - We don't rely on operator== working on double values, as
/// it returns true for things that are clearly not equal, like -0.0 and 0.0.
/// As such, this method can be used to do an exact bit-for-bit comparison of
/// two floating point values.
bool ConstantFPSDNode::isExactlyValue(const APFloat& V) const {
  return Value.bitwiseIsEqual(V);
}

bool ConstantFPSDNode::isValueValidForType(MVT::ValueType VT, 
                                           const APFloat& Val) {
  assert(MVT::isFloatingPoint(VT) && "Can only convert between FP types");
  
  // PPC long double cannot be converted to any other type.
  if (VT == MVT::ppcf128 ||
      &Val.getSemantics() == &APFloat::PPCDoubleDouble)
    return false;
  
  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  return Val2.convert(*MVTToAPFloatSemantics(VT),
                      APFloat::rmNearestTiesToEven) == APFloat::opOK;
}

//===----------------------------------------------------------------------===//
//                              ISD Namespace
//===----------------------------------------------------------------------===//

/// isBuildVectorAllOnes - Return true if the specified node is a
/// BUILD_VECTOR where all of the elements are ~0 or undef.
bool ISD::isBuildVectorAllOnes(const SDNode *N) {
  // Look through a bit convert.
  if (N->getOpcode() == ISD::BIT_CONVERT)
    N = N->getOperand(0).Val;
  
  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;
  
  unsigned i = 0, e = N->getNumOperands();
  
  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).getOpcode() == ISD::UNDEF)
    ++i;
  
  // Do not accept an all-undef vector.
  if (i == e) return false;
  
  // Do not accept build_vectors that aren't all constants or which have non-~0
  // elements.
  SDOperand NotZero = N->getOperand(i);
  if (isa<ConstantSDNode>(NotZero)) {
    if (!cast<ConstantSDNode>(NotZero)->isAllOnesValue())
      return false;
  } else if (isa<ConstantFPSDNode>(NotZero)) {
    if (!cast<ConstantFPSDNode>(NotZero)->getValueAPF().
                convertToAPInt().isAllOnesValue())
      return false;
  } else
    return false;
  
  // Okay, we have at least one ~0 value, check to see if the rest match or are
  // undefs.
  for (++i; i != e; ++i)
    if (N->getOperand(i) != NotZero &&
        N->getOperand(i).getOpcode() != ISD::UNDEF)
      return false;
  return true;
}


/// isBuildVectorAllZeros - Return true if the specified node is a
/// BUILD_VECTOR where all of the elements are 0 or undef.
bool ISD::isBuildVectorAllZeros(const SDNode *N) {
  // Look through a bit convert.
  if (N->getOpcode() == ISD::BIT_CONVERT)
    N = N->getOperand(0).Val;
  
  if (N->getOpcode() != ISD::BUILD_VECTOR) return false;
  
  unsigned i = 0, e = N->getNumOperands();
  
  // Skip over all of the undef values.
  while (i != e && N->getOperand(i).getOpcode() == ISD::UNDEF)
    ++i;
  
  // Do not accept an all-undef vector.
  if (i == e) return false;
  
  // Do not accept build_vectors that aren't all constants or which have non-~0
  // elements.
  SDOperand Zero = N->getOperand(i);
  if (isa<ConstantSDNode>(Zero)) {
    if (!cast<ConstantSDNode>(Zero)->isNullValue())
      return false;
  } else if (isa<ConstantFPSDNode>(Zero)) {
    if (!cast<ConstantFPSDNode>(Zero)->getValueAPF().isPosZero())
      return false;
  } else
    return false;
  
  // Okay, we have at least one ~0 value, check to see if the rest match or are
  // undefs.
  for (++i; i != e; ++i)
    if (N->getOperand(i) != Zero &&
        N->getOperand(i).getOpcode() != ISD::UNDEF)
      return false;
  return true;
}

/// isScalarToVector - Return true if the specified node is a
/// ISD::SCALAR_TO_VECTOR node or a BUILD_VECTOR node where only the low
/// element is not an undef.
bool ISD::isScalarToVector(const SDNode *N) {
  if (N->getOpcode() == ISD::SCALAR_TO_VECTOR)
    return true;

  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;
  if (N->getOperand(0).getOpcode() == ISD::UNDEF)
    return false;
  unsigned NumElems = N->getNumOperands();
  for (unsigned i = 1; i < NumElems; ++i) {
    SDOperand V = N->getOperand(i);
    if (V.getOpcode() != ISD::UNDEF)
      return false;
  }
  return true;
}


/// isDebugLabel - Return true if the specified node represents a debug
/// label (i.e. ISD::LABEL or TargetInstrInfo::LABEL node and third operand
/// is 0).
bool ISD::isDebugLabel(const SDNode *N) {
  SDOperand Zero;
  if (N->getOpcode() == ISD::LABEL)
    Zero = N->getOperand(2);
  else if (N->isTargetOpcode() &&
           N->getTargetOpcode() == TargetInstrInfo::LABEL)
    // Chain moved to last operand.
    Zero = N->getOperand(1);
  else
    return false;
  return isa<ConstantSDNode>(Zero) && cast<ConstantSDNode>(Zero)->isNullValue();
}

/// getSetCCSwappedOperands - Return the operation corresponding to (Y op X)
/// when given the operation for (X op Y).
ISD::CondCode ISD::getSetCCSwappedOperands(ISD::CondCode Operation) {
  // To perform this operation, we just need to swap the L and G bits of the
  // operation.
  unsigned OldL = (Operation >> 2) & 1;
  unsigned OldG = (Operation >> 1) & 1;
  return ISD::CondCode((Operation & ~6) |  // Keep the N, U, E bits
                       (OldL << 1) |       // New G bit
                       (OldG << 2));        // New L bit.
}

/// getSetCCInverse - Return the operation corresponding to !(X op Y), where
/// 'op' is a valid SetCC operation.
ISD::CondCode ISD::getSetCCInverse(ISD::CondCode Op, bool isInteger) {
  unsigned Operation = Op;
  if (isInteger)
    Operation ^= 7;   // Flip L, G, E bits, but not U.
  else
    Operation ^= 15;  // Flip all of the condition bits.
  if (Operation > ISD::SETTRUE2)
    Operation &= ~8;     // Don't let N and U bits get set.
  return ISD::CondCode(Operation);
}


/// isSignedOp - For an integer comparison, return 1 if the comparison is a
/// signed operation and 2 if the result is an unsigned comparison.  Return zero
/// if the operation does not depend on the sign of the input (setne and seteq).
static int isSignedOp(ISD::CondCode Opcode) {
  switch (Opcode) {
  default: assert(0 && "Illegal integer setcc operation!");
  case ISD::SETEQ:
  case ISD::SETNE: return 0;
  case ISD::SETLT:
  case ISD::SETLE:
  case ISD::SETGT:
  case ISD::SETGE: return 1;
  case ISD::SETULT:
  case ISD::SETULE:
  case ISD::SETUGT:
  case ISD::SETUGE: return 2;
  }
}

/// getSetCCOrOperation - Return the result of a logical OR between different
/// comparisons of identical values: ((X op1 Y) | (X op2 Y)).  This function
/// returns SETCC_INVALID if it is not possible to represent the resultant
/// comparison.
ISD::CondCode ISD::getSetCCOrOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                       bool isInteger) {
  if (isInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed integer setcc with an unsigned integer setcc.
    return ISD::SETCC_INVALID;

  unsigned Op = Op1 | Op2;  // Combine all of the condition bits.

  // If the N and U bits get set then the resultant comparison DOES suddenly
  // care about orderedness, and is true when ordered.
  if (Op > ISD::SETTRUE2)
    Op &= ~16;     // Clear the U bit if the N bit is set.
  
  // Canonicalize illegal integer setcc's.
  if (isInteger && Op == ISD::SETUNE)  // e.g. SETUGT | SETULT
    Op = ISD::SETNE;
  
  return ISD::CondCode(Op);
}

/// getSetCCAndOperation - Return the result of a logical AND between different
/// comparisons of identical values: ((X op1 Y) & (X op2 Y)).  This
/// function returns zero if it is not possible to represent the resultant
/// comparison.
ISD::CondCode ISD::getSetCCAndOperation(ISD::CondCode Op1, ISD::CondCode Op2,
                                        bool isInteger) {
  if (isInteger && (isSignedOp(Op1) | isSignedOp(Op2)) == 3)
    // Cannot fold a signed setcc with an unsigned setcc.
    return ISD::SETCC_INVALID;

  // Combine all of the condition bits.
  ISD::CondCode Result = ISD::CondCode(Op1 & Op2);
  
  // Canonicalize illegal integer setcc's.
  if (isInteger) {
    switch (Result) {
    default: break;
    case ISD::SETUO : Result = ISD::SETFALSE; break;  // SETUGT & SETULT
    case ISD::SETOEQ:                                 // SETEQ  & SETU[LG]E
    case ISD::SETUEQ: Result = ISD::SETEQ   ; break;  // SETUGE & SETULE
    case ISD::SETOLT: Result = ISD::SETULT  ; break;  // SETULT & SETNE
    case ISD::SETOGT: Result = ISD::SETUGT  ; break;  // SETUGT & SETNE
    }
  }
  
  return Result;
}

const TargetMachine &SelectionDAG::getTarget() const {
  return TLI.getTargetMachine();
}

//===----------------------------------------------------------------------===//
//                           SDNode Profile Support
//===----------------------------------------------------------------------===//

/// AddNodeIDOpcode - Add the node opcode to the NodeID data.
///
static void AddNodeIDOpcode(FoldingSetNodeID &ID, unsigned OpC)  {
  ID.AddInteger(OpC);
}

/// AddNodeIDValueTypes - Value type lists are intern'd so we can represent them
/// solely with their pointer.
static void AddNodeIDValueTypes(FoldingSetNodeID &ID, SDVTList VTList) {
  ID.AddPointer(VTList.VTs);  
}

/// AddNodeIDOperands - Various routines for adding operands to the NodeID data.
///
static void AddNodeIDOperands(FoldingSetNodeID &ID,
                              SDOperandPtr Ops, unsigned NumOps) {
  for (; NumOps; --NumOps, ++Ops) {
    ID.AddPointer(Ops->Val);
    ID.AddInteger(Ops->ResNo);
  }
}

static void AddNodeIDNode(FoldingSetNodeID &ID,
                          unsigned short OpC, SDVTList VTList, 
                          SDOperandPtr OpList, unsigned N) {
  AddNodeIDOpcode(ID, OpC);
  AddNodeIDValueTypes(ID, VTList);
  AddNodeIDOperands(ID, OpList, N);
}


/// AddNodeIDNode - Generic routine for adding a nodes info to the NodeID
/// data.
static void AddNodeIDNode(FoldingSetNodeID &ID, SDNode *N) {
  AddNodeIDOpcode(ID, N->getOpcode());
  // Add the return value info.
  AddNodeIDValueTypes(ID, N->getVTList());
  // Add the operand info.
  AddNodeIDOperands(ID, N->op_begin(), N->getNumOperands());

  // Handle SDNode leafs with special info.
  switch (N->getOpcode()) {
  default: break;  // Normal nodes don't need extra info.
  case ISD::ARG_FLAGS:
    ID.AddInteger(cast<ARG_FLAGSSDNode>(N)->getArgFlags().getRawBits());
    break;
  case ISD::TargetConstant:
  case ISD::Constant:
    ID.Add(cast<ConstantSDNode>(N)->getAPIntValue());
    break;
  case ISD::TargetConstantFP:
  case ISD::ConstantFP: {
    ID.Add(cast<ConstantFPSDNode>(N)->getValueAPF());
    break;
  }
  case ISD::TargetGlobalAddress:
  case ISD::GlobalAddress:
  case ISD::TargetGlobalTLSAddress:
  case ISD::GlobalTLSAddress: {
    GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(N);
    ID.AddPointer(GA->getGlobal());
    ID.AddInteger(GA->getOffset());
    break;
  }
  case ISD::BasicBlock:
    ID.AddPointer(cast<BasicBlockSDNode>(N)->getBasicBlock());
    break;
  case ISD::Register:
    ID.AddInteger(cast<RegisterSDNode>(N)->getReg());
    break;
  case ISD::SRCVALUE:
    ID.AddPointer(cast<SrcValueSDNode>(N)->getValue());
    break;
  case ISD::MEMOPERAND: {
    const MachineMemOperand &MO = cast<MemOperandSDNode>(N)->MO;
    ID.AddPointer(MO.getValue());
    ID.AddInteger(MO.getFlags());
    ID.AddInteger(MO.getOffset());
    ID.AddInteger(MO.getSize());
    ID.AddInteger(MO.getAlignment());
    break;
  }
  case ISD::FrameIndex:
  case ISD::TargetFrameIndex:
    ID.AddInteger(cast<FrameIndexSDNode>(N)->getIndex());
    break;
  case ISD::JumpTable:
  case ISD::TargetJumpTable:
    ID.AddInteger(cast<JumpTableSDNode>(N)->getIndex());
    break;
  case ISD::ConstantPool:
  case ISD::TargetConstantPool: {
    ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(N);
    ID.AddInteger(CP->getAlignment());
    ID.AddInteger(CP->getOffset());
    if (CP->isMachineConstantPoolEntry())
      CP->getMachineCPVal()->AddSelectionDAGCSEId(ID);
    else
      ID.AddPointer(CP->getConstVal());
    break;
  }
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(N);
    ID.AddInteger(LD->getAddressingMode());
    ID.AddInteger(LD->getExtensionType());
    ID.AddInteger((unsigned int)(LD->getMemoryVT()));
    ID.AddInteger(LD->getAlignment());
    ID.AddInteger(LD->isVolatile());
    break;
  }
  case ISD::STORE: {
    StoreSDNode *ST = cast<StoreSDNode>(N);
    ID.AddInteger(ST->getAddressingMode());
    ID.AddInteger(ST->isTruncatingStore());
    ID.AddInteger((unsigned int)(ST->getMemoryVT()));
    ID.AddInteger(ST->getAlignment());
    ID.AddInteger(ST->isVolatile());
    break;
  }
  }
}

//===----------------------------------------------------------------------===//
//                              SelectionDAG Class
//===----------------------------------------------------------------------===//

/// RemoveDeadNodes - This method deletes all unreachable nodes in the
/// SelectionDAG.
void SelectionDAG::RemoveDeadNodes() {
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted.
  HandleSDNode Dummy(getRoot());

  SmallVector<SDNode*, 128> DeadNodes;
  
  // Add all obviously-dead nodes to the DeadNodes worklist.
  for (allnodes_iterator I = allnodes_begin(), E = allnodes_end(); I != E; ++I)
    if (I->use_empty())
      DeadNodes.push_back(I);

  // Process the worklist, deleting the nodes and adding their uses to the
  // worklist.
  while (!DeadNodes.empty()) {
    SDNode *N = DeadNodes.back();
    DeadNodes.pop_back();
    
    // Take the node out of the appropriate CSE map.
    RemoveNodeFromCSEMaps(N);

    // Next, brutally remove the operand list.  This is safe to do, as there are
    // no cycles in the graph.
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      SDNode *Operand = I->getVal();
      Operand->removeUser(std::distance(N->op_begin(), I), N);
      
      // Now that we removed this operand, see if there are no uses of it left.
      if (Operand->use_empty())
        DeadNodes.push_back(Operand);
    }
    if (N->OperandsNeedDelete) {
      delete[] N->OperandList;
    }
    N->OperandList = 0;
    N->NumOperands = 0;
    
    // Finally, remove N itself.
    AllNodes.erase(N);
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(Dummy.getValue());
}

void SelectionDAG::RemoveDeadNode(SDNode *N, DAGUpdateListener *UpdateListener){
  SmallVector<SDNode*, 16> DeadNodes;
  DeadNodes.push_back(N);

  // Process the worklist, deleting the nodes and adding their uses to the
  // worklist.
  while (!DeadNodes.empty()) {
    SDNode *N = DeadNodes.back();
    DeadNodes.pop_back();
    
    if (UpdateListener)
      UpdateListener->NodeDeleted(N);
    
    // Take the node out of the appropriate CSE map.
    RemoveNodeFromCSEMaps(N);

    // Next, brutally remove the operand list.  This is safe to do, as there are
    // no cycles in the graph.
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      SDNode *Operand = I->getVal();
      Operand->removeUser(std::distance(N->op_begin(), I), N);
      
      // Now that we removed this operand, see if there are no uses of it left.
      if (Operand->use_empty())
        DeadNodes.push_back(Operand);
    }
    if (N->OperandsNeedDelete) {
      delete[] N->OperandList;
    }
    N->OperandList = 0;
    N->NumOperands = 0;
    
    // Finally, remove N itself.
    AllNodes.erase(N);
  }
}

void SelectionDAG::DeleteNode(SDNode *N) {
  assert(N->use_empty() && "Cannot delete a node that is not dead!");

  // First take this out of the appropriate CSE map.
  RemoveNodeFromCSEMaps(N);

  // Finally, remove uses due to operands of this node, remove from the 
  // AllNodes list, and delete the node.
  DeleteNodeNotInCSEMaps(N);
}

void SelectionDAG::DeleteNodeNotInCSEMaps(SDNode *N) {

  // Remove it from the AllNodes list.
  AllNodes.remove(N);
    
  // Drop all of the operands and decrement used nodes use counts.
  for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I)
    I->getVal()->removeUser(std::distance(N->op_begin(), I), N);
  if (N->OperandsNeedDelete) {
    delete[] N->OperandList;
  }
  N->OperandList = 0;
  N->NumOperands = 0;
  
  delete N;
}

/// RemoveNodeFromCSEMaps - Take the specified node out of the CSE map that
/// correspond to it.  This is useful when we're about to delete or repurpose
/// the node.  We don't want future request for structurally identical nodes
/// to return N anymore.
void SelectionDAG::RemoveNodeFromCSEMaps(SDNode *N) {
  bool Erased = false;
  switch (N->getOpcode()) {
  case ISD::HANDLENODE: return;  // noop.
  case ISD::STRING:
    Erased = StringNodes.erase(cast<StringSDNode>(N)->getValue());
    break;
  case ISD::CONDCODE:
    assert(CondCodeNodes[cast<CondCodeSDNode>(N)->get()] &&
           "Cond code doesn't exist!");
    Erased = CondCodeNodes[cast<CondCodeSDNode>(N)->get()] != 0;
    CondCodeNodes[cast<CondCodeSDNode>(N)->get()] = 0;
    break;
  case ISD::ExternalSymbol:
    Erased = ExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::TargetExternalSymbol:
    Erased =
      TargetExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::VALUETYPE: {
    MVT::ValueType VT = cast<VTSDNode>(N)->getVT();
    if (MVT::isExtendedVT(VT)) {
      Erased = ExtendedValueTypeNodes.erase(VT);
    } else {
      Erased = ValueTypeNodes[VT] != 0;
      ValueTypeNodes[VT] = 0;
    }
    break;
  }
  default:
    // Remove it from the CSE Map.
    Erased = CSEMap.RemoveNode(N);
    break;
  }
#ifndef NDEBUG
  // Verify that the node was actually in one of the CSE maps, unless it has a 
  // flag result (which cannot be CSE'd) or is one of the special cases that are
  // not subject to CSE.
  if (!Erased && N->getValueType(N->getNumValues()-1) != MVT::Flag &&
      !N->isTargetOpcode()) {
    N->dump(this);
    cerr << "\n";
    assert(0 && "Node is not in map!");
  }
#endif
}

/// AddNonLeafNodeToCSEMaps - Add the specified node back to the CSE maps.  It
/// has been taken out and modified in some way.  If the specified node already
/// exists in the CSE maps, do not modify the maps, but return the existing node
/// instead.  If it doesn't exist, add it and return null.
///
SDNode *SelectionDAG::AddNonLeafNodeToCSEMaps(SDNode *N) {
  assert(N->getNumOperands() && "This is a leaf node!");
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
  
  SDNode *New = CSEMap.GetOrInsertNode(N);
  if (New != N) return New;  // Node already existed.
  return 0;
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized, 
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, SDOperand Op,
                                           void *&InsertPos) {
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
  
  SDOperand Ops[] = { Op };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, 1);
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}

/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized, 
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, 
                                           SDOperand Op1, SDOperand Op2,
                                           void *&InsertPos) {
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
                                              
  SDOperand Ops[] = { Op1, Op2 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, 2);
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}


/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized, 
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, 
                                           SDOperandPtr Ops,unsigned NumOps,
                                           void *&InsertPos) {
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
  
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, N->getOpcode(), N->getVTList(), Ops, NumOps);
  
  if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    ID.AddInteger(LD->getAddressingMode());
    ID.AddInteger(LD->getExtensionType());
    ID.AddInteger((unsigned int)(LD->getMemoryVT()));
    ID.AddInteger(LD->getAlignment());
    ID.AddInteger(LD->isVolatile());
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    ID.AddInteger(ST->getAddressingMode());
    ID.AddInteger(ST->isTruncatingStore());
    ID.AddInteger((unsigned int)(ST->getMemoryVT()));
    ID.AddInteger(ST->getAlignment());
    ID.AddInteger(ST->isVolatile());
  }
  
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}


SelectionDAG::~SelectionDAG() {
  while (!AllNodes.empty()) {
    SDNode *N = AllNodes.begin();
    N->SetNextInBucket(0);
    if (N->OperandsNeedDelete) {
      delete [] N->OperandList;
    }
    N->OperandList = 0;
    N->NumOperands = 0;
    AllNodes.pop_front();
  }
}

SDOperand SelectionDAG::getZeroExtendInReg(SDOperand Op, MVT::ValueType VT) {
  if (Op.getValueType() == VT) return Op;
  APInt Imm = APInt::getLowBitsSet(Op.getValueSizeInBits(),
                                   MVT::getSizeInBits(VT));
  return getNode(ISD::AND, Op.getValueType(), Op,
                 getConstant(Imm, Op.getValueType()));
}

SDOperand SelectionDAG::getString(const std::string &Val) {
  StringSDNode *&N = StringNodes[Val];
  if (!N) {
    N = new StringSDNode(Val);
    AllNodes.push_back(N);
  }
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstant(uint64_t Val, MVT::ValueType VT, bool isT) {
  MVT::ValueType EltVT =
    MVT::isVector(VT) ? MVT::getVectorElementType(VT) : VT;

  return getConstant(APInt(MVT::getSizeInBits(EltVT), Val), VT, isT);
}

SDOperand SelectionDAG::getConstant(const APInt &Val, MVT::ValueType VT, bool isT) {
  assert(MVT::isInteger(VT) && "Cannot create FP integer constant!");

  MVT::ValueType EltVT =
    MVT::isVector(VT) ? MVT::getVectorElementType(VT) : VT;
  
  assert(Val.getBitWidth() == MVT::getSizeInBits(EltVT) &&
         "APInt size does not match type size!");

  unsigned Opc = isT ? ISD::TargetConstant : ISD::Constant;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), (SDOperand*)0, 0);
  ID.Add(Val);
  void *IP = 0;
  SDNode *N = NULL;
  if ((N = CSEMap.FindNodeOrInsertPos(ID, IP)))
    if (!MVT::isVector(VT))
      return SDOperand(N, 0);
  if (!N) {
    N = new ConstantSDNode(isT, Val, EltVT);
    CSEMap.InsertNode(N, IP);
    AllNodes.push_back(N);
  }

  SDOperand Result(N, 0);
  if (MVT::isVector(VT)) {
    SmallVector<SDOperand, 8> Ops;
    Ops.assign(MVT::getVectorNumElements(VT), Result);
    Result = getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
  }
  return Result;
}

SDOperand SelectionDAG::getIntPtrConstant(uint64_t Val, bool isTarget) {
  return getConstant(Val, TLI.getPointerTy(), isTarget);
}


SDOperand SelectionDAG::getConstantFP(const APFloat& V, MVT::ValueType VT,
                                      bool isTarget) {
  assert(MVT::isFloatingPoint(VT) && "Cannot create integer FP constant!");
                                
  MVT::ValueType EltVT =
    MVT::isVector(VT) ? MVT::getVectorElementType(VT) : VT;

  // Do the map lookup using the actual bit pattern for the floating point
  // value, so that we don't have problems with 0.0 comparing equal to -0.0, and
  // we don't have issues with SNANs.
  unsigned Opc = isTarget ? ISD::TargetConstantFP : ISD::ConstantFP;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(EltVT), (SDOperand*)0, 0);
  ID.Add(V);
  void *IP = 0;
  SDNode *N = NULL;
  if ((N = CSEMap.FindNodeOrInsertPos(ID, IP)))
    if (!MVT::isVector(VT))
      return SDOperand(N, 0);
  if (!N) {
    N = new ConstantFPSDNode(isTarget, V, EltVT);
    CSEMap.InsertNode(N, IP);
    AllNodes.push_back(N);
  }

  SDOperand Result(N, 0);
  if (MVT::isVector(VT)) {
    SmallVector<SDOperand, 8> Ops;
    Ops.assign(MVT::getVectorNumElements(VT), Result);
    Result = getNode(ISD::BUILD_VECTOR, VT, &Ops[0], Ops.size());
  }
  return Result;
}

SDOperand SelectionDAG::getConstantFP(double Val, MVT::ValueType VT,
                                      bool isTarget) {
  MVT::ValueType EltVT =
    MVT::isVector(VT) ? MVT::getVectorElementType(VT) : VT;
  if (EltVT==MVT::f32)
    return getConstantFP(APFloat((float)Val), VT, isTarget);
  else
    return getConstantFP(APFloat(Val), VT, isTarget);
}

SDOperand SelectionDAG::getGlobalAddress(const GlobalValue *GV,
                                         MVT::ValueType VT, int Offset,
                                         bool isTargetGA) {
  unsigned Opc;

  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar) {
    // If GV is an alias then use the aliasee for determining thread-localness.
    if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
      GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal());
  }

  if (GVar && GVar->isThreadLocal())
    Opc = isTargetGA ? ISD::TargetGlobalTLSAddress : ISD::GlobalTLSAddress;
  else
    Opc = isTargetGA ? ISD::TargetGlobalAddress : ISD::GlobalAddress;

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), (SDOperand*)0, 0);
  ID.AddPointer(GV);
  ID.AddInteger(Offset);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
   return SDOperand(E, 0);
  SDNode *N = new GlobalAddressSDNode(isTargetGA, GV, VT, Offset);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getFrameIndex(int FI, MVT::ValueType VT,
                                      bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetFrameIndex : ISD::FrameIndex;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), (SDOperand*)0, 0);
  ID.AddInteger(FI);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new FrameIndexSDNode(FI, VT, isTarget);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getJumpTable(int JTI, MVT::ValueType VT, bool isTarget){
  unsigned Opc = isTarget ? ISD::TargetJumpTable : ISD::JumpTable;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), (SDOperand*)0, 0);
  ID.AddInteger(JTI);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new JumpTableSDNode(JTI, VT, isTarget);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstantPool(Constant *C, MVT::ValueType VT,
                                        unsigned Alignment, int Offset,
                                        bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), (SDOperand*)0, 0);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  ID.AddPointer(C);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantPoolSDNode(isTarget, C, VT, Offset, Alignment);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getConstantPool(MachineConstantPoolValue *C,
                                        MVT::ValueType VT,
                                        unsigned Alignment, int Offset,
                                        bool isTarget) {
  unsigned Opc = isTarget ? ISD::TargetConstantPool : ISD::ConstantPool;
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opc, getVTList(VT), (SDOperand*)0, 0);
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  C->AddSelectionDAGCSEId(ID);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantPoolSDNode(isTarget, C, VT, Offset, Alignment);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getBasicBlock(MachineBasicBlock *MBB) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BasicBlock, getVTList(MVT::Other), (SDOperand*)0, 0);
  ID.AddPointer(MBB);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new BasicBlockSDNode(MBB);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getArgFlags(ISD::ArgFlagsTy Flags) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::ARG_FLAGS, getVTList(MVT::Other), (SDOperand*)0, 0);
  ID.AddInteger(Flags.getRawBits());
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ARG_FLAGSSDNode(Flags);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getValueType(MVT::ValueType VT) {
  if (!MVT::isExtendedVT(VT) && (unsigned)VT >= ValueTypeNodes.size())
    ValueTypeNodes.resize(VT+1);

  SDNode *&N = MVT::isExtendedVT(VT) ?
    ExtendedValueTypeNodes[VT] : ValueTypeNodes[VT];

  if (N) return SDOperand(N, 0);
  N = new VTSDNode(VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getExternalSymbol(const char *Sym, MVT::ValueType VT) {
  SDNode *&N = ExternalSymbols[Sym];
  if (N) return SDOperand(N, 0);
  N = new ExternalSymbolSDNode(false, Sym, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTargetExternalSymbol(const char *Sym,
                                                MVT::ValueType VT) {
  SDNode *&N = TargetExternalSymbols[Sym];
  if (N) return SDOperand(N, 0);
  N = new ExternalSymbolSDNode(true, Sym, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getCondCode(ISD::CondCode Cond) {
  if ((unsigned)Cond >= CondCodeNodes.size())
    CondCodeNodes.resize(Cond+1);
  
  if (CondCodeNodes[Cond] == 0) {
    CondCodeNodes[Cond] = new CondCodeSDNode(Cond);
    AllNodes.push_back(CondCodeNodes[Cond]);
  }
  return SDOperand(CondCodeNodes[Cond], 0);
}

SDOperand SelectionDAG::getRegister(unsigned RegNo, MVT::ValueType VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::Register, getVTList(VT), (SDOperand*)0, 0);
  ID.AddInteger(RegNo);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new RegisterSDNode(RegNo, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getSrcValue(const Value *V) {
  assert((!V || isa<PointerType>(V->getType())) &&
         "SrcValue is not a pointer?");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::SRCVALUE, getVTList(MVT::Other), (SDOperand*)0, 0);
  ID.AddPointer(V);

  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);

  SDNode *N = new SrcValueSDNode(V);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getMemOperand(const MachineMemOperand &MO) {
  const Value *v = MO.getValue();
  assert((!v || isa<PointerType>(v->getType())) &&
         "SrcValue is not a pointer?");

  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::MEMOPERAND, getVTList(MVT::Other), (SDOperand*)0, 0);
  ID.AddPointer(v);
  ID.AddInteger(MO.getFlags());
  ID.AddInteger(MO.getOffset());
  ID.AddInteger(MO.getSize());
  ID.AddInteger(MO.getAlignment());

  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);

  SDNode *N = new MemOperandSDNode(MO);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

/// CreateStackTemporary - Create a stack temporary, suitable for holding the
/// specified value type.
SDOperand SelectionDAG::CreateStackTemporary(MVT::ValueType VT) {
  MachineFrameInfo *FrameInfo = getMachineFunction().getFrameInfo();
  unsigned ByteSize = MVT::getSizeInBits(VT)/8;
  const Type *Ty = MVT::getTypeForValueType(VT);
  unsigned StackAlign = (unsigned)TLI.getTargetData()->getPrefTypeAlignment(Ty);
  int FrameIdx = FrameInfo->CreateStackObject(ByteSize, StackAlign);
  return getFrameIndex(FrameIdx, TLI.getPointerTy());
}


SDOperand SelectionDAG::FoldSetCC(MVT::ValueType VT, SDOperand N1,
                                  SDOperand N2, ISD::CondCode Cond) {
  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return getConstant(0, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2:  return getConstant(1, VT);
    
  case ISD::SETOEQ:
  case ISD::SETOGT:
  case ISD::SETOGE:
  case ISD::SETOLT:
  case ISD::SETOLE:
  case ISD::SETONE:
  case ISD::SETO:
  case ISD::SETUO:
  case ISD::SETUEQ:
  case ISD::SETUNE:
    assert(!MVT::isInteger(N1.getValueType()) && "Illegal setcc for integer!");
    break;
  }
  
  if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val)) {
    const APInt &C2 = N2C->getAPIntValue();
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val)) {
      const APInt &C1 = N1C->getAPIntValue();
      
      switch (Cond) {
      default: assert(0 && "Unknown integer setcc!");
      case ISD::SETEQ:  return getConstant(C1 == C2, VT);
      case ISD::SETNE:  return getConstant(C1 != C2, VT);
      case ISD::SETULT: return getConstant(C1.ult(C2), VT);
      case ISD::SETUGT: return getConstant(C1.ugt(C2), VT);
      case ISD::SETULE: return getConstant(C1.ule(C2), VT);
      case ISD::SETUGE: return getConstant(C1.uge(C2), VT);
      case ISD::SETLT:  return getConstant(C1.slt(C2), VT);
      case ISD::SETGT:  return getConstant(C1.sgt(C2), VT);
      case ISD::SETLE:  return getConstant(C1.sle(C2), VT);
      case ISD::SETGE:  return getConstant(C1.sge(C2), VT);
      }
    }
  }
  if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1.Val)) {
    if (ConstantFPSDNode *N2C = dyn_cast<ConstantFPSDNode>(N2.Val)) {
      // No compile time operations on this type yet.
      if (N1C->getValueType(0) == MVT::ppcf128)
        return SDOperand();

      APFloat::cmpResult R = N1C->getValueAPF().compare(N2C->getValueAPF());
      switch (Cond) {
      default: break;
      case ISD::SETEQ:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOEQ: return getConstant(R==APFloat::cmpEqual, VT);
      case ISD::SETNE:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETONE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpLessThan, VT);
      case ISD::SETLT:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOLT: return getConstant(R==APFloat::cmpLessThan, VT);
      case ISD::SETGT:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOGT: return getConstant(R==APFloat::cmpGreaterThan, VT);
      case ISD::SETLE:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOLE: return getConstant(R==APFloat::cmpLessThan ||
                                           R==APFloat::cmpEqual, VT);
      case ISD::SETGE:  if (R==APFloat::cmpUnordered) 
                          return getNode(ISD::UNDEF, VT);
                        // fall through
      case ISD::SETOGE: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpEqual, VT);
      case ISD::SETO:   return getConstant(R!=APFloat::cmpUnordered, VT);
      case ISD::SETUO:  return getConstant(R==APFloat::cmpUnordered, VT);
      case ISD::SETUEQ: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpEqual, VT);
      case ISD::SETUNE: return getConstant(R!=APFloat::cmpEqual, VT);
      case ISD::SETULT: return getConstant(R==APFloat::cmpUnordered ||
                                           R==APFloat::cmpLessThan, VT);
      case ISD::SETUGT: return getConstant(R==APFloat::cmpGreaterThan ||
                                           R==APFloat::cmpUnordered, VT);
      case ISD::SETULE: return getConstant(R!=APFloat::cmpGreaterThan, VT);
      case ISD::SETUGE: return getConstant(R!=APFloat::cmpLessThan, VT);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      return getSetCC(VT, N2, N1, ISD::getSetCCSwappedOperands(Cond));
    }
  }

  // Could not fold it.
  return SDOperand();
}

/// SignBitIsZero - Return true if the sign bit of Op is known to be zero.  We
/// use this predicate to simplify operations downstream.
bool SelectionDAG::SignBitIsZero(SDOperand Op, unsigned Depth) const {
  unsigned BitWidth = Op.getValueSizeInBits();
  return MaskedValueIsZero(Op, APInt::getSignBit(BitWidth), Depth);
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
bool SelectionDAG::MaskedValueIsZero(SDOperand Op, const APInt &Mask, 
                                     unsigned Depth) const {
  APInt KnownZero, KnownOne;
  ComputeMaskedBits(Op, Mask, KnownZero, KnownOne, Depth);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
  return (KnownZero & Mask) == Mask;
}

/// ComputeMaskedBits - Determine which of the bits specified in Mask are
/// known to be either zero or one and return them in the KnownZero/KnownOne
/// bitsets.  This code only analyzes bits in Mask, in order to short-circuit
/// processing.
void SelectionDAG::ComputeMaskedBits(SDOperand Op, const APInt &Mask, 
                                     APInt &KnownZero, APInt &KnownOne,
                                     unsigned Depth) const {
  unsigned BitWidth = Mask.getBitWidth();
  assert(BitWidth == MVT::getSizeInBits(Op.getValueType()) &&
         "Mask size mismatches value type size!");

  KnownZero = KnownOne = APInt(BitWidth, 0);   // Don't know anything.
  if (Depth == 6 || Mask == 0)
    return;  // Limit search depth.
  
  APInt KnownZero2, KnownOne2;

  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getAPIntValue() & Mask;
    KnownZero = ~KnownOne & Mask;
    return;
  case ISD::AND:
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask & ~KnownZero,
                      KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    return;
  case ISD::OR:
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask & ~KnownOne,
                      KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    return;
  case ISD::XOR: {
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    return;
  }
  case ISD::MUL: {
    APInt Mask2 = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(Op.getOperand(1), Mask2, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask2, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If low bits are zero in either operand, output low known-0 bits.
    // Also compute a conserative estimate for high known-0 bits.
    // More trickiness is possible, but this is sufficient for the
    // interesting case of alignment computation.
    KnownOne.clear();
    unsigned TrailZ = KnownZero.countTrailingOnes() +
                      KnownZero2.countTrailingOnes();
    unsigned LeadZ =  std::max(KnownZero.countLeadingOnes() +
                               KnownZero2.countLeadingOnes(),
                               BitWidth) - BitWidth;

    TrailZ = std::min(TrailZ, BitWidth);
    LeadZ = std::min(LeadZ, BitWidth);
    KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ) |
                APInt::getHighBitsSet(BitWidth, LeadZ);
    KnownZero &= Mask;
    return;
  }
  case ISD::UDIV: {
    // For the purposes of computing leading zeros we can conservatively
    // treat a udiv as a logical right shift by the power of 2 known to
    // be less than the denominator.
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(Op.getOperand(0),
                      AllOnes, KnownZero2, KnownOne2, Depth+1);
    unsigned LeadZ = KnownZero2.countLeadingOnes();

    KnownOne2.clear();
    KnownZero2.clear();
    ComputeMaskedBits(Op.getOperand(1),
                      AllOnes, KnownZero2, KnownOne2, Depth+1);
    unsigned RHSUnknownLeadingOnes = KnownOne2.countLeadingZeros();
    if (RHSUnknownLeadingOnes != BitWidth)
      LeadZ = std::min(BitWidth,
                       LeadZ + BitWidth - RHSUnknownLeadingOnes - 1);

    KnownZero = APInt::getHighBitsSet(BitWidth, LeadZ) & Mask;
    return;
  }
  case ISD::SELECT:
    ComputeMaskedBits(Op.getOperand(2), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case ISD::SELECT_CC:
    ComputeMaskedBits(Op.getOperand(3), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(2), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult &&
        BitWidth > 1)
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    return;
  case ISD::SHL:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        return;

      ComputeMaskedBits(Op.getOperand(0), Mask.lshr(ShAmt),
                        KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero <<= ShAmt;
      KnownOne  <<= ShAmt;
      // low bits known zero.
      KnownZero |= APInt::getLowBitsSet(BitWidth, ShAmt);
    }
    return;
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        return;

      ComputeMaskedBits(Op.getOperand(0), (Mask << ShAmt),
                        KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt) & Mask;
      KnownZero |= HighBits;  // High bits known zero.
    }
    return;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        return;

      APInt InDemandedMask = (Mask << ShAmt);
      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt) & Mask;
      if (HighBits.getBoolValue())
        InDemandedMask |= APInt::getSignBit(BitWidth);
      
      ComputeMaskedBits(Op.getOperand(0), InDemandedMask, KnownZero, KnownOne,
                        Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);
      
      // Handle the sign bits.
      APInt SignBit = APInt::getSignBit(BitWidth);
      SignBit = SignBit.lshr(ShAmt);  // Adjust to where it is now in the mask.
      
      if (KnownZero.intersects(SignBit)) {
        KnownZero |= HighBits;  // New bits are known zero.
      } else if (KnownOne.intersects(SignBit)) {
        KnownOne  |= HighBits;  // New bits are known one.
      }
    }
    return;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    unsigned EBits = MVT::getSizeInBits(EVT);
    
    // Sign extension.  Compute the demanded bits in the result that are not 
    // present in the input.
    APInt NewBits = APInt::getHighBitsSet(BitWidth, BitWidth - EBits) & Mask;

    APInt InSignBit = APInt::getSignBit(EBits);
    APInt InputDemandedBits = Mask & APInt::getLowBitsSet(BitWidth, EBits);
    
    // If the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InSignBit.zext(BitWidth);
    if (NewBits.getBoolValue())
      InputDemandedBits |= InSignBit;
    
    ComputeMaskedBits(Op.getOperand(0), InputDemandedBits,
                      KnownZero, KnownOne, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero.intersects(InSignBit)) {         // Input sign bit known clear
      KnownZero |= NewBits;
      KnownOne  &= ~NewBits;
    } else if (KnownOne.intersects(InSignBit)) {   // Input sign bit known set
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {                              // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne  &= ~NewBits;
    }
    return;
  }
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP: {
    unsigned LowBits = Log2_32(BitWidth)+1;
    KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - LowBits);
    KnownOne  = APInt(BitWidth, 0);
    return;
  }
  case ISD::LOAD: {
    if (ISD::isZEXTLoad(Op.Val)) {
      LoadSDNode *LD = cast<LoadSDNode>(Op);
      MVT::ValueType VT = LD->getMemoryVT();
      unsigned MemBits = MVT::getSizeInBits(VT);
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - MemBits) & Mask;
    }
    return;
  }
  case ISD::ZERO_EXTEND: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    unsigned InBits = MVT::getSizeInBits(InVT);
    APInt NewBits   = APInt::getHighBitsSet(BitWidth, BitWidth - InBits) & Mask;
    APInt InMask    = Mask;
    InMask.trunc(InBits);
    KnownZero.trunc(InBits);
    KnownOne.trunc(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);
    KnownZero.zext(BitWidth);
    KnownOne.zext(BitWidth);
    KnownZero |= NewBits;
    return;
  }
  case ISD::SIGN_EXTEND: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    unsigned InBits = MVT::getSizeInBits(InVT);
    APInt InSignBit = APInt::getSignBit(InBits);
    APInt NewBits   = APInt::getHighBitsSet(BitWidth, BitWidth - InBits) & Mask;
    APInt InMask = Mask;
    InMask.trunc(InBits);

    // If any of the sign extended bits are demanded, we know that the sign
    // bit is demanded. Temporarily set this bit in the mask for our callee.
    if (NewBits.getBoolValue())
      InMask |= InSignBit;

    KnownZero.trunc(InBits);
    KnownOne.trunc(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);

    // Note if the sign bit is known to be zero or one.
    bool SignBitKnownZero = KnownZero.isNegative();
    bool SignBitKnownOne  = KnownOne.isNegative();
    assert(!(SignBitKnownZero && SignBitKnownOne) &&
           "Sign bit can't be known to be both zero and one!");

    // If the sign bit wasn't actually demanded by our caller, we don't
    // want it set in the KnownZero and KnownOne result values. Reset the
    // mask and reapply it to the result values.
    InMask = Mask;
    InMask.trunc(InBits);
    KnownZero &= InMask;
    KnownOne  &= InMask;

    KnownZero.zext(BitWidth);
    KnownOne.zext(BitWidth);

    // If the sign bit is known zero or one, the top bits match.
    if (SignBitKnownZero)
      KnownZero |= NewBits;
    else if (SignBitKnownOne)
      KnownOne  |= NewBits;
    return;
  }
  case ISD::ANY_EXTEND: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    unsigned InBits = MVT::getSizeInBits(InVT);
    APInt InMask = Mask;
    InMask.trunc(InBits);
    KnownZero.trunc(InBits);
    KnownOne.trunc(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);
    KnownZero.zext(BitWidth);
    KnownOne.zext(BitWidth);
    return;
  }
  case ISD::TRUNCATE: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    unsigned InBits = MVT::getSizeInBits(InVT);
    APInt InMask = Mask;
    InMask.zext(InBits);
    KnownZero.zext(InBits);
    KnownOne.zext(InBits);
    ComputeMaskedBits(Op.getOperand(0), InMask, KnownZero, KnownOne, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero.trunc(BitWidth);
    KnownOne.trunc(BitWidth);
    break;
  }
  case ISD::AssertZext: {
    MVT::ValueType VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    APInt InMask = APInt::getLowBitsSet(BitWidth, MVT::getSizeInBits(VT));
    ComputeMaskedBits(Op.getOperand(0), Mask & InMask, KnownZero, 
                      KnownOne, Depth+1);
    KnownZero |= (~InMask) & Mask;
    return;
  }
  case ISD::FGETSIGN:
    // All bits are zero except the low bit.
    KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    return;
  
  case ISD::SUB: {
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0))) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      if (CLHS->getAPIntValue().isNonNegative()) {
        unsigned NLZ = (CLHS->getAPIntValue()+1).countLeadingZeros();
        // NLZ can't be BitWidth with no sign bit
        APInt MaskV = APInt::getHighBitsSet(BitWidth, NLZ+1);
        ComputeMaskedBits(Op.getOperand(1), MaskV, KnownZero2, KnownOne2,
                          Depth+1);

        // If all of the MaskV bits are known to be zero, then we know the
        // output top bits are zero, because we now know that the output is
        // from [0-C].
        if ((KnownZero2 & MaskV) == MaskV) {
          unsigned NLZ2 = CLHS->getAPIntValue().countLeadingZeros();
          // Top bits known zero.
          KnownZero = APInt::getHighBitsSet(BitWidth, NLZ2) & Mask;
        }
      }
    }
  }
  // fall through
  case ISD::ADD: {
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS.  For example, 8+(X<<3) is known to have the
    // low 3 bits clear.
    APInt Mask2 = APInt::getLowBitsSet(BitWidth, Mask.countTrailingOnes());
    ComputeMaskedBits(Op.getOperand(0), Mask2, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    unsigned KnownZeroOut = KnownZero2.countTrailingOnes();

    ComputeMaskedBits(Op.getOperand(1), Mask2, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    KnownZeroOut = std::min(KnownZeroOut,
                            KnownZero2.countTrailingOnes());

    KnownZero |= APInt::getLowBitsSet(BitWidth, KnownZeroOut);
    return;
  }
  case ISD::SREM:
    if (ConstantSDNode *Rem = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt RA = Rem->getAPIntValue();
      if (RA.isPowerOf2() || (-RA).isPowerOf2()) {
        APInt LowBits = RA.isStrictlyPositive() ? (RA - 1) : ~RA;
        APInt Mask2 = LowBits | APInt::getSignBit(BitWidth);
        ComputeMaskedBits(Op.getOperand(0), Mask2,KnownZero2,KnownOne2,Depth+1);

        // The sign of a remainder is equal to the sign of the first
        // operand (zero being positive).
        if (KnownZero2[BitWidth-1] || ((KnownZero2 & LowBits) == LowBits))
          KnownZero2 |= ~LowBits;
        else if (KnownOne2[BitWidth-1])
          KnownOne2 |= ~LowBits;

        KnownZero |= KnownZero2 & Mask;
        KnownOne |= KnownOne2 & Mask;

        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?");
      }
    }
    return;
  case ISD::UREM: {
    if (ConstantSDNode *Rem = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt RA = Rem->getAPIntValue();
      if (RA.isPowerOf2()) {
        APInt LowBits = (RA - 1);
        APInt Mask2 = LowBits & Mask;
        KnownZero |= ~LowBits & Mask;
        ComputeMaskedBits(Op.getOperand(0), Mask2, KnownZero, KnownOne,Depth+1);
        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?");
        break;
      }
    }

    // Since the result is less than or equal to either operand, any leading
    // zero bits in either operand must also exist in the result.
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(Op.getOperand(0), AllOnes, KnownZero, KnownOne,
                      Depth+1);
    ComputeMaskedBits(Op.getOperand(1), AllOnes, KnownZero2, KnownOne2,
                      Depth+1);

    uint32_t Leaders = std::max(KnownZero.countLeadingOnes(),
                                KnownZero2.countLeadingOnes());
    KnownOne.clear();
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders) & Mask;
    return;
  }
  default:
    // Allow the target to implement this method for its nodes.
    if (Op.getOpcode() >= ISD::BUILTIN_OP_END) {
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
      TLI.computeMaskedBitsForTargetNode(Op, Mask, KnownZero, KnownOne, *this);
    }
    return;
  }
}

/// ComputeNumSignBits - Return the number of times the sign bit of the
/// register is replicated into the other bits.  We know that at least 1 bit
/// is always equal to the sign bit (itself), but other cases can give us
/// information.  For example, immediately after an "SRA X, 2", we know that
/// the top 3 bits are all equal to each other, so we return 3.
unsigned SelectionDAG::ComputeNumSignBits(SDOperand Op, unsigned Depth) const{
  MVT::ValueType VT = Op.getValueType();
  assert(MVT::isInteger(VT) && "Invalid VT!");
  unsigned VTBits = MVT::getSizeInBits(VT);
  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;
  
  if (Depth == 6)
    return 1;  // Limit search depth.

  switch (Op.getOpcode()) {
  default: break;
  case ISD::AssertSext:
    Tmp = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    return VTBits-Tmp+1;
  case ISD::AssertZext:
    Tmp = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    return VTBits-Tmp;
    
  case ISD::Constant: {
    const APInt &Val = cast<ConstantSDNode>(Op)->getAPIntValue();
    // If negative, return # leading ones.
    if (Val.isNegative())
      return Val.countLeadingOnes();
    
    // Return # leading zeros.
    return Val.countLeadingZeros();
  }
    
  case ISD::SIGN_EXTEND:
    Tmp = VTBits-MVT::getSizeInBits(Op.getOperand(0).getValueType());
    return ComputeNumSignBits(Op.getOperand(0), Depth+1) + Tmp;
    
  case ISD::SIGN_EXTEND_INREG:
    // Max of the input and what this extends.
    Tmp = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    Tmp = VTBits-Tmp+1;
    
    Tmp2 = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    return std::max(Tmp, Tmp2);

  case ISD::SRA:
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    // SRA X, C   -> adds C sign bits.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      Tmp += C->getValue();
      if (Tmp > VTBits) Tmp = VTBits;
    }
    return Tmp;
  case ISD::SHL:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      // shl destroys sign bits.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (C->getValue() >= VTBits ||      // Bad shift.
          C->getValue() >= Tmp) break;    // Shifted all sign bits out.
      return Tmp - C->getValue();
    }
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:    // NOT is handled here.
    // Logical binary ops preserve the number of sign bits at the worst.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp != 1) {
      Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
      FirstAnswer = std::min(Tmp, Tmp2);
      // We computed what we know about the sign bits as our first
      // answer. Now proceed to the generic code that uses
      // ComputeMaskedBits, and pick whichever answer is better.
    }
    break;

  case ISD::SELECT:
    Tmp = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(Op.getOperand(2), Depth+1);
    return std::min(Tmp, Tmp2);
    
  case ISD::SETCC:
    // If setcc returns 0/-1, all bits are sign bits.
    if (TLI.getSetCCResultContents() ==
        TargetLowering::ZeroOrNegativeOneSetCCResult)
      return VTBits;
    break;
  case ISD::ROTL:
  case ISD::ROTR:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned RotAmt = C->getValue() & (VTBits-1);
      
      // Handle rotate right by N like a rotate left by 32-N.
      if (Op.getOpcode() == ISD::ROTR)
        RotAmt = (VTBits-RotAmt) & (VTBits-1);

      // If we aren't rotating out all of the known-in sign bits, return the
      // number that are left.  This handles rotl(sext(x), 1) for example.
      Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
      if (Tmp > RotAmt+1) return Tmp-RotAmt;
    }
    break;
  case ISD::ADD:
    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
      
    // Special case decrementing a value (ADD X, -1):
    if (ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(Op.getOperand(0)))
      if (CRHS->isAllOnesValue()) {
        APInt KnownZero, KnownOne;
        APInt Mask = APInt::getAllOnesValue(VTBits);
        ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
        
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(VTBits, 1)) == Mask)
          return VTBits;
        
        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (KnownZero.isNegative())
          return Tmp;
      }
      
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;
      return std::min(Tmp, Tmp2)-1;
    break;
    
  case ISD::SUB:
    Tmp2 = ComputeNumSignBits(Op.getOperand(1), Depth+1);
    if (Tmp2 == 1) return 1;
      
    // Handle NEG.
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0)))
      if (CLHS->isNullValue()) {
        APInt KnownZero, KnownOne;
        APInt Mask = APInt::getAllOnesValue(VTBits);
        ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(VTBits, 1)) == Mask)
          return VTBits;
        
        // If the input is known to be positive (the sign bit is known clear),
        // the output of the NEG has the same number of sign bits as the input.
        if (KnownZero.isNegative())
          return Tmp2;
        
        // Otherwise, we treat this like a SUB.
      }
    
    // Sub can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(Op.getOperand(0), Depth+1);
    if (Tmp == 1) return 1;  // Early out.
      return std::min(Tmp, Tmp2)-1;
    break;
  case ISD::TRUNCATE:
    // FIXME: it's tricky to do anything useful for this, but it is an important
    // case for targets like X86.
    break;
  }
  
  // Handle LOADX separately here. EXTLOAD case will fallthrough.
  if (Op.getOpcode() == ISD::LOAD) {
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    unsigned ExtType = LD->getExtensionType();
    switch (ExtType) {
    default: break;
    case ISD::SEXTLOAD:    // '17' bits known
      Tmp = MVT::getSizeInBits(LD->getMemoryVT());
      return VTBits-Tmp+1;
    case ISD::ZEXTLOAD:    // '16' bits known
      Tmp = MVT::getSizeInBits(LD->getMemoryVT());
      return VTBits-Tmp;
    }
  }

  // Allow the target to implement this method for its nodes.
  if (Op.getOpcode() >= ISD::BUILTIN_OP_END ||
      Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN || 
      Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
      Op.getOpcode() == ISD::INTRINSIC_VOID) {
    unsigned NumBits = TLI.ComputeNumSignBitsForTargetNode(Op, Depth);
    if (NumBits > 1) FirstAnswer = std::max(FirstAnswer, NumBits);
  }
  
  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  APInt KnownZero, KnownOne;
  APInt Mask = APInt::getAllOnesValue(VTBits);
  ComputeMaskedBits(Op, Mask, KnownZero, KnownOne, Depth);
  
  if (KnownZero.isNegative()) {        // sign bit is 0
    Mask = KnownZero;
  } else if (KnownOne.isNegative()) {  // sign bit is 1;
    Mask = KnownOne;
  } else {
    // Nothing known.
    return FirstAnswer;
  }
  
  // Okay, we know that the sign bit in Mask is set.  Use CLZ to determine
  // the number of identical bits in the top of the input value.
  Mask = ~Mask;
  Mask <<= Mask.getBitWidth()-VTBits;
  // Return # leading zeros.  We use 'min' here in case Val was zero before
  // shifting.  We don't want to return '64' as for an i32 "0".
  return std::max(FirstAnswer, std::min(VTBits, Mask.countLeadingZeros()));
}


bool SelectionDAG::isVerifiedDebugInfoDesc(SDOperand Op) const {
  GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op);
  if (!GA) return false;
  GlobalVariable *GV = dyn_cast<GlobalVariable>(GA->getGlobal());
  if (!GV) return false;
  MachineModuleInfo *MMI = getMachineModuleInfo();
  return MMI && MMI->hasDebugInfo() && MMI->isVerified(GV);
}


/// getShuffleScalarElt - Returns the scalar element that will make up the ith
/// element of the result of the vector shuffle.
SDOperand SelectionDAG::getShuffleScalarElt(const SDNode *N, unsigned Idx) {
  MVT::ValueType VT = N->getValueType(0);
  SDOperand PermMask = N->getOperand(2);
  unsigned NumElems = PermMask.getNumOperands();
  SDOperand V = (Idx < NumElems) ? N->getOperand(0) : N->getOperand(1);
  Idx %= NumElems;
  if (V.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    return (Idx == 0)
     ? V.getOperand(0) : getNode(ISD::UNDEF, MVT::getVectorElementType(VT));
  }
  if (V.getOpcode() == ISD::VECTOR_SHUFFLE) {
    SDOperand Elt = PermMask.getOperand(Idx);
    if (Elt.getOpcode() == ISD::UNDEF)
      return getNode(ISD::UNDEF, MVT::getVectorElementType(VT));
    return getShuffleScalarElt(V.Val,cast<ConstantSDNode>(Elt)->getValue());
  }
  return SDOperand();
}


/// getNode - Gets or creates the specified node.
///
SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT) {
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, Opcode, getVTList(VT), (SDOperand*)0, 0);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new SDNode(Opcode, SDNode::getSDVTList(VT));
  CSEMap.InsertNode(N, IP);
  
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand Operand) {
  // Constant fold unary operations with an integer constant operand.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Operand.Val)) {
    const APInt &Val = C->getAPIntValue();
    unsigned BitWidth = MVT::getSizeInBits(VT);
    switch (Opcode) {
    default: break;
    case ISD::SIGN_EXTEND:
      return getConstant(APInt(Val).sextOrTrunc(BitWidth), VT);
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::TRUNCATE:
      return getConstant(APInt(Val).zextOrTrunc(BitWidth), VT);
    case ISD::UINT_TO_FP:
    case ISD::SINT_TO_FP: {
      const uint64_t zero[] = {0, 0};
      // No compile time operations on this type.
      if (VT==MVT::ppcf128)
        break;
      APFloat apf = APFloat(APInt(BitWidth, 2, zero));
      (void)apf.convertFromAPInt(Val, 
                                 Opcode==ISD::SINT_TO_FP,
                                 APFloat::rmNearestTiesToEven);
      return getConstantFP(apf, VT);
    }
    case ISD::BIT_CONVERT:
      if (VT == MVT::f32 && C->getValueType(0) == MVT::i32)
        return getConstantFP(Val.bitsToFloat(), VT);
      else if (VT == MVT::f64 && C->getValueType(0) == MVT::i64)
        return getConstantFP(Val.bitsToDouble(), VT);
      break;
    case ISD::BSWAP:
      return getConstant(Val.byteSwap(), VT);
    case ISD::CTPOP:
      return getConstant(Val.countPopulation(), VT);
    case ISD::CTLZ:
      return getConstant(Val.countLeadingZeros(), VT);
    case ISD::CTTZ:
      return getConstant(Val.countTrailingZeros(), VT);
    }
  }

  // Constant fold unary operations with a floating point constant operand.
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand.Val)) {
    APFloat V = C->getValueAPF();    // make copy
    if (VT != MVT::ppcf128 && Operand.getValueType() != MVT::ppcf128) {
      switch (Opcode) {
      case ISD::FNEG:
        V.changeSign();
        return getConstantFP(V, VT);
      case ISD::FABS:
        V.clearSign();
        return getConstantFP(V, VT);
      case ISD::FP_ROUND:
      case ISD::FP_EXTEND:
        // This can return overflow, underflow, or inexact; we don't care.
        // FIXME need to be more flexible about rounding mode.
        (void)V.convert(*MVTToAPFloatSemantics(VT),
                        APFloat::rmNearestTiesToEven);
        return getConstantFP(V, VT);
      case ISD::FP_TO_SINT:
      case ISD::FP_TO_UINT: {
        integerPart x;
        assert(integerPartWidth >= 64);
        // FIXME need to be more flexible about rounding mode.
        APFloat::opStatus s = V.convertToInteger(&x, 64U,
                              Opcode==ISD::FP_TO_SINT,
                              APFloat::rmTowardZero);
        if (s==APFloat::opInvalidOp)     // inexact is OK, in fact usual
          break;
        return getConstant(x, VT);
      }
      case ISD::BIT_CONVERT:
        if (VT == MVT::i32 && C->getValueType(0) == MVT::f32)
          return getConstant((uint32_t)V.convertToAPInt().getZExtValue(), VT);
        else if (VT == MVT::i64 && C->getValueType(0) == MVT::f64)
          return getConstant(V.convertToAPInt().getZExtValue(), VT);
        break;
      }
    }
  }

  unsigned OpOpcode = Operand.Val->getOpcode();
  switch (Opcode) {
  case ISD::TokenFactor:
  case ISD::MERGE_VALUES:
    return Operand;         // Factor or merge of one node?  No need.
  case ISD::FP_ROUND: assert(0 && "Invalid method to make FP_ROUND node");
  case ISD::FP_EXTEND:
    assert(MVT::isFloatingPoint(VT) &&
           MVT::isFloatingPoint(Operand.getValueType()) && "Invalid FP cast!");
    if (Operand.getValueType() == VT) return Operand;  // noop conversion.
    if (Operand.getOpcode() == ISD::UNDEF)
      return getNode(ISD::UNDEF, VT);
    break;
  case ISD::SIGN_EXTEND:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid SIGN_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(MVT::getSizeInBits(Operand.getValueType()) < MVT::getSizeInBits(VT)
           && "Invalid sext node, dst < src!");
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ZERO_EXTEND:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid ZERO_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(MVT::getSizeInBits(Operand.getValueType()) < MVT::getSizeInBits(VT)
           && "Invalid zext node, dst < src!");
    if (OpOpcode == ISD::ZERO_EXTEND)   // (zext (zext x)) -> (zext x)
      return getNode(ISD::ZERO_EXTEND, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ANY_EXTEND:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid ANY_EXTEND!");
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(MVT::getSizeInBits(Operand.getValueType()) < MVT::getSizeInBits(VT)
           && "Invalid anyext node, dst < src!");
    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND)
      // (ext (zext x)) -> (zext x)  and  (ext (sext x)) -> (sext x)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::TRUNCATE:
    assert(MVT::isInteger(VT) && MVT::isInteger(Operand.getValueType()) &&
           "Invalid TRUNCATE!");
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
    assert(MVT::getSizeInBits(Operand.getValueType()) > MVT::getSizeInBits(VT)
           && "Invalid truncate node, src < dst!");
    if (OpOpcode == ISD::TRUNCATE)
      return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
    else if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
             OpOpcode == ISD::ANY_EXTEND) {
      // If the source is smaller than the dest, we still need an extend.
      if (MVT::getSizeInBits(Operand.Val->getOperand(0).getValueType())
          < MVT::getSizeInBits(VT))
        return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
      else if (MVT::getSizeInBits(Operand.Val->getOperand(0).getValueType())
               > MVT::getSizeInBits(VT))
        return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
      else
        return Operand.Val->getOperand(0);
    }
    break;
  case ISD::BIT_CONVERT:
    // Basic sanity checking.
    assert(MVT::getSizeInBits(VT) == MVT::getSizeInBits(Operand.getValueType())
           && "Cannot BIT_CONVERT between types of different sizes!");
    if (VT == Operand.getValueType()) return Operand;  // noop conversion.
    if (OpOpcode == ISD::BIT_CONVERT)  // bitconv(bitconv(x)) -> bitconv(x)
      return getNode(ISD::BIT_CONVERT, VT, Operand.getOperand(0));
    if (OpOpcode == ISD::UNDEF)
      return getNode(ISD::UNDEF, VT);
    break;
  case ISD::SCALAR_TO_VECTOR:
    assert(MVT::isVector(VT) && !MVT::isVector(Operand.getValueType()) &&
           MVT::getVectorElementType(VT) == Operand.getValueType() &&
           "Illegal SCALAR_TO_VECTOR node!");
    if (OpOpcode == ISD::UNDEF)
      return getNode(ISD::UNDEF, VT);
    // scalar_to_vector(extract_vector_elt V, 0) -> V, top bits are undefined.
    if (OpOpcode == ISD::EXTRACT_VECTOR_ELT &&
        isa<ConstantSDNode>(Operand.getOperand(1)) &&
        Operand.getConstantOperandVal(1) == 0 &&
        Operand.getOperand(0).getValueType() == VT)
      return Operand.getOperand(0);
    break;
  case ISD::FNEG:
    if (OpOpcode == ISD::FSUB)   // -(X-Y) -> (Y-X)
      return getNode(ISD::FSUB, VT, Operand.Val->getOperand(1),
                     Operand.Val->getOperand(0));
    if (OpOpcode == ISD::FNEG)  // --X -> X
      return Operand.Val->getOperand(0);
    break;
  case ISD::FABS:
    if (OpOpcode == ISD::FNEG)  // abs(-X) -> abs(X)
      return getNode(ISD::FABS, VT, Operand.Val->getOperand(0));
    break;
  }

  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Flag) { // Don't CSE flag producing nodes
    FoldingSetNodeID ID;
    SDOperand Ops[1] = { Operand };
    AddNodeIDNode(ID, Opcode, VTs, Ops, 1);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new UnarySDNode(Opcode, VTs, Operand);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new UnarySDNode(Opcode, VTs, Operand);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}



SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2) {
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  switch (Opcode) {
  default: break;
  case ISD::TokenFactor:
    assert(VT == MVT::Other && N1.getValueType() == MVT::Other &&
           N2.getValueType() == MVT::Other && "Invalid token factor!");
    // Fold trivial token factors.
    if (N1.getOpcode() == ISD::EntryToken) return N2;
    if (N2.getOpcode() == ISD::EntryToken) return N1;
    break;
  case ISD::AND:
    assert(MVT::isInteger(VT) && N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    // (X & 0) -> 0.  This commonly occurs when legalizing i64 values, so it's
    // worth handling here.
    if (N2C && N2C->isNullValue())
      return N2;
    if (N2C && N2C->isAllOnesValue())  // X & -1 -> X
      return N1;
    break;
  case ISD::OR:
  case ISD::XOR:
    assert(MVT::isInteger(VT) && N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    // (X ^| 0) -> X.  This commonly occurs when legalizing i64 values, so it's
    // worth handling here.
    if (N2C && N2C->isNullValue())
      return N1;
    break;
  case ISD::UDIV:
  case ISD::UREM:
  case ISD::MULHU:
  case ISD::MULHS:
    assert(MVT::isInteger(VT) && "This operator does not apply to FP types!");
    // fall through
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::SREM:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
    assert(N1.getValueType() == N2.getValueType() &&
           N1.getValueType() == VT && "Binary operator types must match!");
    break;
  case ISD::FCOPYSIGN:   // N1 and result must match.  N1/N2 need not match.
    assert(N1.getValueType() == VT &&
           MVT::isFloatingPoint(N1.getValueType()) && 
           MVT::isFloatingPoint(N2.getValueType()) &&
           "Invalid FCOPYSIGN!");
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR:
    assert(VT == N1.getValueType() &&
           "Shift operators return type must be the same as their first arg");
    assert(MVT::isInteger(VT) && MVT::isInteger(N2.getValueType()) &&
           VT != MVT::i1 && "Shifts only work on integers");
    break;
  case ISD::FP_ROUND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg round!");
    assert(MVT::isFloatingPoint(VT) && MVT::isFloatingPoint(EVT) &&
           "Cannot FP_ROUND_INREG integer types");
    assert(MVT::getSizeInBits(EVT) <= MVT::getSizeInBits(VT) &&
           "Not rounding down!");
    if (cast<VTSDNode>(N2)->getVT() == VT) return N1;  // Not actually rounding.
    break;
  }
  case ISD::FP_ROUND:
    assert(MVT::isFloatingPoint(VT) &&
           MVT::isFloatingPoint(N1.getValueType()) &&
           MVT::getSizeInBits(VT) <= MVT::getSizeInBits(N1.getValueType()) &&
           isa<ConstantSDNode>(N2) && "Invalid FP_ROUND!");
    if (N1.getValueType() == VT) return N1;  // noop conversion.
    break;
  case ISD::AssertSext:
  case ISD::AssertZext: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(MVT::isInteger(VT) && MVT::isInteger(EVT) &&
           "Cannot *_EXTEND_INREG FP types");
    assert(MVT::getSizeInBits(EVT) <= MVT::getSizeInBits(VT) &&
           "Not extending!");
    if (VT == EVT) return N1; // noop assertion.
    break;
  }
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(MVT::isInteger(VT) && MVT::isInteger(EVT) &&
           "Cannot *_EXTEND_INREG FP types");
    assert(MVT::getSizeInBits(EVT) <= MVT::getSizeInBits(VT) &&
           "Not extending!");
    if (EVT == VT) return N1;  // Not actually extending

    if (N1C) {
      APInt Val = N1C->getAPIntValue();
      unsigned FromBits = MVT::getSizeInBits(cast<VTSDNode>(N2)->getVT());
      Val <<= Val.getBitWidth()-FromBits;
      Val = Val.ashr(Val.getBitWidth()-FromBits);
      return getConstant(Val, VT);
    }
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    assert(N2C && "Bad EXTRACT_VECTOR_ELT!");

    // EXTRACT_VECTOR_ELT of an UNDEF is an UNDEF.
    if (N1.getOpcode() == ISD::UNDEF)
      return getNode(ISD::UNDEF, VT);
      
    // EXTRACT_VECTOR_ELT of CONCAT_VECTORS is often formed while lowering is
    // expanding copies of large vectors from registers.
    if (N1.getOpcode() == ISD::CONCAT_VECTORS &&
        N1.getNumOperands() > 0) {
      unsigned Factor =
        MVT::getVectorNumElements(N1.getOperand(0).getValueType());
      return getNode(ISD::EXTRACT_VECTOR_ELT, VT,
                     N1.getOperand(N2C->getValue() / Factor),
                     getConstant(N2C->getValue() % Factor, N2.getValueType()));
    }

    // EXTRACT_VECTOR_ELT of BUILD_VECTOR is often formed while lowering is
    // expanding large vector constants.
    if (N1.getOpcode() == ISD::BUILD_VECTOR)
      return N1.getOperand(N2C->getValue());
      
    // EXTRACT_VECTOR_ELT of INSERT_VECTOR_ELT is often formed when vector
    // operations are lowered to scalars.
    if (N1.getOpcode() == ISD::INSERT_VECTOR_ELT)
      if (ConstantSDNode *IEC = dyn_cast<ConstantSDNode>(N1.getOperand(2))) {
        if (IEC == N2C)
          return N1.getOperand(1);
        else
          return getNode(ISD::EXTRACT_VECTOR_ELT, VT, N1.getOperand(0), N2);
      }
    break;
  case ISD::EXTRACT_ELEMENT:
    assert(N2C && (unsigned)N2C->getValue() < 2 && "Bad EXTRACT_ELEMENT!");
    assert(!MVT::isVector(N1.getValueType()) &&
           MVT::isInteger(N1.getValueType()) &&
           !MVT::isVector(VT) && MVT::isInteger(VT) &&
           "EXTRACT_ELEMENT only applies to integers!");

    // EXTRACT_ELEMENT of BUILD_PAIR is often formed while legalize is expanding
    // 64-bit integers into 32-bit parts.  Instead of building the extract of
    // the BUILD_PAIR, only to have legalize rip it apart, just do it now. 
    if (N1.getOpcode() == ISD::BUILD_PAIR)
      return N1.getOperand(N2C->getValue());

    // EXTRACT_ELEMENT of a constant int is also very common.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      unsigned ElementSize = MVT::getSizeInBits(VT);
      unsigned Shift = ElementSize * N2C->getValue();
      APInt ShiftedVal = C->getAPIntValue().lshr(Shift);
      return getConstant(ShiftedVal.trunc(ElementSize), VT);
    }
    break;
  case ISD::EXTRACT_SUBVECTOR:
    if (N1.getValueType() == VT) // Trivial extraction.
      return N1;
    break;
  }

  if (N1C) {
    if (N2C) {
      APInt C1 = N1C->getAPIntValue(), C2 = N2C->getAPIntValue();
      switch (Opcode) {
      case ISD::ADD: return getConstant(C1 + C2, VT);
      case ISD::SUB: return getConstant(C1 - C2, VT);
      case ISD::MUL: return getConstant(C1 * C2, VT);
      case ISD::UDIV:
        if (C2.getBoolValue()) return getConstant(C1.udiv(C2), VT);
        break;
      case ISD::UREM :
        if (C2.getBoolValue()) return getConstant(C1.urem(C2), VT);
        break;
      case ISD::SDIV :
        if (C2.getBoolValue()) return getConstant(C1.sdiv(C2), VT);
        break;
      case ISD::SREM :
        if (C2.getBoolValue()) return getConstant(C1.srem(C2), VT);
        break;
      case ISD::AND  : return getConstant(C1 & C2, VT);
      case ISD::OR   : return getConstant(C1 | C2, VT);
      case ISD::XOR  : return getConstant(C1 ^ C2, VT);
      case ISD::SHL  : return getConstant(C1 << C2, VT);
      case ISD::SRL  : return getConstant(C1.lshr(C2), VT);
      case ISD::SRA  : return getConstant(C1.ashr(C2), VT);
      case ISD::ROTL : return getConstant(C1.rotl(C2), VT);
      case ISD::ROTR : return getConstant(C1.rotr(C2), VT);
      default: break;
      }
    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1C, N2C);
        std::swap(N1, N2);
      }
    }
  }

  // Constant fold FP operations.
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2.Val);
  if (N1CFP) {
    if (!N2CFP && isCommutativeBinOp(Opcode)) {
      // Cannonicalize constant to RHS if commutative
      std::swap(N1CFP, N2CFP);
      std::swap(N1, N2);
    } else if (N2CFP && VT != MVT::ppcf128) {
      APFloat V1 = N1CFP->getValueAPF(), V2 = N2CFP->getValueAPF();
      APFloat::opStatus s;
      switch (Opcode) {
      case ISD::FADD: 
        s = V1.add(V2, APFloat::rmNearestTiesToEven);
        if (s != APFloat::opInvalidOp)
          return getConstantFP(V1, VT);
        break;
      case ISD::FSUB: 
        s = V1.subtract(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp)
          return getConstantFP(V1, VT);
        break;
      case ISD::FMUL:
        s = V1.multiply(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp)
          return getConstantFP(V1, VT);
        break;
      case ISD::FDIV:
        s = V1.divide(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp && s!=APFloat::opDivByZero)
          return getConstantFP(V1, VT);
        break;
      case ISD::FREM :
        s = V1.mod(V2, APFloat::rmNearestTiesToEven);
        if (s!=APFloat::opInvalidOp && s!=APFloat::opDivByZero)
          return getConstantFP(V1, VT);
        break;
      case ISD::FCOPYSIGN:
        V1.copySign(V2);
        return getConstantFP(V1, VT);
      default: break;
      }
    }
  }
  
  // Canonicalize an UNDEF to the RHS, even over a constant.
  if (N1.getOpcode() == ISD::UNDEF) {
    if (isCommutativeBinOp(Opcode)) {
      std::swap(N1, N2);
    } else {
      switch (Opcode) {
      case ISD::FP_ROUND_INREG:
      case ISD::SIGN_EXTEND_INREG:
      case ISD::SUB:
      case ISD::FSUB:
      case ISD::FDIV:
      case ISD::FREM:
      case ISD::SRA:
        return N1;     // fold op(undef, arg2) -> undef
      case ISD::UDIV:
      case ISD::SDIV:
      case ISD::UREM:
      case ISD::SREM:
      case ISD::SRL:
      case ISD::SHL:
        if (!MVT::isVector(VT)) 
          return getConstant(0, VT);    // fold op(undef, arg2) -> 0
        // For vectors, we can't easily build an all zero vector, just return
        // the LHS.
        return N2;
      }
    }
  }
  
  // Fold a bunch of operators when the RHS is undef. 
  if (N2.getOpcode() == ISD::UNDEF) {
    switch (Opcode) {
    case ISD::XOR:
      if (N1.getOpcode() == ISD::UNDEF)
        // Handle undef ^ undef -> 0 special case. This is a common
        // idiom (misuse).
        return getConstant(0, VT);
      // fallthrough
    case ISD::ADD:
    case ISD::ADDC:
    case ISD::ADDE:
    case ISD::SUB:
    case ISD::FADD:
    case ISD::FSUB:
    case ISD::FMUL:
    case ISD::FDIV:
    case ISD::FREM:
    case ISD::UDIV:
    case ISD::SDIV:
    case ISD::UREM:
    case ISD::SREM:
      return N2;       // fold op(arg1, undef) -> undef
    case ISD::MUL: 
    case ISD::AND:
    case ISD::SRL:
    case ISD::SHL:
      if (!MVT::isVector(VT)) 
        return getConstant(0, VT);  // fold op(arg1, undef) -> 0
      // For vectors, we can't easily build an all zero vector, just return
      // the LHS.
      return N1;
    case ISD::OR:
      if (!MVT::isVector(VT)) 
        return getConstant(MVT::getIntVTBitMask(VT), VT);
      // For vectors, we can't easily build an all one vector, just return
      // the LHS.
      return N1;
    case ISD::SRA:
      return N1;
    }
  }

  // Memoize this node if possible.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Flag) {
    SDOperand Ops[] = { N1, N2 };
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, 2);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new BinarySDNode(Opcode, VTs, N1, N2);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new BinarySDNode(Opcode, VTs, N1, N2);
  }

  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3) {
  // Perform various simplifications.
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  switch (Opcode) {
  case ISD::SETCC: {
    // Use FoldSetCC to simplify SETCC's.
    SDOperand Simp = FoldSetCC(VT, N1, N2, cast<CondCodeSDNode>(N3)->get());
    if (Simp.Val) return Simp;
    break;
  }
  case ISD::SELECT:
    if (N1C) {
     if (N1C->getValue())
        return N2;             // select true, X, Y -> X
      else
        return N3;             // select false, X, Y -> Y
    }

    if (N2 == N3) return N2;   // select C, X, X -> X
    break;
  case ISD::BRCOND:
    if (N2C) {
      if (N2C->getValue()) // Unconditional branch
        return getNode(ISD::BR, MVT::Other, N1, N3);
      else
        return N1;         // Never-taken branch
    }
    break;
  case ISD::VECTOR_SHUFFLE:
    assert(VT == N1.getValueType() && VT == N2.getValueType() &&
           MVT::isVector(VT) && MVT::isVector(N3.getValueType()) &&
           N3.getOpcode() == ISD::BUILD_VECTOR &&
           MVT::getVectorNumElements(VT) == N3.getNumOperands() &&
           "Illegal VECTOR_SHUFFLE node!");
    break;
  case ISD::BIT_CONVERT:
    // Fold bit_convert nodes from a type to themselves.
    if (N1.getValueType() == VT)
      return N1;
    break;
  }

  // Memoize node if it doesn't produce a flag.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Flag) {
    SDOperand Ops[] = { N1, N2, N3 };
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, 3);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new TernarySDNode(Opcode, VTs, N1, N2, N3);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new TernarySDNode(Opcode, VTs, N1, N2, N3);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4) {
  SDOperand Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, VT, Ops, 4);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4, SDOperand N5) {
  SDOperand Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, VT, Ops, 5);
}

/// getMemsetValue - Vectorized representation of the memset value
/// operand.
static SDOperand getMemsetValue(SDOperand Value, MVT::ValueType VT,
                                SelectionDAG &DAG) {
  unsigned NumBits = MVT::isVector(VT) ?
    MVT::getSizeInBits(MVT::getVectorElementType(VT)) : MVT::getSizeInBits(VT);
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Value)) {
    APInt Val = APInt(NumBits, C->getValue() & 255);
    unsigned Shift = 8;
    for (unsigned i = NumBits; i > 8; i >>= 1) {
      Val = (Val << Shift) | Val;
      Shift <<= 1;
    }
    if (MVT::isInteger(VT))
      return DAG.getConstant(Val, VT);
    return DAG.getConstantFP(APFloat(Val), VT);
  }

  Value = DAG.getNode(ISD::ZERO_EXTEND, VT, Value);
  unsigned Shift = 8;
  for (unsigned i = NumBits; i > 8; i >>= 1) {
    Value = DAG.getNode(ISD::OR, VT,
                        DAG.getNode(ISD::SHL, VT, Value,
                                    DAG.getConstant(Shift, MVT::i8)), Value);
    Shift <<= 1;
  }

  return Value;
}

/// getMemsetStringVal - Similar to getMemsetValue. Except this is only
/// used when a memcpy is turned into a memset when the source is a constant
/// string ptr.
static SDOperand getMemsetStringVal(MVT::ValueType VT, SelectionDAG &DAG,
                                    const TargetLowering &TLI,
                                    std::string &Str, unsigned Offset) {
  assert(!MVT::isVector(VT) && "Can't handle vector type here!");
  unsigned NumBits = MVT::getSizeInBits(VT);
  unsigned MSB = NumBits / 8;
  uint64_t Val = 0;
  if (TLI.isLittleEndian())
    Offset = Offset + MSB - 1;
  for (unsigned i = 0; i != MSB; ++i) {
    Val = (Val << 8) | (unsigned char)Str[Offset];
    Offset += TLI.isLittleEndian() ? -1 : 1;
  }
  return DAG.getConstant(Val, VT);
}

/// getMemBasePlusOffset - Returns base and offset node for the 
///
static SDOperand getMemBasePlusOffset(SDOperand Base, unsigned Offset,
                                      SelectionDAG &DAG) {
  MVT::ValueType VT = Base.getValueType();
  return DAG.getNode(ISD::ADD, VT, Base, DAG.getConstant(Offset, VT));
}

/// isMemSrcFromString - Returns true if memcpy source is a string constant.
///
static bool isMemSrcFromString(SDOperand Src, std::string &Str,
                               uint64_t &SrcOff) {
  unsigned SrcDelta = 0;
  GlobalAddressSDNode *G = NULL;
  if (Src.getOpcode() == ISD::GlobalAddress)
    G = cast<GlobalAddressSDNode>(Src);
  else if (Src.getOpcode() == ISD::ADD &&
           Src.getOperand(0).getOpcode() == ISD::GlobalAddress &&
           Src.getOperand(1).getOpcode() == ISD::Constant) {
    G = cast<GlobalAddressSDNode>(Src.getOperand(0));
    SrcDelta = cast<ConstantSDNode>(Src.getOperand(1))->getValue();
  }
  if (!G)
    return false;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(G->getGlobal());
  if (GV && GV->isConstant()) {
    Str = GV->getStringValue(false);
    if (!Str.empty()) {
      SrcOff += SrcDelta;
      return true;
    }
  }

  return false;
}

/// MeetsMaxMemopRequirement - Determines if the number of memory ops required
/// to replace the memset / memcpy is below the threshold. It also returns the
/// types of the sequence of memory ops to perform memset / memcpy.
static
bool MeetsMaxMemopRequirement(std::vector<MVT::ValueType> &MemOps,
                              SDOperand Dst, SDOperand Src,
                              unsigned Limit, uint64_t Size, unsigned &Align,
                              SelectionDAG &DAG,
                              const TargetLowering &TLI) {
  bool AllowUnalign = TLI.allowsUnalignedMemoryAccesses();

  std::string Str;
  uint64_t SrcOff = 0;
  bool isSrcStr = isMemSrcFromString(Src, Str, SrcOff);
  bool isSrcConst = isa<ConstantSDNode>(Src);
  MVT::ValueType VT= TLI.getOptimalMemOpType(Size, Align, isSrcConst, isSrcStr);
  if (VT != MVT::iAny) {
    unsigned NewAlign = (unsigned)
      TLI.getTargetData()->getABITypeAlignment(MVT::getTypeForValueType(VT));
    // If source is a string constant, this will require an unaligned load.
    if (NewAlign > Align && (isSrcConst || AllowUnalign)) {
      if (Dst.getOpcode() != ISD::FrameIndex) {
        // Can't change destination alignment. It requires a unaligned store.
        if (AllowUnalign)
          VT = MVT::iAny;
      } else {
        int FI = cast<FrameIndexSDNode>(Dst)->getIndex();
        MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
        if (MFI->isFixedObjectIndex(FI)) {
          // Can't change destination alignment. It requires a unaligned store.
          if (AllowUnalign)
            VT = MVT::iAny;
        } else {
          // Give the stack frame object a larger alignment.
          MFI->setObjectAlignment(FI, NewAlign);
          Align = NewAlign;
        }
      }
    }
  }

  if (VT == MVT::iAny) {
    if (AllowUnalign) {
      VT = MVT::i64;
    } else {
      switch (Align & 7) {
      case 0:  VT = MVT::i64; break;
      case 4:  VT = MVT::i32; break;
      case 2:  VT = MVT::i16; break;
      default: VT = MVT::i8;  break;
      }
    }

    MVT::ValueType LVT = MVT::i64;
    while (!TLI.isTypeLegal(LVT))
      LVT = (MVT::ValueType)((unsigned)LVT - 1);
    assert(MVT::isInteger(LVT));

    if (VT > LVT)
      VT = LVT;
  }

  unsigned NumMemOps = 0;
  while (Size != 0) {
    unsigned VTSize = MVT::getSizeInBits(VT) / 8;
    while (VTSize > Size) {
      // For now, only use non-vector load / store's for the left-over pieces.
      if (MVT::isVector(VT)) {
        VT = MVT::i64;
        while (!TLI.isTypeLegal(VT))
          VT = (MVT::ValueType)((unsigned)VT - 1);         
        VTSize = MVT::getSizeInBits(VT) / 8;
      } else {
        VT = (MVT::ValueType)((unsigned)VT - 1);
        VTSize >>= 1;
      }
    }

    if (++NumMemOps > Limit)
      return false;
    MemOps.push_back(VT);
    Size -= VTSize;
  }

  return true;
}

static SDOperand getMemcpyLoadsAndStores(SelectionDAG &DAG,
                                         SDOperand Chain, SDOperand Dst,
                                         SDOperand Src, uint64_t Size,
                                         unsigned Align, bool AlwaysInline,
                                         const Value *DstSV, uint64_t DstSVOff,
                                         const Value *SrcSV, uint64_t SrcSVOff){
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Expand memcpy to a series of store ops if the size operand falls below
  // a certain threshold.
  std::vector<MVT::ValueType> MemOps;
  uint64_t Limit = -1;
  if (!AlwaysInline)
    Limit = TLI.getMaxStoresPerMemcpy();
  unsigned DstAlign = Align;  // Destination alignment can change.
  if (!MeetsMaxMemopRequirement(MemOps, Dst, Src, Limit, Size, DstAlign,
                                DAG, TLI))
    return SDOperand();

  std::string Str;
  uint64_t SrcOff = 0, DstOff = 0;
  bool CopyFromStr = isMemSrcFromString(Src, Str, SrcOff);

  SmallVector<SDOperand, 8> OutChains;
  unsigned NumMemOps = MemOps.size();
  for (unsigned i = 0; i < NumMemOps; i++) {
    MVT::ValueType VT = MemOps[i];
    unsigned VTSize = MVT::getSizeInBits(VT) / 8;
    SDOperand Value, Store;

    if (CopyFromStr && !MVT::isVector(VT)) {
      // It's unlikely a store of a vector immediate can be done in a single
      // instruction. It would require a load from a constantpool first.
      // FIXME: Handle cases where store of vector immediate is done in a
      // single instruction.
      Value = getMemsetStringVal(VT, DAG, TLI, Str, SrcOff);
      Store = DAG.getStore(Chain, Value,
                           getMemBasePlusOffset(Dst, DstOff, DAG),
                           DstSV, DstSVOff + DstOff);
    } else {
      Value = DAG.getLoad(VT, Chain,
                          getMemBasePlusOffset(Src, SrcOff, DAG),
                          SrcSV, SrcSVOff + SrcOff, false, Align);
      Store = DAG.getStore(Chain, Value,
                           getMemBasePlusOffset(Dst, DstOff, DAG),
                           DstSV, DstSVOff + DstOff, false, DstAlign);
    }
    OutChains.push_back(Store);
    SrcOff += VTSize;
    DstOff += VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, MVT::Other,
                     &OutChains[0], OutChains.size());
}

static SDOperand getMemsetStores(SelectionDAG &DAG,
                                 SDOperand Chain, SDOperand Dst,
                                 SDOperand Src, uint64_t Size,
                                 unsigned Align,
                                 const Value *DstSV, uint64_t DstSVOff) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Expand memset to a series of load/store ops if the size operand
  // falls below a certain threshold.
  std::vector<MVT::ValueType> MemOps;
  if (!MeetsMaxMemopRequirement(MemOps, Dst, Src, TLI.getMaxStoresPerMemset(),
                                Size, Align, DAG, TLI))
    return SDOperand();

  SmallVector<SDOperand, 8> OutChains;
  uint64_t DstOff = 0;

  unsigned NumMemOps = MemOps.size();
  for (unsigned i = 0; i < NumMemOps; i++) {
    MVT::ValueType VT = MemOps[i];
    unsigned VTSize = MVT::getSizeInBits(VT) / 8;
    SDOperand Value = getMemsetValue(Src, VT, DAG);
    SDOperand Store = DAG.getStore(Chain, Value,
                                   getMemBasePlusOffset(Dst, DstOff, DAG),
                                   DstSV, DstSVOff + DstOff);
    OutChains.push_back(Store);
    DstOff += VTSize;
  }

  return DAG.getNode(ISD::TokenFactor, MVT::Other,
                     &OutChains[0], OutChains.size());
}

SDOperand SelectionDAG::getMemcpy(SDOperand Chain, SDOperand Dst,
                                  SDOperand Src, SDOperand Size,
                                  unsigned Align, bool AlwaysInline,
                                  const Value *DstSV, uint64_t DstSVOff,
                                  const Value *SrcSV, uint64_t SrcSVOff) {

  // Check to see if we should lower the memcpy to loads and stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memcpy with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDOperand Result =
      getMemcpyLoadsAndStores(*this, Chain, Dst, Src, ConstantSize->getValue(),
                              Align, false, DstSV, DstSVOff, SrcSV, SrcSVOff);
    if (Result.Val)
      return Result;
  }

  // Then check to see if we should lower the memcpy with target-specific
  // code. If the target chooses to do this, this is the next best.
  SDOperand Result =
    TLI.EmitTargetCodeForMemcpy(*this, Chain, Dst, Src, Size, Align,
                                AlwaysInline,
                                DstSV, DstSVOff, SrcSV, SrcSVOff);
  if (Result.Val)
    return Result;

  // If we really need inline code and the target declined to provide it,
  // use a (potentially long) sequence of loads and stores.
  if (AlwaysInline) {
    assert(ConstantSize && "AlwaysInline requires a constant size!");
    return getMemcpyLoadsAndStores(*this, Chain, Dst, Src,
                                   ConstantSize->getValue(), Align, true,
                                   DstSV, DstSVOff, SrcSV, SrcSVOff);
  }

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = TLI.getTargetData()->getIntPtrType();
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  std::pair<SDOperand,SDOperand> CallResult =
    TLI.LowerCallTo(Chain, Type::VoidTy,
                    false, false, false, CallingConv::C, false,
                    getExternalSymbol("memcpy", TLI.getPointerTy()),
                    Args, *this);
  return CallResult.second;
}

SDOperand SelectionDAG::getMemmove(SDOperand Chain, SDOperand Dst,
                                   SDOperand Src, SDOperand Size,
                                   unsigned Align,
                                   const Value *DstSV, uint64_t DstSVOff,
                                   const Value *SrcSV, uint64_t SrcSVOff) {

  // TODO: Optimize small memmove cases with simple loads and stores,
  // ensuring that all loads precede all stores. This can cause severe
  // register pressure, so targets should be careful with the size limit.

  // Then check to see if we should lower the memmove with target-specific
  // code. If the target chooses to do this, this is the next best.
  SDOperand Result =
    TLI.EmitTargetCodeForMemmove(*this, Chain, Dst, Src, Size, Align,
                                 DstSV, DstSVOff, SrcSV, SrcSVOff);
  if (Result.Val)
    return Result;

  // Emit a library call.
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = TLI.getTargetData()->getIntPtrType();
  Entry.Node = Dst; Args.push_back(Entry);
  Entry.Node = Src; Args.push_back(Entry);
  Entry.Node = Size; Args.push_back(Entry);
  std::pair<SDOperand,SDOperand> CallResult =
    TLI.LowerCallTo(Chain, Type::VoidTy,
                    false, false, false, CallingConv::C, false,
                    getExternalSymbol("memmove", TLI.getPointerTy()),
                    Args, *this);
  return CallResult.second;
}

SDOperand SelectionDAG::getMemset(SDOperand Chain, SDOperand Dst,
                                  SDOperand Src, SDOperand Size,
                                  unsigned Align,
                                  const Value *DstSV, uint64_t DstSVOff) {

  // Check to see if we should lower the memset to stores first.
  // For cases within the target-specified limits, this is the best choice.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (ConstantSize) {
    // Memset with size zero? Just return the original chain.
    if (ConstantSize->isNullValue())
      return Chain;

    SDOperand Result =
      getMemsetStores(*this, Chain, Dst, Src, ConstantSize->getValue(), Align,
                      DstSV, DstSVOff);
    if (Result.Val)
      return Result;
  }

  // Then check to see if we should lower the memset with target-specific
  // code. If the target chooses to do this, this is the next best.
  SDOperand Result =
    TLI.EmitTargetCodeForMemset(*this, Chain, Dst, Src, Size, Align,
                                DstSV, DstSVOff);
  if (Result.Val)
    return Result;

  // Emit a library call.
  const Type *IntPtrTy = TLI.getTargetData()->getIntPtrType();
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Dst; Entry.Ty = IntPtrTy;
  Args.push_back(Entry);
  // Extend or truncate the argument to be an i32 value for the call.
  if (Src.getValueType() > MVT::i32)
    Src = getNode(ISD::TRUNCATE, MVT::i32, Src);
  else
    Src = getNode(ISD::ZERO_EXTEND, MVT::i32, Src);
  Entry.Node = Src; Entry.Ty = Type::Int32Ty; Entry.isSExt = true;
  Args.push_back(Entry);
  Entry.Node = Size; Entry.Ty = IntPtrTy; Entry.isSExt = false;
  Args.push_back(Entry);
  std::pair<SDOperand,SDOperand> CallResult =
    TLI.LowerCallTo(Chain, Type::VoidTy,
                    false, false, false, CallingConv::C, false,
                    getExternalSymbol("memset", TLI.getPointerTy()),
                    Args, *this);
  return CallResult.second;
}

SDOperand SelectionDAG::getAtomic(unsigned Opcode, SDOperand Chain, 
                                  SDOperand Ptr, SDOperand Cmp, 
                                  SDOperand Swp, MVT::ValueType VT) {
  assert(Opcode == ISD::ATOMIC_LCS && "Invalid Atomic Op");
  assert(Cmp.getValueType() == Swp.getValueType() && "Invalid Atomic Op Types");
  SDVTList VTs = getVTList(Cmp.getValueType(), MVT::Other);
  FoldingSetNodeID ID;
  SDOperand Ops[] = {Chain, Ptr, Cmp, Swp};
  AddNodeIDNode(ID, Opcode, VTs, Ops, 4);
  ID.AddInteger((unsigned int)VT);
  void* IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode* N = new AtomicSDNode(Opcode, VTs, Chain, Ptr, Cmp, Swp, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getAtomic(unsigned Opcode, SDOperand Chain, 
                                  SDOperand Ptr, SDOperand Val, 
                                  MVT::ValueType VT) {
  assert((   Opcode == ISD::ATOMIC_LAS || Opcode == ISD::ATOMIC_LSS
          || Opcode == ISD::ATOMIC_SWAP || Opcode == ISD::ATOMIC_LOAD_AND
          || Opcode == ISD::ATOMIC_LOAD_OR || Opcode == ISD::ATOMIC_LOAD_XOR
          || Opcode == ISD::ATOMIC_LOAD_MIN || Opcode == ISD::ATOMIC_LOAD_MAX
          || Opcode == ISD::ATOMIC_LOAD_UMIN || Opcode == ISD::ATOMIC_LOAD_UMAX) 
         && "Invalid Atomic Op");
  SDVTList VTs = getVTList(Val.getValueType(), MVT::Other);
  FoldingSetNodeID ID;
  SDOperand Ops[] = {Chain, Ptr, Val};
  AddNodeIDNode(ID, Opcode, VTs, Ops, 3);
  ID.AddInteger((unsigned int)VT);
  void* IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode* N = new AtomicSDNode(Opcode, VTs, Chain, Ptr, Val, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand
SelectionDAG::getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                      MVT::ValueType VT, SDOperand Chain,
                      SDOperand Ptr, SDOperand Offset,
                      const Value *SV, int SVOffset, MVT::ValueType EVT,
                      bool isVolatile, unsigned Alignment) {
  if (Alignment == 0) { // Ensure that codegen never sees alignment 0
    const Type *Ty = 0;
    if (VT != MVT::iPTR) {
      Ty = MVT::getTypeForValueType(VT);
    } else if (SV) {
      const PointerType *PT = dyn_cast<PointerType>(SV->getType());
      assert(PT && "Value for load must be a pointer");
      Ty = PT->getElementType();
    }
    assert(Ty && "Could not get type information for load");
    Alignment = TLI.getTargetData()->getABITypeAlignment(Ty);
  }

  if (VT == EVT) {
    ExtType = ISD::NON_EXTLOAD;
  } else if (ExtType == ISD::NON_EXTLOAD) {
    assert(VT == EVT && "Non-extending load from different memory type!");
  } else {
    // Extending load.
    if (MVT::isVector(VT))
      assert(EVT == MVT::getVectorElementType(VT) && "Invalid vector extload!");
    else
      assert(MVT::getSizeInBits(EVT) < MVT::getSizeInBits(VT) &&
             "Should only be an extending load, not truncating!");
    assert((ExtType == ISD::EXTLOAD || MVT::isInteger(VT)) &&
           "Cannot sign/zero extend a FP/Vector load!");
    assert(MVT::isInteger(VT) == MVT::isInteger(EVT) &&
           "Cannot convert from FP to Int or Int -> FP!");
  }

  bool Indexed = AM != ISD::UNINDEXED;
  assert(Indexed || Offset.getOpcode() == ISD::UNDEF &&
         "Unindexed load with an offset!");

  SDVTList VTs = Indexed ?
    getVTList(VT, Ptr.getValueType(), MVT::Other) : getVTList(VT, MVT::Other);
  SDOperand Ops[] = { Chain, Ptr, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::LOAD, VTs, Ops, 3);
  ID.AddInteger(AM);
  ID.AddInteger(ExtType);
  ID.AddInteger((unsigned int)EVT);
  ID.AddInteger(Alignment);
  ID.AddInteger(isVolatile);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new LoadSDNode(Ops, VTs, AM, ExtType, EVT, SV, SVOffset,
                             Alignment, isVolatile);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getLoad(MVT::ValueType VT,
                                SDOperand Chain, SDOperand Ptr,
                                const Value *SV, int SVOffset,
                                bool isVolatile, unsigned Alignment) {
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD, VT, Chain, Ptr, Undef,
                 SV, SVOffset, VT, isVolatile, Alignment);
}

SDOperand SelectionDAG::getExtLoad(ISD::LoadExtType ExtType, MVT::ValueType VT,
                                   SDOperand Chain, SDOperand Ptr,
                                   const Value *SV,
                                   int SVOffset, MVT::ValueType EVT,
                                   bool isVolatile, unsigned Alignment) {
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  return getLoad(ISD::UNINDEXED, ExtType, VT, Chain, Ptr, Undef,
                 SV, SVOffset, EVT, isVolatile, Alignment);
}

SDOperand
SelectionDAG::getIndexedLoad(SDOperand OrigLoad, SDOperand Base,
                             SDOperand Offset, ISD::MemIndexedMode AM) {
  LoadSDNode *LD = cast<LoadSDNode>(OrigLoad);
  assert(LD->getOffset().getOpcode() == ISD::UNDEF &&
         "Load is already a indexed load!");
  return getLoad(AM, LD->getExtensionType(), OrigLoad.getValueType(),
                 LD->getChain(), Base, Offset, LD->getSrcValue(),
                 LD->getSrcValueOffset(), LD->getMemoryVT(),
                 LD->isVolatile(), LD->getAlignment());
}

SDOperand SelectionDAG::getStore(SDOperand Chain, SDOperand Val,
                                 SDOperand Ptr, const Value *SV, int SVOffset,
                                 bool isVolatile, unsigned Alignment) {
  MVT::ValueType VT = Val.getValueType();

  if (Alignment == 0) { // Ensure that codegen never sees alignment 0
    const Type *Ty = 0;
    if (VT != MVT::iPTR) {
      Ty = MVT::getTypeForValueType(VT);
    } else if (SV) {
      const PointerType *PT = dyn_cast<PointerType>(SV->getType());
      assert(PT && "Value for store must be a pointer");
      Ty = PT->getElementType();
    }
    assert(Ty && "Could not get type information for store");
    Alignment = TLI.getTargetData()->getABITypeAlignment(Ty);
  }
  SDVTList VTs = getVTList(MVT::Other);
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  SDOperand Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(ISD::UNINDEXED);
  ID.AddInteger(false);
  ID.AddInteger((unsigned int)VT);
  ID.AddInteger(Alignment);
  ID.AddInteger(isVolatile);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new StoreSDNode(Ops, VTs, ISD::UNINDEXED, false,
                              VT, SV, SVOffset, Alignment, isVolatile);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTruncStore(SDOperand Chain, SDOperand Val,
                                      SDOperand Ptr, const Value *SV,
                                      int SVOffset, MVT::ValueType SVT,
                                      bool isVolatile, unsigned Alignment) {
  MVT::ValueType VT = Val.getValueType();

  if (VT == SVT)
    return getStore(Chain, Val, Ptr, SV, SVOffset, isVolatile, Alignment);

  assert(MVT::getSizeInBits(VT) > MVT::getSizeInBits(SVT) &&
         "Not a truncation?");
  assert(MVT::isInteger(VT) == MVT::isInteger(SVT) &&
         "Can't do FP-INT conversion!");

  if (Alignment == 0) { // Ensure that codegen never sees alignment 0
    const Type *Ty = 0;
    if (VT != MVT::iPTR) {
      Ty = MVT::getTypeForValueType(VT);
    } else if (SV) {
      const PointerType *PT = dyn_cast<PointerType>(SV->getType());
      assert(PT && "Value for store must be a pointer");
      Ty = PT->getElementType();
    }
    assert(Ty && "Could not get type information for store");
    Alignment = TLI.getTargetData()->getABITypeAlignment(Ty);
  }
  SDVTList VTs = getVTList(MVT::Other);
  SDOperand Undef = getNode(ISD::UNDEF, Ptr.getValueType());
  SDOperand Ops[] = { Chain, Val, Ptr, Undef };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(ISD::UNINDEXED);
  ID.AddInteger(1);
  ID.AddInteger((unsigned int)SVT);
  ID.AddInteger(Alignment);
  ID.AddInteger(isVolatile);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new StoreSDNode(Ops, VTs, ISD::UNINDEXED, true,
                              SVT, SV, SVOffset, Alignment, isVolatile);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand
SelectionDAG::getIndexedStore(SDOperand OrigStore, SDOperand Base,
                              SDOperand Offset, ISD::MemIndexedMode AM) {
  StoreSDNode *ST = cast<StoreSDNode>(OrigStore);
  assert(ST->getOffset().getOpcode() == ISD::UNDEF &&
         "Store is already a indexed store!");
  SDVTList VTs = getVTList(Base.getValueType(), MVT::Other);
  SDOperand Ops[] = { ST->getChain(), ST->getValue(), Base, Offset };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::STORE, VTs, Ops, 4);
  ID.AddInteger(AM);
  ID.AddInteger(ST->isTruncatingStore());
  ID.AddInteger((unsigned int)(ST->getMemoryVT()));
  ID.AddInteger(ST->getAlignment());
  ID.AddInteger(ST->isVolatile());
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new StoreSDNode(Ops, VTs, AM,
                              ST->isTruncatingStore(), ST->getMemoryVT(),
                              ST->getSrcValue(), ST->getSrcValueOffset(),
                              ST->getAlignment(), ST->isVolatile());
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getVAArg(MVT::ValueType VT,
                                 SDOperand Chain, SDOperand Ptr,
                                 SDOperand SV) {
  SDOperand Ops[] = { Chain, Ptr, SV };
  return getNode(ISD::VAARG, getVTList(VT, MVT::Other), Ops, 3);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperandPtr Ops, unsigned NumOps) {
  switch (NumOps) {
  case 0: return getNode(Opcode, VT);
  case 1: return getNode(Opcode, VT, Ops[0]);
  case 2: return getNode(Opcode, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }
  
  switch (Opcode) {
  default: break;
  case ISD::SELECT_CC: {
    assert(NumOps == 5 && "SELECT_CC takes 5 operands!");
    assert(Ops[0].getValueType() == Ops[1].getValueType() &&
           "LHS and RHS of condition must have same type!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "True and False arms of SelectCC must have same type!");
    assert(Ops[2].getValueType() == VT &&
           "select_cc node must be of same type as true and false value!");
    break;
  }
  case ISD::BR_CC: {
    assert(NumOps == 5 && "BR_CC takes 5 operands!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "LHS/RHS of comparison should match types!");
    break;
  }
  }

  // Memoize nodes.
  SDNode *N;
  SDVTList VTs = getVTList(VT);
  if (VT != MVT::Flag) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTs, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new SDNode(Opcode, VTs, Ops, NumOps);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new SDNode(Opcode, VTs, Ops, NumOps);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode,
                                std::vector<MVT::ValueType> &ResultTys,
                                SDOperandPtr Ops, unsigned NumOps) {
  return getNode(Opcode, getNodeValueTypes(ResultTys), ResultTys.size(),
                 Ops, NumOps);
}

SDOperand SelectionDAG::getNode(unsigned Opcode,
                                const MVT::ValueType *VTs, unsigned NumVTs,
                                SDOperandPtr Ops, unsigned NumOps) {
  if (NumVTs == 1)
    return getNode(Opcode, VTs[0], Ops, NumOps);
  return getNode(Opcode, makeVTList(VTs, NumVTs), Ops, NumOps);
}  
  
SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList,
                                SDOperandPtr Ops, unsigned NumOps) {
  if (VTList.NumVTs == 1)
    return getNode(Opcode, VTList.VTs[0], Ops, NumOps);

  switch (Opcode) {
  // FIXME: figure out how to safely handle things like
  // int foo(int x) { return 1 << (x & 255); }
  // int bar() { return foo(256); }
#if 0
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:
  case ISD::SHL_PARTS:
    if (N3.getOpcode() == ISD::SIGN_EXTEND_INREG &&
        cast<VTSDNode>(N3.getOperand(1))->getVT() != MVT::i1)
      return getNode(Opcode, VT, N1, N2, N3.getOperand(0));
    else if (N3.getOpcode() == ISD::AND)
      if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(N3.getOperand(1))) {
        // If the and is only masking out bits that cannot effect the shift,
        // eliminate the and.
        unsigned NumBits = MVT::getSizeInBits(VT)*2;
        if ((AndRHS->getValue() & (NumBits-1)) == NumBits-1)
          return getNode(Opcode, VT, N1, N2, N3.getOperand(0));
      }
    break;
#endif
  }

  // Memoize the node unless it returns a flag.
  SDNode *N;
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Flag) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    if (NumOps == 1)
      N = new UnarySDNode(Opcode, VTList, Ops[0]);
    else if (NumOps == 2)
      N = new BinarySDNode(Opcode, VTList, Ops[0], Ops[1]);
    else if (NumOps == 3)
      N = new TernarySDNode(Opcode, VTList, Ops[0], Ops[1], Ops[2]);
    else
      N = new SDNode(Opcode, VTList, Ops, NumOps);
    CSEMap.InsertNode(N, IP);
  } else {
    if (NumOps == 1)
      N = new UnarySDNode(Opcode, VTList, Ops[0]);
    else if (NumOps == 2)
      N = new BinarySDNode(Opcode, VTList, Ops[0], Ops[1]);
    else if (NumOps == 3)
      N = new TernarySDNode(Opcode, VTList, Ops[0], Ops[1], Ops[2]);
    else
      N = new SDNode(Opcode, VTList, Ops, NumOps);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList) {
  return getNode(Opcode, VTList, (SDOperand*)0, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList,
                                SDOperand N1) {
  SDOperand Ops[] = { N1 };
  return getNode(Opcode, VTList, Ops, 1);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList,
                                SDOperand N1, SDOperand N2) {
  SDOperand Ops[] = { N1, N2 };
  return getNode(Opcode, VTList, Ops, 2);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList,
                                SDOperand N1, SDOperand N2, SDOperand N3) {
  SDOperand Ops[] = { N1, N2, N3 };
  return getNode(Opcode, VTList, Ops, 3);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4) {
  SDOperand Ops[] = { N1, N2, N3, N4 };
  return getNode(Opcode, VTList, Ops, 4);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, SDVTList VTList,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4, SDOperand N5) {
  SDOperand Ops[] = { N1, N2, N3, N4, N5 };
  return getNode(Opcode, VTList, Ops, 5);
}

SDVTList SelectionDAG::getVTList(MVT::ValueType VT) {
  return makeVTList(SDNode::getValueTypeList(VT), 1);
}

SDVTList SelectionDAG::getVTList(MVT::ValueType VT1, MVT::ValueType VT2) {
  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() == 2 && (*I)[0] == VT1 && (*I)[1] == VT2)
      return makeVTList(&(*I)[0], 2);
  }
  std::vector<MVT::ValueType> V;
  V.push_back(VT1);
  V.push_back(VT2);
  VTList.push_front(V);
  return makeVTList(&(*VTList.begin())[0], 2);
}
SDVTList SelectionDAG::getVTList(MVT::ValueType VT1, MVT::ValueType VT2,
                                 MVT::ValueType VT3) {
  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() == 3 && (*I)[0] == VT1 && (*I)[1] == VT2 &&
        (*I)[2] == VT3)
      return makeVTList(&(*I)[0], 3);
  }
  std::vector<MVT::ValueType> V;
  V.push_back(VT1);
  V.push_back(VT2);
  V.push_back(VT3);
  VTList.push_front(V);
  return makeVTList(&(*VTList.begin())[0], 3);
}

SDVTList SelectionDAG::getVTList(const MVT::ValueType *VTs, unsigned NumVTs) {
  switch (NumVTs) {
    case 0: assert(0 && "Cannot have nodes without results!");
    case 1: return getVTList(VTs[0]);
    case 2: return getVTList(VTs[0], VTs[1]);
    case 3: return getVTList(VTs[0], VTs[1], VTs[2]);
    default: break;
  }

  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() != NumVTs || VTs[0] != (*I)[0] || VTs[1] != (*I)[1]) continue;
   
    bool NoMatch = false;
    for (unsigned i = 2; i != NumVTs; ++i)
      if (VTs[i] != (*I)[i]) {
        NoMatch = true;
        break;
      }
    if (!NoMatch)
      return makeVTList(&*I->begin(), NumVTs);
  }
  
  VTList.push_front(std::vector<MVT::ValueType>(VTs, VTs+NumVTs));
  return makeVTList(&*VTList.begin()->begin(), NumVTs);
}


/// UpdateNodeOperands - *Mutate* the specified node in-place to have the
/// specified operands.  If the resultant node already exists in the DAG,
/// this does not modify the specified node, instead it returns the node that
/// already exists.  If the resultant node does not exist in the DAG, the
/// input node is returned.  As a degenerate case, if you specify the same
/// input operands as the node already has, the input node is returned.
SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand InN, SDOperand Op) {
  SDNode *N = InN.Val;
  assert(N->getNumOperands() == 1 && "Update with wrong number of operands");
  
  // Check to see if there is no change.
  if (Op == N->getOperand(0)) return InN;
  
  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op, InsertPos))
    return SDOperand(Existing, InN.ResNo);
  
  // Nope it doesn't.  Remove the node from it's current place in the maps.
  if (InsertPos)
    RemoveNodeFromCSEMaps(N);
  
  // Now we update the operands.
  N->OperandList[0].getVal()->removeUser(0, N);
  N->OperandList[0] = Op;
  N->OperandList[0].setUser(N);
  Op.Val->addUser(0, N);
  
  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand InN, SDOperand Op1, SDOperand Op2) {
  SDNode *N = InN.Val;
  assert(N->getNumOperands() == 2 && "Update with wrong number of operands");
  
  // Check to see if there is no change.
  if (Op1 == N->getOperand(0) && Op2 == N->getOperand(1))
    return InN;   // No operands changed, just return the input node.
  
  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Op1, Op2, InsertPos))
    return SDOperand(Existing, InN.ResNo);
  
  // Nope it doesn't.  Remove the node from it's current place in the maps.
  if (InsertPos)
    RemoveNodeFromCSEMaps(N);
  
  // Now we update the operands.
  if (N->OperandList[0] != Op1) {
    N->OperandList[0].getVal()->removeUser(0, N);
    N->OperandList[0] = Op1;
    N->OperandList[0].setUser(N);
    Op1.Val->addUser(0, N);
  }
  if (N->OperandList[1] != Op2) {
    N->OperandList[1].getVal()->removeUser(1, N);
    N->OperandList[1] = Op2;
    N->OperandList[1].setUser(N);
    Op2.Val->addUser(1, N);
  }
  
  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2, SDOperand Op3) {
  SDOperand Ops[] = { Op1, Op2, Op3 };
  return UpdateNodeOperands(N, Ops, 3);
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2, 
                   SDOperand Op3, SDOperand Op4) {
  SDOperand Ops[] = { Op1, Op2, Op3, Op4 };
  return UpdateNodeOperands(N, Ops, 4);
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand N, SDOperand Op1, SDOperand Op2,
                   SDOperand Op3, SDOperand Op4, SDOperand Op5) {
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5 };
  return UpdateNodeOperands(N, Ops, 5);
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand InN, SDOperandPtr Ops, unsigned NumOps) {
  SDNode *N = InN.Val;
  assert(N->getNumOperands() == NumOps &&
         "Update with wrong number of operands");
  
  // Check to see if there is no change.
  bool AnyChange = false;
  for (unsigned i = 0; i != NumOps; ++i) {
    if (Ops[i] != N->getOperand(i)) {
      AnyChange = true;
      break;
    }
  }
  
  // No operands changed, just return the input node.
  if (!AnyChange) return InN;
  
  // See if the modified node already exists.
  void *InsertPos = 0;
  if (SDNode *Existing = FindModifiedNodeSlot(N, Ops, NumOps, InsertPos))
    return SDOperand(Existing, InN.ResNo);
  
  // Nope it doesn't.  Remove the node from its current place in the maps.
  if (InsertPos)
    RemoveNodeFromCSEMaps(N);
  
  // Now we update the operands.
  for (unsigned i = 0; i != NumOps; ++i) {
    if (N->OperandList[i] != Ops[i]) {
      N->OperandList[i].getVal()->removeUser(i, N);
      N->OperandList[i] = Ops[i];
      N->OperandList[i].setUser(N);
      Ops[i].Val->addUser(i, N);
    }
  }

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}

/// MorphNodeTo - This frees the operands of the current node, resets the
/// opcode, types, and operands to the specified value.  This should only be
/// used by the SelectionDAG class.
void SDNode::MorphNodeTo(unsigned Opc, SDVTList L,
                         SDOperandPtr Ops, unsigned NumOps) {
  NodeType = Opc;
  ValueList = L.VTs;
  NumValues = L.NumVTs;
  
  // Clear the operands list, updating used nodes to remove this from their
  // use list.
  for (op_iterator I = op_begin(), E = op_end(); I != E; ++I)
    I->getVal()->removeUser(std::distance(op_begin(), I), this);
  
  // If NumOps is larger than the # of operands we currently have, reallocate
  // the operand list.
  if (NumOps > NumOperands) {
    if (OperandsNeedDelete) {
      delete [] OperandList;
    }
    OperandList = new SDUse[NumOps];
    OperandsNeedDelete = true;
  }
  
  // Assign the new operands.
  NumOperands = NumOps;
  
  for (unsigned i = 0, e = NumOps; i != e; ++i) {
    OperandList[i] = Ops[i];
    OperandList[i].setUser(this);
    SDNode *N = OperandList[i].getVal();
    N->addUser(i, this);
    ++N->UsesSize;
  }
}

/// SelectNodeTo - These are used for target selectors to *mutate* the
/// specified node to have the specified return type, Target opcode, and
/// operands.  Note that target opcodes are stored as
/// ISD::BUILTIN_OP_END+TargetOpcode in the node opcode field.
///
/// Note that SelectNodeTo returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.
SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT) {
  SDVTList VTs = getVTList(VT);
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, (SDOperand*)0, 0);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
   
  RemoveNodeFromCSEMaps(N);
  
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, SDOperandPtr(), 0);

  CSEMap.InsertNode(N, IP);
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, SDOperand Op1) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  SDOperand Ops[] = { Op1 };
  
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 1);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 1);
  CSEMap.InsertNode(N, IP);
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, SDOperand Op1,
                                   SDOperand Op2) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  SDOperand Ops[] = { Op1, Op2 };
  
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, SDOperand Op1,
                                   SDOperand Op2, SDOperand Op3) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);

  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT, SDOperandPtr Ops,
                                   unsigned NumOps) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT);
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, NumOps);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, NumOps);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc, 
                                   MVT::ValueType VT1, MVT::ValueType VT2,
                                   SDOperand Op1, SDOperand Op2) {
  SDVTList VTs = getVTList(VT1, VT2);
  FoldingSetNodeID ID;
  SDOperand Ops[] = { Op1, Op2 };
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;

  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 2);
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}

SDNode *SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                   MVT::ValueType VT1, MVT::ValueType VT2,
                                   SDOperand Op1, SDOperand Op2, 
                                   SDOperand Op3) {
  // If an identical node already exists, use it.
  SDVTList VTs = getVTList(VT1, VT2);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  FoldingSetNodeID ID;
  AddNodeIDNode(ID, ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return ON;

  RemoveNodeFromCSEMaps(N);

  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc, VTs, Ops, 3);
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return N;
}


/// getTargetNode - These are used for target selectors to create a new node
/// with specified return type(s), target opcode, and operands.
///
/// Note that getTargetNode returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2,
                                    SDOperand Op3) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperandPtr Ops, unsigned NumOps) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SDOperand Op;
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, &Op, 0).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, &Op1, 1).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SDOperand Ops[] = { Op1, Op2 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, Ops, 2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2, SDOperand Op3) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, Ops, 3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                                    MVT::ValueType VT2,
                                    SDOperandPtr Ops, unsigned NumOps) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 2, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2, VT3);
  SDOperand Ops[] = { Op1, Op2 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 3, Ops, 2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2,
                                    SDOperand Op3) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2, VT3);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 3, Ops, 3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperandPtr Ops, unsigned NumOps) {
  const MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2, VT3);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 3, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    MVT::ValueType VT4,
                                    SDOperandPtr Ops, unsigned NumOps) {
  std::vector<MVT::ValueType> VTList;
  VTList.push_back(VT1);
  VTList.push_back(VT2);
  VTList.push_back(VT3);
  VTList.push_back(VT4);
  const MVT::ValueType *VTs = getNodeValueTypes(VTList);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, 4, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode,
                                    std::vector<MVT::ValueType> &ResultTys,
                                    SDOperandPtr Ops, unsigned NumOps) {
  const MVT::ValueType *VTs = getNodeValueTypes(ResultTys);
  return getNode(ISD::BUILTIN_OP_END+Opcode, VTs, ResultTys.size(),
                 Ops, NumOps).Val;
}

/// getNodeIfExists - Get the specified node if it's already available, or
/// else return NULL.
SDNode *SelectionDAG::getNodeIfExists(unsigned Opcode, SDVTList VTList,
                                      SDOperandPtr Ops, unsigned NumOps) {
  if (VTList.VTs[VTList.NumVTs-1] != MVT::Flag) {
    FoldingSetNodeID ID;
    AddNodeIDNode(ID, Opcode, VTList, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return E;
  }
  return NULL;
}


/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From has a single result value.
///
void SelectionDAG::ReplaceAllUsesWith(SDOperand FromN, SDOperand To,
                                      DAGUpdateListener *UpdateListener) {
  SDNode *From = FromN.Val;
  assert(From->getNumValues() == 1 && FromN.ResNo == 0 && 
         "Cannot replace with this method!");
  assert(From != To.Val && "Cannot replace uses of with self");

  while (!From->use_empty()) {
    SDNode::use_iterator UI = From->use_begin();
    SDNode *U = UI->getUser();

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    int operandNum = 0;
    for (SDNode::op_iterator I = U->op_begin(), E = U->op_end();
         I != E; ++I, ++operandNum)
      if (I->getVal() == From) {
        From->removeUser(operandNum, U);
        *I = To;
        I->setUser(U);
        To.Val->addUser(operandNum, U);
      }    

    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, UpdateListener);
      // U is now dead.  Inform the listener if it exists and delete it.
      if (UpdateListener) 
        UpdateListener->NodeDeleted(U);
      DeleteNodeNotInCSEMaps(U);
    } else {
      // If the node doesn't already exist, we updated it.  Inform a listener if
      // it exists.
      if (UpdateListener) 
        UpdateListener->NodeUpdated(U);
    }
  }
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From/To have matching types and numbers of result
/// values.
///
void SelectionDAG::ReplaceAllUsesWith(SDNode *From, SDNode *To,
                                      DAGUpdateListener *UpdateListener) {
  assert(From != To && "Cannot replace uses of with self");
  assert(From->getNumValues() == To->getNumValues() &&
         "Cannot use this version of ReplaceAllUsesWith!");
  if (From->getNumValues() == 1)   // If possible, use the faster version.
    return ReplaceAllUsesWith(SDOperand(From, 0), SDOperand(To, 0),
                              UpdateListener);
  
  while (!From->use_empty()) {
    SDNode::use_iterator UI = From->use_begin();
    SDNode *U = UI->getUser();

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    int operandNum = 0;
    for (SDNode::op_iterator I = U->op_begin(), E = U->op_end();
         I != E; ++I, ++operandNum)
      if (I->getVal() == From) {
        From->removeUser(operandNum, U);
        I->getVal() = To;
        To->addUser(operandNum, U);
      }

    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, UpdateListener);
      // U is now dead.  Inform the listener if it exists and delete it.
      if (UpdateListener) 
        UpdateListener->NodeDeleted(U);
      DeleteNodeNotInCSEMaps(U);
    } else {
      // If the node doesn't already exist, we updated it.  Inform a listener if
      // it exists.
      if (UpdateListener) 
        UpdateListener->NodeUpdated(U);
    }
  }
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version can replace From with any result values.  To must match the
/// number and types of values returned by From.
void SelectionDAG::ReplaceAllUsesWith(SDNode *From,
                                      SDOperandPtr To,
                                      DAGUpdateListener *UpdateListener) {
  if (From->getNumValues() == 1)  // Handle the simple case efficiently.
    return ReplaceAllUsesWith(SDOperand(From, 0), To[0], UpdateListener);

  while (!From->use_empty()) {
    SDNode::use_iterator UI = From->use_begin();
    SDNode *U = UI->getUser();

    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    int operandNum = 0;
    for (SDNode::op_iterator I = U->op_begin(), E = U->op_end();
         I != E; ++I, ++operandNum)
      if (I->getVal() == From) {
        const SDOperand &ToOp = To[I->getSDOperand().ResNo];
        From->removeUser(operandNum, U);
        *I = ToOp;
        I->setUser(U);
        ToOp.Val->addUser(operandNum, U);
      }

    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, UpdateListener);
      // U is now dead.  Inform the listener if it exists and delete it.
      if (UpdateListener) 
        UpdateListener->NodeDeleted(U);
      DeleteNodeNotInCSEMaps(U);
    } else {
      // If the node doesn't already exist, we updated it.  Inform a listener if
      // it exists.
      if (UpdateListener) 
        UpdateListener->NodeUpdated(U);
    }
  }
}

namespace {
  /// ChainedSetUpdaterListener - This class is a DAGUpdateListener that removes
  /// any deleted nodes from the set passed into its constructor and recursively
  /// notifies another update listener if specified.
  class ChainedSetUpdaterListener : 
  public SelectionDAG::DAGUpdateListener {
    SmallSetVector<SDNode*, 16> &Set;
    SelectionDAG::DAGUpdateListener *Chain;
  public:
    ChainedSetUpdaterListener(SmallSetVector<SDNode*, 16> &set,
                              SelectionDAG::DAGUpdateListener *chain)
      : Set(set), Chain(chain) {}
 
    virtual void NodeDeleted(SDNode *N) {
      Set.remove(N);
      if (Chain) Chain->NodeDeleted(N);
    }
    virtual void NodeUpdated(SDNode *N) {
      if (Chain) Chain->NodeUpdated(N);
    }
  };
}

/// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.Val alone.  The Deleted vector is
/// handled the same way as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValueWith(SDOperand From, SDOperand To,
                                             DAGUpdateListener *UpdateListener){
  assert(From != To && "Cannot replace a value with itself");
  
  // Handle the simple, trivial, case efficiently.
  if (From.Val->getNumValues() == 1) {
    ReplaceAllUsesWith(From, To, UpdateListener);
    return;
  }

  if (From.use_empty()) return;

  // Get all of the users of From.Val.  We want these in a nice,
  // deterministically ordered and uniqued set, so we use a SmallSetVector.
  SmallSetVector<SDNode*, 16> Users;
  for (SDNode::use_iterator UI = From.Val->use_begin(), 
      E = From.Val->use_end(); UI != E; ++UI) {
    SDNode *User = UI->getUser();
    if (!Users.count(User))
      Users.insert(User);
  }

  // When one of the recursive merges deletes nodes from the graph, we need to
  // make sure that UpdateListener is notified *and* that the node is removed
  // from Users if present.  CSUL does this.
  ChainedSetUpdaterListener CSUL(Users, UpdateListener);
  
  while (!Users.empty()) {
    // We know that this user uses some value of From.  If it is the right
    // value, update it.
    SDNode *User = Users.back();
    Users.pop_back();
    
    // Scan for an operand that matches From.
    SDNode::op_iterator Op = User->op_begin(), E = User->op_end();
    for (; Op != E; ++Op)
      if (*Op == From) break;
    
    // If there are no matches, the user must use some other result of From.
    if (Op == E) continue;
      
    // Okay, we know this user needs to be updated.  Remove its old self
    // from the CSE maps.
    RemoveNodeFromCSEMaps(User);
    
    // Update all operands that match "From" in case there are multiple uses.
    for (; Op != E; ++Op) {
      if (*Op == From) {
        From.Val->removeUser(Op-User->op_begin(), User);
        *Op = To;
        Op->setUser(User);
        To.Val->addUser(Op-User->op_begin(), User);
      }
    }
               
    // Now that we have modified User, add it back to the CSE maps.  If it
    // already exists there, recursively merge the results together.
    SDNode *Existing = AddNonLeafNodeToCSEMaps(User);
    if (!Existing) {
      if (UpdateListener) UpdateListener->NodeUpdated(User);
      continue;  // Continue on to next user.
    }
    
    // If there was already an existing matching node, use ReplaceAllUsesWith
    // to replace the dead one with the existing one.  This can cause
    // recursive merging of other unrelated nodes down the line.  The merging
    // can cause deletion of nodes that used the old value.  To handle this, we
    // use CSUL to remove them from the Users set.
    ReplaceAllUsesWith(User, Existing, &CSUL);
    
    // User is now dead.  Notify a listener if present.
    if (UpdateListener) UpdateListener->NodeDeleted(User);
    DeleteNodeNotInCSEMaps(User);
  }
}

/// AssignNodeIds - Assign a unique node id for each node in the DAG based on
/// their allnodes order. It returns the maximum id.
unsigned SelectionDAG::AssignNodeIds() {
  unsigned Id = 0;
  for (allnodes_iterator I = allnodes_begin(), E = allnodes_end(); I != E; ++I){
    SDNode *N = I;
    N->setNodeId(Id++);
  }
  return Id;
}

/// AssignTopologicalOrder - Assign a unique node id for each node in the DAG
/// based on their topological order. It returns the maximum id and a vector
/// of the SDNodes* in assigned order by reference.
unsigned SelectionDAG::AssignTopologicalOrder(std::vector<SDNode*> &TopOrder) {
  unsigned DAGSize = AllNodes.size();
  std::vector<unsigned> InDegree(DAGSize);
  std::vector<SDNode*> Sources;

  // Use a two pass approach to avoid using a std::map which is slow.
  unsigned Id = 0;
  for (allnodes_iterator I = allnodes_begin(),E = allnodes_end(); I != E; ++I){
    SDNode *N = I;
    N->setNodeId(Id++);
    unsigned Degree = N->use_size();
    InDegree[N->getNodeId()] = Degree;
    if (Degree == 0)
      Sources.push_back(N);
  }

  TopOrder.clear();
  while (!Sources.empty()) {
    SDNode *N = Sources.back();
    Sources.pop_back();
    TopOrder.push_back(N);
    for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
      SDNode *P = I->getVal();
      unsigned Degree = --InDegree[P->getNodeId()];
      if (Degree == 0)
        Sources.push_back(P);
    }
  }

  // Second pass, assign the actual topological order as node ids.
  Id = 0;
  for (std::vector<SDNode*>::iterator TI = TopOrder.begin(),TE = TopOrder.end();
       TI != TE; ++TI)
    (*TI)->setNodeId(Id++);

  return Id;
}



//===----------------------------------------------------------------------===//
//                              SDNode Class
//===----------------------------------------------------------------------===//

// Out-of-line virtual method to give class a home.
void SDNode::ANCHOR() {}
void UnarySDNode::ANCHOR() {}
void BinarySDNode::ANCHOR() {}
void TernarySDNode::ANCHOR() {}
void HandleSDNode::ANCHOR() {}
void StringSDNode::ANCHOR() {}
void ConstantSDNode::ANCHOR() {}
void ConstantFPSDNode::ANCHOR() {}
void GlobalAddressSDNode::ANCHOR() {}
void FrameIndexSDNode::ANCHOR() {}
void JumpTableSDNode::ANCHOR() {}
void ConstantPoolSDNode::ANCHOR() {}
void BasicBlockSDNode::ANCHOR() {}
void SrcValueSDNode::ANCHOR() {}
void MemOperandSDNode::ANCHOR() {}
void RegisterSDNode::ANCHOR() {}
void ExternalSymbolSDNode::ANCHOR() {}
void CondCodeSDNode::ANCHOR() {}
void ARG_FLAGSSDNode::ANCHOR() {}
void VTSDNode::ANCHOR() {}
void LoadSDNode::ANCHOR() {}
void StoreSDNode::ANCHOR() {}
void AtomicSDNode::ANCHOR() {}

HandleSDNode::~HandleSDNode() {
  SDVTList VTs = { 0, 0 };
  MorphNodeTo(ISD::HANDLENODE, VTs, SDOperandPtr(), 0);  // Drops operand uses.
}

GlobalAddressSDNode::GlobalAddressSDNode(bool isTarget, const GlobalValue *GA,
                                         MVT::ValueType VT, int o)
  : SDNode(isa<GlobalVariable>(GA) &&
           cast<GlobalVariable>(GA)->isThreadLocal() ?
           // Thread Local
           (isTarget ? ISD::TargetGlobalTLSAddress : ISD::GlobalTLSAddress) :
           // Non Thread Local
           (isTarget ? ISD::TargetGlobalAddress : ISD::GlobalAddress),
           getSDVTList(VT)), Offset(o) {
  TheGlobal = const_cast<GlobalValue*>(GA);
}

/// getMemOperand - Return a MachineMemOperand object describing the memory
/// reference performed by this load or store.
MachineMemOperand LSBaseSDNode::getMemOperand() const {
  int Size = (MVT::getSizeInBits(getMemoryVT()) + 7) >> 3;
  int Flags =
    getOpcode() == ISD::LOAD ? MachineMemOperand::MOLoad :
                               MachineMemOperand::MOStore;
  if (IsVolatile) Flags |= MachineMemOperand::MOVolatile;

  // Check if the load references a frame index, and does not have
  // an SV attached.
  const FrameIndexSDNode *FI =
    dyn_cast<const FrameIndexSDNode>(getBasePtr().Val);
  if (!getSrcValue() && FI)
    return MachineMemOperand(PseudoSourceValue::getFixedStack(), Flags,
                             FI->getIndex(), Size, Alignment);
  else
    return MachineMemOperand(getSrcValue(), Flags,
                             getSrcValueOffset(), Size, Alignment);
}

/// Profile - Gather unique data for the node.
///
void SDNode::Profile(FoldingSetNodeID &ID) {
  AddNodeIDNode(ID, this);
}

/// getValueTypeList - Return a pointer to the specified value type.
///
const MVT::ValueType *SDNode::getValueTypeList(MVT::ValueType VT) {
  if (MVT::isExtendedVT(VT)) {
    static std::set<MVT::ValueType> EVTs;
    return &(*EVTs.insert(VT).first);
  } else {
    static MVT::ValueType VTs[MVT::LAST_VALUETYPE];
    VTs[VT] = VT;
    return &VTs[VT];
  }
}

/// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
/// indicated value.  This method ignores uses of other values defined by this
/// operation.
bool SDNode::hasNUsesOfValue(unsigned NUses, unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  // If there is only one value, this is easy.
  if (getNumValues() == 1)
    return use_size() == NUses;
  if (use_size() < NUses) return false;

  SDOperand TheValue(const_cast<SDNode *>(this), Value);

  SmallPtrSet<SDNode*, 32> UsersHandled;

  // TODO: Only iterate over uses of a given value of the node
  for (SDNode::use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI) {
    if (*UI == TheValue) {
      if (NUses == 0)
        return false;
      --NUses;
    }
  }

  // Found exactly the right number of uses?
  return NUses == 0;
}


/// hasAnyUseOfValue - Return true if there are any use of the indicated
/// value. This method ignores uses of other values defined by this operation.
bool SDNode::hasAnyUseOfValue(unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  if (use_empty()) return false;

  SDOperand TheValue(const_cast<SDNode *>(this), Value);

  SmallPtrSet<SDNode*, 32> UsersHandled;

  for (SDNode::use_iterator UI = use_begin(), E = use_end(); UI != E; ++UI) {
    SDNode *User = UI->getUser();
    if (User->getNumOperands() == 1 ||
        UsersHandled.insert(User))     // First time we've seen this?
      for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
        if (User->getOperand(i) == TheValue) {
          return true;
        }
  }

  return false;
}


/// isOnlyUseOf - Return true if this node is the only use of N.
///
bool SDNode::isOnlyUseOf(SDNode *N) const {
  bool Seen = false;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDNode *User = I->getUser();
    if (User == this)
      Seen = true;
    else
      return false;
  }

  return Seen;
}

/// isOperand - Return true if this node is an operand of N.
///
bool SDOperand::isOperandOf(SDNode *N) const {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (*this == N->getOperand(i))
      return true;
  return false;
}

bool SDNode::isOperandOf(SDNode *N) const {
  for (unsigned i = 0, e = N->NumOperands; i != e; ++i)
    if (this == N->OperandList[i].getVal())
      return true;
  return false;
}

/// reachesChainWithoutSideEffects - Return true if this operand (which must
/// be a chain) reaches the specified operand without crossing any 
/// side-effecting instructions.  In practice, this looks through token
/// factors and non-volatile loads.  In order to remain efficient, this only
/// looks a couple of nodes in, it does not do an exhaustive search.
bool SDOperand::reachesChainWithoutSideEffects(SDOperand Dest, 
                                               unsigned Depth) const {
  if (*this == Dest) return true;
  
  // Don't search too deeply, we just want to be able to see through
  // TokenFactor's etc.
  if (Depth == 0) return false;
  
  // If this is a token factor, all inputs to the TF happen in parallel.  If any
  // of the operands of the TF reach dest, then we can do the xform.
  if (getOpcode() == ISD::TokenFactor) {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (getOperand(i).reachesChainWithoutSideEffects(Dest, Depth-1))
        return true;
    return false;
  }
  
  // Loads don't have side effects, look through them.
  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(*this)) {
    if (!Ld->isVolatile())
      return Ld->getChain().reachesChainWithoutSideEffects(Dest, Depth-1);
  }
  return false;
}


static void findPredecessor(SDNode *N, const SDNode *P, bool &found,
                            SmallPtrSet<SDNode *, 32> &Visited) {
  if (found || !Visited.insert(N))
    return;

  for (unsigned i = 0, e = N->getNumOperands(); !found && i != e; ++i) {
    SDNode *Op = N->getOperand(i).Val;
    if (Op == P) {
      found = true;
      return;
    }
    findPredecessor(Op, P, found, Visited);
  }
}

/// isPredecessorOf - Return true if this node is a predecessor of N. This node
/// is either an operand of N or it can be reached by recursively traversing
/// up the operands.
/// NOTE: this is an expensive method. Use it carefully.
bool SDNode::isPredecessorOf(SDNode *N) const {
  SmallPtrSet<SDNode *, 32> Visited;
  bool found = false;
  findPredecessor(N, this, found, Visited);
  return found;
}

uint64_t SDNode::getConstantOperandVal(unsigned Num) const {
  assert(Num < NumOperands && "Invalid child # of SDNode!");
  return cast<ConstantSDNode>(OperandList[Num])->getValue();
}

std::string SDNode::getOperationName(const SelectionDAG *G) const {
  switch (getOpcode()) {
  default:
    if (getOpcode() < ISD::BUILTIN_OP_END)
      return "<<Unknown DAG Node>>";
    else {
      if (G) {
        if (const TargetInstrInfo *TII = G->getTarget().getInstrInfo())
          if (getOpcode()-ISD::BUILTIN_OP_END < TII->getNumOpcodes())
            return TII->get(getOpcode()-ISD::BUILTIN_OP_END).getName();

        TargetLowering &TLI = G->getTargetLoweringInfo();
        const char *Name =
          TLI.getTargetNodeName(getOpcode());
        if (Name) return Name;
      }

      return "<<Unknown Target Node>>";
    }
   
  case ISD::PREFETCH:      return "Prefetch";
  case ISD::MEMBARRIER:    return "MemBarrier";
  case ISD::ATOMIC_LCS:    return "AtomicLCS";
  case ISD::ATOMIC_LAS:    return "AtomicLAS";
  case ISD::ATOMIC_LSS:    return "AtomicLSS";
  case ISD::ATOMIC_LOAD_AND:  return "AtomicLoadAnd";
  case ISD::ATOMIC_LOAD_OR:   return "AtomicLoadOr";
  case ISD::ATOMIC_LOAD_XOR:  return "AtomicLoadXor";
  case ISD::ATOMIC_LOAD_MIN:  return "AtomicLoadMin";
  case ISD::ATOMIC_LOAD_MAX:  return "AtomicLoadMax";
  case ISD::ATOMIC_LOAD_UMIN: return "AtomicLoadUMin";
  case ISD::ATOMIC_LOAD_UMAX: return "AtomicLoadUMax";
  case ISD::ATOMIC_SWAP:   return "AtomicSWAP";
  case ISD::PCMARKER:      return "PCMarker";
  case ISD::READCYCLECOUNTER: return "ReadCycleCounter";
  case ISD::SRCVALUE:      return "SrcValue";
  case ISD::MEMOPERAND:    return "MemOperand";
  case ISD::EntryToken:    return "EntryToken";
  case ISD::TokenFactor:   return "TokenFactor";
  case ISD::AssertSext:    return "AssertSext";
  case ISD::AssertZext:    return "AssertZext";

  case ISD::STRING:        return "String";
  case ISD::BasicBlock:    return "BasicBlock";
  case ISD::ARG_FLAGS:     return "ArgFlags";
  case ISD::VALUETYPE:     return "ValueType";
  case ISD::Register:      return "Register";

  case ISD::Constant:      return "Constant";
  case ISD::ConstantFP:    return "ConstantFP";
  case ISD::GlobalAddress: return "GlobalAddress";
  case ISD::GlobalTLSAddress: return "GlobalTLSAddress";
  case ISD::FrameIndex:    return "FrameIndex";
  case ISD::JumpTable:     return "JumpTable";
  case ISD::GLOBAL_OFFSET_TABLE: return "GLOBAL_OFFSET_TABLE";
  case ISD::RETURNADDR: return "RETURNADDR";
  case ISD::FRAMEADDR: return "FRAMEADDR";
  case ISD::FRAME_TO_ARGS_OFFSET: return "FRAME_TO_ARGS_OFFSET";
  case ISD::EXCEPTIONADDR: return "EXCEPTIONADDR";
  case ISD::EHSELECTION: return "EHSELECTION";
  case ISD::EH_RETURN: return "EH_RETURN";
  case ISD::ConstantPool:  return "ConstantPool";
  case ISD::ExternalSymbol: return "ExternalSymbol";
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(getOperand(0))->getValue();
    return Intrinsic::getName((Intrinsic::ID)IID);
  }
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(getOperand(1))->getValue();
    return Intrinsic::getName((Intrinsic::ID)IID);
  }

  case ISD::BUILD_VECTOR:   return "BUILD_VECTOR";
  case ISD::TargetConstant: return "TargetConstant";
  case ISD::TargetConstantFP:return "TargetConstantFP";
  case ISD::TargetGlobalAddress: return "TargetGlobalAddress";
  case ISD::TargetGlobalTLSAddress: return "TargetGlobalTLSAddress";
  case ISD::TargetFrameIndex: return "TargetFrameIndex";
  case ISD::TargetJumpTable:  return "TargetJumpTable";
  case ISD::TargetConstantPool:  return "TargetConstantPool";
  case ISD::TargetExternalSymbol: return "TargetExternalSymbol";

  case ISD::CopyToReg:     return "CopyToReg";
  case ISD::CopyFromReg:   return "CopyFromReg";
  case ISD::UNDEF:         return "undef";
  case ISD::MERGE_VALUES:  return "merge_values";
  case ISD::INLINEASM:     return "inlineasm";
  case ISD::LABEL:         return "label";
  case ISD::DECLARE:       return "declare";
  case ISD::HANDLENODE:    return "handlenode";
  case ISD::FORMAL_ARGUMENTS: return "formal_arguments";
  case ISD::CALL:          return "call";
    
  // Unary operators
  case ISD::FABS:   return "fabs";
  case ISD::FNEG:   return "fneg";
  case ISD::FSQRT:  return "fsqrt";
  case ISD::FSIN:   return "fsin";
  case ISD::FCOS:   return "fcos";
  case ISD::FPOWI:  return "fpowi";
  case ISD::FPOW:   return "fpow";

  // Binary operators
  case ISD::ADD:    return "add";
  case ISD::SUB:    return "sub";
  case ISD::MUL:    return "mul";
  case ISD::MULHU:  return "mulhu";
  case ISD::MULHS:  return "mulhs";
  case ISD::SDIV:   return "sdiv";
  case ISD::UDIV:   return "udiv";
  case ISD::SREM:   return "srem";
  case ISD::UREM:   return "urem";
  case ISD::SMUL_LOHI:  return "smul_lohi";
  case ISD::UMUL_LOHI:  return "umul_lohi";
  case ISD::SDIVREM:    return "sdivrem";
  case ISD::UDIVREM:    return "divrem";
  case ISD::AND:    return "and";
  case ISD::OR:     return "or";
  case ISD::XOR:    return "xor";
  case ISD::SHL:    return "shl";
  case ISD::SRA:    return "sra";
  case ISD::SRL:    return "srl";
  case ISD::ROTL:   return "rotl";
  case ISD::ROTR:   return "rotr";
  case ISD::FADD:   return "fadd";
  case ISD::FSUB:   return "fsub";
  case ISD::FMUL:   return "fmul";
  case ISD::FDIV:   return "fdiv";
  case ISD::FREM:   return "frem";
  case ISD::FCOPYSIGN: return "fcopysign";
  case ISD::FGETSIGN:  return "fgetsign";

  case ISD::SETCC:       return "setcc";
  case ISD::VSETCC:      return "vsetcc";
  case ISD::SELECT:      return "select";
  case ISD::SELECT_CC:   return "select_cc";
  case ISD::INSERT_VECTOR_ELT:   return "insert_vector_elt";
  case ISD::EXTRACT_VECTOR_ELT:  return "extract_vector_elt";
  case ISD::CONCAT_VECTORS:      return "concat_vectors";
  case ISD::EXTRACT_SUBVECTOR:   return "extract_subvector";
  case ISD::SCALAR_TO_VECTOR:    return "scalar_to_vector";
  case ISD::VECTOR_SHUFFLE:      return "vector_shuffle";
  case ISD::CARRY_FALSE:         return "carry_false";
  case ISD::ADDC:        return "addc";
  case ISD::ADDE:        return "adde";
  case ISD::SUBC:        return "subc";
  case ISD::SUBE:        return "sube";
  case ISD::SHL_PARTS:   return "shl_parts";
  case ISD::SRA_PARTS:   return "sra_parts";
  case ISD::SRL_PARTS:   return "srl_parts";
  
  case ISD::EXTRACT_SUBREG:     return "extract_subreg";
  case ISD::INSERT_SUBREG:      return "insert_subreg";
  
  // Conversion operators.
  case ISD::SIGN_EXTEND: return "sign_extend";
  case ISD::ZERO_EXTEND: return "zero_extend";
  case ISD::ANY_EXTEND:  return "any_extend";
  case ISD::SIGN_EXTEND_INREG: return "sign_extend_inreg";
  case ISD::TRUNCATE:    return "truncate";
  case ISD::FP_ROUND:    return "fp_round";
  case ISD::FLT_ROUNDS_: return "flt_rounds";
  case ISD::FP_ROUND_INREG: return "fp_round_inreg";
  case ISD::FP_EXTEND:   return "fp_extend";

  case ISD::SINT_TO_FP:  return "sint_to_fp";
  case ISD::UINT_TO_FP:  return "uint_to_fp";
  case ISD::FP_TO_SINT:  return "fp_to_sint";
  case ISD::FP_TO_UINT:  return "fp_to_uint";
  case ISD::BIT_CONVERT: return "bit_convert";

    // Control flow instructions
  case ISD::BR:      return "br";
  case ISD::BRIND:   return "brind";
  case ISD::BR_JT:   return "br_jt";
  case ISD::BRCOND:  return "brcond";
  case ISD::BR_CC:   return "br_cc";
  case ISD::RET:     return "ret";
  case ISD::CALLSEQ_START:  return "callseq_start";
  case ISD::CALLSEQ_END:    return "callseq_end";

    // Other operators
  case ISD::LOAD:               return "load";
  case ISD::STORE:              return "store";
  case ISD::VAARG:              return "vaarg";
  case ISD::VACOPY:             return "vacopy";
  case ISD::VAEND:              return "vaend";
  case ISD::VASTART:            return "vastart";
  case ISD::DYNAMIC_STACKALLOC: return "dynamic_stackalloc";
  case ISD::EXTRACT_ELEMENT:    return "extract_element";
  case ISD::BUILD_PAIR:         return "build_pair";
  case ISD::STACKSAVE:          return "stacksave";
  case ISD::STACKRESTORE:       return "stackrestore";
  case ISD::TRAP:               return "trap";

  // Bit manipulation
  case ISD::BSWAP:   return "bswap";
  case ISD::CTPOP:   return "ctpop";
  case ISD::CTTZ:    return "cttz";
  case ISD::CTLZ:    return "ctlz";

  // Debug info
  case ISD::LOCATION: return "location";
  case ISD::DEBUG_LOC: return "debug_loc";

  // Trampolines
  case ISD::TRAMPOLINE: return "trampoline";

  case ISD::CONDCODE:
    switch (cast<CondCodeSDNode>(this)->get()) {
    default: assert(0 && "Unknown setcc condition!");
    case ISD::SETOEQ:  return "setoeq";
    case ISD::SETOGT:  return "setogt";
    case ISD::SETOGE:  return "setoge";
    case ISD::SETOLT:  return "setolt";
    case ISD::SETOLE:  return "setole";
    case ISD::SETONE:  return "setone";

    case ISD::SETO:    return "seto";
    case ISD::SETUO:   return "setuo";
    case ISD::SETUEQ:  return "setue";
    case ISD::SETUGT:  return "setugt";
    case ISD::SETUGE:  return "setuge";
    case ISD::SETULT:  return "setult";
    case ISD::SETULE:  return "setule";
    case ISD::SETUNE:  return "setune";

    case ISD::SETEQ:   return "seteq";
    case ISD::SETGT:   return "setgt";
    case ISD::SETGE:   return "setge";
    case ISD::SETLT:   return "setlt";
    case ISD::SETLE:   return "setle";
    case ISD::SETNE:   return "setne";
    }
  }
}

const char *SDNode::getIndexedModeName(ISD::MemIndexedMode AM) {
  switch (AM) {
  default:
    return "";
  case ISD::PRE_INC:
    return "<pre-inc>";
  case ISD::PRE_DEC:
    return "<pre-dec>";
  case ISD::POST_INC:
    return "<post-inc>";
  case ISD::POST_DEC:
    return "<post-dec>";
  }
}

std::string ISD::ArgFlagsTy::getArgFlagsString() {
  std::string S = "< ";

  if (isZExt())
    S += "zext ";
  if (isSExt())
    S += "sext ";
  if (isInReg())
    S += "inreg ";
  if (isSRet())
    S += "sret ";
  if (isByVal())
    S += "byval ";
  if (isNest())
    S += "nest ";
  if (getByValAlign())
    S += "byval-align:" + utostr(getByValAlign()) + " ";
  if (getOrigAlign())
    S += "orig-align:" + utostr(getOrigAlign()) + " ";
  if (getByValSize())
    S += "byval-size:" + utostr(getByValSize()) + " ";
  return S + ">";
}

void SDNode::dump() const { dump(0); }
void SDNode::dump(const SelectionDAG *G) const {
  cerr << (void*)this << ": ";

  for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
    if (i) cerr << ",";
    if (getValueType(i) == MVT::Other)
      cerr << "ch";
    else
      cerr << MVT::getValueTypeString(getValueType(i));
  }
  cerr << " = " << getOperationName(G);

  cerr << " ";
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    if (i) cerr << ", ";
    cerr << (void*)getOperand(i).Val;
    if (unsigned RN = getOperand(i).ResNo)
      cerr << ":" << RN;
  }

  if (!isTargetOpcode() && getOpcode() == ISD::VECTOR_SHUFFLE) {
    SDNode *Mask = getOperand(2).Val;
    cerr << "<";
    for (unsigned i = 0, e = Mask->getNumOperands(); i != e; ++i) {
      if (i) cerr << ",";
      if (Mask->getOperand(i).getOpcode() == ISD::UNDEF)
        cerr << "u";
      else
        cerr << cast<ConstantSDNode>(Mask->getOperand(i))->getValue();
    }
    cerr << ">";
  }

  if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(this)) {
    cerr << "<" << CSDN->getValue() << ">";
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(this)) {
    if (&CSDN->getValueAPF().getSemantics()==&APFloat::IEEEsingle)
      cerr << "<" << CSDN->getValueAPF().convertToFloat() << ">";
    else if (&CSDN->getValueAPF().getSemantics()==&APFloat::IEEEdouble)
      cerr << "<" << CSDN->getValueAPF().convertToDouble() << ">";
    else {
      cerr << "<APFloat(";
      CSDN->getValueAPF().convertToAPInt().dump();
      cerr << ")>";
    }
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(this)) {
    int offset = GADN->getOffset();
    cerr << "<";
    WriteAsOperand(*cerr.stream(), GADN->getGlobal()) << ">";
    if (offset > 0)
      cerr << " + " << offset;
    else
      cerr << " " << offset;
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(this)) {
    cerr << "<" << FIDN->getIndex() << ">";
  } else if (const JumpTableSDNode *JTDN = dyn_cast<JumpTableSDNode>(this)) {
    cerr << "<" << JTDN->getIndex() << ">";
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(this)){
    int offset = CP->getOffset();
    if (CP->isMachineConstantPoolEntry())
      cerr << "<" << *CP->getMachineCPVal() << ">";
    else
      cerr << "<" << *CP->getConstVal() << ">";
    if (offset > 0)
      cerr << " + " << offset;
    else
      cerr << " " << offset;
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(this)) {
    cerr << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      cerr << LBB->getName() << " ";
    cerr << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(this)) {
    if (G && R->getReg() &&
        TargetRegisterInfo::isPhysicalRegister(R->getReg())) {
      cerr << " " << G->getTarget().getRegisterInfo()->getName(R->getReg());
    } else {
      cerr << " #" << R->getReg();
    }
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(this)) {
    cerr << "'" << ES->getSymbol() << "'";
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(this)) {
    if (M->getValue())
      cerr << "<" << M->getValue() << ">";
    else
      cerr << "<null>";
  } else if (const MemOperandSDNode *M = dyn_cast<MemOperandSDNode>(this)) {
    if (M->MO.getValue())
      cerr << "<" << M->MO.getValue() << ":" << M->MO.getOffset() << ">";
    else
      cerr << "<null:" << M->MO.getOffset() << ">";
  } else if (const ARG_FLAGSSDNode *N = dyn_cast<ARG_FLAGSSDNode>(this)) {
    cerr << N->getArgFlags().getArgFlagsString();
  } else if (const VTSDNode *N = dyn_cast<VTSDNode>(this)) {
    cerr << ":" << MVT::getValueTypeString(N->getVT());
  } else if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(this)) {
    const Value *SrcValue = LD->getSrcValue();
    int SrcOffset = LD->getSrcValueOffset();
    cerr << " <";
    if (SrcValue)
      cerr << SrcValue;
    else
      cerr << "null";
    cerr << ":" << SrcOffset << ">";

    bool doExt = true;
    switch (LD->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD:
      cerr << " <anyext ";
      break;
    case ISD::SEXTLOAD:
      cerr << " <sext ";
      break;
    case ISD::ZEXTLOAD:
      cerr << " <zext ";
      break;
    }
    if (doExt)
      cerr << MVT::getValueTypeString(LD->getMemoryVT()) << ">";

    const char *AM = getIndexedModeName(LD->getAddressingMode());
    if (*AM)
      cerr << " " << AM;
    if (LD->isVolatile())
      cerr << " <volatile>";
    cerr << " alignment=" << LD->getAlignment();
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(this)) {
    const Value *SrcValue = ST->getSrcValue();
    int SrcOffset = ST->getSrcValueOffset();
    cerr << " <";
    if (SrcValue)
      cerr << SrcValue;
    else
      cerr << "null";
    cerr << ":" << SrcOffset << ">";

    if (ST->isTruncatingStore())
      cerr << " <trunc "
           << MVT::getValueTypeString(ST->getMemoryVT()) << ">";

    const char *AM = getIndexedModeName(ST->getAddressingMode());
    if (*AM)
      cerr << " " << AM;
    if (ST->isVolatile())
      cerr << " <volatile>";
    cerr << " alignment=" << ST->getAlignment();
  }
}

static void DumpNodes(const SDNode *N, unsigned indent, const SelectionDAG *G) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (N->getOperand(i).Val->hasOneUse())
      DumpNodes(N->getOperand(i).Val, indent+2, G);
    else
      cerr << "\n" << std::string(indent+2, ' ')
           << (void*)N->getOperand(i).Val << ": <multiple use>";


  cerr << "\n" << std::string(indent, ' ');
  N->dump(G);
}

void SelectionDAG::dump() const {
  cerr << "SelectionDAG has " << AllNodes.size() << " nodes:";
  std::vector<const SDNode*> Nodes;
  for (allnodes_const_iterator I = allnodes_begin(), E = allnodes_end();
       I != E; ++I)
    Nodes.push_back(I);
  
  std::sort(Nodes.begin(), Nodes.end());

  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    if (!Nodes[i]->hasOneUse() && Nodes[i] != getRoot().Val)
      DumpNodes(Nodes[i], 2, this);
  }

  if (getRoot().Val) DumpNodes(getRoot().Val, 2, this);

  cerr << "\n\n";
}

const Type *ConstantPoolSDNode::getType() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}
