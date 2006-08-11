//===-- SelectionDAG.cpp - Implement the SelectionDAG data structures -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Intrinsics.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <iostream>
#include <set>
#include <cmath>
#include <algorithm>
using namespace llvm;

static bool isCommutativeBinOp(unsigned Opcode) {
  switch (Opcode) {
  case ISD::ADD:
  case ISD::MUL:
  case ISD::MULHU:
  case ISD::MULHS:
  case ISD::FADD:
  case ISD::FMUL:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: return true;
  default: return false; // FIXME: Need commutative info for user ops!
  }
}

// isInvertibleForFree - Return true if there is no cost to emitting the logical
// inverse of this node.
static bool isInvertibleForFree(SDOperand N) {
  if (isa<ConstantSDNode>(N.Val)) return true;
  if (N.Val->getOpcode() == ISD::SETCC && N.Val->hasOneUse())
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
//                              ConstantFPSDNode Class
//===----------------------------------------------------------------------===//

/// isExactlyValue - We don't rely on operator== working on double values, as
/// it returns true for things that are clearly not equal, like -0.0 and 0.0.
/// As such, this method can be used to do an exact bit-for-bit comparison of
/// two floating point values.
bool ConstantFPSDNode::isExactlyValue(double V) const {
  return DoubleToBits(V) == DoubleToBits(Value);
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
    MVT::ValueType VT = NotZero.getValueType();
    if (VT== MVT::f64) {
      if (DoubleToBits(cast<ConstantFPSDNode>(NotZero)->getValue()) !=
          (uint64_t)-1)
        return false;
    } else {
      if (FloatToBits(cast<ConstantFPSDNode>(NotZero)->getValue()) !=
          (uint32_t)-1)
        return false;
    }
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
    if (!cast<ConstantFPSDNode>(Zero)->isExactlyValue(0.0))
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
      SDNode *Operand = I->Val;
      Operand->removeUser(N);
      
      // Now that we removed this operand, see if there are no uses of it left.
      if (Operand->use_empty())
        DeadNodes.push_back(Operand);
    }
    delete[] N->OperandList;
    N->OperandList = 0;
    N->NumOperands = 0;
    
    // Finally, remove N itself.
    AllNodes.erase(N);
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(Dummy.getValue());
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
    I->Val->removeUser(N);
  delete[] N->OperandList;
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
  case ISD::VALUETYPE:
    Erased = ValueTypeNodes[cast<VTSDNode>(N)->getVT()] != 0;
    ValueTypeNodes[cast<VTSDNode>(N)->getVT()] = 0;
    break;
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
    N->dump();
    std::cerr << "\n";
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
  
  SelectionDAGCSEMap::NodeID ID;
  ID.SetOpcode(N->getOpcode());
  ID.SetValueTypes(N->value_begin());
  ID.SetOperands(Op);
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
                                              
  SelectionDAGCSEMap::NodeID ID;
  ID.SetOpcode(N->getOpcode());
  ID.SetValueTypes(N->value_begin());
  ID.SetOperands(Op1, Op2);
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}


/// FindModifiedNodeSlot - Find a slot for the specified node if its operands
/// were replaced with those specified.  If this node is never memoized, 
/// return null, otherwise return a pointer to the slot it would take.  If a
/// node already exists with these operands, the slot will be non-null.
SDNode *SelectionDAG::FindModifiedNodeSlot(SDNode *N, 
                                           const SDOperand *Ops,unsigned NumOps,
                                           void *&InsertPos) {
  if (N->getOpcode() == ISD::HANDLENODE || N->getValueType(0) == MVT::Flag)
    return 0;    // Never add these nodes.
  
  // Check that remaining values produced are not flags.
  for (unsigned i = 1, e = N->getNumValues(); i != e; ++i)
    if (N->getValueType(i) == MVT::Flag)
      return 0;   // Never CSE anything that produces a flag.
  
  SelectionDAGCSEMap::NodeID ID;
  ID.SetOpcode(N->getOpcode());
  ID.SetValueTypes(N->value_begin());
  ID.SetOperands(Ops, NumOps);
  return CSEMap.FindNodeOrInsertPos(ID, InsertPos);
}


SelectionDAG::~SelectionDAG() {
  while (!AllNodes.empty()) {
    SDNode *N = AllNodes.begin();
    delete [] N->OperandList;
    N->OperandList = 0;
    N->NumOperands = 0;
    AllNodes.pop_front();
  }
}

SDOperand SelectionDAG::getZeroExtendInReg(SDOperand Op, MVT::ValueType VT) {
  if (Op.getValueType() == VT) return Op;
  int64_t Imm = ~0ULL >> (64-MVT::getSizeInBits(VT));
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
  assert(MVT::isInteger(VT) && "Cannot create FP integer constant!");
  assert(!MVT::isVector(VT) && "Cannot create Vector ConstantSDNodes!");
  
  // Mask out any bits that are not valid for this constant.
  Val &= MVT::getIntVTBitMask(VT);

  unsigned Opc = isT ? ISD::TargetConstant : ISD::Constant;
  SelectionDAGCSEMap::NodeID ID(Opc, getNodeValueTypes(VT));
  ID.AddInteger(Val);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantSDNode(isT, Val, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getConstantFP(double Val, MVT::ValueType VT,
                                      bool isTarget) {
  assert(MVT::isFloatingPoint(VT) && "Cannot create integer FP constant!");
  if (VT == MVT::f32)
    Val = (float)Val;  // Mask out extra precision.

  // Do the map lookup using the actual bit pattern for the floating point
  // value, so that we don't have problems with 0.0 comparing equal to -0.0, and
  // we don't have issues with SNANs.
  unsigned Opc = isTarget ? ISD::TargetConstantFP : ISD::ConstantFP;
  SelectionDAGCSEMap::NodeID ID(Opc, getNodeValueTypes(VT));
  ID.AddInteger(DoubleToBits(Val));
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantFPSDNode(isTarget, Val, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getGlobalAddress(const GlobalValue *GV,
                                         MVT::ValueType VT, int Offset,
                                         bool isTargetGA) {
  unsigned Opc = isTargetGA ? ISD::TargetGlobalAddress : ISD::GlobalAddress;
  SelectionDAGCSEMap::NodeID ID(Opc, getNodeValueTypes(VT));
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
  SelectionDAGCSEMap::NodeID ID(Opc, getNodeValueTypes(VT));
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
  SelectionDAGCSEMap::NodeID ID(Opc, getNodeValueTypes(VT));
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
  SelectionDAGCSEMap::NodeID ID(Opc, getNodeValueTypes(VT));
  ID.AddInteger(Alignment);
  ID.AddInteger(Offset);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new ConstantPoolSDNode(isTarget, C, VT, Offset, Alignment);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getBasicBlock(MachineBasicBlock *MBB) {
  SelectionDAGCSEMap::NodeID ID(ISD::BasicBlock, getNodeValueTypes(MVT::Other));
  ID.AddPointer(MBB);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new BasicBlockSDNode(MBB);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getValueType(MVT::ValueType VT) {
  if ((unsigned)VT >= ValueTypeNodes.size())
    ValueTypeNodes.resize(VT+1);
  if (ValueTypeNodes[VT] == 0) {
    ValueTypeNodes[VT] = new VTSDNode(VT);
    AllNodes.push_back(ValueTypeNodes[VT]);
  }

  return SDOperand(ValueTypeNodes[VT], 0);
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
  SelectionDAGCSEMap::NodeID ID(ISD::Register, getNodeValueTypes(VT));
  ID.AddInteger(RegNo);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new RegisterSDNode(RegNo, VT);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getSrcValue(const Value *V, int Offset) {
  assert((!V || isa<PointerType>(V->getType())) &&
         "SrcValue is not a pointer?");

  SelectionDAGCSEMap::NodeID ID(ISD::SRCVALUE, getNodeValueTypes(MVT::Other));
  ID.AddPointer(V);
  ID.AddInteger(Offset);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new SrcValueSDNode(V, Offset);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SimplifySetCC(MVT::ValueType VT, SDOperand N1,
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
    uint64_t C2 = N2C->getValue();
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val)) {
      uint64_t C1 = N1C->getValue();

      // Sign extend the operands if required
      if (ISD::isSignedIntSetCC(Cond)) {
        C1 = N1C->getSignExtended();
        C2 = N2C->getSignExtended();
      }

      switch (Cond) {
      default: assert(0 && "Unknown integer setcc!");
      case ISD::SETEQ:  return getConstant(C1 == C2, VT);
      case ISD::SETNE:  return getConstant(C1 != C2, VT);
      case ISD::SETULT: return getConstant(C1 <  C2, VT);
      case ISD::SETUGT: return getConstant(C1 >  C2, VT);
      case ISD::SETULE: return getConstant(C1 <= C2, VT);
      case ISD::SETUGE: return getConstant(C1 >= C2, VT);
      case ISD::SETLT:  return getConstant((int64_t)C1 <  (int64_t)C2, VT);
      case ISD::SETGT:  return getConstant((int64_t)C1 >  (int64_t)C2, VT);
      case ISD::SETLE:  return getConstant((int64_t)C1 <= (int64_t)C2, VT);
      case ISD::SETGE:  return getConstant((int64_t)C1 >= (int64_t)C2, VT);
      }
    } else {
      // If the LHS is a ZERO_EXTEND, perform the comparison on the input.
      if (N1.getOpcode() == ISD::ZERO_EXTEND) {
        unsigned InSize = MVT::getSizeInBits(N1.getOperand(0).getValueType());

        // If the comparison constant has bits in the upper part, the
        // zero-extended value could never match.
        if (C2 & (~0ULL << InSize)) {
          unsigned VSize = MVT::getSizeInBits(N1.getValueType());
          switch (Cond) {
          case ISD::SETUGT:
          case ISD::SETUGE:
          case ISD::SETEQ: return getConstant(0, VT);
          case ISD::SETULT:
          case ISD::SETULE:
          case ISD::SETNE: return getConstant(1, VT);
          case ISD::SETGT:
          case ISD::SETGE:
            // True if the sign bit of C2 is set.
            return getConstant((C2 & (1ULL << VSize)) != 0, VT);
          case ISD::SETLT:
          case ISD::SETLE:
            // True if the sign bit of C2 isn't set.
            return getConstant((C2 & (1ULL << VSize)) == 0, VT);
          default:
            break;
          }
        }

        // Otherwise, we can perform the comparison with the low bits.
        switch (Cond) {
        case ISD::SETEQ:
        case ISD::SETNE:
        case ISD::SETUGT:
        case ISD::SETUGE:
        case ISD::SETULT:
        case ISD::SETULE:
          return getSetCC(VT, N1.getOperand(0),
                          getConstant(C2, N1.getOperand(0).getValueType()),
                          Cond);
        default:
          break;   // todo, be more careful with signed comparisons
        }
      } else if (N1.getOpcode() == ISD::SIGN_EXTEND_INREG &&
                 (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
        MVT::ValueType ExtSrcTy = cast<VTSDNode>(N1.getOperand(1))->getVT();
        unsigned ExtSrcTyBits = MVT::getSizeInBits(ExtSrcTy);
        MVT::ValueType ExtDstTy = N1.getValueType();
        unsigned ExtDstTyBits = MVT::getSizeInBits(ExtDstTy);

        // If the extended part has any inconsistent bits, it cannot ever
        // compare equal.  In other words, they have to be all ones or all
        // zeros.
        uint64_t ExtBits =
          (~0ULL >> (64-ExtSrcTyBits)) & (~0ULL << (ExtDstTyBits-1));
        if ((C2 & ExtBits) != 0 && (C2 & ExtBits) != ExtBits)
          return getConstant(Cond == ISD::SETNE, VT);
        
        // Otherwise, make this a use of a zext.
        return getSetCC(VT, getZeroExtendInReg(N1.getOperand(0), ExtSrcTy),
                        getConstant(C2 & (~0ULL>>(64-ExtSrcTyBits)), ExtDstTy),
                        Cond);
      }

      uint64_t MinVal, MaxVal;
      unsigned OperandBitSize = MVT::getSizeInBits(N2C->getValueType(0));
      if (ISD::isSignedIntSetCC(Cond)) {
        MinVal = 1ULL << (OperandBitSize-1);
        if (OperandBitSize != 1)   // Avoid X >> 64, which is undefined.
          MaxVal = ~0ULL >> (65-OperandBitSize);
        else
          MaxVal = 0;
      } else {
        MinVal = 0;
        MaxVal = ~0ULL >> (64-OperandBitSize);
      }

      // Canonicalize GE/LE comparisons to use GT/LT comparisons.
      if (Cond == ISD::SETGE || Cond == ISD::SETUGE) {
        if (C2 == MinVal) return getConstant(1, VT);   // X >= MIN --> true
        --C2;                                          // X >= C1 --> X > (C1-1)
        return getSetCC(VT, N1, getConstant(C2, N2.getValueType()),
                        (Cond == ISD::SETGE) ? ISD::SETGT : ISD::SETUGT);
      }

      if (Cond == ISD::SETLE || Cond == ISD::SETULE) {
        if (C2 == MaxVal) return getConstant(1, VT);   // X <= MAX --> true
        ++C2;                                          // X <= C1 --> X < (C1+1)
        return getSetCC(VT, N1, getConstant(C2, N2.getValueType()),
                        (Cond == ISD::SETLE) ? ISD::SETLT : ISD::SETULT);
      }

      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C2 == MinVal)
        return getConstant(0, VT);      // X < MIN --> false

      // Canonicalize setgt X, Min --> setne X, Min
      if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C2 == MinVal)
        return getSetCC(VT, N1, N2, ISD::SETNE);

      // If we have setult X, 1, turn it into seteq X, 0
      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C2 == MinVal+1)
        return getSetCC(VT, N1, getConstant(MinVal, N1.getValueType()),
                        ISD::SETEQ);
      // If we have setugt X, Max-1, turn it into seteq X, Max
      else if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C2 == MaxVal-1)
        return getSetCC(VT, N1, getConstant(MaxVal, N1.getValueType()),
                        ISD::SETEQ);

      // If we have "setcc X, C1", check to see if we can shrink the immediate
      // by changing cc.

      // SETUGT X, SINTMAX  -> SETLT X, 0
      if (Cond == ISD::SETUGT && OperandBitSize != 1 &&
          C2 == (~0ULL >> (65-OperandBitSize)))
        return getSetCC(VT, N1, getConstant(0, N2.getValueType()), ISD::SETLT);

      // FIXME: Implement the rest of these.


      // Fold bit comparisons when we can.
      if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
          VT == N1.getValueType() && N1.getOpcode() == ISD::AND)
        if (ConstantSDNode *AndRHS =
                    dyn_cast<ConstantSDNode>(N1.getOperand(1))) {
          if (Cond == ISD::SETNE && C2 == 0) {// (X & 8) != 0  -->  (X & 8) >> 3
            // Perform the xform if the AND RHS is a single bit.
            if ((AndRHS->getValue() & (AndRHS->getValue()-1)) == 0) {
              return getNode(ISD::SRL, VT, N1,
                             getConstant(Log2_64(AndRHS->getValue()),
                                                   TLI.getShiftAmountTy()));
            }
          } else if (Cond == ISD::SETEQ && C2 == AndRHS->getValue()) {
            // (X & 8) == 8  -->  (X & 8) >> 3
            // Perform the xform if C2 is a single bit.
            if ((C2 & (C2-1)) == 0) {
              return getNode(ISD::SRL, VT, N1,
                             getConstant(Log2_64(C2),TLI.getShiftAmountTy()));
            }
          }
        }
    }
  } else if (isa<ConstantSDNode>(N1.Val)) {
      // Ensure that the constant occurs on the RHS.
    return getSetCC(VT, N2, N1, ISD::getSetCCSwappedOperands(Cond));
  }

  if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1.Val))
    if (ConstantFPSDNode *N2C = dyn_cast<ConstantFPSDNode>(N2.Val)) {
      double C1 = N1C->getValue(), C2 = N2C->getValue();

      switch (Cond) {
      default: break; // FIXME: Implement the rest of these!
      case ISD::SETEQ:  return getConstant(C1 == C2, VT);
      case ISD::SETNE:  return getConstant(C1 != C2, VT);
      case ISD::SETLT:  return getConstant(C1 < C2, VT);
      case ISD::SETGT:  return getConstant(C1 > C2, VT);
      case ISD::SETLE:  return getConstant(C1 <= C2, VT);
      case ISD::SETGE:  return getConstant(C1 >= C2, VT);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      return getSetCC(VT, N2, N1, ISD::getSetCCSwappedOperands(Cond));
    }

  // Could not fold it.
  return SDOperand();
}

/// getNode - Gets or creates the specified node.
///
SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT) {
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(Opcode, VTs);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new SDNode(Opcode, VT);
  CSEMap.InsertNode(N, IP);
  
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand Operand) {
  unsigned Tmp1;
  // Constant fold unary operations with an integer constant operand.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Operand.Val)) {
    uint64_t Val = C->getValue();
    switch (Opcode) {
    default: break;
    case ISD::SIGN_EXTEND: return getConstant(C->getSignExtended(), VT);
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND: return getConstant(Val, VT);
    case ISD::TRUNCATE:    return getConstant(Val, VT);
    case ISD::SINT_TO_FP:  return getConstantFP(C->getSignExtended(), VT);
    case ISD::UINT_TO_FP:  return getConstantFP(C->getValue(), VT);
    case ISD::BIT_CONVERT:
      if (VT == MVT::f32 && C->getValueType(0) == MVT::i32)
        return getConstantFP(BitsToFloat(Val), VT);
      else if (VT == MVT::f64 && C->getValueType(0) == MVT::i64)
        return getConstantFP(BitsToDouble(Val), VT);
      break;
    case ISD::BSWAP:
      switch(VT) {
      default: assert(0 && "Invalid bswap!"); break;
      case MVT::i16: return getConstant(ByteSwap_16((unsigned short)Val), VT);
      case MVT::i32: return getConstant(ByteSwap_32((unsigned)Val), VT);
      case MVT::i64: return getConstant(ByteSwap_64(Val), VT);
      }
      break;
    case ISD::CTPOP:
      switch(VT) {
      default: assert(0 && "Invalid ctpop!"); break;
      case MVT::i1: return getConstant(Val != 0, VT);
      case MVT::i8: 
        Tmp1 = (unsigned)Val & 0xFF;
        return getConstant(CountPopulation_32(Tmp1), VT);
      case MVT::i16:
        Tmp1 = (unsigned)Val & 0xFFFF;
        return getConstant(CountPopulation_32(Tmp1), VT);
      case MVT::i32:
        return getConstant(CountPopulation_32((unsigned)Val), VT);
      case MVT::i64:
        return getConstant(CountPopulation_64(Val), VT);
      }
    case ISD::CTLZ:
      switch(VT) {
      default: assert(0 && "Invalid ctlz!"); break;
      case MVT::i1: return getConstant(Val == 0, VT);
      case MVT::i8: 
        Tmp1 = (unsigned)Val & 0xFF;
        return getConstant(CountLeadingZeros_32(Tmp1)-24, VT);
      case MVT::i16:
        Tmp1 = (unsigned)Val & 0xFFFF;
        return getConstant(CountLeadingZeros_32(Tmp1)-16, VT);
      case MVT::i32:
        return getConstant(CountLeadingZeros_32((unsigned)Val), VT);
      case MVT::i64:
        return getConstant(CountLeadingZeros_64(Val), VT);
      }
    case ISD::CTTZ:
      switch(VT) {
      default: assert(0 && "Invalid cttz!"); break;
      case MVT::i1: return getConstant(Val == 0, VT);
      case MVT::i8: 
        Tmp1 = (unsigned)Val | 0x100;
        return getConstant(CountTrailingZeros_32(Tmp1), VT);
      case MVT::i16:
        Tmp1 = (unsigned)Val | 0x10000;
        return getConstant(CountTrailingZeros_32(Tmp1), VT);
      case MVT::i32:
        return getConstant(CountTrailingZeros_32((unsigned)Val), VT);
      case MVT::i64:
        return getConstant(CountTrailingZeros_64(Val), VT);
      }
    }
  }

  // Constant fold unary operations with an floating point constant operand.
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand.Val))
    switch (Opcode) {
    case ISD::FNEG:
      return getConstantFP(-C->getValue(), VT);
    case ISD::FABS:
      return getConstantFP(fabs(C->getValue()), VT);
    case ISD::FP_ROUND:
    case ISD::FP_EXTEND:
      return getConstantFP(C->getValue(), VT);
    case ISD::FP_TO_SINT:
      return getConstant((int64_t)C->getValue(), VT);
    case ISD::FP_TO_UINT:
      return getConstant((uint64_t)C->getValue(), VT);
    case ISD::BIT_CONVERT:
      if (VT == MVT::i32 && C->getValueType(0) == MVT::f32)
        return getConstant(FloatToBits(C->getValue()), VT);
      else if (VT == MVT::i64 && C->getValueType(0) == MVT::f64)
        return getConstant(DoubleToBits(C->getValue()), VT);
      break;
    }

  unsigned OpOpcode = Operand.Val->getOpcode();
  switch (Opcode) {
  case ISD::TokenFactor:
    return Operand;         // Factor of one node?  No factor.
  case ISD::SIGN_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType() < VT && "Invalid sext node, dst < src!");
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ZERO_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType() < VT && "Invalid zext node, dst < src!");
    if (OpOpcode == ISD::ZERO_EXTEND)   // (zext (zext x)) -> (zext x)
      return getNode(ISD::ZERO_EXTEND, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ANY_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    assert(Operand.getValueType() < VT && "Invalid anyext node, dst < src!");
    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND)
      // (ext (zext x)) -> (zext x)  and  (ext (sext x)) -> (sext x)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::TRUNCATE:
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
    assert(Operand.getValueType() > VT && "Invalid truncate node, src < dst!");
    if (OpOpcode == ISD::TRUNCATE)
      return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
    else if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND ||
             OpOpcode == ISD::ANY_EXTEND) {
      // If the source is smaller than the dest, we still need an extend.
      if (Operand.Val->getOperand(0).getValueType() < VT)
        return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
      else if (Operand.Val->getOperand(0).getValueType() > VT)
        return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
      else
        return Operand.Val->getOperand(0);
    }
    break;
  case ISD::BIT_CONVERT:
    // Basic sanity checking.
    assert(MVT::getSizeInBits(VT) == MVT::getSizeInBits(Operand.getValueType())
           && "Cannot BIT_CONVERT between two different types!");
    if (VT == Operand.getValueType()) return Operand;  // noop conversion.
    if (OpOpcode == ISD::BIT_CONVERT)  // bitconv(bitconv(x)) -> bitconv(x)
      return getNode(ISD::BIT_CONVERT, VT, Operand.getOperand(0));
    if (OpOpcode == ISD::UNDEF)
      return getNode(ISD::UNDEF, VT);
    break;
  case ISD::SCALAR_TO_VECTOR:
    assert(MVT::isVector(VT) && !MVT::isVector(Operand.getValueType()) &&
           MVT::getVectorBaseType(VT) == Operand.getValueType() &&
           "Illegal SCALAR_TO_VECTOR node!");
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
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  if (VT != MVT::Flag) { // Don't CSE flag producing nodes
    SelectionDAGCSEMap::NodeID ID(Opcode, VTs, Operand);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new SDNode(Opcode, Operand);
    N->setValueTypes(VTs, 1);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new SDNode(Opcode, Operand);
    N->setValueTypes(VTs, 1);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}



SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2) {
#ifndef NDEBUG
  switch (Opcode) {
  case ISD::TokenFactor:
    assert(VT == MVT::Other && N1.getValueType() == MVT::Other &&
           N2.getValueType() == MVT::Other && "Invalid token factor!");
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
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
    assert(MVT::isInteger(N1.getValueType()) && "Should use F* for FP ops");
    // fall through.
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
    assert(EVT <= VT && "Not rounding down!");
    break;
  }
  case ISD::AssertSext:
  case ISD::AssertZext:
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    assert(VT == N1.getValueType() && "Not an inreg extend!");
    assert(MVT::isInteger(VT) && MVT::isInteger(EVT) &&
           "Cannot *_EXTEND_INREG FP types");
    assert(EVT <= VT && "Not extending!");
  }

  default: break;
  }
#endif

  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  if (N1C) {
    if (Opcode == ISD::SIGN_EXTEND_INREG) {
      int64_t Val = N1C->getValue();
      unsigned FromBits = MVT::getSizeInBits(cast<VTSDNode>(N2)->getVT());
      Val <<= 64-FromBits;
      Val >>= 64-FromBits;
      return getConstant(Val, VT);
    }
    
    if (N2C) {
      uint64_t C1 = N1C->getValue(), C2 = N2C->getValue();
      switch (Opcode) {
      case ISD::ADD: return getConstant(C1 + C2, VT);
      case ISD::SUB: return getConstant(C1 - C2, VT);
      case ISD::MUL: return getConstant(C1 * C2, VT);
      case ISD::UDIV:
        if (C2) return getConstant(C1 / C2, VT);
        break;
      case ISD::UREM :
        if (C2) return getConstant(C1 % C2, VT);
        break;
      case ISD::SDIV :
        if (C2) return getConstant(N1C->getSignExtended() /
                                   N2C->getSignExtended(), VT);
        break;
      case ISD::SREM :
        if (C2) return getConstant(N1C->getSignExtended() %
                                   N2C->getSignExtended(), VT);
        break;
      case ISD::AND  : return getConstant(C1 & C2, VT);
      case ISD::OR   : return getConstant(C1 | C2, VT);
      case ISD::XOR  : return getConstant(C1 ^ C2, VT);
      case ISD::SHL  : return getConstant(C1 << C2, VT);
      case ISD::SRL  : return getConstant(C1 >> C2, VT);
      case ISD::SRA  : return getConstant(N1C->getSignExtended() >>(int)C2, VT);
      case ISD::ROTL : 
        return getConstant((C1 << C2) | (C1 >> (MVT::getSizeInBits(VT) - C2)),
                           VT);
      case ISD::ROTR : 
        return getConstant((C1 >> C2) | (C1 << (MVT::getSizeInBits(VT) - C2)), 
                           VT);
      default: break;
      }
    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1C, N2C);
        std::swap(N1, N2);
      }
    }
  }

  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2.Val);
  if (N1CFP) {
    if (N2CFP) {
      double C1 = N1CFP->getValue(), C2 = N2CFP->getValue();
      switch (Opcode) {
      case ISD::FADD: return getConstantFP(C1 + C2, VT);
      case ISD::FSUB: return getConstantFP(C1 - C2, VT);
      case ISD::FMUL: return getConstantFP(C1 * C2, VT);
      case ISD::FDIV:
        if (C2) return getConstantFP(C1 / C2, VT);
        break;
      case ISD::FREM :
        if (C2) return getConstantFP(fmod(C1, C2), VT);
        break;
      case ISD::FCOPYSIGN: {
        union {
          double   F;
          uint64_t I;
        } u1;
        union {
          double  F;
          int64_t I;
        } u2;
        u1.F = C1;
        u2.F = C2;
        if (u2.I < 0)  // Sign bit of RHS set?
          u1.I |= 1ULL << 63;      // Set the sign bit of the LHS.
        else 
          u1.I &= (1ULL << 63)-1;  // Clear the sign bit of the LHS.
        return getConstantFP(u1.F, VT);
      }
      default: break;
      }
    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1CFP, N2CFP);
        std::swap(N1, N2);
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
        return getConstant(0, VT);    // fold op(undef, arg2) -> 0
      }
    }
  }
  
  // Fold a bunch of operators when the RHS is undef. 
  if (N2.getOpcode() == ISD::UNDEF) {
    switch (Opcode) {
    case ISD::ADD:
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
    case ISD::XOR:
      return N2;       // fold op(arg1, undef) -> undef
    case ISD::MUL: 
    case ISD::AND:
    case ISD::SRL:
    case ISD::SHL:
      return getConstant(0, VT);  // fold op(arg1, undef) -> 0
    case ISD::OR:
      return getConstant(MVT::getIntVTBitMask(VT), VT);
    case ISD::SRA:
      return N1;
    }
  }

  // Finally, fold operations that do not require constants.
  switch (Opcode) {
  case ISD::FP_ROUND_INREG:
    if (cast<VTSDNode>(N2)->getVT() == VT) return N1;  // Not actually rounding.
    break;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    if (EVT == VT) return N1;  // Not actually extending
    break;
  }

  // FIXME: figure out how to safely handle things like
  // int foo(int x) { return 1 << (x & 255); }
  // int bar() { return foo(256); }
#if 0
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    if (N2.getOpcode() == ISD::SIGN_EXTEND_INREG &&
        cast<VTSDNode>(N2.getOperand(1))->getVT() != MVT::i1)
      return getNode(Opcode, VT, N1, N2.getOperand(0));
    else if (N2.getOpcode() == ISD::AND)
      if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(N2.getOperand(1))) {
        // If the and is only masking out bits that cannot effect the shift,
        // eliminate the and.
        unsigned NumBits = MVT::getSizeInBits(VT);
        if ((AndRHS->getValue() & (NumBits-1)) == NumBits-1)
          return getNode(Opcode, VT, N1, N2.getOperand(0));
      }
    break;
#endif
  }

  // Memoize this node if possible.
  SDNode *N;
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  if (VT != MVT::Flag) {
    SelectionDAGCSEMap::NodeID ID(Opcode, VTs, N1, N2);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new SDNode(Opcode, N1, N2);
    N->setValueTypes(VTs, 1);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new SDNode(Opcode, N1, N2);
    N->setValueTypes(VTs, 1);
  }

  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3) {
  // Perform various simplifications.
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  //ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.Val);
  switch (Opcode) {
  case ISD::SETCC: {
    // Use SimplifySetCC  to simplify SETCC's.
    SDOperand Simp = SimplifySetCC(VT, N1, N2, cast<CondCodeSDNode>(N3)->get());
    if (Simp.Val) return Simp;
    break;
  }
  case ISD::SELECT:
    if (N1C)
      if (N1C->getValue())
        return N2;             // select true, X, Y -> X
      else
        return N3;             // select false, X, Y -> Y

    if (N2 == N3) return N2;   // select C, X, X -> X
    break;
  case ISD::BRCOND:
    if (N2C)
      if (N2C->getValue()) // Unconditional branch
        return getNode(ISD::BR, MVT::Other, N1, N3);
      else
        return N1;         // Never-taken branch
    break;
  case ISD::VECTOR_SHUFFLE:
    assert(VT == N1.getValueType() && VT == N2.getValueType() &&
           MVT::isVector(VT) && MVT::isVector(N3.getValueType()) &&
           N3.getOpcode() == ISD::BUILD_VECTOR &&
           MVT::getVectorNumElements(VT) == N3.getNumOperands() &&
           "Illegal VECTOR_SHUFFLE node!");
    break;
  }

  // Memoize node if it doesn't produce a flag.
  SDNode *N;
  MVT::ValueType *VTs = getNodeValueTypes(VT);

  if (VT != MVT::Flag) {
    SelectionDAGCSEMap::NodeID ID(Opcode, VTs, N1, N2, N3);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new SDNode(Opcode, N1, N2, N3);
    N->setValueTypes(VTs, 1);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new SDNode(Opcode, N1, N2, N3);
    N->setValueTypes(VTs, 1);
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

SDOperand SelectionDAG::getLoad(MVT::ValueType VT,
                                SDOperand Chain, SDOperand Ptr,
                                SDOperand SV) {
  MVT::ValueType *VTs = getNodeValueTypes(VT, MVT::Other);
  
  SelectionDAGCSEMap::NodeID ID(ISD::LOAD, VTs, Chain, Ptr, SV);
  void *IP = 0;
  if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(E, 0);
  SDNode *N = new SDNode(ISD::LOAD, Chain, Ptr, SV);
  N->setValueTypes(VTs, 2);
  CSEMap.InsertNode(N, IP);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getVecLoad(unsigned Count, MVT::ValueType EVT,
                                   SDOperand Chain, SDOperand Ptr,
                                   SDOperand SV) {
  SDOperand Ops[] = { Chain, Ptr, SV, getConstant(Count, MVT::i32), 
                      getValueType(EVT) };
  std::vector<MVT::ValueType> VTs;
  VTs.reserve(2);
  VTs.push_back(MVT::Vector); VTs.push_back(MVT::Other);  // Add token chain.
  return getNode(ISD::VLOAD, VTs, Ops, 5);
}

SDOperand SelectionDAG::getExtLoad(unsigned Opcode, MVT::ValueType VT,
                                   SDOperand Chain, SDOperand Ptr, SDOperand SV,
                                   MVT::ValueType EVT) {
  SDOperand Ops[] = { Chain, Ptr, SV, getValueType(EVT) };
  std::vector<MVT::ValueType> VTs;
  VTs.reserve(2);
  VTs.push_back(VT); VTs.push_back(MVT::Other);  // Add token chain.
  return getNode(Opcode, VTs, Ops, 4);
}

SDOperand SelectionDAG::getVAArg(MVT::ValueType VT,
                                 SDOperand Chain, SDOperand Ptr,
                                 SDOperand SV) {
  SDOperand Ops[] = { Chain, Ptr, SV };
  std::vector<MVT::ValueType> VTs;
  VTs.reserve(2);
  VTs.push_back(VT); VTs.push_back(MVT::Other);  // Add token chain.
  return getNode(ISD::VAARG, VTs, Ops, 3);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                const SDOperand *Ops, unsigned NumOps) {
  switch (NumOps) {
  case 0: return getNode(Opcode, VT);
  case 1: return getNode(Opcode, VT, Ops[0]);
  case 2: return getNode(Opcode, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }
  
  switch (Opcode) {
  default: break;
  case ISD::TRUNCSTORE: {
    assert(NumOps == 5 && "TRUNCSTORE takes 5 operands!");
    MVT::ValueType EVT = cast<VTSDNode>(Ops[4])->getVT();
#if 0 // FIXME: If the target supports EVT natively, convert to a truncate/store
    // If this is a truncating store of a constant, convert to the desired type
    // and store it instead.
    if (isa<Constant>(Ops[0])) {
      SDOperand Op = getNode(ISD::TRUNCATE, EVT, N1);
      if (isa<Constant>(Op))
        N1 = Op;
    }
    // Also for ConstantFP?
#endif
    if (Ops[0].getValueType() == EVT)       // Normal store?
      return getNode(ISD::STORE, VT, Ops[0], Ops[1], Ops[2], Ops[3]);
    assert(Ops[1].getValueType() > EVT && "Not a truncation?");
    assert(MVT::isInteger(Ops[1].getValueType()) == MVT::isInteger(EVT) &&
           "Can't do FP-INT conversion!");
    break;
  }
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
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  if (VT != MVT::Flag) {
    SelectionDAGCSEMap::NodeID ID(Opcode, VTs, Ops, NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new SDNode(Opcode, Ops, NumOps);
    N->setValueTypes(VTs, 1);
    CSEMap.InsertNode(N, IP);
  } else {
    N = new SDNode(Opcode, Ops, NumOps);
    N->setValueTypes(VTs, 1);
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode,
                                std::vector<MVT::ValueType> &ResultTys,
                                const SDOperand *Ops, unsigned NumOps) {
  if (ResultTys.size() == 1)
    return getNode(Opcode, ResultTys[0], Ops, NumOps);

  switch (Opcode) {
  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD: {
    MVT::ValueType EVT = cast<VTSDNode>(Ops[3])->getVT();
    assert(NumOps == 4 && ResultTys.size() == 2 && "Bad *EXTLOAD!");
    // If they are asking for an extending load from/to the same thing, return a
    // normal load.
    if (ResultTys[0] == EVT)
      return getLoad(ResultTys[0], Ops[0], Ops[1], Ops[2]);
    if (MVT::isVector(ResultTys[0])) {
      assert(EVT == MVT::getVectorBaseType(ResultTys[0]) &&
             "Invalid vector extload!");
    } else {
      assert(EVT < ResultTys[0] &&
             "Should only be an extending load, not truncating!");
    }
    assert((Opcode == ISD::EXTLOAD || MVT::isInteger(ResultTys[0])) &&
           "Cannot sign/zero extend a FP/Vector load!");
    assert(MVT::isInteger(ResultTys[0]) == MVT::isInteger(EVT) &&
           "Cannot convert from FP to Int or Int -> FP!");
    break;
  }

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
  MVT::ValueType *VTs = getNodeValueTypes(ResultTys);
  if (ResultTys.back() != MVT::Flag) {
    SelectionDAGCSEMap::NodeID ID;
    ID.SetOpcode(Opcode);
    ID.SetValueTypes(VTs);
    ID.SetOperands(&Ops[0], NumOps);
    void *IP = 0;
    if (SDNode *E = CSEMap.FindNodeOrInsertPos(ID, IP))
      return SDOperand(E, 0);
    N = new SDNode(Opcode, Ops, NumOps);
    N->setValueTypes(VTs, ResultTys.size());
    CSEMap.InsertNode(N, IP);
  } else {
    N = new SDNode(Opcode, Ops, NumOps);
    N->setValueTypes(VTs, ResultTys.size());
  }
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


MVT::ValueType *SelectionDAG::getNodeValueTypes(MVT::ValueType VT) {
  return SDNode::getValueTypeList(VT);
}

MVT::ValueType *SelectionDAG::getNodeValueTypes(
                                        std::vector<MVT::ValueType> &RetVals) {
  switch (RetVals.size()) {
  case 0: assert(0 && "Cannot have nodes without results!");
  case 1: return SDNode::getValueTypeList(RetVals[0]);
  case 2: return getNodeValueTypes(RetVals[0], RetVals[1]);
  default: break;
  }
  
  std::list<std::vector<MVT::ValueType> >::iterator I =
    std::find(VTList.begin(), VTList.end(), RetVals);
  if (I == VTList.end()) {
    VTList.push_front(RetVals);
    I = VTList.begin();
  }

  return &(*I)[0];
}

MVT::ValueType *SelectionDAG::getNodeValueTypes(MVT::ValueType VT1, 
                                                MVT::ValueType VT2) {
  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() == 2 && (*I)[0] == VT1 && (*I)[1] == VT2)
      return &(*I)[0];
  }
  std::vector<MVT::ValueType> V;
  V.push_back(VT1);
  V.push_back(VT2);
  VTList.push_front(V);
  return &(*VTList.begin())[0];
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
  N->OperandList[0].Val->removeUser(N);
  Op.Val->addUser(N);
  N->OperandList[0] = Op;
  
  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}

SDOperand SelectionDAG::
UpdateNodeOperands(SDOperand InN, SDOperand Op1, SDOperand Op2) {
  SDNode *N = InN.Val;
  assert(N->getNumOperands() == 2 && "Update with wrong number of operands");
  
  // Check to see if there is no change.
  bool AnyChange = false;
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
    N->OperandList[0].Val->removeUser(N);
    Op1.Val->addUser(N);
    N->OperandList[0] = Op1;
  }
  if (N->OperandList[1] != Op2) {
    N->OperandList[1].Val->removeUser(N);
    Op2.Val->addUser(N);
    N->OperandList[1] = Op2;
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
UpdateNodeOperands(SDOperand InN, SDOperand *Ops, unsigned NumOps) {
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
  
  // Nope it doesn't.  Remove the node from it's current place in the maps.
  if (InsertPos)
    RemoveNodeFromCSEMaps(N);
  
  // Now we update the operands.
  for (unsigned i = 0; i != NumOps; ++i) {
    if (N->OperandList[i] != Ops[i]) {
      N->OperandList[i].Val->removeUser(N);
      Ops[i].Val->addUser(N);
      N->OperandList[i] = Ops[i];
    }
  }

  // If this gets put into a CSE map, add it.
  if (InsertPos) CSEMap.InsertNode(N, InsertPos);
  return InN;
}




/// SelectNodeTo - These are used for target selectors to *mutate* the
/// specified node to have the specified return type, Target opcode, and
/// operands.  Note that target opcodes are stored as
/// ISD::BUILTIN_OP_END+TargetOpcode in the node opcode field.
///
/// Note that SelectNodeTo returns the resultant node.  If there is already a
/// node of the specified opcode and operands, it returns that node instead of
/// the current one.
SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT) {
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
   
  RemoveNodeFromCSEMaps(N);
  
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(getNodeValueTypes(VT), 1);

  CSEMap.InsertNode(N, IP);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs, Op1);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(getNodeValueTypes(VT), 1);
  N->setOperands(Op1);
  CSEMap.InsertNode(N, IP);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1,
                                     SDOperand Op2) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs, Op1, Op2);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 1);
  N->setOperands(Op1, Op2);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1,
                                     SDOperand Op2, SDOperand Op3) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs, Op1, Op2, Op3);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 1);
  N->setOperands(Op1, Op2, Op3);

  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1,
                                     SDOperand Op2, SDOperand Op3,
                                     SDOperand Op4) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  ID.AddOperand(Op1);
  ID.AddOperand(Op2);
  ID.AddOperand(Op3);
  ID.AddOperand(Op4);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
  
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 1);
  N->setOperands(Op1, Op2, Op3, Op4);

  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1,
                                     SDOperand Op2, SDOperand Op3,
                                     SDOperand Op4, SDOperand Op5) {
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  ID.AddOperand(Op1);
  ID.AddOperand(Op2);
  ID.AddOperand(Op3);
  ID.AddOperand(Op4);
  ID.AddOperand(Op5);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 1);
  N->setOperands(Op1, Op2, Op3, Op4, Op5);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1,
                                     SDOperand Op2, SDOperand Op3,SDOperand Op4,
                                     SDOperand Op5, SDOperand Op6) {
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  ID.AddOperand(Op1);
  ID.AddOperand(Op2);
  ID.AddOperand(Op3);
  ID.AddOperand(Op4);
  ID.AddOperand(Op5);
  ID.AddOperand(Op6);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 1);
  N->setOperands(Op1, Op2, Op3, Op4, Op5, Op6);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1,
                                     SDOperand Op2, SDOperand Op3,SDOperand Op4,
                                     SDOperand Op5, SDOperand Op6,
				     SDOperand Op7) {
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  // If an identical node already exists, use it.
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  ID.AddOperand(Op1);
  ID.AddOperand(Op2);
  ID.AddOperand(Op3);
  ID.AddOperand(Op4);
  ID.AddOperand(Op5);
  ID.AddOperand(Op6);
  ID.AddOperand(Op7);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 1);
  N->setOperands(Op1, Op2, Op3, Op4, Op5, Op6, Op7);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}
SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT, SDOperand Op1,
                                     SDOperand Op2, SDOperand Op3,SDOperand Op4,
                                     SDOperand Op5, SDOperand Op6,
				     SDOperand Op7, SDOperand Op8) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  ID.AddOperand(Op1);
  ID.AddOperand(Op2);
  ID.AddOperand(Op3);
  ID.AddOperand(Op4);
  ID.AddOperand(Op5);
  ID.AddOperand(Op6);
  ID.AddOperand(Op7);
  ID.AddOperand(Op8);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 1);
  N->setOperands(Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc, 
                                     MVT::ValueType VT1, MVT::ValueType VT2,
                                     SDOperand Op1, SDOperand Op2) {
  MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs, Op1, Op2);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);

  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 2);
  N->setOperands(Op1, Op2);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT1, MVT::ValueType VT2,
                                     SDOperand Op1, SDOperand Op2, 
                                     SDOperand Op3) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs,
                                Op1, Op2, Op3);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);

  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 2);
  N->setOperands(Op1, Op2, Op3);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT1, MVT::ValueType VT2,
                                     SDOperand Op1, SDOperand Op2,
                                     SDOperand Op3, SDOperand Op4) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  ID.AddOperand(Op1);
  ID.AddOperand(Op2);
  ID.AddOperand(Op3);
  ID.AddOperand(Op4);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 2);
  N->setOperands(Op1, Op2, Op3, Op4);

  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                     MVT::ValueType VT1, MVT::ValueType VT2,
                                     SDOperand Op1, SDOperand Op2,
                                     SDOperand Op3, SDOperand Op4, 
                                     SDOperand Op5) {
  // If an identical node already exists, use it.
  MVT::ValueType *VTs = getNodeValueTypes(VT1, VT2);
  SelectionDAGCSEMap::NodeID ID(ISD::BUILTIN_OP_END+TargetOpc, VTs);
  ID.AddOperand(Op1);
  ID.AddOperand(Op2);
  ID.AddOperand(Op3);
  ID.AddOperand(Op4);
  ID.AddOperand(Op5);
  void *IP = 0;
  if (SDNode *ON = CSEMap.FindNodeOrInsertPos(ID, IP))
    return SDOperand(ON, 0);
                                       
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VTs, 2);
  N->setOperands(Op1, Op2, Op3, Op4, Op5);
  
  CSEMap.InsertNode(N, IP);   // Memoize the new node.
  return SDOperand(N, 0);
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
                                    SDOperand Op1, SDOperand Op2, SDOperand Op3) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2, SDOperand Op3,
                                    SDOperand Op4) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3, Op4).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2, SDOperand Op3,
                                    SDOperand Op4, SDOperand Op5) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Op1, Op2, Op3, Op4, Op5).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2, SDOperand Op3,
                                    SDOperand Op4, SDOperand Op5,
                                    SDOperand Op6) {
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5, Op6 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops, 6).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2, SDOperand Op3,
                                    SDOperand Op4, SDOperand Op5, SDOperand Op6,
                                    SDOperand Op7) {
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5, Op6, Op7 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops, 7).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    SDOperand Op1, SDOperand Op2, SDOperand Op3,
                                    SDOperand Op4, SDOperand Op5, SDOperand Op6,
                                    SDOperand Op7, SDOperand Op8) {
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops, 8).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT,
                                    const SDOperand *Ops, unsigned NumOps) {
  return getNode(ISD::BUILTIN_OP_END+Opcode, VT, Ops, NumOps).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, &Op1, 1).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  SDOperand Ops[] = { Op1, Op2 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2, SDOperand Op3) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  SDOperand Ops[] = { Op1, Op2, Op3 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 3).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2, SDOperand Op3, 
                                    SDOperand Op4) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  SDOperand Ops[] = { Op1, Op2, Op3, Op4 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 4).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2, SDOperand Op3, SDOperand Op4,
                                    SDOperand Op5) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 5).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2, SDOperand Op3, SDOperand Op4,
                                    SDOperand Op5, SDOperand Op6) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5, Op6 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 6).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, SDOperand Op1,
                                    SDOperand Op2, SDOperand Op3, SDOperand Op4,
                                    SDOperand Op5, SDOperand Op6,
                                    SDOperand Op7) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5, Op6, Op7 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 7).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  ResultTys.push_back(VT3);
  SDOperand Ops[] = { Op1, Op2 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 2).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2,
                                    SDOperand Op3, SDOperand Op4,
                                    SDOperand Op5) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  ResultTys.push_back(VT3);
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 5).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2,
                                    SDOperand Op3, SDOperand Op4, SDOperand Op5,
                                    SDOperand Op6) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  ResultTys.push_back(VT3);
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5, Op6 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 6).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1,
                                    MVT::ValueType VT2, MVT::ValueType VT3,
                                    SDOperand Op1, SDOperand Op2,
                                    SDOperand Op3, SDOperand Op4, SDOperand Op5,
                                    SDOperand Op6, SDOperand Op7) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  ResultTys.push_back(VT3);
  SDOperand Ops[] = { Op1, Op2, Op3, Op4, Op5, Op6, Op7 };
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, 7).Val;
}
SDNode *SelectionDAG::getTargetNode(unsigned Opcode, MVT::ValueType VT1, 
                                    MVT::ValueType VT2,
                                    const SDOperand *Ops, unsigned NumOps) {
  std::vector<MVT::ValueType> ResultTys;
  ResultTys.push_back(VT1);
  ResultTys.push_back(VT2);
  return getNode(ISD::BUILTIN_OP_END+Opcode, ResultTys, Ops, NumOps).Val;
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version assumes From/To have a single result value.
///
void SelectionDAG::ReplaceAllUsesWith(SDOperand FromN, SDOperand ToN,
                                      std::vector<SDNode*> *Deleted) {
  SDNode *From = FromN.Val, *To = ToN.Val;
  assert(From->getNumValues() == 1 && To->getNumValues() == 1 &&
         "Cannot replace with this method!");
  assert(From != To && "Cannot replace uses of with self");
  
  while (!From->use_empty()) {
    // Process users until they are all gone.
    SDNode *U = *From->use_begin();
    
    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    
    for (SDOperand *I = U->OperandList, *E = U->OperandList+U->NumOperands;
         I != E; ++I)
      if (I->Val == From) {
        From->removeUser(U);
        I->Val = To;
        To->addUser(U);
      }

    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, Deleted);
      // U is now dead.
      if (Deleted) Deleted->push_back(U);
      DeleteNodeNotInCSEMaps(U);
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
                                      std::vector<SDNode*> *Deleted) {
  assert(From != To && "Cannot replace uses of with self");
  assert(From->getNumValues() == To->getNumValues() &&
         "Cannot use this version of ReplaceAllUsesWith!");
  if (From->getNumValues() == 1) {  // If possible, use the faster version.
    ReplaceAllUsesWith(SDOperand(From, 0), SDOperand(To, 0), Deleted);
    return;
  }
  
  while (!From->use_empty()) {
    // Process users until they are all gone.
    SDNode *U = *From->use_begin();
    
    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    
    for (SDOperand *I = U->OperandList, *E = U->OperandList+U->NumOperands;
         I != E; ++I)
      if (I->Val == From) {
        From->removeUser(U);
        I->Val = To;
        To->addUser(U);
      }
        
    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, Deleted);
      // U is now dead.
      if (Deleted) Deleted->push_back(U);
      DeleteNodeNotInCSEMaps(U);
    }
  }
}

/// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
/// This can cause recursive merging of nodes in the DAG.
///
/// This version can replace From with any result values.  To must match the
/// number and types of values returned by From.
void SelectionDAG::ReplaceAllUsesWith(SDNode *From,
                                      const SDOperand *To,
                                      std::vector<SDNode*> *Deleted) {
  if (From->getNumValues() == 1 && To[0].Val->getNumValues() == 1) {
    // Degenerate case handled above.
    ReplaceAllUsesWith(SDOperand(From, 0), To[0], Deleted);
    return;
  }

  while (!From->use_empty()) {
    // Process users until they are all gone.
    SDNode *U = *From->use_begin();
    
    // This node is about to morph, remove its old self from the CSE maps.
    RemoveNodeFromCSEMaps(U);
    
    for (SDOperand *I = U->OperandList, *E = U->OperandList+U->NumOperands;
         I != E; ++I)
      if (I->Val == From) {
        const SDOperand &ToOp = To[I->ResNo];
        From->removeUser(U);
        *I = ToOp;
        ToOp.Val->addUser(U);
      }
        
    // Now that we have modified U, add it back to the CSE maps.  If it already
    // exists there, recursively merge the results together.
    if (SDNode *Existing = AddNonLeafNodeToCSEMaps(U)) {
      ReplaceAllUsesWith(U, Existing, Deleted);
      // U is now dead.
      if (Deleted) Deleted->push_back(U);
      DeleteNodeNotInCSEMaps(U);
    }
  }
}

/// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
/// uses of other values produced by From.Val alone.  The Deleted vector is
/// handled the same was as for ReplaceAllUsesWith.
void SelectionDAG::ReplaceAllUsesOfValueWith(SDOperand From, SDOperand To,
                                             std::vector<SDNode*> &Deleted) {
  assert(From != To && "Cannot replace a value with itself");
  // Handle the simple, trivial, case efficiently.
  if (From.Val->getNumValues() == 1 && To.Val->getNumValues() == 1) {
    ReplaceAllUsesWith(From, To, &Deleted);
    return;
  }
  
  // Get all of the users in a nice, deterministically ordered, uniqued set.
  SetVector<SDNode*> Users(From.Val->use_begin(), From.Val->use_end());

  while (!Users.empty()) {
    // We know that this user uses some value of From.  If it is the right
    // value, update it.
    SDNode *User = Users.back();
    Users.pop_back();
    
    for (SDOperand *Op = User->OperandList,
         *E = User->OperandList+User->NumOperands; Op != E; ++Op) {
      if (*Op == From) {
        // Okay, we know this user needs to be updated.  Remove its old self
        // from the CSE maps.
        RemoveNodeFromCSEMaps(User);
        
        // Update all operands that match "From".
        for (; Op != E; ++Op) {
          if (*Op == From) {
            From.Val->removeUser(User);
            *Op = To;
            To.Val->addUser(User);
          }
        }
                   
        // Now that we have modified User, add it back to the CSE maps.  If it
        // already exists there, recursively merge the results together.
        if (SDNode *Existing = AddNonLeafNodeToCSEMaps(User)) {
          unsigned NumDeleted = Deleted.size();
          ReplaceAllUsesWith(User, Existing, &Deleted);
          
          // User is now dead.
          Deleted.push_back(User);
          DeleteNodeNotInCSEMaps(User);
          
          // We have to be careful here, because ReplaceAllUsesWith could have
          // deleted a user of From, which means there may be dangling pointers
          // in the "Users" setvector.  Scan over the deleted node pointers and
          // remove them from the setvector.
          for (unsigned i = NumDeleted, e = Deleted.size(); i != e; ++i)
            Users.remove(Deleted[i]);
        }
        break;   // Exit the operand scanning loop.
      }
    }
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
      SDNode *P = I->Val;
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
void SDNode::ANCHOR() {
}

/// getValueTypeList - Return a pointer to the specified value type.
///
MVT::ValueType *SDNode::getValueTypeList(MVT::ValueType VT) {
  static MVT::ValueType VTs[MVT::LAST_VALUETYPE];
  VTs[VT] = VT;
  return &VTs[VT];
}
  
/// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
/// indicated value.  This method ignores uses of other values defined by this
/// operation.
bool SDNode::hasNUsesOfValue(unsigned NUses, unsigned Value) const {
  assert(Value < getNumValues() && "Bad value!");

  // If there is only one value, this is easy.
  if (getNumValues() == 1)
    return use_size() == NUses;
  if (Uses.size() < NUses) return false;

  SDOperand TheValue(const_cast<SDNode *>(this), Value);

  std::set<SDNode*> UsersHandled;

  for (std::vector<SDNode*>::const_iterator UI = Uses.begin(), E = Uses.end();
       UI != E; ++UI) {
    SDNode *User = *UI;
    if (User->getNumOperands() == 1 ||
        UsersHandled.insert(User).second)     // First time we've seen this?
      for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i)
        if (User->getOperand(i) == TheValue) {
          if (NUses == 0)
            return false;   // too many uses
          --NUses;
        }
  }

  // Found exactly the right number of uses?
  return NUses == 0;
}


// isOnlyUse - Return true if this node is the only use of N.
bool SDNode::isOnlyUse(SDNode *N) const {
  bool Seen = false;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDNode *User = *I;
    if (User == this)
      Seen = true;
    else
      return false;
  }

  return Seen;
}

// isOperand - Return true if this node is an operand of N.
bool SDOperand::isOperand(SDNode *N) const {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (*this == N->getOperand(i))
      return true;
  return false;
}

bool SDNode::isOperand(SDNode *N) const {
  for (unsigned i = 0, e = N->NumOperands; i != e; ++i)
    if (this == N->OperandList[i].Val)
      return true;
  return false;
}

const char *SDNode::getOperationName(const SelectionDAG *G) const {
  switch (getOpcode()) {
  default:
    if (getOpcode() < ISD::BUILTIN_OP_END)
      return "<<Unknown DAG Node>>";
    else {
      if (G) {
        if (const TargetInstrInfo *TII = G->getTarget().getInstrInfo())
          if (getOpcode()-ISD::BUILTIN_OP_END < TII->getNumOpcodes())
            return TII->getName(getOpcode()-ISD::BUILTIN_OP_END);

        TargetLowering &TLI = G->getTargetLoweringInfo();
        const char *Name =
          TLI.getTargetNodeName(getOpcode());
        if (Name) return Name;
      }

      return "<<Unknown Target Node>>";
    }
   
  case ISD::PCMARKER:      return "PCMarker";
  case ISD::READCYCLECOUNTER: return "ReadCycleCounter";
  case ISD::SRCVALUE:      return "SrcValue";
  case ISD::EntryToken:    return "EntryToken";
  case ISD::TokenFactor:   return "TokenFactor";
  case ISD::AssertSext:    return "AssertSext";
  case ISD::AssertZext:    return "AssertZext";

  case ISD::STRING:        return "String";
  case ISD::BasicBlock:    return "BasicBlock";
  case ISD::VALUETYPE:     return "ValueType";
  case ISD::Register:      return "Register";

  case ISD::Constant:      return "Constant";
  case ISD::ConstantFP:    return "ConstantFP";
  case ISD::GlobalAddress: return "GlobalAddress";
  case ISD::FrameIndex:    return "FrameIndex";
  case ISD::JumpTable:     return "JumpTable";
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
  case ISD::TargetFrameIndex: return "TargetFrameIndex";
  case ISD::TargetJumpTable:  return "TargetJumpTable";
  case ISD::TargetConstantPool:  return "TargetConstantPool";
  case ISD::TargetExternalSymbol: return "TargetExternalSymbol";

  case ISD::CopyToReg:     return "CopyToReg";
  case ISD::CopyFromReg:   return "CopyFromReg";
  case ISD::UNDEF:         return "undef";
  case ISD::MERGE_VALUES:  return "mergevalues";
  case ISD::INLINEASM:     return "inlineasm";
  case ISD::HANDLENODE:    return "handlenode";
  case ISD::FORMAL_ARGUMENTS: return "formal_arguments";
  case ISD::CALL:          return "call";
    
  // Unary operators
  case ISD::FABS:   return "fabs";
  case ISD::FNEG:   return "fneg";
  case ISD::FSQRT:  return "fsqrt";
  case ISD::FSIN:   return "fsin";
  case ISD::FCOS:   return "fcos";

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
  case ISD::VADD:   return "vadd";
  case ISD::VSUB:   return "vsub";
  case ISD::VMUL:   return "vmul";
  case ISD::VSDIV:  return "vsdiv";
  case ISD::VUDIV:  return "vudiv";
  case ISD::VAND:   return "vand";
  case ISD::VOR:    return "vor";
  case ISD::VXOR:   return "vxor";

  case ISD::SETCC:       return "setcc";
  case ISD::SELECT:      return "select";
  case ISD::SELECT_CC:   return "select_cc";
  case ISD::VSELECT:     return "vselect";
  case ISD::INSERT_VECTOR_ELT:   return "insert_vector_elt";
  case ISD::VINSERT_VECTOR_ELT:  return "vinsert_vector_elt";
  case ISD::EXTRACT_VECTOR_ELT:  return "extract_vector_elt";
  case ISD::VEXTRACT_VECTOR_ELT: return "vextract_vector_elt";
  case ISD::SCALAR_TO_VECTOR:    return "scalar_to_vector";
  case ISD::VBUILD_VECTOR:       return "vbuild_vector";
  case ISD::VECTOR_SHUFFLE:      return "vector_shuffle";
  case ISD::VVECTOR_SHUFFLE:     return "vvector_shuffle";
  case ISD::VBIT_CONVERT:        return "vbit_convert";
  case ISD::ADDC:        return "addc";
  case ISD::ADDE:        return "adde";
  case ISD::SUBC:        return "subc";
  case ISD::SUBE:        return "sube";
  case ISD::SHL_PARTS:   return "shl_parts";
  case ISD::SRA_PARTS:   return "sra_parts";
  case ISD::SRL_PARTS:   return "srl_parts";

  // Conversion operators.
  case ISD::SIGN_EXTEND: return "sign_extend";
  case ISD::ZERO_EXTEND: return "zero_extend";
  case ISD::ANY_EXTEND:  return "any_extend";
  case ISD::SIGN_EXTEND_INREG: return "sign_extend_inreg";
  case ISD::TRUNCATE:    return "truncate";
  case ISD::FP_ROUND:    return "fp_round";
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
  case ISD::BRCOND:  return "brcond";
  case ISD::BR_CC:   return "br_cc";
  case ISD::RET:     return "ret";
  case ISD::CALLSEQ_START:  return "callseq_start";
  case ISD::CALLSEQ_END:    return "callseq_end";

    // Other operators
  case ISD::LOAD:               return "load";
  case ISD::STORE:              return "store";
  case ISD::VLOAD:              return "vload";
  case ISD::EXTLOAD:            return "extload";
  case ISD::SEXTLOAD:           return "sextload";
  case ISD::ZEXTLOAD:           return "zextload";
  case ISD::TRUNCSTORE:         return "truncstore";
  case ISD::VAARG:              return "vaarg";
  case ISD::VACOPY:             return "vacopy";
  case ISD::VAEND:              return "vaend";
  case ISD::VASTART:            return "vastart";
  case ISD::DYNAMIC_STACKALLOC: return "dynamic_stackalloc";
  case ISD::EXTRACT_ELEMENT:    return "extract_element";
  case ISD::BUILD_PAIR:         return "build_pair";
  case ISD::STACKSAVE:          return "stacksave";
  case ISD::STACKRESTORE:       return "stackrestore";
    
  // Block memory operations.
  case ISD::MEMSET:  return "memset";
  case ISD::MEMCPY:  return "memcpy";
  case ISD::MEMMOVE: return "memmove";

  // Bit manipulation
  case ISD::BSWAP:   return "bswap";
  case ISD::CTPOP:   return "ctpop";
  case ISD::CTTZ:    return "cttz";
  case ISD::CTLZ:    return "ctlz";

  // Debug info
  case ISD::LOCATION: return "location";
  case ISD::DEBUG_LOC: return "debug_loc";
  case ISD::DEBUG_LABEL: return "debug_label";

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

void SDNode::dump() const { dump(0); }
void SDNode::dump(const SelectionDAG *G) const {
  std::cerr << (void*)this << ": ";

  for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
    if (i) std::cerr << ",";
    if (getValueType(i) == MVT::Other)
      std::cerr << "ch";
    else
      std::cerr << MVT::getValueTypeString(getValueType(i));
  }
  std::cerr << " = " << getOperationName(G);

  std::cerr << " ";
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    if (i) std::cerr << ", ";
    std::cerr << (void*)getOperand(i).Val;
    if (unsigned RN = getOperand(i).ResNo)
      std::cerr << ":" << RN;
  }

  if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(this)) {
    std::cerr << "<" << CSDN->getValue() << ">";
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(this)) {
    std::cerr << "<" << CSDN->getValue() << ">";
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(this)) {
    int offset = GADN->getOffset();
    std::cerr << "<";
    WriteAsOperand(std::cerr, GADN->getGlobal()) << ">";
    if (offset > 0)
      std::cerr << " + " << offset;
    else
      std::cerr << " " << offset;
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(this)) {
    std::cerr << "<" << FIDN->getIndex() << ">";
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(this)){
    int offset = CP->getOffset();
    std::cerr << "<" << *CP->get() << ">";
    if (offset > 0)
      std::cerr << " + " << offset;
    else
      std::cerr << " " << offset;
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(this)) {
    std::cerr << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      std::cerr << LBB->getName() << " ";
    std::cerr << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(this)) {
    if (G && R->getReg() && MRegisterInfo::isPhysicalRegister(R->getReg())) {
      std::cerr << " " <<G->getTarget().getRegisterInfo()->getName(R->getReg());
    } else {
      std::cerr << " #" << R->getReg();
    }
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(this)) {
    std::cerr << "'" << ES->getSymbol() << "'";
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(this)) {
    if (M->getValue())
      std::cerr << "<" << M->getValue() << ":" << M->getOffset() << ">";
    else
      std::cerr << "<null:" << M->getOffset() << ">";
  } else if (const VTSDNode *N = dyn_cast<VTSDNode>(this)) {
    std::cerr << ":" << getValueTypeString(N->getVT());
  }
}

static void DumpNodes(const SDNode *N, unsigned indent, const SelectionDAG *G) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (N->getOperand(i).Val->hasOneUse())
      DumpNodes(N->getOperand(i).Val, indent+2, G);
    else
      std::cerr << "\n" << std::string(indent+2, ' ')
                << (void*)N->getOperand(i).Val << ": <multiple use>";


  std::cerr << "\n" << std::string(indent, ' ');
  N->dump(G);
}

void SelectionDAG::dump() const {
  std::cerr << "SelectionDAG has " << AllNodes.size() << " nodes:";
  std::vector<const SDNode*> Nodes;
  for (allnodes_const_iterator I = allnodes_begin(), E = allnodes_end();
       I != E; ++I)
    Nodes.push_back(I);
  
  std::sort(Nodes.begin(), Nodes.end());

  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    if (!Nodes[i]->hasOneUse() && Nodes[i] != getRoot().Val)
      DumpNodes(Nodes[i], 2, this);
  }

  DumpNodes(getRoot().Val, 2, this);

  std::cerr << "\n\n";
}

/// InsertISelMapEntry - A helper function to insert a key / element pair
/// into a SDOperand to SDOperand map. This is added to avoid the map
/// insertion operator from being inlined.
void SelectionDAG::InsertISelMapEntry(std::map<SDOperand, SDOperand> &Map,
                                      SDNode *Key, unsigned KeyResNo,
                                      SDNode *Element, unsigned ElementResNo) {
  Map.insert(std::make_pair(SDOperand(Key, KeyResNo),
                            SDOperand(Element, ElementResNo)));
}
