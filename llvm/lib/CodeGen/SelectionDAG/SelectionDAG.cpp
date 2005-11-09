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
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <iostream>
#include <set>
#include <cmath>
#include <algorithm>
using namespace llvm;

static bool isCommutativeBinOp(unsigned Opcode) {
  switch (Opcode) {
  case ISD::ADD:
  case ISD::MUL:
  case ISD::FADD:
  case ISD::FMUL:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: return true;
  default: return false; // FIXME: Need commutative info for user ops!
  }
}

static bool isAssociativeBinOp(unsigned Opcode) {
  switch (Opcode) {
  case ISD::ADD:
  case ISD::MUL:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: return true;
  default: return false; // FIXME: Need associative info for user ops!
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
//                              ISD Class
//===----------------------------------------------------------------------===//

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
    Op &= ~16;     // Clear the N bit.
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
  return ISD::CondCode(Op1 & Op2);
}

const TargetMachine &SelectionDAG::getTarget() const {
  return TLI.getTargetMachine();
}

//===----------------------------------------------------------------------===//
//                              SelectionDAG Class
//===----------------------------------------------------------------------===//

/// RemoveDeadNodes - This method deletes all unreachable nodes in the
/// SelectionDAG, including nodes (like loads) that have uses of their token
/// chain but no other uses and no side effect.  If a node is passed in as an
/// argument, it is used as the seed for node deletion.
void SelectionDAG::RemoveDeadNodes(SDNode *N) {
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted.
  HandleSDNode Dummy(getRoot());

  bool MadeChange = false;
  
  // If we have a hint to start from, use it.
  if (N && N->use_empty()) {
    DestroyDeadNode(N);
    MadeChange = true;
  }

  for (allnodes_iterator I = allnodes_begin(), E = allnodes_end(); I != E; ++I)
    if (I->use_empty() && I->getOpcode() != 65535) {
      // Node is dead, recursively delete newly dead uses.
      DestroyDeadNode(I);
      MadeChange = true;
    }
  
  // Walk the nodes list, removing the nodes we've marked as dead.
  if (MadeChange) {
    for (allnodes_iterator I = allnodes_begin(), E = allnodes_end(); I != E; ) {
      SDNode *N = I++;
      if (N->use_empty())
        AllNodes.erase(N);
    }
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(Dummy.getValue());
}

/// DestroyDeadNode - We know that N is dead.  Nuke it from the CSE maps for the
/// graph.  If it is the last user of any of its operands, recursively process
/// them the same way.
/// 
void SelectionDAG::DestroyDeadNode(SDNode *N) {
  // Okay, we really are going to delete this node.  First take this out of the
  // appropriate CSE map.
  RemoveNodeFromCSEMaps(N);
  
  // Next, brutally remove the operand list.  This is safe to do, as there are
  // no cycles in the graph.
  for (SDNode::op_iterator I = N->op_begin(), E = N->op_end(); I != E; ++I) {
    SDNode *O = I->Val;
    O->removeUser(N);
    
    // Now that we removed this operand, see if there are no uses of it left.
    if (O->use_empty())
      DestroyDeadNode(O);
  }
  delete[] N->OperandList;
  N->OperandList = 0;
  N->NumOperands = 0;

  // Mark the node as dead.
  N->MorphNodeTo(65535);
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
  case ISD::Constant:
    Erased = Constants.erase(std::make_pair(cast<ConstantSDNode>(N)->getValue(),
                                            N->getValueType(0)));
    break;
  case ISD::TargetConstant:
    Erased = TargetConstants.erase(std::make_pair(
                                    cast<ConstantSDNode>(N)->getValue(),
                                                  N->getValueType(0)));
    break;
  case ISD::ConstantFP: {
    uint64_t V = DoubleToBits(cast<ConstantFPSDNode>(N)->getValue());
    Erased = ConstantFPs.erase(std::make_pair(V, N->getValueType(0)));
    break;
  }
  case ISD::CONDCODE:
    assert(CondCodeNodes[cast<CondCodeSDNode>(N)->get()] &&
           "Cond code doesn't exist!");
    Erased = CondCodeNodes[cast<CondCodeSDNode>(N)->get()] != 0;
    CondCodeNodes[cast<CondCodeSDNode>(N)->get()] = 0;
    break;
  case ISD::GlobalAddress:
    Erased = GlobalValues.erase(cast<GlobalAddressSDNode>(N)->getGlobal());
    break;
  case ISD::TargetGlobalAddress:
    Erased =TargetGlobalValues.erase(cast<GlobalAddressSDNode>(N)->getGlobal());
    break;
  case ISD::FrameIndex:
    Erased = FrameIndices.erase(cast<FrameIndexSDNode>(N)->getIndex());
    break;
  case ISD::TargetFrameIndex:
    Erased = TargetFrameIndices.erase(cast<FrameIndexSDNode>(N)->getIndex());
    break;
  case ISD::ConstantPool:
    Erased = ConstantPoolIndices.erase(cast<ConstantPoolSDNode>(N)->get());
    break;
  case ISD::TargetConstantPool:
    Erased =TargetConstantPoolIndices.erase(cast<ConstantPoolSDNode>(N)->get());
    break;
  case ISD::BasicBlock:
    Erased = BBNodes.erase(cast<BasicBlockSDNode>(N)->getBasicBlock());
    break;
  case ISD::ExternalSymbol:
    Erased = ExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::TargetExternalSymbol:
    Erased = TargetExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;
  case ISD::VALUETYPE:
    Erased = ValueTypeNodes[cast<VTSDNode>(N)->getVT()] != 0;
    ValueTypeNodes[cast<VTSDNode>(N)->getVT()] = 0;
    break;
  case ISD::Register:
    Erased = RegNodes.erase(std::make_pair(cast<RegisterSDNode>(N)->getReg(),
                                           N->getValueType(0)));
    break;
  case ISD::SRCVALUE: {
    SrcValueSDNode *SVN = cast<SrcValueSDNode>(N);
    Erased =ValueNodes.erase(std::make_pair(SVN->getValue(), SVN->getOffset()));
    break;
  }    
  case ISD::LOAD:
    Erased = Loads.erase(std::make_pair(N->getOperand(1),
                                        std::make_pair(N->getOperand(0),
                                                       N->getValueType(0))));
    break;
  default:
    if (N->getNumValues() == 1) {
      if (N->getNumOperands() == 0) {
        Erased = NullaryOps.erase(std::make_pair(N->getOpcode(),
                                                 N->getValueType(0)));
      } else if (N->getNumOperands() == 1) {
        Erased = 
          UnaryOps.erase(std::make_pair(N->getOpcode(),
                                        std::make_pair(N->getOperand(0),
                                                       N->getValueType(0))));
      } else if (N->getNumOperands() == 2) {
        Erased = 
          BinaryOps.erase(std::make_pair(N->getOpcode(),
                                         std::make_pair(N->getOperand(0),
                                                        N->getOperand(1))));
      } else { 
        std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
        Erased = 
          OneResultNodes.erase(std::make_pair(N->getOpcode(),
                                              std::make_pair(N->getValueType(0),
                                                             Ops)));
      }
    } else {
      // Remove the node from the ArbitraryNodes map.
      std::vector<MVT::ValueType> RV(N->value_begin(), N->value_end());
      std::vector<SDOperand>     Ops(N->op_begin(), N->op_end());
      Erased =
        ArbitraryNodes.erase(std::make_pair(N->getOpcode(),
                                            std::make_pair(RV, Ops)));
    }
    break;
  }
#ifndef NDEBUG
  // Verify that the node was actually in one of the CSE maps, unless it has a 
  // flag result (which cannot be CSE'd) or is one of the special cases that are
  // not subject to CSE.
  if (!Erased && N->getValueType(N->getNumValues()-1) != MVT::Flag &&
      N->getOpcode() != ISD::CALL && N->getOpcode() != ISD::CALLSEQ_START &&
      N->getOpcode() != ISD::CALLSEQ_END && !N->isTargetOpcode()) {
    
    N->dump();
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
  if (N->getOpcode() == ISD::LOAD) {
    SDNode *&L = Loads[std::make_pair(N->getOperand(1),
                                      std::make_pair(N->getOperand(0),
                                                     N->getValueType(0)))];
    if (L) return L;
    L = N;
  } else if (N->getOpcode() == ISD::HANDLENODE) {
    return 0;  // never add it.
  } else if (N->getNumOperands() == 1) {
    SDNode *&U = UnaryOps[std::make_pair(N->getOpcode(),
                                         std::make_pair(N->getOperand(0),
                                                        N->getValueType(0)))];
    if (U) return U;
    U = N;
  } else if (N->getNumOperands() == 2) {
    SDNode *&B = BinaryOps[std::make_pair(N->getOpcode(),
                                          std::make_pair(N->getOperand(0),
                                                         N->getOperand(1)))];
    if (B) return B;
    B = N;
  } else if (N->getNumValues() == 1) {
    std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
    SDNode *&ORN = OneResultNodes[std::make_pair(N->getOpcode(),
                                  std::make_pair(N->getValueType(0), Ops))];
    if (ORN) return ORN;
    ORN = N;
  } else {
    // Remove the node from the ArbitraryNodes map.
    std::vector<MVT::ValueType> RV(N->value_begin(), N->value_end());
    std::vector<SDOperand>     Ops(N->op_begin(), N->op_end());
    SDNode *&AN = ArbitraryNodes[std::make_pair(N->getOpcode(),
                                                std::make_pair(RV, Ops))];
    if (AN) return AN;
    AN = N;
  }
  return 0;
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

SDOperand SelectionDAG::getConstant(uint64_t Val, MVT::ValueType VT) {
  assert(MVT::isInteger(VT) && "Cannot create FP integer constant!");
  // Mask out any bits that are not valid for this constant.
  if (VT != MVT::i64)
    Val &= ((uint64_t)1 << MVT::getSizeInBits(VT)) - 1;

  SDNode *&N = Constants[std::make_pair(Val, VT)];
  if (N) return SDOperand(N, 0);
  N = new ConstantSDNode(false, Val, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTargetConstant(uint64_t Val, MVT::ValueType VT) {
  assert(MVT::isInteger(VT) && "Cannot create FP integer constant!");
  // Mask out any bits that are not valid for this constant.
  if (VT != MVT::i64)
    Val &= ((uint64_t)1 << MVT::getSizeInBits(VT)) - 1;
  
  SDNode *&N = TargetConstants[std::make_pair(Val, VT)];
  if (N) return SDOperand(N, 0);
  N = new ConstantSDNode(true, Val, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstantFP(double Val, MVT::ValueType VT) {
  assert(MVT::isFloatingPoint(VT) && "Cannot create integer FP constant!");
  if (VT == MVT::f32)
    Val = (float)Val;  // Mask out extra precision.

  // Do the map lookup using the actual bit pattern for the floating point
  // value, so that we don't have problems with 0.0 comparing equal to -0.0, and
  // we don't have issues with SNANs.
  SDNode *&N = ConstantFPs[std::make_pair(DoubleToBits(Val), VT)];
  if (N) return SDOperand(N, 0);
  N = new ConstantFPSDNode(Val, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}



SDOperand SelectionDAG::getGlobalAddress(const GlobalValue *GV,
                                         MVT::ValueType VT) {
  SDNode *&N = GlobalValues[GV];
  if (N) return SDOperand(N, 0);
  N = new GlobalAddressSDNode(false, GV, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTargetGlobalAddress(const GlobalValue *GV,
                                               MVT::ValueType VT) {
  SDNode *&N = TargetGlobalValues[GV];
  if (N) return SDOperand(N, 0);
  N = new GlobalAddressSDNode(true, GV, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getFrameIndex(int FI, MVT::ValueType VT) {
  SDNode *&N = FrameIndices[FI];
  if (N) return SDOperand(N, 0);
  N = new FrameIndexSDNode(FI, VT, false);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTargetFrameIndex(int FI, MVT::ValueType VT) {
  SDNode *&N = TargetFrameIndices[FI];
  if (N) return SDOperand(N, 0);
  N = new FrameIndexSDNode(FI, VT, true);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstantPool(Constant *C, MVT::ValueType VT) {
  SDNode *&N = ConstantPoolIndices[C];
  if (N) return SDOperand(N, 0);
  N = new ConstantPoolSDNode(C, VT, false);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getTargetConstantPool(Constant *C, MVT::ValueType VT) {
  SDNode *&N = TargetConstantPoolIndices[C];
  if (N) return SDOperand(N, 0);
  N = new ConstantPoolSDNode(C, VT, true);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getBasicBlock(MachineBasicBlock *MBB) {
  SDNode *&N = BBNodes[MBB];
  if (N) return SDOperand(N, 0);
  N = new BasicBlockSDNode(MBB);
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

SDOperand SelectionDAG::getTargetExternalSymbol(const char *Sym, MVT::ValueType VT) {
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
  RegisterSDNode *&Reg = RegNodes[std::make_pair(RegNo, VT)];
  if (!Reg) {
    Reg = new RegisterSDNode(RegNo, VT);
    AllNodes.push_back(Reg);
  }
  return SDOperand(Reg, 0);
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
  SDNode *&N = NullaryOps[std::make_pair(Opcode, VT)];
  if (!N) {
    N = new SDNode(Opcode, VT);
    AllNodes.push_back(N);
  }
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand Operand) {
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
    }
  }

  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand.Val))
    switch (Opcode) {
    case ISD::FNEG:
      return getConstantFP(-C->getValue(), VT);
    case ISD::FP_ROUND:
    case ISD::FP_EXTEND:
      return getConstantFP(C->getValue(), VT);
    case ISD::FP_TO_SINT:
      return getConstant((int64_t)C->getValue(), VT);
    case ISD::FP_TO_UINT:
      return getConstant((uint64_t)C->getValue(), VT);
    }

  unsigned OpOpcode = Operand.Val->getOpcode();
  switch (Opcode) {
  case ISD::TokenFactor:
    return Operand;         // Factor of one node?  No factor.
  case ISD::SIGN_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ZERO_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    if (OpOpcode == ISD::ZERO_EXTEND)   // (zext (zext x)) -> (zext x)
      return getNode(ISD::ZERO_EXTEND, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ANY_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND)
      // (ext (zext x)) -> (zext x)  and  (ext (sext x)) -> (sext x)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::TRUNCATE:
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
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
  if (VT != MVT::Flag) { // Don't CSE flag producing nodes
    SDNode *&E = UnaryOps[std::make_pair(Opcode, std::make_pair(Operand, VT))];
    if (E) return SDOperand(E, 0);
    E = N = new SDNode(Opcode, Operand);
  } else {
    N = new SDNode(Opcode, Operand);
  }
  N->setValueTypes(VT);
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

  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
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
      default: break;
      }
    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1CFP, N2CFP);
        std::swap(N1, N2);
      }
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
  if (Opcode != ISD::CALLSEQ_START && Opcode != ISD::CALLSEQ_END &&
      VT != MVT::Flag) {
    SDNode *&BON = BinaryOps[std::make_pair(Opcode, std::make_pair(N1, N2))];
    if (BON) return SDOperand(BON, 0);

    BON = N = new SDNode(Opcode, N1, N2);
  } else {
    N = new SDNode(Opcode, N1, N2);
  }

  N->setValueTypes(VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

// setAdjCallChain - This method changes the token chain of an
// CALLSEQ_START/END node to be the specified operand.
void SDNode::setAdjCallChain(SDOperand N) {
  assert(N.getValueType() == MVT::Other);
  assert((getOpcode() == ISD::CALLSEQ_START ||
          getOpcode() == ISD::CALLSEQ_END) && "Cannot adjust this node!");

  OperandList[0].Val->removeUser(this);
  OperandList[0] = N;
  OperandList[0].Val->Uses.push_back(this);
}



SDOperand SelectionDAG::getLoad(MVT::ValueType VT,
                                SDOperand Chain, SDOperand Ptr,
                                SDOperand SV) {
  SDNode *&N = Loads[std::make_pair(Ptr, std::make_pair(Chain, VT))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(ISD::LOAD, Chain, Ptr, SV);

  // Loads have a token chain.
  setNodeValueTypes(N, VT, MVT::Other);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getExtLoad(unsigned Opcode, MVT::ValueType VT,
                                   SDOperand Chain, SDOperand Ptr, SDOperand SV,
                                   MVT::ValueType EVT) {
  std::vector<SDOperand> Ops;
  Ops.reserve(4);
  Ops.push_back(Chain);
  Ops.push_back(Ptr);
  Ops.push_back(SV);
  Ops.push_back(getValueType(EVT));
  std::vector<MVT::ValueType> VTs;
  VTs.reserve(2);
  VTs.push_back(VT); VTs.push_back(MVT::Other);  // Add token chain.
  return getNode(Opcode, VTs, Ops);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3) {
  // Perform various simplifications.
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.Val);
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
  }

  std::vector<SDOperand> Ops;
  Ops.reserve(3);
  Ops.push_back(N1);
  Ops.push_back(N2);
  Ops.push_back(N3);

  // Memoize node if it doesn't produce a flag.
  SDNode *N;
  if (VT != MVT::Flag) {
    SDNode *&E = OneResultNodes[std::make_pair(Opcode,std::make_pair(VT, Ops))];
    if (E) return SDOperand(E, 0);
    E = N = new SDNode(Opcode, N1, N2, N3);
  } else {
    N = new SDNode(Opcode, N1, N2, N3);
  }
  N->setValueTypes(VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4) {
  std::vector<SDOperand> Ops;
  Ops.reserve(4);
  Ops.push_back(N1);
  Ops.push_back(N2);
  Ops.push_back(N3);
  Ops.push_back(N4);
  return getNode(Opcode, VT, Ops);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3,
                                SDOperand N4, SDOperand N5) {
  std::vector<SDOperand> Ops;
  Ops.reserve(5);
  Ops.push_back(N1);
  Ops.push_back(N2);
  Ops.push_back(N3);
  Ops.push_back(N4);
  Ops.push_back(N5);
  return getNode(Opcode, VT, Ops);
}


SDOperand SelectionDAG::getSrcValue(const Value *V, int Offset) {
  assert((!V || isa<PointerType>(V->getType())) &&
         "SrcValue is not a pointer?");
  SDNode *&N = ValueNodes[std::make_pair(V, Offset)];
  if (N) return SDOperand(N, 0);

  N = new SrcValueSDNode(V, Offset);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                std::vector<SDOperand> &Ops) {
  switch (Ops.size()) {
  case 0: return getNode(Opcode, VT);
  case 1: return getNode(Opcode, VT, Ops[0]);
  case 2: return getNode(Opcode, VT, Ops[0], Ops[1]);
  case 3: return getNode(Opcode, VT, Ops[0], Ops[1], Ops[2]);
  default: break;
  }
  
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(Ops[1].Val);
  switch (Opcode) {
  default: break;
  case ISD::BRCONDTWOWAY:
    if (N1C)
      if (N1C->getValue()) // Unconditional branch to true dest.
        return getNode(ISD::BR, MVT::Other, Ops[0], Ops[2]);
      else                 // Unconditional branch to false dest.
        return getNode(ISD::BR, MVT::Other, Ops[0], Ops[3]);
    break;
  case ISD::BRTWOWAY_CC:
    assert(Ops.size() == 6 && "BRTWOWAY_CC takes 6 operands!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "LHS and RHS of comparison must have same type!");
    break;
  case ISD::TRUNCSTORE: {
    assert(Ops.size() == 5 && "TRUNCSTORE takes 5 operands!");
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
    assert(Ops.size() == 5 && "SELECT_CC takes 5 operands!");
    assert(Ops[0].getValueType() == Ops[1].getValueType() &&
           "LHS and RHS of condition must have same type!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "True and False arms of SelectCC must have same type!");
    assert(Ops[2].getValueType() == VT &&
           "select_cc node must be of same type as true and false value!");
    break;
  }
  case ISD::BR_CC: {
    assert(Ops.size() == 5 && "BR_CC takes 5 operands!");
    assert(Ops[2].getValueType() == Ops[3].getValueType() &&
           "LHS/RHS of comparison should match types!");
    break;
  }
  }

  // Memoize nodes.
  SDNode *N;
  if (VT != MVT::Flag) {
    SDNode *&E =
      OneResultNodes[std::make_pair(Opcode, std::make_pair(VT, Ops))];
    if (E) return SDOperand(E, 0);
    E = N = new SDNode(Opcode, Ops);
  } else {
    N = new SDNode(Opcode, Ops);
  }
  N->setValueTypes(VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode,
                                std::vector<MVT::ValueType> &ResultTys,
                                std::vector<SDOperand> &Ops) {
  if (ResultTys.size() == 1)
    return getNode(Opcode, ResultTys[0], Ops);

  switch (Opcode) {
  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD: {
    MVT::ValueType EVT = cast<VTSDNode>(Ops[3])->getVT();
    assert(Ops.size() == 4 && ResultTys.size() == 2 && "Bad *EXTLOAD!");
    // If they are asking for an extending load from/to the same thing, return a
    // normal load.
    if (ResultTys[0] == EVT)
      return getLoad(ResultTys[0], Ops[0], Ops[1], Ops[2]);
    assert(EVT < ResultTys[0] &&
           "Should only be an extending load, not truncating!");
    assert((Opcode == ISD::EXTLOAD || MVT::isInteger(ResultTys[0])) &&
           "Cannot sign/zero extend a FP load!");
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
  if (ResultTys.back() != MVT::Flag) {
    SDNode *&E =
      ArbitraryNodes[std::make_pair(Opcode, std::make_pair(ResultTys, Ops))];
    if (E) return SDOperand(E, 0);
    E = N = new SDNode(Opcode, Ops);
  } else {
    N = new SDNode(Opcode, Ops);
  }
  setNodeValueTypes(N, ResultTys);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

void SelectionDAG::setNodeValueTypes(SDNode *N, 
                                     std::vector<MVT::ValueType> &RetVals) {
  switch (RetVals.size()) {
  case 0: return;
  case 1: N->setValueTypes(RetVals[0]); return;
  case 2: setNodeValueTypes(N, RetVals[0], RetVals[1]); return;
  default: break;
  }
  
  std::list<std::vector<MVT::ValueType> >::iterator I =
    std::find(VTList.begin(), VTList.end(), RetVals);
  if (I == VTList.end()) {
    VTList.push_front(RetVals);
    I = VTList.begin();
  }

  N->setValueTypes(&(*I)[0], I->size());
}

void SelectionDAG::setNodeValueTypes(SDNode *N, MVT::ValueType VT1, 
                                     MVT::ValueType VT2) {
  for (std::list<std::vector<MVT::ValueType> >::iterator I = VTList.begin(),
       E = VTList.end(); I != E; ++I) {
    if (I->size() == 2 && (*I)[0] == VT1 && (*I)[1] == VT2) {
      N->setValueTypes(&(*I)[0], 2);
      return;
    }
  }
  std::vector<MVT::ValueType> V;
  V.push_back(VT1);
  V.push_back(VT2);
  VTList.push_front(V);
  N->setValueTypes(&(*VTList.begin())[0], 2);
}


/// SelectNodeTo - These are used for target selectors to *mutate* the
/// specified node to have the specified return type, Target opcode, and
/// operands.  Note that target opcodes are stored as
/// ISD::BUILTIN_OP_END+TargetOpcode in the node opcode field.
void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                MVT::ValueType VT) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VT);
}
void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                MVT::ValueType VT, SDOperand Op1) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VT);
  N->setOperands(Op1);
}
void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                MVT::ValueType VT, SDOperand Op1,
                                SDOperand Op2) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VT);
  N->setOperands(Op1, Op2);
}
void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc, 
                                MVT::ValueType VT1, MVT::ValueType VT2,
                                SDOperand Op1, SDOperand Op2) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  setNodeValueTypes(N, VT1, VT2);
  N->setOperands(Op1, Op2);
}
void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                MVT::ValueType VT, SDOperand Op1,
                                SDOperand Op2, SDOperand Op3) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VT);
  N->setOperands(Op1, Op2, Op3);
}
void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                MVT::ValueType VT1, MVT::ValueType VT2,
                                SDOperand Op1, SDOperand Op2, SDOperand Op3) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  setNodeValueTypes(N, VT1, VT2);
  N->setOperands(Op1, Op2, Op3);
}

void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                MVT::ValueType VT, SDOperand Op1,
                                SDOperand Op2, SDOperand Op3, SDOperand Op4) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VT);
  N->setOperands(Op1, Op2, Op3, Op4);
}
void SelectionDAG::SelectNodeTo(SDNode *N, unsigned TargetOpc,
                                MVT::ValueType VT, SDOperand Op1,
                                SDOperand Op2, SDOperand Op3, SDOperand Op4,
                                SDOperand Op5) {
  RemoveNodeFromCSEMaps(N);
  N->MorphNodeTo(ISD::BUILTIN_OP_END+TargetOpc);
  N->setValueTypes(VT);
  N->setOperands(Op1, Op2, Op3, Op4, Op5);
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
                                      const std::vector<SDOperand> &To,
                                      std::vector<SDNode*> *Deleted) {
  assert(From->getNumValues() == To.size() &&
         "Incorrect number of values to replace with!");
  if (To.size() == 1 && To[0].Val->getNumValues() == 1) {
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


//===----------------------------------------------------------------------===//
//                              SDNode Class
//===----------------------------------------------------------------------===//


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
bool SDNode::hasNUsesOfValue(unsigned NUses, unsigned Value) {
  assert(Value < getNumValues() && "Bad value!");

  // If there is only one value, this is easy.
  if (getNumValues() == 1)
    return use_size() == NUses;
  if (Uses.size() < NUses) return false;

  SDOperand TheValue(this, Value);

  std::set<SDNode*> UsersHandled;

  for (std::vector<SDNode*>::iterator UI = Uses.begin(), E = Uses.end();
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


const char *SDNode::getOperationName(const SelectionDAG *G) const {
  switch (getOpcode()) {
  default:
    if (getOpcode() < ISD::BUILTIN_OP_END)
      return "<<Unknown DAG Node>>";
    else {
      if (G)
        if (const TargetInstrInfo *TII = G->getTarget().getInstrInfo())
          if (getOpcode()-ISD::BUILTIN_OP_END < TII->getNumOpcodes())
            return TII->getName(getOpcode()-ISD::BUILTIN_OP_END);
      return "<<Unknown Target Node>>";
    }
   
  case ISD::PCMARKER:      return "PCMarker";
  case ISD::SRCVALUE:      return "SrcValue";
  case ISD::VALUETYPE:     return "ValueType";
  case ISD::EntryToken:    return "EntryToken";
  case ISD::TokenFactor:   return "TokenFactor";
  case ISD::AssertSext:    return "AssertSext";
  case ISD::AssertZext:    return "AssertZext";
  case ISD::Constant:      return "Constant";
  case ISD::TargetConstant: return "TargetConstant";
  case ISD::ConstantFP:    return "ConstantFP";
  case ISD::GlobalAddress: return "GlobalAddress";
  case ISD::TargetGlobalAddress: return "TargetGlobalAddress";
  case ISD::FrameIndex:    return "FrameIndex";
  case ISD::TargetFrameIndex: return "TargetFrameIndex";
  case ISD::BasicBlock:    return "BasicBlock";
  case ISD::Register:      return "Register";
  case ISD::ExternalSymbol: return "ExternalSymbol";
  case ISD::TargetExternalSymbol: return "TargetExternalSymbol";
  case ISD::ConstantPool:  return "ConstantPool";
  case ISD::TargetConstantPool:  return "TargetConstantPool";
  case ISD::CopyToReg:     return "CopyToReg";
  case ISD::CopyFromReg:   return "CopyFromReg";
  case ISD::ImplicitDef:   return "ImplicitDef";
  case ISD::UNDEF:         return "undef";

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
  case ISD::FADD:   return "fadd";
  case ISD::FSUB:   return "fsub";
  case ISD::FMUL:   return "fmul";
  case ISD::FDIV:   return "fdiv";
  case ISD::FREM:   return "frem";
    
  case ISD::SETCC:       return "setcc";
  case ISD::SELECT:      return "select";
  case ISD::SELECT_CC:   return "select_cc";
  case ISD::ADD_PARTS:   return "add_parts";
  case ISD::SUB_PARTS:   return "sub_parts";
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

    // Control flow instructions
  case ISD::BR:      return "br";
  case ISD::BRCOND:  return "brcond";
  case ISD::BRCONDTWOWAY:  return "brcondtwoway";
  case ISD::BR_CC:  return "br_cc";
  case ISD::BRTWOWAY_CC:  return "brtwoway_cc";
  case ISD::RET:     return "ret";
  case ISD::CALL:    return "call";
  case ISD::TAILCALL:return "tailcall";
  case ISD::CALLSEQ_START:  return "callseq_start";
  case ISD::CALLSEQ_END:    return "callseq_end";

    // Other operators
  case ISD::LOAD:    return "load";
  case ISD::STORE:   return "store";
  case ISD::EXTLOAD:    return "extload";
  case ISD::SEXTLOAD:   return "sextload";
  case ISD::ZEXTLOAD:   return "zextload";
  case ISD::TRUNCSTORE: return "truncstore";

  case ISD::DYNAMIC_STACKALLOC: return "dynamic_stackalloc";
  case ISD::EXTRACT_ELEMENT: return "extract_element";
  case ISD::BUILD_PAIR: return "build_pair";
  case ISD::MEMSET:  return "memset";
  case ISD::MEMCPY:  return "memcpy";
  case ISD::MEMMOVE: return "memmove";

  // Bit counting
  case ISD::CTPOP:   return "ctpop";
  case ISD::CTTZ:    return "cttz";
  case ISD::CTLZ:    return "ctlz";

  // IO Intrinsics
  case ISD::READPORT: return "readport";
  case ISD::WRITEPORT: return "writeport";
  case ISD::READIO: return "readio";
  case ISD::WRITEIO: return "writeio";

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
    std::cerr << "<";
    WriteAsOperand(std::cerr, GADN->getGlobal()) << ">";
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(this)) {
    std::cerr << "<" << FIDN->getIndex() << ">";
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(this)){
    std::cerr << "<" << *CP->get() << ">";
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(this)) {
    std::cerr << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      std::cerr << LBB->getName() << " ";
    std::cerr << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(this)) {
    if (G && MRegisterInfo::isPhysicalRegister(R->getReg())) {
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

