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
#include <iostream>
#include <set>
#include <cmath>
using namespace llvm;

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

/// RemoveDeadNodes - This method deletes all unreachable nodes in the
/// SelectionDAG, including nodes (like loads) that have uses of their token
/// chain but no other uses and no side effect.  If a node is passed in as an
/// argument, it is used as the seed for node deletion.
void SelectionDAG::RemoveDeadNodes(SDNode *N) {
  std::set<SDNode*> AllNodeSet(AllNodes.begin(), AllNodes.end());

  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted.
  SDNode *DummyNode = new SDNode(ISD::EntryToken, getRoot());

  DeleteNodeIfDead(N, &AllNodeSet);

 Restart:
  unsigned NumNodes = AllNodeSet.size();
  for (std::set<SDNode*>::iterator I = AllNodeSet.begin(), E = AllNodeSet.end();
       I != E; ++I) {
    // Try to delete this node.
    DeleteNodeIfDead(*I, &AllNodeSet);

    // If we actually deleted any nodes, do not use invalid iterators in
    // AllNodeSet.
    if (AllNodeSet.size() != NumNodes)
      goto Restart;
  }

  // Restore AllNodes.
  if (AllNodes.size() != NumNodes)
    AllNodes.assign(AllNodeSet.begin(), AllNodeSet.end());

  // If the root changed (e.g. it was a dead load, update the root).
  setRoot(DummyNode->getOperand(0));

  // Now that we are done with the dummy node, delete it.
  DummyNode->getOperand(0).Val->removeUser(DummyNode);
  delete DummyNode;
}

void SelectionDAG::DeleteNodeIfDead(SDNode *N, void *NodeSet) {
  if (!N->use_empty())
    return;

  // Okay, we really are going to delete this node.  First take this out of the
  // appropriate CSE map.
  switch (N->getOpcode()) {
  case ISD::Constant:
    Constants.erase(std::make_pair(cast<ConstantSDNode>(N)->getValue(),
                                   N->getValueType(0)));
    break;
  case ISD::ConstantFP:
    ConstantFPs.erase(std::make_pair(cast<ConstantFPSDNode>(N)->getValue(),
                                     N->getValueType(0)));
    break;
  case ISD::GlobalAddress:
    GlobalValues.erase(cast<GlobalAddressSDNode>(N)->getGlobal());
    break;
  case ISD::FrameIndex:
    FrameIndices.erase(cast<FrameIndexSDNode>(N)->getIndex());
    break;
  case ISD::ConstantPool:
    ConstantPoolIndices.erase(cast<ConstantPoolSDNode>(N)->getIndex());
    break;
  case ISD::BasicBlock:
    BBNodes.erase(cast<BasicBlockSDNode>(N)->getBasicBlock());
    break;
  case ISD::ExternalSymbol:
    ExternalSymbols.erase(cast<ExternalSymbolSDNode>(N)->getSymbol());
    break;

  case ISD::LOAD:
    Loads.erase(std::make_pair(N->getOperand(1),
                               std::make_pair(N->getOperand(0),
                                              N->getValueType(0))));
    break;
  case ISD::SETCC:
    SetCCs.erase(std::make_pair(std::make_pair(N->getOperand(0),
                                               N->getOperand(1)),
                                cast<SetCCSDNode>(N)->getCondition()));
    break;
  default:
    if (N->getNumOperands() == 1)
      UnaryOps.erase(std::make_pair(N->getOpcode(),
                                    std::make_pair(N->getOperand(0),
                                                   N->getValueType(0))));
    else if (N->getNumOperands() == 2)
      BinaryOps.erase(std::make_pair(N->getOpcode(),
                                     std::make_pair(N->getOperand(0),
                                                    N->getOperand(1))));
    break;
  }

  // Next, brutally remove the operand list.
  while (!N->Operands.empty()) {
    SDNode *O = N->Operands.back().Val;
    N->Operands.pop_back();
    O->removeUser(N);

    // Now that we removed this operand, see if there are no uses of it left.
    DeleteNodeIfDead(O, NodeSet);
  }
  
  // Remove the node from the nodes set and delete it.
  std::set<SDNode*> &AllNodeSet = *(std::set<SDNode*>*)NodeSet;
  AllNodeSet.erase(N);

  // Now that the node is gone, check to see if any of the operands of this node
  // are dead now.
  delete N;
}


SelectionDAG::~SelectionDAG() {
  for (unsigned i = 0, e = AllNodes.size(); i != e; ++i)
    delete AllNodes[i];
}

SDOperand SelectionDAG::getConstant(uint64_t Val, MVT::ValueType VT) {
  assert(MVT::isInteger(VT) && "Cannot create FP integer constant!");
  // Mask out any bits that are not valid for this constant.
  Val &= (1ULL << MVT::getSizeInBits(VT)) - 1;

  SDNode *&N = Constants[std::make_pair(Val, VT)];
  if (N) return SDOperand(N, 0);
  N = new ConstantSDNode(Val, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstantFP(double Val, MVT::ValueType VT) {
  assert(MVT::isFloatingPoint(VT) && "Cannot create integer FP constant!");
  if (VT == MVT::f32)
    Val = (float)Val;  // Mask out extra precision.

  SDNode *&N = ConstantFPs[std::make_pair(Val, VT)];
  if (N) return SDOperand(N, 0);
  N = new ConstantFPSDNode(Val, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}



SDOperand SelectionDAG::getGlobalAddress(const GlobalValue *GV,
                                         MVT::ValueType VT) {
  SDNode *&N = GlobalValues[GV];
  if (N) return SDOperand(N, 0);
  N = new GlobalAddressSDNode(GV,VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getFrameIndex(int FI, MVT::ValueType VT) {
  SDNode *&N = FrameIndices[FI];
  if (N) return SDOperand(N, 0);
  N = new FrameIndexSDNode(FI, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getConstantPool(unsigned CPIdx, MVT::ValueType VT) {
  SDNode *N = ConstantPoolIndices[CPIdx];
  if (N) return SDOperand(N, 0);
  N = new ConstantPoolSDNode(CPIdx, VT);
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

SDOperand SelectionDAG::getExternalSymbol(const char *Sym, MVT::ValueType VT) {
  SDNode *&N = ExternalSymbols[Sym];
  if (N) return SDOperand(N, 0);
  N = new ExternalSymbolSDNode(Sym, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getSetCC(ISD::CondCode Cond, SDOperand N1,
                                 SDOperand N2) {
  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return getConstant(0, MVT::i1);
  case ISD::SETTRUE:
  case ISD::SETTRUE2:  return getConstant(1, MVT::i1);
  }

  if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val))
    if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val)) {
      uint64_t C1 = N1C->getValue(), C2 = N2C->getValue();
      
      // Sign extend the operands if required
      if (ISD::isSignedIntSetCC(Cond)) {
        C1 = N1C->getSignExtended();
        C2 = N2C->getSignExtended();
      }

      switch (Cond) {
      default: assert(0 && "Unknown integer setcc!");
      case ISD::SETEQ:  return getConstant(C1 == C2, MVT::i1);
      case ISD::SETNE:  return getConstant(C1 != C2, MVT::i1);
      case ISD::SETULT: return getConstant(C1 <  C2, MVT::i1);
      case ISD::SETUGT: return getConstant(C1 >  C2, MVT::i1);
      case ISD::SETULE: return getConstant(C1 <= C2, MVT::i1);
      case ISD::SETUGE: return getConstant(C1 >= C2, MVT::i1);
      case ISD::SETLT:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      case ISD::SETGT:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      case ISD::SETLE:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      case ISD::SETGE:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      Cond = ISD::getSetCCSwappedOperands(Cond);
      std::swap(N1, N2);
    }

  if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1.Val))
    if (ConstantFPSDNode *N2C = dyn_cast<ConstantFPSDNode>(N2.Val)) {
      double C1 = N1C->getValue(), C2 = N2C->getValue();
      
      switch (Cond) {
      default: break; // FIXME: Implement the rest of these!
      case ISD::SETEQ:  return getConstant(C1 == C2, MVT::i1);
      case ISD::SETNE:  return getConstant(C1 != C2, MVT::i1);
      case ISD::SETLT:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      case ISD::SETGT:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      case ISD::SETLE:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      case ISD::SETGE:  return getConstant((int64_t)C1 < (int64_t)C2, MVT::i1);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      Cond = ISD::getSetCCSwappedOperands(Cond);
      std::swap(N1, N2);
    }

  if (N1 == N2) {
    // We can always fold X == Y for integer setcc's.
    if (MVT::isInteger(N1.getValueType()))
      return getConstant(ISD::isTrueWhenEqual(Cond), MVT::i1);
    unsigned UOF = ISD::getUnorderedFlavor(Cond);
    if (UOF == 2)   // FP operators that are undefined on NaNs.
      return getConstant(ISD::isTrueWhenEqual(Cond), MVT::i1);
    if (UOF == ISD::isTrueWhenEqual(Cond))
      return getConstant(UOF, MVT::i1);
    // Otherwise, we can't fold it.  However, we can simplify it to SETUO/SETO
    // if it is not already.
    Cond = UOF == 0 ? ISD::SETUO : ISD::SETO;
  }


  SetCCSDNode *&N = SetCCs[std::make_pair(std::make_pair(N1, N2), Cond)];
  if (N) return SDOperand(N, 0);
  N = new SetCCSDNode(Cond, N1, N2);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}



/// getNode - Gets or creates the specified node.
///
SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT) {
  SDNode *N = new SDNode(Opcode, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

static const Type *getTypeFor(MVT::ValueType VT) {
  switch (VT) {
  default: assert(0 && "Unknown MVT!");
  case MVT::i1: return Type::BoolTy;
  case MVT::i8: return Type::UByteTy;
  case MVT::i16: return Type::UShortTy;
  case MVT::i32: return Type::UIntTy;
  case MVT::i64: return Type::ULongTy;
  case MVT::f32: return Type::FloatTy;
  case MVT::f64: return Type::DoubleTy;
  }
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand Operand) {
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Operand.Val)) {
    uint64_t Val = C->getValue();
    switch (Opcode) {
    default: break;
    case ISD::SIGN_EXTEND: return getConstant(C->getSignExtended(), VT);
    case ISD::ZERO_EXTEND: return getConstant(Val, VT);
    case ISD::TRUNCATE:    return getConstant(Val, VT);
    }
  }

  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Operand.Val))
    switch (Opcode) {
    case ISD::FP_ROUND:
    case ISD::FP_EXTEND:
      return getConstantFP(C->getValue(), VT);
    }

  unsigned OpOpcode = Operand.Val->getOpcode();
  switch (Opcode) {
  case ISD::SIGN_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::ZERO_EXTEND:
    if (Operand.getValueType() == VT) return Operand;   // noop extension
    if (OpOpcode == ISD::SIGN_EXTEND || OpOpcode == ISD::ZERO_EXTEND)
      return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
    break;
  case ISD::TRUNCATE:
    if (Operand.getValueType() == VT) return Operand;   // noop truncate
    if (OpOpcode == ISD::TRUNCATE)
      return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
    else if (OpOpcode == ISD::ZERO_EXTEND || OpOpcode == ISD::SIGN_EXTEND) {
      // If the source is smaller than the dest, we still need an extend.
      if (Operand.Val->getOperand(0).getValueType() < VT)
        return getNode(OpOpcode, VT, Operand.Val->getOperand(0));
      else if (Operand.Val->getOperand(0).getValueType() > VT)
        return getNode(ISD::TRUNCATE, VT, Operand.Val->getOperand(0));
      else
        return Operand.Val->getOperand(0);
    }
    break;
  }

  SDNode *&N = UnaryOps[std::make_pair(Opcode, std::make_pair(Operand, VT))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(Opcode, Operand);
  N->setValueTypes(VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

static bool isCommutativeBinOp(unsigned Opcode) {
  switch (Opcode) {
  case ISD::ADD:
  case ISD::MUL:
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

static unsigned ExactLog2(uint64_t Val) {
  unsigned Count = 0;
  while (Val != 1) {
    Val >>= 1;
    ++Count;
  }
  return Count;
}

// isInvertibleForFree - Return true if there is no cost to emitting the logical
// inverse of this node.
static bool isInvertibleForFree(SDOperand N) {
  if (isa<ConstantSDNode>(N.Val)) return true;
  if (isa<SetCCSDNode>(N.Val) && N.Val->hasOneUse())
    return true;
  return false;  
}


SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2) {
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
      default: break;
      }

    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1C, N2C);
        std::swap(N1, N2);
      }
    }
  }

  if (N2C) {
    uint64_t C2 = N2C->getValue();

    switch (Opcode) {
    case ISD::ADD:
      if (!C2) return N1;         // add X, 0 -> X
      break;
    case ISD::SUB:
      if (!C2) return N1;         // sub X, 0 -> X
      break;
    case ISD::MUL:
      if (!C2) return N2;         // mul X, 0 -> 0
      if (N2C->isAllOnesValue()) // mul X, -1 -> 0-X
        return getNode(ISD::SUB, VT, getConstant(0, VT), N1);

      // FIXME: This should only be done if the target supports shift
      // operations.
      if ((C2 & C2-1) == 0) {
        SDOperand ShAmt = getConstant(ExactLog2(C2), MVT::i8);
        return getNode(ISD::SHL, VT, N1, ShAmt);
      }
      break;

    case ISD::UDIV:
      // FIXME: This should only be done if the target supports shift
      // operations.
      if ((C2 & C2-1) == 0 && C2) {
        SDOperand ShAmt = getConstant(ExactLog2(C2), MVT::i8);
        return getNode(ISD::SRL, VT, N1, ShAmt);
      }
      break;

    case ISD::AND:
      if (!C2) return N2;         // X and 0 -> 0
      if (N2C->isAllOnesValue())
	return N1;                // X and -1 -> X
      break;
    case ISD::OR:
      if (!C2)return N1;          // X or 0 -> X
      if (N2C->isAllOnesValue())
	return N2;                // X or -1 -> -1
      break;
    case ISD::XOR:
      if (!C2) return N1;        // X xor 0 -> X
      if (N2C->isAllOnesValue()) {
        if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(N1.Val)){
          // !(X op Y) -> (X !op Y)
          bool isInteger = MVT::isInteger(SetCC->getOperand(0).getValueType());
          return getSetCC(ISD::getSetCCInverse(SetCC->getCondition(),isInteger),
                          SetCC->getOperand(0), SetCC->getOperand(1));
        } else if (N1.getOpcode() == ISD::AND || N1.getOpcode() == ISD::OR) {
          SDNode *Op = N1.Val;
          // !(X or Y) -> (!X and !Y) iff X or Y are freely invertible
          // !(X and Y) -> (!X or !Y) iff X or Y are freely invertible
          SDOperand LHS = Op->getOperand(0), RHS = Op->getOperand(1);
          if (isInvertibleForFree(RHS) || isInvertibleForFree(LHS)) {
            LHS = getNode(ISD::XOR, VT, LHS, N2);  // RHS = ~LHS
            RHS = getNode(ISD::XOR, VT, RHS, N2);  // RHS = ~RHS
            if (Op->getOpcode() == ISD::AND)
              return getNode(ISD::OR, VT, LHS, RHS);
            return getNode(ISD::AND, VT, LHS, RHS);
          }
        }
	// X xor -1 -> not(x)  ?
      }
      break;
    }

    // Reassociate ((X op C1) op C2) if possible.
    if (N1.getOpcode() == Opcode && isAssociativeBinOp(Opcode))
      if (ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N1.Val->getOperand(1)))
        return getNode(Opcode, VT, N1.Val->getOperand(0),
                       getNode(Opcode, VT, N2, N1.Val->getOperand(1)));
  }

  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1.Val);
  ConstantFPSDNode *N2CFP = dyn_cast<ConstantFPSDNode>(N2.Val);
  if (N1CFP)
    if (N2CFP) {
      double C1 = N1CFP->getValue(), C2 = N2CFP->getValue();
      switch (Opcode) {
      case ISD::ADD: return getConstantFP(C1 + C2, VT);
      case ISD::SUB: return getConstantFP(C1 - C2, VT);
      case ISD::MUL: return getConstantFP(C1 * C2, VT);
      case ISD::SDIV:
        if (C2) return getConstantFP(C1 / C2, VT);
        break;
      case ISD::SREM :
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

  // Finally, fold operations that do not require constants.
  switch (Opcode) {
  case ISD::AND:
  case ISD::OR:
    if (SetCCSDNode *LHS = dyn_cast<SetCCSDNode>(N1.Val))
      if (SetCCSDNode *RHS = dyn_cast<SetCCSDNode>(N2.Val)) {
        SDOperand LL = LHS->getOperand(0), RL = RHS->getOperand(0);
        SDOperand LR = LHS->getOperand(1), RR = RHS->getOperand(1);
        ISD::CondCode Op2 = RHS->getCondition();

        // (X op1 Y) | (Y op2 X) -> (X op1 Y) | (X swapop2 Y)
        if (LL == RR && LR == RL) {
          Op2 = ISD::getSetCCSwappedOperands(Op2);
          goto MatchedBackwards;
        }
      
        if (LL == RL && LR == RR) {
        MatchedBackwards:
          ISD::CondCode Result;
          bool isInteger = MVT::isInteger(LL.getValueType());
          if (Opcode == ISD::OR)
            Result = ISD::getSetCCOrOperation(LHS->getCondition(), Op2,
                                              isInteger);
          else
            Result = ISD::getSetCCAndOperation(LHS->getCondition(), Op2,
                                               isInteger);
          if (Result != ISD::SETCC_INVALID)
            return getSetCC(Result, LL, LR);
        }
      }
    break;
  case ISD::XOR:
    if (N1 == N2) return getConstant(0, VT);  // xor X, Y -> 0
    break;
  }

  SDNode *&N = BinaryOps[std::make_pair(Opcode, std::make_pair(N1, N2))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(Opcode, N1, N2);
  N->setValueTypes(VT);

  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getLoad(MVT::ValueType VT,
                                SDOperand Chain, SDOperand Ptr) {
  SDNode *&N = Loads[std::make_pair(Ptr, std::make_pair(Chain, VT))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(ISD::LOAD, Chain, Ptr);

  // Loads have a token chain.
  N->setValueTypes(VT, MVT::Other);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}


SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                SDOperand N1, SDOperand N2, SDOperand N3) {
  // Perform various simplifications.
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.Val);
  switch (Opcode) {
  case ISD::SELECT:
    if (N1C)
      if (N1C->getValue())
        return N2;             // select true, X, Y -> X
      else 
        return N3;             // select false, X, Y -> Y

    if (N2 == N3) return N2;   // select C, X, X -> X

    if (VT == MVT::i1) {  // Boolean SELECT
      if (N2C) {
        if (N3C) {
          if (N2C->getValue()) // select C, 1, 0 -> C
            return N1;
          return getNode(ISD::XOR, VT, N1, N3); // select C, 0, 1 -> ~C
        }

        if (N2C->getValue())   // select C, 1, X -> C | X
          return getNode(ISD::OR, VT, N1, N3);
        else                   // select C, 0, X -> ~C & X
          return getNode(ISD::AND, VT,
                         getNode(ISD::XOR, N1.getValueType(), N1,
                                 getConstant(1, N1.getValueType())), N3);
      } else if (N3C) {
        if (N3C->getValue())   // select C, X, 1 -> ~C | X
          return getNode(ISD::OR, VT,
                         getNode(ISD::XOR, N1.getValueType(), N1,
                                 getConstant(1, N1.getValueType())), N2);
        else                   // select C, X, 0 -> C & X
          return getNode(ISD::AND, VT, N1, N2);
      }
    }

    break;
  case ISD::BRCOND:
    if (N2C)
      if (N2C->getValue()) // Unconditional branch
        return getNode(ISD::BR, MVT::Other, N1, N3);
      else
        return N1;         // Never-taken branch
    break;
  }

  SDNode *N = new SDNode(Opcode, N1, N2, N3);
  switch (Opcode) {
  default: 
    N->setValueTypes(VT);
    break;
  case ISD::DYNAMIC_STACKALLOC: // DYNAMIC_STACKALLOC produces pointer and chain
    N->setValueTypes(VT, MVT::Other);
    break;
  }

  // FIXME: memoize NODES
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT,
                                std::vector<SDOperand> &Children) {
  switch (Children.size()) {
  case 0: return getNode(Opcode, VT);
  case 1: return getNode(Opcode, VT, Children[0]);
  case 2: return getNode(Opcode, VT, Children[0], Children[1]);
  case 3: return getNode(Opcode, VT, Children[0], Children[1], Children[2]);
  default:
    // FIXME: MEMOIZE!!
    SDNode *N = new SDNode(Opcode, Children);
    N->setValueTypes(VT);
    AllNodes.push_back(N);
    return SDOperand(N, 0);
  }
}



void SDNode::dump() const {
  std::cerr << (void*)this << ": ";

  for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
    if (i) std::cerr << ",";
    switch (getValueType(i)) {
    default: assert(0 && "Unknown value type!");
    case MVT::i1:    std::cerr << "i1"; break;
    case MVT::i8:    std::cerr << "i8"; break;
    case MVT::i16:   std::cerr << "i16"; break;
    case MVT::i32:   std::cerr << "i32"; break;
    case MVT::i64:   std::cerr << "i64"; break;
    case MVT::f32:   std::cerr << "f32"; break;
    case MVT::f64:   std::cerr << "f64"; break;
    case MVT::Other: std::cerr << "ch"; break;
    }
  }
  std::cerr << " = ";

  switch (getOpcode()) {
  default: std::cerr << "<<Unknown>>"; break;
  case ISD::EntryToken:    std::cerr << "EntryToken"; break;
  case ISD::Constant:      std::cerr << "Constant"; break;
  case ISD::ConstantFP:    std::cerr << "ConstantFP"; break;
  case ISD::GlobalAddress: std::cerr << "GlobalAddress"; break;
  case ISD::FrameIndex:    std::cerr << "FrameIndex"; break;
  case ISD::BasicBlock:    std::cerr << "BasicBlock"; break;
  case ISD::ExternalSymbol: std::cerr << "ExternalSymbol"; break;
  case ISD::ConstantPool:  std::cerr << "ConstantPoolIndex"; break;
  case ISD::CopyToReg:     std::cerr << "CopyToReg"; break;
  case ISD::CopyFromReg:   std::cerr << "CopyFromReg"; break;

  case ISD::ADD:    std::cerr << "add"; break;
  case ISD::SUB:    std::cerr << "sub"; break;
  case ISD::MUL:    std::cerr << "mul"; break;
  case ISD::SDIV:   std::cerr << "sdiv"; break;
  case ISD::UDIV:   std::cerr << "udiv"; break;
  case ISD::SREM:   std::cerr << "srem"; break;
  case ISD::UREM:   std::cerr << "urem"; break;
  case ISD::AND:    std::cerr << "and"; break;
  case ISD::OR:     std::cerr << "or"; break;
  case ISD::XOR:    std::cerr << "xor"; break;
  case ISD::SHL:    std::cerr << "shl"; break;
  case ISD::SRA:    std::cerr << "sra"; break;
  case ISD::SRL:    std::cerr << "srl"; break;

  case ISD::SETCC:  std::cerr << "setcc"; break;
  case ISD::SELECT: std::cerr << "select"; break;
  case ISD::ADDC:   std::cerr << "addc"; break;
  case ISD::SUBB:   std::cerr << "subb"; break;

    // Conversion operators.
  case ISD::SIGN_EXTEND: std::cerr << "sign_extend"; break;
  case ISD::ZERO_EXTEND: std::cerr << "zero_extend"; break;
  case ISD::TRUNCATE:    std::cerr << "truncate"; break;
  case ISD::FP_ROUND:    std::cerr << "fp_round"; break;
  case ISD::FP_EXTEND:   std::cerr << "fp_extend"; break;

    // Control flow instructions
  case ISD::BR:      std::cerr << "br"; break;
  case ISD::BRCOND:  std::cerr << "brcond"; break;
  case ISD::RET:     std::cerr << "ret"; break;
  case ISD::CALL:    std::cerr << "call"; break;
  case ISD::ADJCALLSTACKDOWN:  std::cerr << "adjcallstackdown"; break;
  case ISD::ADJCALLSTACKUP:    std::cerr << "adjcallstackup"; break;

    // Other operators
  case ISD::LOAD:    std::cerr << "load"; break;
  case ISD::STORE:   std::cerr << "store"; break;
  case ISD::DYNAMIC_STACKALLOC: std::cerr << "dynamic_stackalloc"; break;
  case ISD::EXTRACT_ELEMENT: std::cerr << "extract_element"; break;
  case ISD::BUILD_PAIR: std::cerr << "build_pair"; break;
  }

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
  } else if (const FrameIndexSDNode *FIDN =
	     dyn_cast<FrameIndexSDNode>(this)) {
    std::cerr << "<" << FIDN->getIndex() << ">";
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(this)){
    std::cerr << "<" << CP->getIndex() << ">";
  } else if (const BasicBlockSDNode *BBDN = 
	     dyn_cast<BasicBlockSDNode>(this)) {
    std::cerr << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      std::cerr << LBB->getName() << " ";
    std::cerr << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const CopyRegSDNode *C2V = dyn_cast<CopyRegSDNode>(this)) {
    std::cerr << "<reg #" << C2V->getReg() << ">";
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(this)) {
    std::cerr << "'" << ES->getSymbol() << "'";
  } else if (const SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(this)) {
    std::cerr << " - condition = ";
    switch (SetCC->getCondition()) {
    default: assert(0 && "Unknown setcc condition!");
    case ISD::SETOEQ:  std::cerr << "setoeq"; break;
    case ISD::SETOGT:  std::cerr << "setogt"; break;
    case ISD::SETOGE:  std::cerr << "setoge"; break;
    case ISD::SETOLT:  std::cerr << "setolt"; break;
    case ISD::SETOLE:  std::cerr << "setole"; break;
    case ISD::SETONE:  std::cerr << "setone"; break;
      
    case ISD::SETO:    std::cerr << "seto";  break;
    case ISD::SETUO:   std::cerr << "setuo"; break;
    case ISD::SETUEQ:  std::cerr << "setue"; break;
    case ISD::SETUGT:  std::cerr << "setugt"; break;
    case ISD::SETUGE:  std::cerr << "setuge"; break;
    case ISD::SETULT:  std::cerr << "setult"; break;
    case ISD::SETULE:  std::cerr << "setule"; break;
    case ISD::SETUNE:  std::cerr << "setune"; break;
      
    case ISD::SETEQ:   std::cerr << "seteq"; break;
    case ISD::SETGT:   std::cerr << "setgt"; break;
    case ISD::SETGE:   std::cerr << "setge"; break;
    case ISD::SETLT:   std::cerr << "setlt"; break;
    case ISD::SETLE:   std::cerr << "setle"; break;
    case ISD::SETNE:   std::cerr << "setne"; break;
    }
  }

}

void SelectionDAG::dump() const {
  std::cerr << "SelectionDAG has " << AllNodes.size() << " nodes:";
  for (unsigned i = 0, e = AllNodes.size(); i != e; ++i) {
    std::cerr << "\n  ";
    AllNodes[i]->dump();
  }
  std::cerr << "\n\n";
}

