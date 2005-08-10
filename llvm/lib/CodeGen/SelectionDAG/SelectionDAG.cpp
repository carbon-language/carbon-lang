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
#include "llvm/Target/TargetLowering.h"
#include <iostream>
#include <set>
#include <cmath>
#include <algorithm>
using namespace llvm;

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

// isInvertibleForFree - Return true if there is no cost to emitting the logical
// inverse of this node.
static bool isInvertibleForFree(SDOperand N) {
  if (isa<ConstantSDNode>(N.Val)) return true;
  if (N.Val->getOpcode() == ISD::SETCC && N.Val->hasOneUse())
    return true;
  return false;
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
  case ISD::ConstantFP: {
    union {
      double DV;
      uint64_t IV;
    };
    DV = cast<ConstantFPSDNode>(N)->getValue();
    ConstantFPs.erase(std::make_pair(IV, N->getValueType(0)));
    break;
  }
  case ISD::CONDCODE:
    assert(CondCodeNodes[cast<CondCodeSDNode>(N)->get()] &&
           "Cond code doesn't exist!");
    CondCodeNodes[cast<CondCodeSDNode>(N)->get()] = 0;
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
  case ISD::VALUETYPE:
    ValueTypeNodes[cast<VTSDNode>(N)->getVT()] = 0;
    break;
  case ISD::SRCVALUE: {
    SrcValueSDNode *SVN = cast<SrcValueSDNode>(N);
    ValueNodes.erase(std::make_pair(SVN->getValue(), SVN->getOffset()));
    break;
  }    
  case ISD::LOAD:
    Loads.erase(std::make_pair(N->getOperand(1),
                               std::make_pair(N->getOperand(0),
                                              N->getValueType(0))));
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
    else if (N->getNumValues() == 1) {
      std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
      OneResultNodes.erase(std::make_pair(N->getOpcode(),
                                          std::make_pair(N->getValueType(0),
                                                         Ops)));
    } else {
      // Remove the node from the ArbitraryNodes map.
      std::vector<MVT::ValueType> RV(N->value_begin(), N->value_end());
      std::vector<SDOperand>     Ops(N->op_begin(), N->op_end());
      ArbitraryNodes.erase(std::make_pair(N->getOpcode(),
                                          std::make_pair(RV, Ops)));
    }
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
  N = new ConstantSDNode(Val, VT);
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
  union {
    double DV;
    uint64_t IV;
  };

  DV = Val;

  SDNode *&N = ConstantFPs[std::make_pair(IV, VT)];
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
  N = new ExternalSymbolSDNode(Sym, VT);
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
      // If the LHS is a ZERO_EXTEND and if this is an ==/!= comparison, perform
      // the comparison on the input.
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

  if (N1 == N2) {
    // We can always fold X == Y for integer setcc's.
    if (MVT::isInteger(N1.getValueType()))
      return getConstant(ISD::isTrueWhenEqual(Cond), VT);
    unsigned UOF = ISD::getUnorderedFlavor(Cond);
    if (UOF == 2)   // FP operators that are undefined on NaNs.
      return getConstant(ISD::isTrueWhenEqual(Cond), VT);
    if (UOF == unsigned(ISD::isTrueWhenEqual(Cond)))
      return getConstant(UOF, VT);
    // Otherwise, we can't fold it.  However, we can simplify it to SETUO/SETO
    // if it is not already.
    ISD::CondCode NewCond = UOF == 0 ? ISD::SETUO : ISD::SETO;
    if (NewCond != Cond)
      return getSetCC(VT, N1, N2, NewCond);
  }

  if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
      MVT::isInteger(N1.getValueType())) {
    if (N1.getOpcode() == ISD::ADD || N1.getOpcode() == ISD::SUB ||
        N1.getOpcode() == ISD::XOR) {
      // Simplify (X+Y) == (X+Z) -->  Y == Z
      if (N1.getOpcode() == N2.getOpcode()) {
        if (N1.getOperand(0) == N2.getOperand(0))
          return getSetCC(VT, N1.getOperand(1), N2.getOperand(1), Cond);
        if (N1.getOperand(1) == N2.getOperand(1))
          return getSetCC(VT, N1.getOperand(0), N2.getOperand(0), Cond);
        if (isCommutativeBinOp(N1.getOpcode())) {
          // If X op Y == Y op X, try other combinations.
          if (N1.getOperand(0) == N2.getOperand(1))
            return getSetCC(VT, N1.getOperand(1), N2.getOperand(0), Cond);
          if (N1.getOperand(1) == N2.getOperand(0))
            return getSetCC(VT, N1.getOperand(1), N2.getOperand(1), Cond);
        }
      }

      // FIXME: move this stuff to the DAG Combiner when it exists!

      // Simplify (X+Z) == X -->  Z == 0
      if (N1.getOperand(0) == N2)
        return getSetCC(VT, N1.getOperand(1),
                        getConstant(0, N1.getValueType()), Cond);
      if (N1.getOperand(1) == N2) {
        if (isCommutativeBinOp(N1.getOpcode()))
          return getSetCC(VT, N1.getOperand(0),
                          getConstant(0, N1.getValueType()), Cond);
        else {
          assert(N1.getOpcode() == ISD::SUB && "Unexpected operation!");
          // (Z-X) == X  --> Z == X<<1
          return getSetCC(VT, N1.getOperand(0),
                          getNode(ISD::SHL, N2.getValueType(),
                                  N2, getConstant(1, TLI.getShiftAmountTy())),
                          Cond);
        }
      }
    }

    if (N2.getOpcode() == ISD::ADD || N2.getOpcode() == ISD::SUB ||
        N2.getOpcode() == ISD::XOR) {
      // Simplify  X == (X+Z) -->  Z == 0
      if (N2.getOperand(0) == N1) {
        return getSetCC(VT, N2.getOperand(1),
                        getConstant(0, N2.getValueType()), Cond);
      } else if (N2.getOperand(1) == N1) {
        if (isCommutativeBinOp(N2.getOpcode())) {
          return getSetCC(VT, N2.getOperand(0),
                          getConstant(0, N2.getValueType()), Cond);
        } else {
          assert(N2.getOpcode() == ISD::SUB && "Unexpected operation!");
          // X == (Z-X)  --> X<<1 == Z
          return getSetCC(VT, getNode(ISD::SHL, N2.getValueType(), N1, 
                                      getConstant(1, TLI.getShiftAmountTy())),
                          N2.getOperand(0), Cond);
        }
      }
    }
  }

  // Fold away ALL boolean setcc's.
  if (N1.getValueType() == MVT::i1) {
    switch (Cond) {
    default: assert(0 && "Unknown integer setcc!");
    case ISD::SETEQ:  // X == Y  -> (X^Y)^1
      N1 = getNode(ISD::XOR, MVT::i1,
                   getNode(ISD::XOR, MVT::i1, N1, N2),
                   getConstant(1, MVT::i1));
      break;
    case ISD::SETNE:  // X != Y   -->  (X^Y)
      N1 = getNode(ISD::XOR, MVT::i1, N1, N2);
      break;
    case ISD::SETGT:  // X >s Y   -->  X == 0 & Y == 1  -->  X^1 & Y
    case ISD::SETULT: // X <u Y   -->  X == 0 & Y == 1  -->  X^1 & Y
      N1 = getNode(ISD::AND, MVT::i1, N2,
                   getNode(ISD::XOR, MVT::i1, N1, getConstant(1, MVT::i1)));
      break;
    case ISD::SETLT:  // X <s Y   --> X == 1 & Y == 0  -->  Y^1 & X
    case ISD::SETUGT: // X >u Y   --> X == 1 & Y == 0  -->  Y^1 & X
      N1 = getNode(ISD::AND, MVT::i1, N1,
                   getNode(ISD::XOR, MVT::i1, N2, getConstant(1, MVT::i1)));
      break;
    case ISD::SETULE: // X <=u Y  --> X == 0 | Y == 1  -->  X^1 | Y
    case ISD::SETGE:  // X >=s Y  --> X == 0 | Y == 1  -->  X^1 | Y
      N1 = getNode(ISD::OR, MVT::i1, N2,
                   getNode(ISD::XOR, MVT::i1, N1, getConstant(1, MVT::i1)));
      break;
    case ISD::SETUGE: // X >=u Y  --> X == 1 | Y == 0  -->  Y^1 | X
    case ISD::SETLE:  // X <=s Y  --> X == 1 | Y == 0  -->  Y^1 | X
      N1 = getNode(ISD::OR, MVT::i1, N1,
                   getNode(ISD::XOR, MVT::i1, N2, getConstant(1, MVT::i1)));
      break;
    }
    if (VT != MVT::i1)
      N1 = getNode(ISD::ZERO_EXTEND, VT, N1);
    return N1;
  }

  // Could not fold it.
  return SDOperand();
}



/// getNode - Gets or creates the specified node.
///
SDOperand SelectionDAG::getNode(unsigned Opcode, MVT::ValueType VT) {
  SDNode *N = new SDNode(Opcode, VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
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
  case ISD::FNEG:
    if (OpOpcode == ISD::SUB)   // -(X-Y) -> (Y-X)
      return getNode(ISD::SUB, VT, Operand.Val->getOperand(1),
                     Operand.Val->getOperand(0));
    if (OpOpcode == ISD::FNEG)  // --X -> X
      return Operand.Val->getOperand(0);
    break;
  case ISD::FABS:
    if (OpOpcode == ISD::FNEG)  // abs(-X) -> abs(X)
      return getNode(ISD::FABS, VT, Operand.Val->getOperand(0));
    break;
  }

  SDNode *&N = UnaryOps[std::make_pair(Opcode, std::make_pair(Operand, VT))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(Opcode, Operand);
  N->setValueTypes(VT);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  V and Mask are known to
/// be the same type.
static bool MaskedValueIsZero(const SDOperand &Op, uint64_t Mask,
                              const TargetLowering &TLI) {
  unsigned SrcBits;
  if (Mask == 0) return true;

  // If we know the result of a setcc has the top bits zero, use this info.
  switch (Op.getOpcode()) {
  case ISD::Constant:
    return (cast<ConstantSDNode>(Op)->getValue() & Mask) == 0;

  case ISD::SETCC:
    return ((Mask & 1) == 0) &&
           TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult;

  case ISD::ZEXTLOAD:
    SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(3))->getVT());
    return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
  case ISD::ZERO_EXTEND:
    SrcBits = MVT::getSizeInBits(Op.getOperand(0).getValueType());
    return MaskedValueIsZero(Op.getOperand(0),Mask & ((1ULL << SrcBits)-1),TLI);

  case ISD::AND:
    // (X & C1) & C2 == 0   iff   C1 & C2 == 0.
    if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      return MaskedValueIsZero(Op.getOperand(0),AndRHS->getValue() & Mask, TLI);

    // FALL THROUGH
  case ISD::OR:
  case ISD::XOR:
    return MaskedValueIsZero(Op.getOperand(0), Mask, TLI) &&
           MaskedValueIsZero(Op.getOperand(1), Mask, TLI);
  case ISD::SELECT:
    return MaskedValueIsZero(Op.getOperand(1), Mask, TLI) &&
           MaskedValueIsZero(Op.getOperand(2), Mask, TLI);

  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  X & (C2 << C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask << ShAmt->getValue();
      SrcBits = MVT::getSizeInBits(Op.getValueType());
      if (SrcBits != 64) NewVal &= (1ULL << SrcBits)-1;
      return MaskedValueIsZero(Op.getOperand(0), NewVal, TLI);
    }
    return false;
  case ISD::SHL:
    // (ushl X, C1) & C2 == 0   iff  X & (C2 >> C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask >> ShAmt->getValue();
      return MaskedValueIsZero(Op.getOperand(0), NewVal, TLI);
    }
    return false;
    // TODO we could handle some SRA cases here.
  default: break;
  }

  return false;
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
      case ISD::SHL  : return getConstant(C1 << (int)C2, VT);
      case ISD::SRL  : return getConstant(C1 >> (unsigned)C2, VT);
      case ISD::SRA  : return getConstant(N1C->getSignExtended() >>(int)C2, VT);
      default: break;
      }

    } else {      // Cannonicalize constant to RHS if commutative
      if (isCommutativeBinOp(Opcode)) {
        std::swap(N1C, N2C);
        std::swap(N1, N2);
      }
    }

    switch (Opcode) {
    default: break;
    case ISD::SHL:    // shl  0, X -> 0
      if (N1C->isNullValue()) return N1;
      break;
    case ISD::SRL:    // srl  0, X -> 0
      if (N1C->isNullValue()) return N1;
      break;
    case ISD::SRA:    // sra -1, X -> -1
      if (N1C->isAllOnesValue()) return N1;
      break;
    case ISD::SIGN_EXTEND_INREG:  // SIGN_EXTEND_INREG N1C, EVT
      // Extending a constant?  Just return the extended constant.
      SDOperand Tmp = getNode(ISD::TRUNCATE, cast<VTSDNode>(N2)->getVT(), N1);
      return getNode(ISD::SIGN_EXTEND, VT, Tmp);
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
      return getNode(ISD::ADD, VT, N1, getConstant(-C2, VT));
    case ISD::MUL:
      if (!C2) return N2;         // mul X, 0 -> 0
      if (N2C->isAllOnesValue()) // mul X, -1 -> 0-X
        return getNode(ISD::SUB, VT, getConstant(0, VT), N1);

      // FIXME: Move this to the DAG combiner when it exists.
      if ((C2 & C2-1) == 0) {
        SDOperand ShAmt = getConstant(Log2_64(C2), TLI.getShiftAmountTy());
        return getNode(ISD::SHL, VT, N1, ShAmt);
      }
      break;

    case ISD::MULHU:
    case ISD::MULHS:
      if (!C2) return N2;         // mul X, 0 -> 0

      if (C2 == 1)                // 0X*01 -> 0X  hi(0X) == 0
        return getConstant(0, VT);

      // Many others could be handled here, including -1, powers of 2, etc.
      break;

    case ISD::UDIV:
      // FIXME: Move this to the DAG combiner when it exists.
      if ((C2 & C2-1) == 0 && C2) {
        SDOperand ShAmt = getConstant(Log2_64(C2), TLI.getShiftAmountTy());
        return getNode(ISD::SRL, VT, N1, ShAmt);
      }
      break;

    case ISD::SHL:
    case ISD::SRL:
    case ISD::SRA:
      // If the shift amount is bigger than the size of the data, then all the
      // bits are shifted out.  Simplify to undef.
      if (C2 >= MVT::getSizeInBits(N1.getValueType())) {
        return getNode(ISD::UNDEF, N1.getValueType());
      }
      if (C2 == 0) return N1;

      if (Opcode == ISD::SHL && N1.getNumOperands() == 2)
        if (ConstantSDNode *OpSA = dyn_cast<ConstantSDNode>(N1.getOperand(1))) {
          unsigned OpSAC = OpSA->getValue();
          if (N1.getOpcode() == ISD::SHL) {
            if (C2+OpSAC >= MVT::getSizeInBits(N1.getValueType()))
              return getConstant(0, N1.getValueType());
            return getNode(ISD::SHL, N1.getValueType(), N1.getOperand(0),
                           getConstant(C2+OpSAC, N2.getValueType()));
          } else if (N1.getOpcode() == ISD::SRL) {
            // (X >> C1) << C2:  if C2 > C1, ((X & ~0<<C1) << C2-C1)
            SDOperand Mask = getNode(ISD::AND, VT, N1.getOperand(0),
                                     getConstant(~0ULL << OpSAC, VT));
            if (C2 > OpSAC) {
              return getNode(ISD::SHL, VT, Mask,
                             getConstant(C2-OpSAC, N2.getValueType()));
            } else {
              // (X >> C1) << C2:  if C2 <= C1, ((X & ~0<<C1) >> C1-C2)
              return getNode(ISD::SRL, VT, Mask,
                             getConstant(OpSAC-C2, N2.getValueType()));
            }
          } else if (N1.getOpcode() == ISD::SRA) {
            // if C1 == C2, just mask out low bits.
            if (C2 == OpSAC)
              return getNode(ISD::AND, VT, N1.getOperand(0),
                             getConstant(~0ULL << C2, VT));
          }
        }
      break;

    case ISD::AND:
      if (!C2) return N2;         // X and 0 -> 0
      if (N2C->isAllOnesValue())
        return N1;                // X and -1 -> X

      if (MaskedValueIsZero(N1, C2, TLI))  // X and 0 -> 0
        return getConstant(0, VT);

      {
        uint64_t NotC2 = ~C2;
        if (VT != MVT::i64)
          NotC2 &= (1ULL << MVT::getSizeInBits(VT))-1;

        if (MaskedValueIsZero(N1, NotC2, TLI))
          return N1;                // if (X & ~C2) -> 0, the and is redundant
      }

      // FIXME: Should add a corresponding version of this for
      // ZERO_EXTEND/SIGN_EXTEND by converting them to an ANY_EXTEND node which
      // we don't have yet.

      // and (sign_extend_inreg x:16:32), 1 -> and x, 1
      if (N1.getOpcode() == ISD::SIGN_EXTEND_INREG) {
        // If we are masking out the part of our input that was extended, just
        // mask the input to the extension directly.
        unsigned ExtendBits =
          MVT::getSizeInBits(cast<VTSDNode>(N1.getOperand(1))->getVT());
        if ((C2 & (~0ULL << ExtendBits)) == 0)
          return getNode(ISD::AND, VT, N1.getOperand(0), N2);
      } else if (N1.getOpcode() == ISD::OR) {
        if (ConstantSDNode *ORI = dyn_cast<ConstantSDNode>(N1.getOperand(1)))
          if ((ORI->getValue() & C2) == C2) {
            // If the 'or' is setting all of the bits that we are masking for,
            // we know the result of the AND will be the AND mask itself.
            return N2;
          }
      }
      break;
    case ISD::OR:
      if (!C2)return N1;          // X or 0 -> X
      if (N2C->isAllOnesValue())
        return N2;                // X or -1 -> -1
      break;
    case ISD::XOR:
      if (!C2) return N1;        // X xor 0 -> X
      if (N2C->isAllOnesValue()) {
        if (N1.Val->getOpcode() == ISD::SETCC){
          SDNode *SetCC = N1.Val;
          // !(X op Y) -> (X !op Y)
          bool isInteger = MVT::isInteger(SetCC->getOperand(0).getValueType());
          ISD::CondCode CC = cast<CondCodeSDNode>(SetCC->getOperand(2))->get();
          return getSetCC(SetCC->getValueType(0),
                          SetCC->getOperand(0), SetCC->getOperand(1),
                          ISD::getSetCCInverse(CC, isInteger));
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
  if (N1CFP) {
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

    if (Opcode == ISD::FP_ROUND_INREG)
      return getNode(ISD::FP_EXTEND, VT,
                     getNode(ISD::FP_ROUND, cast<VTSDNode>(N2)->getVT(), N1));
  }

  // Finally, fold operations that do not require constants.
  switch (Opcode) {
  case ISD::TokenFactor:
    if (N1.getOpcode() == ISD::EntryToken)
      return N2;
    if (N2.getOpcode() == ISD::EntryToken)
      return N1;
    break;

  case ISD::AND:
  case ISD::OR:
    if (N1.Val->getOpcode() == ISD::SETCC && N2.Val->getOpcode() == ISD::SETCC){
      SDNode *LHS = N1.Val, *RHS = N2.Val;
      SDOperand LL = LHS->getOperand(0), RL = RHS->getOperand(0);
      SDOperand LR = LHS->getOperand(1), RR = RHS->getOperand(1);
      ISD::CondCode Op1 = cast<CondCodeSDNode>(LHS->getOperand(2))->get();
      ISD::CondCode Op2 = cast<CondCodeSDNode>(RHS->getOperand(2))->get();

      if (LR == RR && isa<ConstantSDNode>(LR) &&
          Op2 == Op1 && MVT::isInteger(LL.getValueType())) {
        // (X != 0) | (Y != 0) -> (X|Y != 0)
        // (X == 0) & (Y == 0) -> (X|Y == 0)
        // (X <  0) | (Y <  0) -> (X|Y < 0)
        if (cast<ConstantSDNode>(LR)->getValue() == 0 &&
            ((Op2 == ISD::SETEQ && Opcode == ISD::AND) ||
             (Op2 == ISD::SETNE && Opcode == ISD::OR) ||
             (Op2 == ISD::SETLT && Opcode == ISD::OR)))
          return getSetCC(VT, getNode(ISD::OR, LR.getValueType(), LL, RL), LR,
                          Op2);

        if (cast<ConstantSDNode>(LR)->isAllOnesValue()) {
          // (X == -1) & (Y == -1) -> (X&Y == -1)
          // (X != -1) | (Y != -1) -> (X&Y != -1)
          // (X >  -1) | (Y >  -1) -> (X&Y >  -1)
          if ((Opcode == ISD::AND && Op2 == ISD::SETEQ) ||
              (Opcode == ISD::OR  && Op2 == ISD::SETNE) ||
              (Opcode == ISD::OR  && Op2 == ISD::SETGT))
            return getSetCC(VT, getNode(ISD::AND, LR.getValueType(), LL, RL),
                            LR, Op2);
          // (X >  -1) & (Y >  -1) -> (X|Y > -1)
          if (Opcode == ISD::AND && Op2 == ISD::SETGT)
            return getSetCC(VT, getNode(ISD::OR, LR.getValueType(), LL, RL),
                            LR, Op2);
        }
      }

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
          Result = ISD::getSetCCOrOperation(Op1, Op2, isInteger);
        else
          Result = ISD::getSetCCAndOperation(Op1, Op2, isInteger);

        if (Result != ISD::SETCC_INVALID)
          return getSetCC(LHS->getValueType(0), LL, LR, Result);
      }
    }

    // and/or zext(a), zext(b) -> zext(and/or a, b)
    if (N1.getOpcode() == ISD::ZERO_EXTEND &&
        N2.getOpcode() == ISD::ZERO_EXTEND &&
        N1.getOperand(0).getValueType() == N2.getOperand(0).getValueType())
      return getNode(ISD::ZERO_EXTEND, VT,
                     getNode(Opcode, N1.getOperand(0).getValueType(),
                             N1.getOperand(0), N2.getOperand(0)));
    break;
  case ISD::XOR:
    if (N1 == N2) return getConstant(0, VT);  // xor X, Y -> 0
    break;
  case ISD::ADD:
    if (N2.getOpcode() == ISD::FNEG)          // (A+ (-B) -> A-B
      return getNode(ISD::SUB, VT, N1, N2.getOperand(0));
    if (N1.getOpcode() == ISD::FNEG)          // ((-A)+B) -> B-A
      return getNode(ISD::SUB, VT, N2, N1.getOperand(0));
    if (N1.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N1.getOperand(0)) &&
        cast<ConstantSDNode>(N1.getOperand(0))->getValue() == 0)
      return getNode(ISD::SUB, VT, N2, N1.getOperand(1)); // (0-A)+B -> B-A
    if (N2.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N2.getOperand(0)) &&
        cast<ConstantSDNode>(N2.getOperand(0))->getValue() == 0)
      return getNode(ISD::SUB, VT, N1, N2.getOperand(1)); // A+(0-B) -> A-B
    if (N2.getOpcode() == ISD::SUB && N1 == N2.Val->getOperand(1) &&
        !MVT::isFloatingPoint(N2.getValueType()))
      return N2.Val->getOperand(0); // A+(B-A) -> B
    break;
  case ISD::SUB:
    if (N1.getOpcode() == ISD::ADD) {
      if (N1.Val->getOperand(0) == N2 &&
          !MVT::isFloatingPoint(N2.getValueType()))
        return N1.Val->getOperand(1);         // (A+B)-A == B
      if (N1.Val->getOperand(1) == N2 &&
          !MVT::isFloatingPoint(N2.getValueType()))
        return N1.Val->getOperand(0);         // (A+B)-B == A
    }
    if (N2.getOpcode() == ISD::FNEG)          // (A- (-B) -> A+B
      return getNode(ISD::ADD, VT, N1, N2.getOperand(0));
    break;
  case ISD::FP_ROUND_INREG:
    if (cast<VTSDNode>(N2)->getVT() == VT) return N1;  // Not actually rounding.
    break;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(N2)->getVT();
    if (EVT == VT) return N1;  // Not actually extending

    // If we are sign extending an extension, use the original source.
    if (N1.getOpcode() == ISD::SIGN_EXTEND_INREG)
      if (cast<VTSDNode>(N1.getOperand(1))->getVT() <= EVT)
        return N1;

    // If we are sign extending a sextload, return just the load.
    if (N1.getOpcode() == ISD::SEXTLOAD)
      if (cast<VTSDNode>(N1.getOperand(3))->getVT() <= EVT)
        return N1;

    // If we are extending the result of a setcc, and we already know the
    // contents of the top bits, eliminate the extension.
    if (N1.getOpcode() == ISD::SETCC &&
        TLI.getSetCCResultContents() ==
                        TargetLowering::ZeroOrNegativeOneSetCCResult)
      return N1;

    // If we are sign extending the result of an (and X, C) operation, and we
    // know the extended bits are zeros already, don't do the extend.
    if (N1.getOpcode() == ISD::AND)
      if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getOperand(1))) {
        uint64_t Mask = N1C->getValue();
        unsigned NumBits = MVT::getSizeInBits(EVT);
        if ((Mask & (~0ULL << (NumBits-1))) == 0)
          return N1;
      }
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
  if (Opcode != ISD::CALLSEQ_START && Opcode != ISD::CALLSEQ_END) {
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

  Operands[0].Val->removeUser(this);
  Operands[0] = N;
  N.Val->Uses.push_back(this);
}



SDOperand SelectionDAG::getLoad(MVT::ValueType VT,
                                SDOperand Chain, SDOperand Ptr,
                                SDOperand SV) {
  SDNode *&N = Loads[std::make_pair(Ptr, std::make_pair(Chain, VT))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(ISD::LOAD, Chain, Ptr, SV);

  // Loads have a token chain.
  N->setValueTypes(VT, MVT::Other);
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

    if (VT == MVT::i1) {  // Boolean SELECT
      if (N2C) {
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

      if (N1 == N2)   // X ? X : Y --> X ? 1 : Y --> X | Y
        return getNode(ISD::OR, VT, N1, N3);
      if (N1 == N3)   // X ? Y : X --> X ? Y : 0 --> X & Y
        return getNode(ISD::AND, VT, N1, N2);
    }

    // If this is a selectcc, check to see if we can simplify the result.
    if (N1.Val->getOpcode() == ISD::SETCC) {
      SDNode *SetCC = N1.Val;
      ISD::CondCode CC = cast<CondCodeSDNode>(SetCC->getOperand(2))->get();
      if (ConstantFPSDNode *CFP =
          dyn_cast<ConstantFPSDNode>(SetCC->getOperand(1)))
        if (CFP->getValue() == 0.0) {   // Allow either -0.0 or 0.0
          // select (setg[te] X, +/-0.0), X, fneg(X) -> fabs
          if ((CC == ISD::SETGE || CC == ISD::SETGT) &&
              N2 == SetCC->getOperand(0) && N3.getOpcode() == ISD::FNEG &&
              N3.getOperand(0) == N2)
            return getNode(ISD::FABS, VT, N2);

          // select (setl[te] X, +/-0.0), fneg(X), X -> fabs
          if ((CC == ISD::SETLT || CC == ISD::SETLE) &&
              N3 == SetCC->getOperand(0) && N2.getOpcode() == ISD::FNEG &&
              N2.getOperand(0) == N3)
            return getNode(ISD::FABS, VT, N3);
        }
      // select (setlt X, 0), A, 0 -> and (sra X, size(X)-1), A
      if (ConstantSDNode *CN =
          dyn_cast<ConstantSDNode>(SetCC->getOperand(1)))
        if (CN->getValue() == 0 && N3C && N3C->getValue() == 0)
          if (CC == ISD::SETLT) {
            MVT::ValueType XType = SetCC->getOperand(0).getValueType();
            MVT::ValueType AType = N2.getValueType();
            if (XType >= AType) {
              // and (sra X, size(X)-1, A) -> "and (srl X, C2), A" iff A is a
              // single-bit constant.  FIXME: remove once the dag combiner
              // exists.
              if (ConstantSDNode *AC = dyn_cast<ConstantSDNode>(N2))
                if ((AC->getValue() & (AC->getValue()-1)) == 0) {
                  unsigned ShCtV = Log2_64(AC->getValue());
                  ShCtV = MVT::getSizeInBits(XType)-ShCtV-1;
                  SDOperand ShCt = getConstant(ShCtV, TLI.getShiftAmountTy());
                  SDOperand Shift = getNode(ISD::SRL, XType,
                                            SetCC->getOperand(0), ShCt);
                  if (XType > AType)
                    Shift = getNode(ISD::TRUNCATE, AType, Shift);
                  return getNode(ISD::AND, AType, Shift, N2);
                }


              SDOperand Shift = getNode(ISD::SRA, XType, SetCC->getOperand(0),
                getConstant(MVT::getSizeInBits(XType)-1,
                            TLI.getShiftAmountTy()));
              if (XType > AType)
                Shift = getNode(ISD::TRUNCATE, AType, Shift);
              return getNode(ISD::AND, AType, Shift, N2);
            }
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

  std::vector<SDOperand> Ops;
  Ops.reserve(3);
  Ops.push_back(N1);
  Ops.push_back(N2);
  Ops.push_back(N3);

  // Memoize nodes.
  SDNode *&N = OneResultNodes[std::make_pair(Opcode, std::make_pair(VT, Ops))];
  if (N) return SDOperand(N, 0);

  N = new SDNode(Opcode, N1, N2, N3);
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
  }

  // Memoize nodes.
  SDNode *&N = OneResultNodes[std::make_pair(Opcode, std::make_pair(VT, Ops))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(Opcode, Ops);
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

  // Memoize the node.
  SDNode *&N = ArbitraryNodes[std::make_pair(Opcode, std::make_pair(ResultTys,
                                                                    Ops))];
  if (N) return SDOperand(N, 0);
  N = new SDNode(Opcode, Ops);
  N->setValueTypes(ResultTys);
  AllNodes.push_back(N);
  return SDOperand(N, 0);
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


const char *SDNode::getOperationName() const {
  switch (getOpcode()) {
  default: return "<<Unknown>>";
  case ISD::PCMARKER:      return "PCMarker";
  case ISD::SRCVALUE:      return "SrcValue";
  case ISD::EntryToken:    return "EntryToken";
  case ISD::TokenFactor:   return "TokenFactor";
  case ISD::Constant:      return "Constant";
  case ISD::ConstantFP:    return "ConstantFP";
  case ISD::GlobalAddress: return "GlobalAddress";
  case ISD::FrameIndex:    return "FrameIndex";
  case ISD::BasicBlock:    return "BasicBlock";
  case ISD::ExternalSymbol: return "ExternalSymbol";
  case ISD::ConstantPool:  return "ConstantPoolIndex";
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

  case ISD::SETCC:       return "setcc";
  case ISD::SELECT:      return "select";
  case ISD::ADD_PARTS:   return "add_parts";
  case ISD::SUB_PARTS:   return "sub_parts";
  case ISD::SHL_PARTS:   return "shl_parts";
  case ISD::SRA_PARTS:   return "sra_parts";
  case ISD::SRL_PARTS:   return "srl_parts";

  // Conversion operators.
  case ISD::SIGN_EXTEND: return "sign_extend";
  case ISD::ZERO_EXTEND: return "zero_extend";
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

void SDNode::dump() const {
  std::cerr << (void*)this << ": ";

  for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
    if (i) std::cerr << ",";
    if (getValueType(i) == MVT::Other)
      std::cerr << "ch";
    else
      std::cerr << MVT::getValueTypeString(getValueType(i));
  }
  std::cerr << " = " << getOperationName();

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
    std::cerr << "<" << CP->getIndex() << ">";
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(this)) {
    std::cerr << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      std::cerr << LBB->getName() << " ";
    std::cerr << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const RegSDNode *C2V = dyn_cast<RegSDNode>(this)) {
    std::cerr << "<reg #" << C2V->getReg() << ">";
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(this)) {
    std::cerr << "'" << ES->getSymbol() << "'";
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(this)) {
    if (M->getValue())
      std::cerr << "<" << M->getValue() << ":" << M->getOffset() << ">";
    else
      std::cerr << "<null:" << M->getOffset() << ">";
  }
}

static void DumpNodes(SDNode *N, unsigned indent) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    if (N->getOperand(i).Val->hasOneUse())
      DumpNodes(N->getOperand(i).Val, indent+2);
    else
      std::cerr << "\n" << std::string(indent+2, ' ')
                << (void*)N->getOperand(i).Val << ": <multiple use>";


  std::cerr << "\n" << std::string(indent, ' ');
  N->dump();
}

void SelectionDAG::dump() const {
  std::cerr << "SelectionDAG has " << AllNodes.size() << " nodes:";
  std::vector<SDNode*> Nodes(AllNodes);
  std::sort(Nodes.begin(), Nodes.end());

  for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
    if (!Nodes[i]->hasOneUse() && Nodes[i] != getRoot().Val)
      DumpNodes(Nodes[i], 2);
  }

  DumpNodes(getRoot().Val, 2);

  std::cerr << "\n\n";
}

