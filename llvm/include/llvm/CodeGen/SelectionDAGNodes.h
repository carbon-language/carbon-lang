//===-- llvm/CodeGen/SelectionDAGNodes.h - SelectionDAG Nodes ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the SDNode class and derived classes, which are used to
// represent the nodes and operations present in a SelectionDAG.  These nodes
// and operations are machine code level operations, with some similarities to
// the GCC RTL representation.
//
// Clients should include the SelectionDAG.h file instead of this file directly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAGNODES_H
#define LLVM_CODEGEN_SELECTIONDAGNODES_H

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/support/DataTypes.h"
#include <cassert>
#include <vector>

namespace llvm {

class SelectionDAG;
class GlobalValue;
class MachineBasicBlock;
class SDNode;
template <typename T> struct simplify_type;

/// ISD namespace - This namespace contains an enum which represents all of the
/// SelectionDAG node types and value types.
///
namespace ISD {
  //===--------------------------------------------------------------------===//
  /// ISD::NodeType enum - This enum defines all of the operators valid in a
  /// SelectionDAG.
  ///
  enum NodeType {
    // Leaf nodes
    EntryToken, Constant, ConstantFP, GlobalAddress, FrameIndex, ConstantPool,
    BasicBlock, ExternalSymbol,

    // CopyToReg - This node has chain and child nodes, and an associated
    // register number.  The instruction selector must guarantee that the value
    // of the value node is available in the virtual register stored in the
    // CopyRegSDNode object.
    CopyToReg,

    // CopyFromReg - This node indicates that the input value is a virtual or
    // physical register that is defined outside of the scope of this
    // SelectionDAG.  The virtual register is available from the
    // CopyRegSDNode object.
    CopyFromReg,

    // EXTRACT_ELEMENT - This is used to get the first or second (determined by
    // a Constant, which is required to be operand #1), element of the aggregate
    // value specified as operand #0.  This is only for use before legalization,
    // for values that will be broken into multiple registers.
    EXTRACT_ELEMENT,

    // BUILD_PAIR - This is the opposite of EXTRACT_ELEMENT in some ways.  Given
    // two values of the same integer value type, this produces a value twice as
    // big.  Like EXTRACT_ELEMENT, this can only be used before legalization.
    BUILD_PAIR,


    // Simple binary arithmetic operators.
    ADD, SUB, MUL, SDIV, UDIV, SREM, UREM,

    // Bitwise operators.
    AND, OR, XOR, SHL, SRA, SRL,

    // Select operator.
    SELECT,

    // SetCC operator - This evaluates to a boolean (i1) true value if the
    // condition is true.  These nodes are instances of the
    // SetCCSDNode class, which contains the condition code as extra
    // state.
    SETCC,

    // addc - Three input, two output operator: (X, Y, C) -> (X+Y+C,
    // Cout).  X,Y are integer inputs of agreeing size, C is a one bit
    // value, and two values are produced: the sum and a carry out.
    ADDC, SUBB,

    // Conversion operators.  These are all single input single output
    // operations.  For all of these, the result type must be strictly
    // wider or narrower (depending on the operation) than the source
    // type.

    // SIGN_EXTEND - Used for integer types, replicating the sign bit
    // into new bits.
    SIGN_EXTEND,

    // ZERO_EXTEND - Used for integer types, zeroing the new bits.
    ZERO_EXTEND,

    // TRUNCATE - Completely drop the high bits.
    TRUNCATE,

    // FP_ROUND - Perform a rounding operation from the current
    // precision down to the specified precision.
    FP_ROUND,

    // FP_EXTEND - Extend a smaller FP type into a larger FP type.
    FP_EXTEND,

    // Other operators.  LOAD and STORE have token chains.
    LOAD, STORE,

    // DYNAMIC_STACKALLOC - Allocate some number of bytes on the stack aligned
    // to a specified boundary.  The first operand is the token chain, the
    // second is the number of bytes to allocate, and the third is the alignment
    // boundary.
    DYNAMIC_STACKALLOC,

    // Control flow instructions.  These all have token chains.
    
    // BR - Unconditional branch.  The first operand is the chain
    // operand, the second is the MBB to branch to.
    BR,

    // BRCOND - Conditional branch.  The first operand is the chain,
    // the second is the condition, the third is the block to branch
    // to if the condition is true.
    BRCOND,

    // RET - Return from function.  The first operand is the chain,
    // and any subsequent operands are the return values for the
    // function.  This operation can have variable number of operands.
    RET,

    // CALL - Call to a function pointer.  The first operand is the chain, the
    // second is the destination function pointer (a GlobalAddress for a direct
    // call).  Arguments have already been lowered to explicit DAGs according to
    // the calling convention in effect here.
    CALL,
    
    // ADJCALLSTACKDOWN/ADJCALLSTACKUP - These operators mark the beginning and
    // end of a call sequence and indicate how much the stack pointer needs to
    // be adjusted for that particular call.  The first operand is a chain, the
    // second is a ConstantSDNode of intptr type.
    ADJCALLSTACKDOWN,  // Beginning of a call sequence
    ADJCALLSTACKUP,    // End of a call sequence


    // BUILTIN_OP_END - This must be the last enum value in this list.
    BUILTIN_OP_END,
  };

  //===--------------------------------------------------------------------===//
  /// ISD::CondCode enum - These are ordered carefully to make the bitfields
  /// below work out, when considering SETFALSE (something that never exists
  /// dynamically) as 0.  "U" -> Unsigned (for integer operands) or Unordered
  /// (for floating point), "L" -> Less than, "G" -> Greater than, "E" -> Equal
  /// to.  If the "N" column is 1, the result of the comparison is undefined if
  /// the input is a NAN.
  ///
  /// All of these (except for the 'always folded ops') should be handled for
  /// floating point.  For integer, only the SETEQ,SETNE,SETLT,SETLE,SETGT,
  /// SETGE,SETULT,SETULE,SETUGT, and SETUGE opcodes are used.
  ///
  /// Note that these are laid out in a specific order to allow bit-twiddling
  /// to transform conditions.
  enum CondCode {
    // Opcode          N U L G E       Intuitive operation
    SETFALSE,      //    0 0 0 0       Always false (always folded)
    SETOEQ,        //    0 0 0 1       True if ordered and equal
    SETOGT,        //    0 0 1 0       True if ordered and greater than
    SETOGE,        //    0 0 1 1       True if ordered and greater than or equal
    SETOLT,        //    0 1 0 0       True if ordered and less than
    SETOLE,        //    0 1 0 1       True if ordered and less than or equal
    SETONE,        //    0 1 1 0       True if ordered and operands are unequal
    SETO,          //    0 1 1 1       True if ordered (no nans)
    SETUO,         //    1 0 0 0       True if unordered: isnan(X) | isnan(Y)
    SETUEQ,        //    1 0 0 1       True if unordered or equal
    SETUGT,        //    1 0 1 0       True if unordered or greater than
    SETUGE,        //    1 0 1 1       True if unordered, greater than, or equal
    SETULT,        //    1 1 0 0       True if unordered or less than
    SETULE,        //    1 1 0 1       True if unordered, less than, or equal 
    SETUNE,        //    1 1 1 0       True if unordered or not equal
    SETTRUE,       //    1 1 1 1       Always true (always folded)
    // Don't care operations: undefined if the input is a nan.
    SETFALSE2,     //  1 X 0 0 0       Always false (always folded)
    SETEQ,         //  1 X 0 0 1       True if equal
    SETGT,         //  1 X 0 1 0       True if greater than
    SETGE,         //  1 X 0 1 1       True if greater than or equal
    SETLT,         //  1 X 1 0 0       True if less than
    SETLE,         //  1 X 1 0 1       True if less than or equal 
    SETNE,         //  1 X 1 1 0       True if not equal
    SETTRUE2,      //  1 X 1 1 1       Always true (always folded)

    SETCC_INVALID,      // Marker value.
  };

  /// isSignedIntSetCC - Return true if this is a setcc instruction that
  /// performs a signed comparison when used with integer operands.
  inline bool isSignedIntSetCC(CondCode Code) {
    return Code == SETGT || Code == SETGE || Code == SETLT || Code == SETLE;
  }

  /// isUnsignedIntSetCC - Return true if this is a setcc instruction that
  /// performs an unsigned comparison when used with integer operands.
  inline bool isUnsignedIntSetCC(CondCode Code) {
    return Code == SETUGT || Code == SETUGE || Code == SETULT || Code == SETULE;
  }

  /// isTrueWhenEqual - Return true if the specified condition returns true if
  /// the two operands to the condition are equal.  Note that if one of the two
  /// operands is a NaN, this value is meaningless.
  inline bool isTrueWhenEqual(CondCode Cond) {
    return ((int)Cond & 1) != 0;
  }

  /// getUnorderedFlavor - This function returns 0 if the condition is always
  /// false if an operand is a NaN, 1 if the condition is always true if the
  /// operand is a NaN, and 2 if the condition is undefined if the operand is a
  /// NaN.
  inline unsigned getUnorderedFlavor(CondCode Cond) {
    return ((int)Cond >> 3) & 3;
  }

  /// getSetCCInverse - Return the operation corresponding to !(X op Y), where
  /// 'op' is a valid SetCC operation.
  CondCode getSetCCInverse(CondCode Operation, bool isInteger);

  /// getSetCCSwappedOperands - Return the operation corresponding to (Y op X)
  /// when given the operation for (X op Y).
  CondCode getSetCCSwappedOperands(CondCode Operation);

  /// getSetCCOrOperation - Return the result of a logical OR between different
  /// comparisons of identical values: ((X op1 Y) | (X op2 Y)).  This
  /// function returns SETCC_INVALID if it is not possible to represent the
  /// resultant comparison.
  CondCode getSetCCOrOperation(CondCode Op1, CondCode Op2, bool isInteger);

  /// getSetCCAndOperation - Return the result of a logical AND between
  /// different comparisons of identical values: ((X op1 Y) & (X op2 Y)).  This
  /// function returns SETCC_INVALID if it is not possible to represent the
  /// resultant comparison.
  CondCode getSetCCAndOperation(CondCode Op1, CondCode Op2, bool isInteger);
}  // end llvm::ISD namespace


//===----------------------------------------------------------------------===//
/// SDOperand - Unlike LLVM values, Selection DAG nodes may return multiple
/// values as the result of a computation.  Many nodes return multiple values,
/// from loads (which define a token and a return value) to ADDC (which returns
/// a result and a carry value), to calls (which may return an arbitrary number
/// of values).
///
/// As such, each use of a SelectionDAG computation must indicate the node that
/// computes it as well as which return value to use from that node.  This pair
/// of information is represented with the SDOperand value type.
///
struct SDOperand {
  SDNode *Val;        // The node defining the value we are using.
  unsigned ResNo;     // Which return value of the node we are using.

  SDOperand() : Val(0) {}
  SDOperand(SDNode *val, unsigned resno) : Val(val), ResNo(resno) {}

  bool operator==(const SDOperand &O) const {
    return Val == O.Val && ResNo == O.ResNo;
  }
  bool operator!=(const SDOperand &O) const {
    return !operator==(O);
  }
  bool operator<(const SDOperand &O) const {
    return Val < O.Val || (Val == O.Val && ResNo < O.ResNo);
  }

  SDOperand getValue(unsigned R) const {
    return SDOperand(Val, R);
  }

  /// getValueType - Return the ValueType of the referenced return value.
  ///
  inline MVT::ValueType getValueType() const;

  // Forwarding methods - These forward to the corresponding methods in SDNode.
  inline unsigned getOpcode() const;
  inline unsigned getNumOperands() const;
  inline const SDOperand &getOperand(unsigned i) const;
};


/// simplify_type specializations - Allow casting operators to work directly on
/// SDOperands as if they were SDNode*'s.
template<> struct simplify_type<SDOperand> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDOperand &Val) {
    return static_cast<SimpleType>(Val.Val);
  }
};
template<> struct simplify_type<const SDOperand> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDOperand &Val) {
    return static_cast<SimpleType>(Val.Val);
  }
};


/// SDNode - Represents one node in the SelectionDAG.
///
class SDNode {
  unsigned NodeType;
  std::vector<SDOperand> Operands;

  /// Values - The types of the values this node defines.  SDNode's may define
  /// multiple values simultaneously.
  std::vector<MVT::ValueType> Values;

  /// Uses - These are all of the SDNode's that use a value produced by this
  /// node.
  std::vector<SDNode*> Uses;
public:

  //===--------------------------------------------------------------------===//
  //  Accessors
  //
  unsigned getOpcode()  const { return NodeType; }

  size_t use_size() const { return Uses.size(); }
  bool use_empty() const { return Uses.empty(); }
  bool hasOneUse() const { return Uses.size() == 1; }

  /// getNumOperands - Return the number of values used by this operation.
  ///
  unsigned getNumOperands() const { return Operands.size(); }

  const SDOperand &getOperand(unsigned Num) {
    assert(Num < Operands.size() && "Invalid child # of SDNode!");
    return Operands[Num];
  }

  const SDOperand &getOperand(unsigned Num) const {
    assert(Num < Operands.size() && "Invalid child # of SDNode!");
    return Operands[Num];
  }

  /// getNumValues - Return the number of values defined/returned by this
  /// operator.
  ///
  unsigned getNumValues() const { return Values.size(); }

  /// getValueType - Return the type of a specified result.
  ///
  MVT::ValueType getValueType(unsigned ResNo) const {
    assert(ResNo < Values.size() && "Illegal result number!");
    return Values[ResNo];
  }

  void dump() const;

  static bool classof(const SDNode *) { return true; }

protected:
  friend class SelectionDAG;

  SDNode(unsigned NT, MVT::ValueType VT) : NodeType(NT) {
    Values.reserve(1);
    Values.push_back(VT);
  }

  SDNode(unsigned NT, SDOperand Op)
    : NodeType(NT) {
    Operands.reserve(1); Operands.push_back(Op);
    Op.Val->Uses.push_back(this);
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2)
    : NodeType(NT) {
    Operands.reserve(2); Operands.push_back(N1); Operands.push_back(N2);
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2, SDOperand N3)
    : NodeType(NT) {
    Operands.reserve(3); Operands.push_back(N1); Operands.push_back(N2);
    Operands.push_back(N3);
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    N3.Val->Uses.push_back(this);
  }
  SDNode(unsigned NT, std::vector<SDOperand> &Nodes) : NodeType(NT) {
    Operands.swap(Nodes);
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
      Nodes[i].Val->Uses.push_back(this);
  }

  virtual ~SDNode() {
    // FIXME: Drop uses.
  }

  void setValueTypes(MVT::ValueType VT) {
    Values.reserve(1);
    Values.push_back(VT);
  }
  void setValueTypes(MVT::ValueType VT1, MVT::ValueType VT2) {
    Values.reserve(2);
    Values.push_back(VT1);
    Values.push_back(VT2);
  }
  /// Note: this method destroys the vector passed in.
  void setValueTypes(std::vector<MVT::ValueType> &VTs) {
    std::swap(Values, VTs);
  }
};


// Define inline functions from the SDOperand class.

inline unsigned SDOperand::getOpcode() const {
  return Val->getOpcode();
}
inline MVT::ValueType SDOperand::getValueType() const {
  return Val->getValueType(ResNo);
}
inline unsigned SDOperand::getNumOperands() const {
  return Val->getNumOperands();
}
inline const SDOperand &SDOperand::getOperand(unsigned i) const {
  return Val->getOperand(i);
}



class ConstantSDNode : public SDNode {
  uint64_t Value;
protected:
  friend class SelectionDAG;
  ConstantSDNode(uint64_t val, MVT::ValueType VT)
    : SDNode(ISD::Constant, VT), Value(val) {
  }
public:

  uint64_t getValue() const { return Value; }

  int64_t getSignExtended() const {
    unsigned Bits = MVT::getSizeInBits(getValueType(0));
    return ((int64_t)Value << 64-Bits) >> 64-Bits;
  }

  bool isNullValue() const { return Value == 0; }
  bool isAllOnesValue() const {
    return Value == (1ULL << MVT::getSizeInBits(getValueType(0)))-1;
  }

  static bool classof(const ConstantSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Constant;
  }
};

class ConstantFPSDNode : public SDNode {
  double Value;
protected:
  friend class SelectionDAG;
  ConstantFPSDNode(double val, MVT::ValueType VT)
    : SDNode(ISD::ConstantFP, VT), Value(val) {
  }
public:

  double getValue() const { return Value; }

  /// isExactlyValue - We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.
  bool isExactlyValue(double V) const {
    union {
      double V;
      uint64_t I;
    } T1;
    T1.V = Value;
    union {
      double V;
      uint64_t I;
    } T2;
    T2.V = V;
    return T1.I == T2.I;
  }

  static bool classof(const ConstantFPSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantFP;
  }
};

class GlobalAddressSDNode : public SDNode {
  GlobalValue *TheGlobal;
protected:
  friend class SelectionDAG;
  GlobalAddressSDNode(const GlobalValue *GA, MVT::ValueType VT)
    : SDNode(ISD::GlobalAddress, VT) {
    TheGlobal = const_cast<GlobalValue*>(GA);

  }
public:

  GlobalValue *getGlobal() const { return TheGlobal; }

  static bool classof(const GlobalAddressSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::GlobalAddress;
  }
};


class FrameIndexSDNode : public SDNode {
  int FI;
protected:
  friend class SelectionDAG;
  FrameIndexSDNode(int fi, MVT::ValueType VT)
    : SDNode(ISD::FrameIndex, VT), FI(fi) {}
public:

  int getIndex() const { return FI; }

  static bool classof(const FrameIndexSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::FrameIndex;
  }
};

class ConstantPoolSDNode : public SDNode {
  unsigned CPI;
protected:
  friend class SelectionDAG;
  ConstantPoolSDNode(unsigned cpi, MVT::ValueType VT)
    : SDNode(ISD::ConstantPool, VT), CPI(cpi) {}
public:

  unsigned getIndex() const { return CPI; }

  static bool classof(const ConstantPoolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantPool;
  }
};

class BasicBlockSDNode : public SDNode {
  MachineBasicBlock *MBB;
protected:
  friend class SelectionDAG;
  BasicBlockSDNode(MachineBasicBlock *mbb)
    : SDNode(ISD::BasicBlock, MVT::Other), MBB(mbb) {}
public:

  MachineBasicBlock *getBasicBlock() const { return MBB; }

  static bool classof(const BasicBlockSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BasicBlock;
  }
};


class CopyRegSDNode : public SDNode {
  unsigned Reg;
protected:
  friend class SelectionDAG;
  CopyRegSDNode(SDOperand Chain, SDOperand Src, unsigned reg)
    : SDNode(ISD::CopyToReg, Chain, Src), Reg(reg) {
    setValueTypes(MVT::Other);  // Just a token chain.
  }
  CopyRegSDNode(unsigned reg, MVT::ValueType VT)
    : SDNode(ISD::CopyFromReg, VT), Reg(reg) {
  }
public:

  unsigned getReg() const { return Reg; }

  static bool classof(const CopyRegSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CopyToReg ||
           N->getOpcode() == ISD::CopyFromReg;
  }
};

class ExternalSymbolSDNode : public SDNode {
  const char *Symbol;
protected:
  friend class SelectionDAG;
  ExternalSymbolSDNode(const char *Sym, MVT::ValueType VT)
    : SDNode(ISD::ExternalSymbol, VT), Symbol(Sym) {
    }
public:

  const char *getSymbol() const { return Symbol; }

  static bool classof(const ExternalSymbolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ExternalSymbol;
  }
};

class SetCCSDNode : public SDNode {
  ISD::CondCode Condition;
protected:
  friend class SelectionDAG;
  SetCCSDNode(ISD::CondCode Cond, SDOperand LHS, SDOperand RHS)
    : SDNode(ISD::SETCC, LHS, RHS), Condition(Cond) {
    setValueTypes(MVT::i1);
  }
public:

  ISD::CondCode getCondition() const { return Condition; }

  static bool classof(const SetCCSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::SETCC;
  }
};

} // end llvm namespace

#endif
