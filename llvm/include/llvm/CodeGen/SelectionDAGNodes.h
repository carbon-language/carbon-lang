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
#include "llvm/Value.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/iterator"
#include "llvm/Support/DataTypes.h"
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
    // EntryToken - This is the marker used to indicate the start of the region.
    EntryToken,

    // Token factor - This node is takes multiple tokens as input and produces a
    // single token result.  This is used to represent the fact that the operand
    // operators are independent of each other.
    TokenFactor,

    // Various leaf nodes.
    Constant, ConstantFP, GlobalAddress, FrameIndex, ConstantPool,
    BasicBlock, ExternalSymbol,

    // CopyToReg - This node has chain and child nodes, and an associated
    // register number.  The instruction selector must guarantee that the value
    // of the value node is available in the register stored in the RegSDNode
    // object.
    CopyToReg,

    // CopyFromReg - This node indicates that the input value is a virtual or
    // physical register that is defined outside of the scope of this
    // SelectionDAG.  The register is available from the RegSDNode object.
    CopyFromReg,

    // ImplicitDef - This node indicates that the specified register is
    // implicitly defined by some operation (e.g. its a live-in argument).  This
    // register is indicated in the RegSDNode object.  The only operand to this
    // is the token chain coming in, the only result is the token chain going
    // out.
    ImplicitDef,

    // UNDEF - An undefined node
    UNDEF,

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

    // MULHU/MULHS - Multiply high - Multiply two integers of type iN, producing
    // an unsigned/signed value of type i[2*n], then return the top part.
    MULHU, MULHS,

    // Bitwise operators.
    AND, OR, XOR, SHL, SRA, SRL,

    // Counting operators
    CTTZ, CTLZ, CTPOP,

    // Select operator.
    SELECT,

    // SetCC operator - This evaluates to a boolean (i1) true value if the
    // condition is true.  These nodes are instances of the
    // SetCCSDNode class, which contains the condition code as extra
    // state.
    SETCC,

    // ADD_PARTS/SUB_PARTS - These operators take two logical operands which are
    // broken into a multiple pieces each, and return the resulting pieces of
    // doing an atomic add/sub operation.  This is used to handle add/sub of
    // expanded types.  The operation ordering is:
    //       [Lo,Hi] = op [LoLHS,HiLHS], [LoRHS,HiRHS]
    ADD_PARTS, SUB_PARTS,

    // SHL_PARTS/SRA_PARTS/SRL_PARTS - These operators are used for expanded
    // integer shift operations, just like ADD/SUB_PARTS.  The operation
    // ordering is:
    //       [Lo,Hi] = op [LoLHS,HiLHS], Amt
    SHL_PARTS, SRA_PARTS, SRL_PARTS,

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

    // [SU]INT_TO_FP - These operators convert integers (whose interpreted sign
    // depends on the first letter) to floating point.
    SINT_TO_FP,
    UINT_TO_FP,

    // SIGN_EXTEND_INREG - This operator atomically performs a SHL/SRA pair to
    // sign extend a small value in a large integer register (e.g. sign
    // extending the low 8 bits of a 32-bit register to fill the top 24 bits
    // with the 7th bit).  The size of the smaller type is indicated by the
    // ExtraValueType in the MVTSDNode for the operator.
    SIGN_EXTEND_INREG,

    // FP_TO_[US]INT - Convert a floating point value to a signed or unsigned
    // integer.
    FP_TO_SINT,
    FP_TO_UINT,

    // FP_ROUND - Perform a rounding operation from the current
    // precision down to the specified precision (currently always 64->32).
    FP_ROUND,

    // FP_ROUND_INREG - This operator takes a floating point register, and
    // rounds it to a floating point value.  It then promotes it and returns it
    // in a register of the same size.  This operation effectively just discards
    // excess precision.  The type to round down to is specified by the
    // ExtraValueType in the MVTSDNode (currently always 64->32->64).
    FP_ROUND_INREG,

    // FP_EXTEND - Extend a smaller FP type into a larger FP type.
    FP_EXTEND,

    // FNEG, FABS, FSQRT, FSIN, FCOS - Perform unary floating point negation,
    // absolute value, square root, sine and cosine operations.
    FNEG, FABS, FSQRT, FSIN, FCOS,

    // Other operators.  LOAD and STORE have token chains as their first
    // operand, then the same operands as an LLVM load/store instruction.
    LOAD, STORE,

    // EXTLOAD, SEXTLOAD, ZEXTLOAD - These three operators are instances of the
    // MVTSDNode.  All of these load a value from memory and extend them to a
    // larger value (e.g. load a byte into a word register).  All three of these
    // have two operands, a chain and a pointer to load from.  The extra value
    // type is the source type being loaded.
    //
    // SEXTLOAD loads the integer operand and sign extends it to a larger
    //          integer result type.
    // ZEXTLOAD loads the integer operand and zero extends it to a larger
    //          integer result type.
    // EXTLOAD  is used for two things: floating point extending loads, and
    //          integer extending loads where it doesn't matter what the high
    //          bits are set to.  The code generator is allowed to codegen this
    //          into whichever operation is more efficient.
    EXTLOAD, SEXTLOAD, ZEXTLOAD,

    // TRUNCSTORE - This operators truncates (for integer) or rounds (for FP) a
    // value and stores it to memory in one operation.  This can be used for
    // either integer or floating point operands, and the stored type
    // represented as the 'extra' value type in the MVTSDNode representing the
    // operator.  This node has the same three operands as a standard store.
    TRUNCSTORE,

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

    // BRCONDTWOWAY - Two-way conditional branch.  The first operand is the
    // chain, the second is the condition, the third is the block to branch to
    // if true, and the forth is the block to branch to if false.  Targets
    // usually do not implement this, preferring to have legalize demote the
    // operation to BRCOND/BR pairs when necessary.
    BRCONDTWOWAY,

    // RET - Return from function.  The first operand is the chain,
    // and any subsequent operands are the return values for the
    // function.  This operation can have variable number of operands.
    RET,

    // CALL - Call to a function pointer.  The first operand is the chain, the
    // second is the destination function pointer (a GlobalAddress for a direct
    // call).  Arguments have already been lowered to explicit DAGs according to
    // the calling convention in effect here.
    CALL,

    // MEMSET/MEMCPY/MEMMOVE - The first operand is the chain, and the rest
    // correspond to the operands of the LLVM intrinsic functions.  The only
    // result is a token chain.  The alignment argument is guaranteed to be a
    // Constant node.
    MEMSET,
    MEMMOVE,
    MEMCPY,

    // ADJCALLSTACKDOWN/ADJCALLSTACKUP - These operators mark the beginning and
    // end of a call sequence and indicate how much the stack pointer needs to
    // be adjusted for that particular call.  The first operand is a chain, the
    // second is a ConstantSDNode of intptr type.
    ADJCALLSTACKDOWN,  // Beginning of a call sequence
    ADJCALLSTACKUP,    // End of a call sequence

    // SRCVALUE - This corresponds to a Value*, and is used to associate memory
    // locations with their value.  This allows one use alias analysis
    // information in the backend.
    SRCVALUE,

    // PCMARKER - This corresponds to the pcmarker intrinsic.
    PCMARKER,

    // READPORT, WRITEPORT, READIO, WRITEIO - These correspond to the LLVM
    // intrinsics of the same name.  The first operand is a token chain, the
    // other operands match the intrinsic.  These produce a token chain in
    // addition to a value (if any).
    READPORT, WRITEPORT, READIO, WRITEIO,

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
class SDOperand {
public:
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
  inline unsigned getNodeDepth() const;
  inline unsigned getNumOperands() const;
  inline const SDOperand &getOperand(unsigned i) const;

  /// hasOneUse - Return true if there is exactly one operation using this
  /// result value of the defining operator.
  inline bool hasOneUse() const;
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
  /// NodeType - The operation that this node performs.
  ///
  unsigned short NodeType;

  /// NodeDepth - Node depth is defined as MAX(Node depth of children)+1.  This
  /// means that leaves have a depth of 1, things that use only leaves have a
  /// depth of 2, etc.
  unsigned short NodeDepth;

  /// Operands - The values that are used by this operation.
  ///
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

  /// getNodeDepth - Return the distance from this node to the leaves in the
  /// graph.  The leaves have a depth of 1.
  unsigned getNodeDepth() const { return NodeDepth; }

  typedef std::vector<SDNode*>::const_iterator use_iterator;
  use_iterator use_begin() const { return Uses.begin(); }
  use_iterator use_end() const { return Uses.end(); }

  /// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
  /// indicated value.  This method ignores uses of other values defined by this
  /// operation.
  bool hasNUsesOfValue(unsigned NUses, unsigned Value);

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

  /// getOperationName - Return the opcode of this operation for printing.
  ///
  const char* getOperationName() const;
  void dump() const;

  static bool classof(const SDNode *) { return true; }

protected:
  friend class SelectionDAG;

  SDNode(unsigned NT, MVT::ValueType VT) : NodeType(NT), NodeDepth(1) {
    Values.reserve(1);
    Values.push_back(VT);
  }
  SDNode(unsigned NT, SDOperand Op)
    : NodeType(NT), NodeDepth(Op.Val->getNodeDepth()+1) {
    Operands.reserve(1); Operands.push_back(Op);
    Op.Val->Uses.push_back(this);
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2)
    : NodeType(NT) {
    if (N1.Val->getNodeDepth() > N2.Val->getNodeDepth())
      NodeDepth = N1.Val->getNodeDepth()+1;
    else
      NodeDepth = N2.Val->getNodeDepth()+1;
    Operands.reserve(2); Operands.push_back(N1); Operands.push_back(N2);
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2, SDOperand N3)
    : NodeType(NT) {
    unsigned ND = N1.Val->getNodeDepth();
    if (ND < N2.Val->getNodeDepth())
      ND = N2.Val->getNodeDepth();
    if (ND < N3.Val->getNodeDepth())
      ND = N3.Val->getNodeDepth();
    NodeDepth = ND+1;

    Operands.reserve(3); Operands.push_back(N1); Operands.push_back(N2);
    Operands.push_back(N3);
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    N3.Val->Uses.push_back(this);
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2, SDOperand N3, SDOperand N4)
    : NodeType(NT) {
    unsigned ND = N1.Val->getNodeDepth();
    if (ND < N2.Val->getNodeDepth())
      ND = N2.Val->getNodeDepth();
    if (ND < N3.Val->getNodeDepth())
      ND = N3.Val->getNodeDepth();
    if (ND < N4.Val->getNodeDepth())
      ND = N4.Val->getNodeDepth();
    NodeDepth = ND+1;

    Operands.reserve(4); Operands.push_back(N1); Operands.push_back(N2);
    Operands.push_back(N3); Operands.push_back(N4);
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    N3.Val->Uses.push_back(this); N4.Val->Uses.push_back(this);
  }
  SDNode(unsigned NT, std::vector<SDOperand> &Nodes) : NodeType(NT) {
    Operands.swap(Nodes);
    unsigned ND = 0;
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      Operands[i].Val->Uses.push_back(this);
      if (ND < Operands[i].Val->getNodeDepth())
        ND = Operands[i].Val->getNodeDepth();
    }
    NodeDepth = ND+1;
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

  void removeUser(SDNode *User) {
    // Remove this user from the operand's use list.
    for (unsigned i = Uses.size(); ; --i) {
      assert(i != 0 && "Didn't find user!");
      if (Uses[i-1] == User) {
        Uses.erase(Uses.begin()+i-1);
        break;
      }
    }
  }
};


// Define inline functions from the SDOperand class.

inline unsigned SDOperand::getOpcode() const {
  return Val->getOpcode();
}
inline unsigned SDOperand::getNodeDepth() const {
  return Val->getNodeDepth();
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
inline bool SDOperand::hasOneUse() const {
  return Val->hasNUsesOfValue(1, ResNo);
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
    return ((int64_t)Value << (64-Bits)) >> (64-Bits);
  }

  bool isNullValue() const { return Value == 0; }
  bool isAllOnesValue() const {
    int NumBits = MVT::getSizeInBits(getValueType(0));
    if (NumBits == 64) return Value+1 == 0;
    return Value == (1ULL << NumBits)-1;
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

class SrcValueSDNode : public SDNode {
  const Value *V;
  int offset;
protected:
  friend class SelectionDAG;
  SrcValueSDNode(const Value* v, int o)
    : SDNode(ISD::SRCVALUE, MVT::Other), V(v), offset(o) {}

public:
  const Value *getValue() const { return V; }
  int getOffset() const { return offset; }

  static bool classof(const SrcValueSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::SRCVALUE;
  }
};


class RegSDNode : public SDNode {
  unsigned Reg;
protected:
  friend class SelectionDAG;
  RegSDNode(unsigned Opc, SDOperand Chain, SDOperand Src, unsigned reg)
    : SDNode(Opc, Chain, Src), Reg(reg) {
  }
  RegSDNode(unsigned Opc, SDOperand Chain, unsigned reg)
    : SDNode(Opc, Chain), Reg(reg) {}
public:

  unsigned getReg() const { return Reg; }

  static bool classof(const RegSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CopyToReg ||
           N->getOpcode() == ISD::CopyFromReg ||
           N->getOpcode() == ISD::ImplicitDef;
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
  }
public:

  ISD::CondCode getCondition() const { return Condition; }

  static bool classof(const SetCCSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::SETCC;
  }
};

/// MVTSDNode - This class is used for operators that require an extra
/// value-type to be kept with the node.
class MVTSDNode : public SDNode {
  MVT::ValueType ExtraValueType;
protected:
  friend class SelectionDAG;
  MVTSDNode(unsigned Opc, MVT::ValueType VT1, SDOperand Op0, MVT::ValueType EVT)
    : SDNode(Opc, Op0), ExtraValueType(EVT) {
    setValueTypes(VT1);
  }
  MVTSDNode(unsigned Opc, MVT::ValueType VT1, MVT::ValueType VT2,
            SDOperand Op0, SDOperand Op1, SDOperand Op2, MVT::ValueType EVT)
    : SDNode(Opc, Op0, Op1, Op2), ExtraValueType(EVT) {
    setValueTypes(VT1, VT2);
  }

  MVTSDNode(unsigned Opc, MVT::ValueType VT,
            SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3, MVT::ValueType EVT)
    : SDNode(Opc, Op0, Op1, Op2, Op3), ExtraValueType(EVT) {
    setValueTypes(VT);
  }
public:

  MVT::ValueType getExtraValueType() const { return ExtraValueType; }

  static bool classof(const MVTSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return
      N->getOpcode() == ISD::SIGN_EXTEND_INREG ||
      N->getOpcode() == ISD::FP_ROUND_INREG ||
      N->getOpcode() == ISD::EXTLOAD  ||
      N->getOpcode() == ISD::SEXTLOAD ||
      N->getOpcode() == ISD::ZEXTLOAD ||
      N->getOpcode() == ISD::TRUNCSTORE;
  }
};

class SDNodeIterator : public forward_iterator<SDNode, ptrdiff_t> {
  SDNode *Node;
  unsigned Operand;

  SDNodeIterator(SDNode *N, unsigned Op) : Node(N), Operand(Op) {}
public:
  bool operator==(const SDNodeIterator& x) const {
    return Operand == x.Operand;
  }
  bool operator!=(const SDNodeIterator& x) const { return !operator==(x); }

  const SDNodeIterator &operator=(const SDNodeIterator &I) {
    assert(I.Node == Node && "Cannot assign iterators to two different nodes!");
    Operand = I.Operand;
    return *this;
  }

  pointer operator*() const {
    return Node->getOperand(Operand).Val;
  }
  pointer operator->() const { return operator*(); }

  SDNodeIterator& operator++() {                // Preincrement
    ++Operand;
    return *this;
  }
  SDNodeIterator operator++(int) { // Postincrement
    SDNodeIterator tmp = *this; ++*this; return tmp;
  }

  static SDNodeIterator begin(SDNode *N) { return SDNodeIterator(N, 0); }
  static SDNodeIterator end  (SDNode *N) {
    return SDNodeIterator(N, N->getNumOperands());
  }

  unsigned getOperand() const { return Operand; }
  const SDNode *getNode() const { return Node; }
};

template <> struct GraphTraits<SDNode*> {
  typedef SDNode NodeType;
  typedef SDNodeIterator ChildIteratorType;
  static inline NodeType *getEntryNode(SDNode *N) { return N; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return SDNodeIterator::begin(N);
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return SDNodeIterator::end(N);
  }
};




} // end llvm namespace

#endif
