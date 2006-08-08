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
template <typename T> struct ilist_traits;
template<typename NodeTy, typename Traits> class iplist;
template<typename NodeTy> class ilist_iterator;

/// ISD namespace - This namespace contains an enum which represents all of the
/// SelectionDAG node types and value types.
///
namespace ISD {
  //===--------------------------------------------------------------------===//
  /// ISD::NodeType enum - This enum defines all of the operators valid in a
  /// SelectionDAG.
  ///
  enum NodeType {
    // DELETED_NODE - This is an illegal flag value that is used to catch
    // errors.  This opcode is not a legal opcode for any node.
    DELETED_NODE,
    
    // EntryToken - This is the marker used to indicate the start of the region.
    EntryToken,

    // Token factor - This node takes multiple tokens as input and produces a
    // single token result.  This is used to represent the fact that the operand
    // operators are independent of each other.
    TokenFactor,
    
    // AssertSext, AssertZext - These nodes record if a register contains a 
    // value that has already been zero or sign extended from a narrower type.  
    // These nodes take two operands.  The first is the node that has already 
    // been extended, and the second is a value type node indicating the width
    // of the extension
    AssertSext, AssertZext,

    // Various leaf nodes.
    STRING, BasicBlock, VALUETYPE, CONDCODE, Register,
    Constant, ConstantFP,
    GlobalAddress, FrameIndex, JumpTable, ConstantPool, ExternalSymbol,

    // TargetConstant* - Like Constant*, but the DAG does not do any folding or
    // simplification of the constant.
    TargetConstant,
    TargetConstantFP,
    
    // TargetGlobalAddress - Like GlobalAddress, but the DAG does no folding or
    // anything else with this node, and this is valid in the target-specific
    // dag, turning into a GlobalAddress operand.
    TargetGlobalAddress,
    TargetFrameIndex,
    TargetJumpTable,
    TargetConstantPool,
    TargetExternalSymbol,
    
    /// RESULT = INTRINSIC_WO_CHAIN(INTRINSICID, arg1, arg2, ...)
    /// This node represents a target intrinsic function with no side effects.
    /// The first operand is the ID number of the intrinsic from the
    /// llvm::Intrinsic namespace.  The operands to the intrinsic follow.  The
    /// node has returns the result of the intrinsic.
    INTRINSIC_WO_CHAIN,
    
    /// RESULT,OUTCHAIN = INTRINSIC_W_CHAIN(INCHAIN, INTRINSICID, arg1, ...)
    /// This node represents a target intrinsic function with side effects that
    /// returns a result.  The first operand is a chain pointer.  The second is
    /// the ID number of the intrinsic from the llvm::Intrinsic namespace.  The
    /// operands to the intrinsic follow.  The node has two results, the result
    /// of the intrinsic and an output chain.
    INTRINSIC_W_CHAIN,

    /// OUTCHAIN = INTRINSIC_VOID(INCHAIN, INTRINSICID, arg1, arg2, ...)
    /// This node represents a target intrinsic function with side effects that
    /// does not return a result.  The first operand is a chain pointer.  The
    /// second is the ID number of the intrinsic from the llvm::Intrinsic
    /// namespace.  The operands to the intrinsic follow.
    INTRINSIC_VOID,
    
    // CopyToReg - This node has three operands: a chain, a register number to
    // set to this value, and a value.  
    CopyToReg,

    // CopyFromReg - This node indicates that the input value is a virtual or
    // physical register that is defined outside of the scope of this
    // SelectionDAG.  The register is available from the RegSDNode object.
    CopyFromReg,

    // UNDEF - An undefined node
    UNDEF,
    
    /// FORMAL_ARGUMENTS(CHAIN, CC#, ISVARARG) - This node represents the formal
    /// arguments for a function.  CC# is a Constant value indicating the
    /// calling convention of the function, and ISVARARG is a flag that
    /// indicates whether the function is varargs or not.  This node has one
    /// result value for each incoming argument, plus one for the output chain.
    /// It must be custom legalized.
    /// 
    FORMAL_ARGUMENTS,
    
    /// RV1, RV2...RVn, CHAIN = CALL(CHAIN, CC#, ISVARARG, ISTAILCALL, CALLEE,
    ///                              ARG0, SIGN0, ARG1, SIGN1, ... ARGn, SIGNn)
    /// This node represents a fully general function call, before the legalizer
    /// runs.  This has one result value for each argument / signness pair, plus
    /// a chain result. It must be custom legalized.
    CALL,

    // EXTRACT_ELEMENT - This is used to get the first or second (determined by
    // a Constant, which is required to be operand #1), element of the aggregate
    // value specified as operand #0.  This is only for use before legalization,
    // for values that will be broken into multiple registers.
    EXTRACT_ELEMENT,

    // BUILD_PAIR - This is the opposite of EXTRACT_ELEMENT in some ways.  Given
    // two values of the same integer value type, this produces a value twice as
    // big.  Like EXTRACT_ELEMENT, this can only be used before legalization.
    BUILD_PAIR,
    
    // MERGE_VALUES - This node takes multiple discrete operands and returns
    // them all as its individual results.  This nodes has exactly the same
    // number of inputs and outputs, and is only valid before legalization.
    // This node is useful for some pieces of the code generator that want to
    // think about a single node with multiple results, not multiple nodes.
    MERGE_VALUES,

    // Simple integer binary arithmetic operators.
    ADD, SUB, MUL, SDIV, UDIV, SREM, UREM,
    
    // Carry-setting nodes for multiple precision addition and subtraction.
    // These nodes take two operands of the same value type, and produce two
    // results.  The first result is the normal add or sub result, the second
    // result is the carry flag result.
    ADDC, SUBC,
    
    // Carry-using nodes for multiple precision addition and subtraction.  These
    // nodes take three operands: The first two are the normal lhs and rhs to
    // the add or sub, and the third is the input carry flag.  These nodes
    // produce two results; the normal result of the add or sub, and the output
    // carry flag.  These nodes both read and write a carry flag to allow them
    // to them to be chained together for add and sub of arbitrarily large
    // values.
    ADDE, SUBE,
    
    // Simple binary floating point operators.
    FADD, FSUB, FMUL, FDIV, FREM,

    // FCOPYSIGN(X, Y) - Return the value of X with the sign of Y.  NOTE: This
    // DAG node does not require that X and Y have the same type, just that they
    // are both floating point.  X and the result must have the same type.
    // FCOPYSIGN(f32, f64) is allowed.
    FCOPYSIGN,

    /// VBUILD_VECTOR(ELT1, ELT2, ELT3, ELT4,...,  COUNT,TYPE) - Return a vector
    /// with the specified, possibly variable, elements.  The number of elements
    /// is required to be a power of two.
    VBUILD_VECTOR,

    /// BUILD_VECTOR(ELT1, ELT2, ELT3, ELT4,...) - Return a vector
    /// with the specified, possibly variable, elements.  The number of elements
    /// is required to be a power of two.
    BUILD_VECTOR,
    
    /// VINSERT_VECTOR_ELT(VECTOR, VAL, IDX,  COUNT,TYPE) - Given a vector
    /// VECTOR, an element ELEMENT, and a (potentially variable) index IDX,
    /// return an vector with the specified element of VECTOR replaced with VAL.
    /// COUNT and TYPE specify the type of vector, as is standard for V* nodes.
    VINSERT_VECTOR_ELT,
    
    /// INSERT_VECTOR_ELT(VECTOR, VAL, IDX) - Returns VECTOR (a legal packed
    /// type) with the element at IDX replaced with VAL.
    INSERT_VECTOR_ELT,

    /// VEXTRACT_VECTOR_ELT(VECTOR, IDX) - Returns a single element from VECTOR
    /// (an MVT::Vector value) identified by the (potentially variable) element
    /// number IDX.
    VEXTRACT_VECTOR_ELT,
    
    /// EXTRACT_VECTOR_ELT(VECTOR, IDX) - Returns a single element from VECTOR
    /// (a legal packed type vector) identified by the (potentially variable)
    /// element number IDX.
    EXTRACT_VECTOR_ELT,
    
    /// VVECTOR_SHUFFLE(VEC1, VEC2, SHUFFLEVEC, COUNT,TYPE) - Returns a vector,
    /// of the same type as VEC1/VEC2.  SHUFFLEVEC is a VBUILD_VECTOR of
    /// constant int values that indicate which value each result element will
    /// get.  The elements of VEC1/VEC2 are enumerated in order.  This is quite
    /// similar to the Altivec 'vperm' instruction, except that the indices must
    /// be constants and are in terms of the element size of VEC1/VEC2, not in
    /// terms of bytes.
    VVECTOR_SHUFFLE,

    /// VECTOR_SHUFFLE(VEC1, VEC2, SHUFFLEVEC) - Returns a vector, of the same
    /// type as VEC1/VEC2.  SHUFFLEVEC is a BUILD_VECTOR of constant int values
    /// (regardless of whether its datatype is legal or not) that indicate
    /// which value each result element will get.  The elements of VEC1/VEC2 are
    /// enumerated in order.  This is quite similar to the Altivec 'vperm'
    /// instruction, except that the indices must be constants and are in terms
    /// of the element size of VEC1/VEC2, not in terms of bytes.
    VECTOR_SHUFFLE,
    
    /// X = VBIT_CONVERT(Y)  and X = VBIT_CONVERT(Y, COUNT,TYPE) - This node
    /// represents a conversion from or to an ISD::Vector type.
    ///
    /// This is lowered to a BIT_CONVERT of the appropriate input/output types.
    /// The input and output are required to have the same size and at least one
    /// is required to be a vector (if neither is a vector, just use
    /// BIT_CONVERT).
    ///
    /// If the result is a vector, this takes three operands (like any other
    /// vector producer) which indicate the size and type of the vector result.
    /// Otherwise it takes one input.
    VBIT_CONVERT,
    
    /// BINOP(LHS, RHS,  COUNT,TYPE)
    /// Simple abstract vector operators.  Unlike the integer and floating point
    /// binary operators, these nodes also take two additional operands:
    /// a constant element count, and a value type node indicating the type of
    /// the elements.  The order is count, type, op0, op1.  All vector opcodes,
    /// including VLOAD and VConstant must currently have count and type as
    /// their last two operands.
    VADD, VSUB, VMUL, VSDIV, VUDIV,
    VAND, VOR, VXOR,
    
    /// VSELECT(COND,LHS,RHS,  COUNT,TYPE) - Select for MVT::Vector values.
    /// COND is a boolean value.  This node return LHS if COND is true, RHS if
    /// COND is false.
    VSELECT,
    
    /// SCALAR_TO_VECTOR(VAL) - This represents the operation of loading a
    /// scalar value into the low element of the resultant vector type.  The top
    /// elements of the vector are undefined.
    SCALAR_TO_VECTOR,
    
    // MULHU/MULHS - Multiply high - Multiply two integers of type iN, producing
    // an unsigned/signed value of type i[2*n], then return the top part.
    MULHU, MULHS,

    // Bitwise operators - logical and, logical or, logical xor, shift left,
    // shift right algebraic (shift in sign bits), shift right logical (shift in
    // zeroes), rotate left, rotate right, and byteswap.
    AND, OR, XOR, SHL, SRA, SRL, ROTL, ROTR, BSWAP,

    // Counting operators
    CTTZ, CTLZ, CTPOP,

    // Select(COND, TRUEVAL, FALSEVAL)
    SELECT, 
    
    // Select with condition operator - This selects between a true value and 
    // a false value (ops #2 and #3) based on the boolean result of comparing
    // the lhs and rhs (ops #0 and #1) of a conditional expression with the 
    // condition code in op #4, a CondCodeSDNode.
    SELECT_CC,

    // SetCC operator - This evaluates to a boolean (i1) true value if the
    // condition is true.  The operands to this are the left and right operands
    // to compare (ops #0, and #1) and the condition code to compare them with
    // (op #2) as a CondCodeSDNode.
    SETCC,

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

    // ANY_EXTEND - Used for integer types.  The high bits are undefined.
    ANY_EXTEND,
    
    // TRUNCATE - Completely drop the high bits.
    TRUNCATE,

    // [SU]INT_TO_FP - These operators convert integers (whose interpreted sign
    // depends on the first letter) to floating point.
    SINT_TO_FP,
    UINT_TO_FP,

    // SIGN_EXTEND_INREG - This operator atomically performs a SHL/SRA pair to
    // sign extend a small value in a large integer register (e.g. sign
    // extending the low 8 bits of a 32-bit register to fill the top 24 bits
    // with the 7th bit).  The size of the smaller type is indicated by the 1th
    // operand, a ValueType node.
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
    // excess precision.  The type to round down to is specified by the 1th
    // operation, a VTSDNode (currently always 64->32->64).
    FP_ROUND_INREG,

    // FP_EXTEND - Extend a smaller FP type into a larger FP type.
    FP_EXTEND,

    // BIT_CONVERT - Theis operator converts between integer and FP values, as
    // if one was stored to memory as integer and the other was loaded from the
    // same address (or equivalently for vector format conversions, etc).  The 
    // source and result are required to have the same bit size (e.g. 
    // f32 <-> i32).  This can also be used for int-to-int or fp-to-fp 
    // conversions, but that is a noop, deleted by getNode().
    BIT_CONVERT,
    
    // FNEG, FABS, FSQRT, FSIN, FCOS - Perform unary floating point negation,
    // absolute value, square root, sine and cosine operations.
    FNEG, FABS, FSQRT, FSIN, FCOS,
    
    // Other operators.  LOAD and STORE have token chains as their first
    // operand, then the same operands as an LLVM load/store instruction, then a
    // SRCVALUE node that provides alias analysis information.
    LOAD, STORE,
    
    // Abstract vector version of LOAD.  VLOAD has a constant element count as
    // the first operand, followed by a value type node indicating the type of
    // the elements, a token chain, a pointer operand, and a SRCVALUE node.
    VLOAD,

    // EXTLOAD, SEXTLOAD, ZEXTLOAD - These three operators all load a value from
    // memory and extend them to a larger value (e.g. load a byte into a word
    // register).  All three of these have four operands, a token chain, a
    // pointer to load from, a SRCVALUE for alias analysis, and a VALUETYPE node
    // indicating the type to load.
    //
    // SEXTLOAD loads the integer operand and sign extends it to a larger
    //          integer result type.
    // ZEXTLOAD loads the integer operand and zero extends it to a larger
    //          integer result type.
    // EXTLOAD  is used for three things: floating point extending loads, 
    //          integer extending loads [the top bits are undefined], and vector
    //          extending loads [load into low elt].
    EXTLOAD, SEXTLOAD, ZEXTLOAD,

    // TRUNCSTORE - This operators truncates (for integer) or rounds (for FP) a
    // value and stores it to memory in one operation.  This can be used for
    // either integer or floating point operands.  The first four operands of
    // this are the same as a standard store.  The fifth is the ValueType to
    // store it as (which will be smaller than the source value).
    TRUNCSTORE,

    // DYNAMIC_STACKALLOC - Allocate some number of bytes on the stack aligned
    // to a specified boundary.  The first operand is the token chain, the
    // second is the number of bytes to allocate, and the third is the alignment
    // boundary.  The size is guaranteed to be a multiple of the stack 
    // alignment, and the alignment is guaranteed to be bigger than the stack 
    // alignment (if required) or 0 to get standard stack alignment.
    DYNAMIC_STACKALLOC,

    // Control flow instructions.  These all have token chains.

    // BR - Unconditional branch.  The first operand is the chain
    // operand, the second is the MBB to branch to.
    BR,

    // BRIND - Indirect branch.  The first operand is the chain, the second
    // is the value to branch to, which must be of the same type as the target's
    // pointer type.
    BRIND,
    
    // BRCOND - Conditional branch.  The first operand is the chain,
    // the second is the condition, the third is the block to branch
    // to if the condition is true.
    BRCOND,

    // BR_CC - Conditional branch.  The behavior is like that of SELECT_CC, in
    // that the condition is represented as condition code, and two nodes to
    // compare, rather than as a combined SetCC node.  The operands in order are
    // chain, cc, lhs, rhs, block to branch to if condition is true.
    BR_CC,
    
    // RET - Return from function.  The first operand is the chain,
    // and any subsequent operands are pairs of return value and return value
    // signness for the function.  This operation can have variable number of
    // operands.
    RET,

    // INLINEASM - Represents an inline asm block.  This node always has two
    // return values: a chain and a flag result.  The inputs are as follows:
    //   Operand #0   : Input chain.
    //   Operand #1   : a ExternalSymbolSDNode with a pointer to the asm string.
    //   Operand #2n+2: A RegisterNode.
    //   Operand #2n+3: A TargetConstant, indicating if the reg is a use/def
    //   Operand #last: Optional, an incoming flag.
    INLINEASM,

    // STACKSAVE - STACKSAVE has one operand, an input chain.  It produces a
    // value, the same type as the pointer type for the system, and an output
    // chain.
    STACKSAVE,
    
    // STACKRESTORE has two operands, an input chain and a pointer to restore to
    // it returns an output chain.
    STACKRESTORE,
    
    // MEMSET/MEMCPY/MEMMOVE - The first operand is the chain, and the rest
    // correspond to the operands of the LLVM intrinsic functions.  The only
    // result is a token chain.  The alignment argument is guaranteed to be a
    // Constant node.
    MEMSET,
    MEMMOVE,
    MEMCPY,

    // CALLSEQ_START/CALLSEQ_END - These operators mark the beginning and end of
    // a call sequence, and carry arbitrary information that target might want
    // to know.  The first operand is a chain, the rest are specified by the
    // target and not touched by the DAG optimizers.
    CALLSEQ_START,  // Beginning of a call sequence
    CALLSEQ_END,    // End of a call sequence
    
    // VAARG - VAARG has three operands: an input chain, a pointer, and a 
    // SRCVALUE.  It returns a pair of values: the vaarg value and a new chain.
    VAARG,
    
    // VACOPY - VACOPY has five operands: an input chain, a destination pointer,
    // a source pointer, a SRCVALUE for the destination, and a SRCVALUE for the
    // source.
    VACOPY,
    
    // VAEND, VASTART - VAEND and VASTART have three operands: an input chain, a
    // pointer, and a SRCVALUE.
    VAEND, VASTART,

    // SRCVALUE - This corresponds to a Value*, and is used to associate memory
    // locations with their value.  This allows one use alias analysis
    // information in the backend.
    SRCVALUE,

    // PCMARKER - This corresponds to the pcmarker intrinsic.
    PCMARKER,

    // READCYCLECOUNTER - This corresponds to the readcyclecounter intrinsic.
    // The only operand is a chain and a value and a chain are produced.  The
    // value is the contents of the architecture specific cycle counter like 
    // register (or other high accuracy low latency clock source)
    READCYCLECOUNTER,

    // HANDLENODE node - Used as a handle for various purposes.
    HANDLENODE,

    // LOCATION - This node is used to represent a source location for debug
    // info.  It takes token chain as input, then a line number, then a column
    // number, then a filename, then a working dir.  It produces a token chain
    // as output.
    LOCATION,
    
    // DEBUG_LOC - This node is used to represent source line information
    // embedded in the code.  It takes a token chain as input, then a line
    // number, then a column then a file id (provided by MachineDebugInfo.) It
    // produces a token chain as output.
    DEBUG_LOC,
    
    // DEBUG_LABEL - This node is used to mark a location in the code where a
    // label should be generated for use by the debug information.  It takes a
    // token chain as input and then a unique id (provided by MachineDebugInfo.)
    // It produces a token chain as output.
    DEBUG_LABEL,
    
    // BUILTIN_OP_END - This must be the last enum value in this list.
    BUILTIN_OP_END
  };

  /// Node predicates

  /// isBuildVectorAllOnes - Return true if the specified node is a
  /// BUILD_VECTOR where all of the elements are ~0 or undef.
  bool isBuildVectorAllOnes(const SDNode *N);

  /// isBuildVectorAllZeros - Return true if the specified node is a
  /// BUILD_VECTOR where all of the elements are 0 or undef.
  bool isBuildVectorAllZeros(const SDNode *N);
  
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

    SETCC_INVALID       // Marker value.
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

  SDOperand() : Val(0), ResNo(0) {}
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

  // isOperand - Return true if this node is an operand of N.
  bool isOperand(SDNode *N) const;

  /// getValueType - Return the ValueType of the referenced return value.
  ///
  inline MVT::ValueType getValueType() const;

  // Forwarding methods - These forward to the corresponding methods in SDNode.
  inline unsigned getOpcode() const;
  inline unsigned getNumOperands() const;
  inline const SDOperand &getOperand(unsigned i) const;
  inline bool isTargetOpcode() const;
  inline unsigned getTargetOpcode() const;

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

  /// NodeId - Unique id per SDNode in the DAG.
  int NodeId;

  /// OperandList - The values that are used by this operation.
  ///
  SDOperand *OperandList;
  
  /// ValueList - The types of the values this node defines.  SDNode's may
  /// define multiple values simultaneously.
  MVT::ValueType *ValueList;

  /// NumOperands/NumValues - The number of entries in the Operand/Value list.
  unsigned short NumOperands, NumValues;
  
  /// Prev/Next pointers - These pointers form the linked list of of the
  /// AllNodes list in the current DAG.
  SDNode *Prev, *Next;
  friend struct ilist_traits<SDNode>;

  /// NextInBucket - This is used by the SelectionDAGCSEMap.
  void *NextInBucket;
  
  /// Uses - These are all of the SDNode's that use a value produced by this
  /// node.
  std::vector<SDNode*> Uses;
  
  // Out-of-line virtual method to give class a home.
  virtual void ANCHOR();
public:
  virtual ~SDNode() {
    assert(NumOperands == 0 && "Operand list not cleared before deletion");
    NodeType = ISD::DELETED_NODE;
  }
  
  //===--------------------------------------------------------------------===//
  //  Accessors
  //
  unsigned getOpcode()  const { return NodeType; }
  bool isTargetOpcode() const { return NodeType >= ISD::BUILTIN_OP_END; }
  unsigned getTargetOpcode() const {
    assert(isTargetOpcode() && "Not a target opcode!");
    return NodeType - ISD::BUILTIN_OP_END;
  }

  size_t use_size() const { return Uses.size(); }
  bool use_empty() const { return Uses.empty(); }
  bool hasOneUse() const { return Uses.size() == 1; }

  /// getNodeId - Return the unique node id.
  ///
  int getNodeId() const { return NodeId; }

  typedef std::vector<SDNode*>::const_iterator use_iterator;
  use_iterator use_begin() const { return Uses.begin(); }
  use_iterator use_end() const { return Uses.end(); }

  /// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
  /// indicated value.  This method ignores uses of other values defined by this
  /// operation.
  bool hasNUsesOfValue(unsigned NUses, unsigned Value) const;

  // isOnlyUse - Return true if this node is the only use of N.
  bool isOnlyUse(SDNode *N) const;

  // isOperand - Return true if this node is an operand of N.
  bool isOperand(SDNode *N) const;

  /// getNumOperands - Return the number of values used by this operation.
  ///
  unsigned getNumOperands() const { return NumOperands; }

  const SDOperand &getOperand(unsigned Num) const {
    assert(Num < NumOperands && "Invalid child # of SDNode!");
    return OperandList[Num];
  }
  typedef const SDOperand* op_iterator;
  op_iterator op_begin() const { return OperandList; }
  op_iterator op_end() const { return OperandList+NumOperands; }


  /// getNumValues - Return the number of values defined/returned by this
  /// operator.
  ///
  unsigned getNumValues() const { return NumValues; }

  /// getValueType - Return the type of a specified result.
  ///
  MVT::ValueType getValueType(unsigned ResNo) const {
    assert(ResNo < NumValues && "Illegal result number!");
    return ValueList[ResNo];
  }

  typedef const MVT::ValueType* value_iterator;
  value_iterator value_begin() const { return ValueList; }
  value_iterator value_end() const { return ValueList+NumValues; }

  /// getOperationName - Return the opcode of this operation for printing.
  ///
  const char* getOperationName(const SelectionDAG *G = 0) const;
  void dump() const;
  void dump(const SelectionDAG *G) const;

  static bool classof(const SDNode *) { return true; }

  
  /// NextInBucket accessors, these are private to SelectionDAGCSEMap.
  void *getNextInBucket() const { return NextInBucket; }
  void SetNextInBucket(void *N) { NextInBucket = N; }
  
protected:
  friend class SelectionDAG;
  
  /// getValueTypeList - Return a pointer to the specified value type.
  ///
  static MVT::ValueType *getValueTypeList(MVT::ValueType VT);

  SDNode(unsigned NT, MVT::ValueType VT) : NodeType(NT), NodeId(-1) {
    OperandList = 0; NumOperands = 0;
    ValueList = getValueTypeList(VT);
    NumValues = 1;
    Prev = 0; Next = 0;
    NextInBucket = 0;
  }
  SDNode(unsigned NT, SDOperand Op)
    : NodeType(NT), NodeId(-1) {
    OperandList = new SDOperand[1];
    OperandList[0] = Op;
    NumOperands = 1;
    Op.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
    NextInBucket = 0;
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2)
    : NodeType(NT), NodeId(-1) {
    OperandList = new SDOperand[2];
    OperandList[0] = N1;
    OperandList[1] = N2;
    NumOperands = 2;
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
    NextInBucket = 0;
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2, SDOperand N3)
    : NodeType(NT), NodeId(-1) {
    OperandList = new SDOperand[3];
    OperandList[0] = N1;
    OperandList[1] = N2;
    OperandList[2] = N3;
    NumOperands = 3;
    
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    N3.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
    NextInBucket = 0;
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2, SDOperand N3, SDOperand N4)
    : NodeType(NT), NodeId(-1) {
    OperandList = new SDOperand[4];
    OperandList[0] = N1;
    OperandList[1] = N2;
    OperandList[2] = N3;
    OperandList[3] = N4;
    NumOperands = 4;
    
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    N3.Val->Uses.push_back(this); N4.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
    NextInBucket = 0;
  }
  SDNode(unsigned Opc, const SDOperand *Ops, unsigned NumOps)
    : NodeType(Opc), NodeId(-1) {
    NumOperands = NumOps;
    OperandList = new SDOperand[NumOperands];
    
    for (unsigned i = 0, e = NumOps; i != e; ++i) {
      OperandList[i] = Ops[i];
      SDNode *N = OperandList[i].Val;
      N->Uses.push_back(this);
    }
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
    NextInBucket = 0;
  }

  /// MorphNodeTo - This clears the return value and operands list, and sets the
  /// opcode of the node to the specified value.  This should only be used by
  /// the SelectionDAG class.
  void MorphNodeTo(unsigned Opc) {
    NodeType = Opc;
    ValueList = 0;
    NumValues = 0;
    
    // Clear the operands list, updating used nodes to remove this from their
    // use list.
    for (op_iterator I = op_begin(), E = op_end(); I != E; ++I)
      I->Val->removeUser(this);
    delete [] OperandList;
    OperandList = 0;
    NumOperands = 0;
  }
  
  void setValueTypes(MVT::ValueType *List, unsigned NumVal) {
    assert(NumValues == 0 && "Should not have values yet!");
    ValueList = List;
    NumValues = NumVal;
  }
  
  void setOperands(SDOperand Op0) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[1];
    OperandList[0] = Op0;
    NumOperands = 1;
    Op0.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[2];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    NumOperands = 2;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[3];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    NumOperands = 3;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[4];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    NumOperands = 4;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[5];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    NumOperands = 5;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4, SDOperand Op5) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[6];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    OperandList[5] = Op5;
    NumOperands = 6;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this); Op5.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4, SDOperand Op5, SDOperand Op6) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[7];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    OperandList[5] = Op5;
    OperandList[6] = Op6;
    NumOperands = 7;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this); Op5.Val->Uses.push_back(this);
    Op6.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4, SDOperand Op5, SDOperand Op6, SDOperand Op7) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[8];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    OperandList[5] = Op5;
    OperandList[6] = Op6;
    OperandList[7] = Op7;
    NumOperands = 8;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this); Op5.Val->Uses.push_back(this);
    Op6.Val->Uses.push_back(this); Op7.Val->Uses.push_back(this);
  }

  void addUser(SDNode *User) {
    Uses.push_back(User);
  }
  void removeUser(SDNode *User) {
    // Remove this user from the operand's use list.
    for (unsigned i = Uses.size(); ; --i) {
      assert(i != 0 && "Didn't find user!");
      if (Uses[i-1] == User) {
        Uses[i-1] = Uses.back();
        Uses.pop_back();
        return;
      }
    }
  }

  void setNodeId(int Id) {
    NodeId = Id;
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
inline bool SDOperand::isTargetOpcode() const {
  return Val->isTargetOpcode();
}
inline unsigned SDOperand::getTargetOpcode() const {
  return Val->getTargetOpcode();
}
inline bool SDOperand::hasOneUse() const {
  return Val->hasNUsesOfValue(1, ResNo);
}

/// HandleSDNode - This class is used to form a handle around another node that
/// is persistant and is updated across invocations of replaceAllUsesWith on its
/// operand.  This node should be directly created by end-users and not added to
/// the AllNodes list.
class HandleSDNode : public SDNode {
public:
  HandleSDNode(SDOperand X) : SDNode(ISD::HANDLENODE, X) {}
  ~HandleSDNode() {
    MorphNodeTo(ISD::HANDLENODE);  // Drops operand uses.
  }
  
  SDOperand getValue() const { return getOperand(0); }
};

class StringSDNode : public SDNode {
  std::string Value;
protected:
  friend class SelectionDAG;
  StringSDNode(const std::string &val)
    : SDNode(ISD::STRING, MVT::Other), Value(val) {
  }
public:
  const std::string &getValue() const { return Value; }
  static bool classof(const StringSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::STRING;
  }
};  

class ConstantSDNode : public SDNode {
  uint64_t Value;
protected:
  friend class SelectionDAG;
  ConstantSDNode(bool isTarget, uint64_t val, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetConstant : ISD::Constant, VT), Value(val) {
  }
public:

  uint64_t getValue() const { return Value; }

  int64_t getSignExtended() const {
    unsigned Bits = MVT::getSizeInBits(getValueType(0));
    return ((int64_t)Value << (64-Bits)) >> (64-Bits);
  }

  bool isNullValue() const { return Value == 0; }
  bool isAllOnesValue() const {
    return Value == MVT::getIntVTBitMask(getValueType(0));
  }

  static bool classof(const ConstantSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Constant ||
           N->getOpcode() == ISD::TargetConstant;
  }
};

class ConstantFPSDNode : public SDNode {
  double Value;
protected:
  friend class SelectionDAG;
  ConstantFPSDNode(bool isTarget, double val, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetConstantFP : ISD::ConstantFP, VT), 
      Value(val) {
  }
public:

  double getValue() const { return Value; }

  /// isExactlyValue - We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.
  bool isExactlyValue(double V) const;

  static bool classof(const ConstantFPSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantFP || 
           N->getOpcode() == ISD::TargetConstantFP;
  }
};

class GlobalAddressSDNode : public SDNode {
  GlobalValue *TheGlobal;
  int Offset;
protected:
  friend class SelectionDAG;
  GlobalAddressSDNode(bool isTarget, const GlobalValue *GA, MVT::ValueType VT,
                      int o=0)
    : SDNode(isTarget ? ISD::TargetGlobalAddress : ISD::GlobalAddress, VT),
      Offset(o) {
    TheGlobal = const_cast<GlobalValue*>(GA);
  }
public:

  GlobalValue *getGlobal() const { return TheGlobal; }
  int getOffset() const { return Offset; }

  static bool classof(const GlobalAddressSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::GlobalAddress ||
           N->getOpcode() == ISD::TargetGlobalAddress;
  }
};


class FrameIndexSDNode : public SDNode {
  int FI;
protected:
  friend class SelectionDAG;
  FrameIndexSDNode(int fi, MVT::ValueType VT, bool isTarg)
    : SDNode(isTarg ? ISD::TargetFrameIndex : ISD::FrameIndex, VT), FI(fi) {}
public:

  int getIndex() const { return FI; }

  static bool classof(const FrameIndexSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::FrameIndex ||
           N->getOpcode() == ISD::TargetFrameIndex;
  }
};

class JumpTableSDNode : public SDNode {
  int JTI;
protected:
  friend class SelectionDAG;
  JumpTableSDNode(int jti, MVT::ValueType VT, bool isTarg)
    : SDNode(isTarg ? ISD::TargetJumpTable : ISD::JumpTable, VT), 
    JTI(jti) {}
public:
    
    int getIndex() const { return JTI; }
  
  static bool classof(const JumpTableSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::JumpTable ||
           N->getOpcode() == ISD::TargetJumpTable;
  }
};

class ConstantPoolSDNode : public SDNode {
  Constant *C;
  int Offset;
  unsigned Alignment;
protected:
  friend class SelectionDAG;
  ConstantPoolSDNode(bool isTarget, Constant *c, MVT::ValueType VT,
                     int o=0)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool, VT),
      C(c), Offset(o), Alignment(0) {}
  ConstantPoolSDNode(bool isTarget, Constant *c, MVT::ValueType VT, int o,
                     unsigned Align)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool, VT),
      C(c), Offset(o), Alignment(Align) {}
public:

  Constant *get() const { return C; }
  int getOffset() const { return Offset; }
  
  // Return the alignment of this constant pool object, which is either 0 (for
  // default alignment) or log2 of the desired value.
  unsigned getAlignment() const { return Alignment; }

  static bool classof(const ConstantPoolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantPool ||
           N->getOpcode() == ISD::TargetConstantPool;
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


class RegisterSDNode : public SDNode {
  unsigned Reg;
protected:
  friend class SelectionDAG;
  RegisterSDNode(unsigned reg, MVT::ValueType VT)
    : SDNode(ISD::Register, VT), Reg(reg) {}
public:

  unsigned getReg() const { return Reg; }

  static bool classof(const RegisterSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Register;
  }
};

class ExternalSymbolSDNode : public SDNode {
  const char *Symbol;
protected:
  friend class SelectionDAG;
  ExternalSymbolSDNode(bool isTarget, const char *Sym, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetExternalSymbol : ISD::ExternalSymbol, VT),
      Symbol(Sym) {
    }
public:

  const char *getSymbol() const { return Symbol; }

  static bool classof(const ExternalSymbolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ExternalSymbol ||
           N->getOpcode() == ISD::TargetExternalSymbol;
  }
};

class CondCodeSDNode : public SDNode {
  ISD::CondCode Condition;
protected:
  friend class SelectionDAG;
  CondCodeSDNode(ISD::CondCode Cond)
    : SDNode(ISD::CONDCODE, MVT::Other), Condition(Cond) {
  }
public:

  ISD::CondCode get() const { return Condition; }

  static bool classof(const CondCodeSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CONDCODE;
  }
};

/// VTSDNode - This class is used to represent MVT::ValueType's, which are used
/// to parameterize some operations.
class VTSDNode : public SDNode {
  MVT::ValueType ValueType;
protected:
  friend class SelectionDAG;
  VTSDNode(MVT::ValueType VT)
    : SDNode(ISD::VALUETYPE, MVT::Other), ValueType(VT) {}
public:

  MVT::ValueType getVT() const { return ValueType; }

  static bool classof(const VTSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::VALUETYPE;
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

template<>
struct ilist_traits<SDNode> {
  static SDNode *getPrev(const SDNode *N) { return N->Prev; }
  static SDNode *getNext(const SDNode *N) { return N->Next; }
  
  static void setPrev(SDNode *N, SDNode *Prev) { N->Prev = Prev; }
  static void setNext(SDNode *N, SDNode *Next) { N->Next = Next; }
  
  static SDNode *createSentinel() {
    return new SDNode(ISD::EntryToken, MVT::Other);
  }
  static void destroySentinel(SDNode *N) { delete N; }
  //static SDNode *createNode(const SDNode &V) { return new SDNode(V); }
  
  
  void addNodeToList(SDNode *NTy) {}
  void removeNodeFromList(SDNode *NTy) {}
  void transferNodesFromList(iplist<SDNode, ilist_traits> &L2,
                             const ilist_iterator<SDNode> &X,
                             const ilist_iterator<SDNode> &Y) {}
};

} // end llvm namespace

#endif
