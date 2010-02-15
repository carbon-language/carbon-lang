//===-- llvm/CodeGen/SelectionDAGNodes.h - SelectionDAG Nodes ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "llvm/Constants.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/System/DataTypes.h"
#include "llvm/Support/DebugLoc.h"
#include <cassert>

namespace llvm {

class SelectionDAG;
class GlobalValue;
class MachineBasicBlock;
class MachineConstantPoolValue;
class SDNode;
class Value;
template <typename T> struct DenseMapInfo;
template <typename T> struct simplify_type;
template <typename T> struct ilist_traits;

void checkForCycles(const SDNode *N);
  
/// SDVTList - This represents a list of ValueType's that has been intern'd by
/// a SelectionDAG.  Instances of this simple value class are returned by
/// SelectionDAG::getVTList(...).
///
struct SDVTList {
  const EVT *VTs;
  unsigned int NumVTs;
};

/// ISD namespace - This namespace contains an enum which represents all of the
/// SelectionDAG node types and value types.
///
namespace ISD {

  //===--------------------------------------------------------------------===//
  /// ISD::NodeType enum - This enum defines the target-independent operators
  /// for a SelectionDAG.
  ///
  /// Targets may also define target-dependent operator codes for SDNodes. For
  /// example, on x86, these are the enum values in the X86ISD namespace.
  /// Targets should aim to use target-independent operators to model their
  /// instruction sets as much as possible, and only use target-dependent
  /// operators when they have special requirements.
  ///
  /// Finally, during and after selection proper, SNodes may use special
  /// operator codes that correspond directly with MachineInstr opcodes. These
  /// are used to represent selected instructions. See the isMachineOpcode()
  /// and getMachineOpcode() member functions of SDNode.
  ///
  enum NodeType {
    // DELETED_NODE - This is an illegal value that is used to catch
    // errors.  This opcode is not a legal opcode for any node.
    DELETED_NODE,

    // EntryToken - This is the marker used to indicate the start of the region.
    EntryToken,

    // TokenFactor - This node takes multiple tokens as input and produces a
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
    BasicBlock, VALUETYPE, CONDCODE, Register,
    Constant, ConstantFP,
    GlobalAddress, GlobalTLSAddress, FrameIndex,
    JumpTable, ConstantPool, ExternalSymbol, BlockAddress,

    // The address of the GOT
    GLOBAL_OFFSET_TABLE,

    // FRAMEADDR, RETURNADDR - These nodes represent llvm.frameaddress and
    // llvm.returnaddress on the DAG.  These nodes take one operand, the index
    // of the frame or return address to return.  An index of zero corresponds
    // to the current function's frame or return address, an index of one to the
    // parent's frame or return address, and so on.
    FRAMEADDR, RETURNADDR,

    // FRAME_TO_ARGS_OFFSET - This node represents offset from frame pointer to
    // first (possible) on-stack argument. This is needed for correct stack
    // adjustment during unwind.
    FRAME_TO_ARGS_OFFSET,

    // RESULT, OUTCHAIN = EXCEPTIONADDR(INCHAIN) - This node represents the
    // address of the exception block on entry to an landing pad block.
    EXCEPTIONADDR,

    // RESULT, OUTCHAIN = LSDAADDR(INCHAIN) - This node represents the
    // address of the Language Specific Data Area for the enclosing function.
    LSDAADDR,

    // RESULT, OUTCHAIN = EHSELECTION(INCHAIN, EXCEPTION) - This node represents
    // the selection index of the exception thrown.
    EHSELECTION,

    // OUTCHAIN = EH_RETURN(INCHAIN, OFFSET, HANDLER) - This node represents
    // 'eh_return' gcc dwarf builtin, which is used to return from
    // exception. The general meaning is: adjust stack by OFFSET and pass
    // execution to HANDLER. Many platform-related details also :)
    EH_RETURN,

    // TargetConstant* - Like Constant*, but the DAG does not do any folding or
    // simplification of the constant.
    TargetConstant,
    TargetConstantFP,

    // TargetGlobalAddress - Like GlobalAddress, but the DAG does no folding or
    // anything else with this node, and this is valid in the target-specific
    // dag, turning into a GlobalAddress operand.
    TargetGlobalAddress,
    TargetGlobalTLSAddress,
    TargetFrameIndex,
    TargetJumpTable,
    TargetConstantPool,
    TargetExternalSymbol,
    TargetBlockAddress,

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
    // SelectionDAG.  The register is available from the RegisterSDNode object.
    CopyFromReg,

    // UNDEF - An undefined node
    UNDEF,

    // EXTRACT_ELEMENT - This is used to get the lower or upper (determined by
    // a Constant, which is required to be operand #1) half of the integer or
    // float value specified as operand #0.  This is only for use before
    // legalization, for values that will be broken into multiple registers.
    EXTRACT_ELEMENT,

    // BUILD_PAIR - This is the opposite of EXTRACT_ELEMENT in some ways.  Given
    // two values of the same integer value type, this produces a value twice as
    // big.  Like EXTRACT_ELEMENT, this can only be used before legalization.
    BUILD_PAIR,

    // MERGE_VALUES - This node takes multiple discrete operands and returns
    // them all as its individual results.  This nodes has exactly the same
    // number of inputs and outputs. This node is useful for some pieces of the
    // code generator that want to think about a single node with multiple
    // results, not multiple nodes.
    MERGE_VALUES,

    // Simple integer binary arithmetic operators.
    ADD, SUB, MUL, SDIV, UDIV, SREM, UREM,

    // SMUL_LOHI/UMUL_LOHI - Multiply two integers of type iN, producing
    // a signed/unsigned value of type i[2*N], and return the full value as
    // two results, each of type iN.
    SMUL_LOHI, UMUL_LOHI,

    // SDIVREM/UDIVREM - Divide two integers and produce both a quotient and
    // remainder result.
    SDIVREM, UDIVREM,

    // CARRY_FALSE - This node is used when folding other nodes,
    // like ADDC/SUBC, which indicate the carry result is always false.
    CARRY_FALSE,

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

    // RESULT, BOOL = [SU]ADDO(LHS, RHS) - Overflow-aware nodes for addition.
    // These nodes take two operands: the normal LHS and RHS to the add. They
    // produce two results: the normal result of the add, and a boolean that
    // indicates if an overflow occured (*not* a flag, because it may be stored
    // to memory, etc.).  If the type of the boolean is not i1 then the high
    // bits conform to getBooleanContents.
    // These nodes are generated from the llvm.[su]add.with.overflow intrinsics.
    SADDO, UADDO,

    // Same for subtraction
    SSUBO, USUBO,

    // Same for multiplication
    SMULO, UMULO,

    // Simple binary floating point operators.
    FADD, FSUB, FMUL, FDIV, FREM,

    // FCOPYSIGN(X, Y) - Return the value of X with the sign of Y.  NOTE: This
    // DAG node does not require that X and Y have the same type, just that they
    // are both floating point.  X and the result must have the same type.
    // FCOPYSIGN(f32, f64) is allowed.
    FCOPYSIGN,

    // INT = FGETSIGN(FP) - Return the sign bit of the specified floating point
    // value as an integer 0/1 value.
    FGETSIGN,

    /// BUILD_VECTOR(ELT0, ELT1, ELT2, ELT3,...) - Return a vector with the
    /// specified, possibly variable, elements.  The number of elements is
    /// required to be a power of two.  The types of the operands must all be
    /// the same and must match the vector element type, except that integer
    /// types are allowed to be larger than the element type, in which case
    /// the operands are implicitly truncated.
    BUILD_VECTOR,

    /// INSERT_VECTOR_ELT(VECTOR, VAL, IDX) - Returns VECTOR with the element
    /// at IDX replaced with VAL.  If the type of VAL is larger than the vector
    /// element type then VAL is truncated before replacement.
    INSERT_VECTOR_ELT,

    /// EXTRACT_VECTOR_ELT(VECTOR, IDX) - Returns a single element from VECTOR
    /// identified by the (potentially variable) element number IDX.  If the
    /// return type is an integer type larger than the element type of the
    /// vector, the result is extended to the width of the return type.
    EXTRACT_VECTOR_ELT,

    /// CONCAT_VECTORS(VECTOR0, VECTOR1, ...) - Given a number of values of
    /// vector type with the same length and element type, this produces a
    /// concatenated vector result value, with length equal to the sum of the
    /// lengths of the input vectors.
    CONCAT_VECTORS,

    /// EXTRACT_SUBVECTOR(VECTOR, IDX) - Returns a subvector from VECTOR (an
    /// vector value) starting with the (potentially variable) element number
    /// IDX, which must be a multiple of the result vector length.
    EXTRACT_SUBVECTOR,

    /// VECTOR_SHUFFLE(VEC1, VEC2) - Returns a vector, of the same type as 
    /// VEC1/VEC2.  A VECTOR_SHUFFLE node also contains an array of constant int
    /// values that indicate which value (or undef) each result element will
    /// get.  These constant ints are accessible through the 
    /// ShuffleVectorSDNode class.  This is quite similar to the Altivec 
    /// 'vperm' instruction, except that the indices must be constants and are
    /// in terms of the element size of VEC1/VEC2, not in terms of bytes.
    VECTOR_SHUFFLE,

    /// SCALAR_TO_VECTOR(VAL) - This represents the operation of loading a
    /// scalar value into element 0 of the resultant vector type.  The top
    /// elements 1 to N-1 of the N-element vector are undefined.  The type
    /// of the operand must match the vector element type, except when they
    /// are integer types.  In this case the operand is allowed to be wider
    /// than the vector element type, and is implicitly truncated to it.
    SCALAR_TO_VECTOR,

    // MULHU/MULHS - Multiply high - Multiply two integers of type iN, producing
    // an unsigned/signed value of type i[2*N], then return the top part.
    MULHU, MULHS,

    // Bitwise operators - logical and, logical or, logical xor, shift left,
    // shift right algebraic (shift in sign bits), shift right logical (shift in
    // zeroes), rotate left, rotate right, and byteswap.
    AND, OR, XOR, SHL, SRA, SRL, ROTL, ROTR, BSWAP,

    // Counting operators
    CTTZ, CTLZ, CTPOP,

    // Select(COND, TRUEVAL, FALSEVAL).  If the type of the boolean COND is not
    // i1 then the high bits must conform to getBooleanContents.
    SELECT,

    // Select with condition operator - This selects between a true value and
    // a false value (ops #2 and #3) based on the boolean result of comparing
    // the lhs and rhs (ops #0 and #1) of a conditional expression with the
    // condition code in op #4, a CondCodeSDNode.
    SELECT_CC,

    // SetCC operator - This evaluates to a true value iff the condition is
    // true.  If the result value type is not i1 then the high bits conform
    // to getBooleanContents.  The operands to this are the left and right
    // operands to compare (ops #0, and #1) and the condition code to compare
    // them with (op #2) as a CondCodeSDNode.
    SETCC,

    // RESULT = VSETCC(LHS, RHS, COND) operator - This evaluates to a vector of
    // integer elements with all bits of the result elements set to true if the
    // comparison is true or all cleared if the comparison is false.  The
    // operands to this are the left and right operands to compare (LHS/RHS) and
    // the condition code to compare them with (COND) as a CondCodeSDNode.
    VSETCC,

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

    /// FP_TO_[US]INT - Convert a floating point value to a signed or unsigned
    /// integer.
    FP_TO_SINT,
    FP_TO_UINT,

    /// X = FP_ROUND(Y, TRUNC) - Rounding 'Y' from a larger floating point type
    /// down to the precision of the destination VT.  TRUNC is a flag, which is
    /// always an integer that is zero or one.  If TRUNC is 0, this is a
    /// normal rounding, if it is 1, this FP_ROUND is known to not change the
    /// value of Y.
    ///
    /// The TRUNC = 1 case is used in cases where we know that the value will
    /// not be modified by the node, because Y is not using any of the extra
    /// precision of source type.  This allows certain transformations like
    /// FP_EXTEND(FP_ROUND(X,1)) -> X which are not safe for
    /// FP_EXTEND(FP_ROUND(X,0)) because the extra bits aren't removed.
    FP_ROUND,

    // FLT_ROUNDS_ - Returns current rounding mode:
    // -1 Undefined
    //  0 Round to 0
    //  1 Round to nearest
    //  2 Round to +inf
    //  3 Round to -inf
    FLT_ROUNDS_,

    /// X = FP_ROUND_INREG(Y, VT) - This operator takes an FP register, and
    /// rounds it to a floating point value.  It then promotes it and returns it
    /// in a register of the same size.  This operation effectively just
    /// discards excess precision.  The type to round down to is specified by
    /// the VT operand, a VTSDNode.
    FP_ROUND_INREG,

    /// X = FP_EXTEND(Y) - Extend a smaller FP type into a larger FP type.
    FP_EXTEND,

    // BIT_CONVERT - This operator converts between integer, vector and FP
    // values, as if the value was stored to memory with one type and loaded
    // from the same address with the other type (or equivalently for vector
    // format conversions, etc).  The source and result are required to have
    // the same bit size (e.g.  f32 <-> i32).  This can also be used for
    // int-to-int or fp-to-fp conversions, but that is a noop, deleted by
    // getNode().
    BIT_CONVERT,

    // CONVERT_RNDSAT - This operator is used to support various conversions
    // between various types (float, signed, unsigned and vectors of those
    // types) with rounding and saturation. NOTE: Avoid using this operator as
    // most target don't support it and the operator might be removed in the
    // future. It takes the following arguments:
    //   0) value
    //   1) dest type (type to convert to)
    //   2) src type (type to convert from)
    //   3) rounding imm
    //   4) saturation imm
    //   5) ISD::CvtCode indicating the type of conversion to do
    CONVERT_RNDSAT,

    // FNEG, FABS, FSQRT, FSIN, FCOS, FPOWI, FPOW,
    // FLOG, FLOG2, FLOG10, FEXP, FEXP2,
    // FCEIL, FTRUNC, FRINT, FNEARBYINT, FFLOOR - Perform various unary floating
    // point operations. These are inspired by libm.
    FNEG, FABS, FSQRT, FSIN, FCOS, FPOWI, FPOW,
    FLOG, FLOG2, FLOG10, FEXP, FEXP2,
    FCEIL, FTRUNC, FRINT, FNEARBYINT, FFLOOR,

    // LOAD and STORE have token chains as their first operand, then the same
    // operands as an LLVM load/store instruction, then an offset node that
    // is added / subtracted from the base pointer to form the address (for
    // indexed memory ops).
    LOAD, STORE,

    // DYNAMIC_STACKALLOC - Allocate some number of bytes on the stack aligned
    // to a specified boundary.  This node always has two return values: a new
    // stack pointer value and a chain. The first operand is the token chain,
    // the second is the number of bytes to allocate, and the third is the
    // alignment boundary.  The size is guaranteed to be a multiple of the stack
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

    // BR_JT - Jumptable branch. The first operand is the chain, the second
    // is the jumptable index, the last one is the jumptable entry index.
    BR_JT,

    // BRCOND - Conditional branch.  The first operand is the chain, the
    // second is the condition, the third is the block to branch to if the
    // condition is true.  If the type of the condition is not i1, then the
    // high bits must conform to getBooleanContents.
    BRCOND,

    // BR_CC - Conditional branch.  The behavior is like that of SELECT_CC, in
    // that the condition is represented as condition code, and two nodes to
    // compare, rather than as a combined SetCC node.  The operands in order are
    // chain, cc, lhs, rhs, block to branch to if condition is true.
    BR_CC,

    // INLINEASM - Represents an inline asm block.  This node always has two
    // return values: a chain and a flag result.  The inputs are as follows:
    //   Operand #0   : Input chain.
    //   Operand #1   : a ExternalSymbolSDNode with a pointer to the asm string.
    //   Operand #2n+2: A RegisterNode.
    //   Operand #2n+3: A TargetConstant, indicating if the reg is a use/def
    //   Operand #last: Optional, an incoming flag.
    INLINEASM,

    // EH_LABEL - Represents a label in mid basic block used to track
    // locations needed for debug and exception handling tables.  These nodes
    // take a chain as input and return a chain.
    EH_LABEL,

    // STACKSAVE - STACKSAVE has one operand, an input chain.  It produces a
    // value, the same type as the pointer type for the system, and an output
    // chain.
    STACKSAVE,

    // STACKRESTORE has two operands, an input chain and a pointer to restore to
    // it returns an output chain.
    STACKRESTORE,

    // CALLSEQ_START/CALLSEQ_END - These operators mark the beginning and end of
    // a call sequence, and carry arbitrary information that target might want
    // to know.  The first operand is a chain, the rest are specified by the
    // target and not touched by the DAG optimizers.
    // CALLSEQ_START..CALLSEQ_END pairs may not be nested.
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

    // SRCVALUE - This is a node type that holds a Value* that is used to
    // make reference to a value in the LLVM IR.
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

    // TRAMPOLINE - This corresponds to the init_trampoline intrinsic.
    // It takes as input a token chain, the pointer to the trampoline,
    // the pointer to the nested function, the pointer to pass for the
    // 'nest' parameter, a SRCVALUE for the trampoline and another for
    // the nested function (allowing targets to access the original
    // Function*).  It produces the result of the intrinsic and a token
    // chain as output.
    TRAMPOLINE,

    // TRAP - Trapping instruction
    TRAP,

    // PREFETCH - This corresponds to a prefetch intrinsic. It takes chains are
    // their first operand. The other operands are the address to prefetch,
    // read / write specifier, and locality specifier.
    PREFETCH,

    // OUTCHAIN = MEMBARRIER(INCHAIN, load-load, load-store, store-load,
    //                       store-store, device)
    // This corresponds to the memory.barrier intrinsic.
    // it takes an input chain, 4 operands to specify the type of barrier, an
    // operand specifying if the barrier applies to device and uncached memory
    // and produces an output chain.
    MEMBARRIER,

    // Val, OUTCHAIN = ATOMIC_CMP_SWAP(INCHAIN, ptr, cmp, swap)
    // this corresponds to the atomic.lcs intrinsic.
    // cmp is compared to *ptr, and if equal, swap is stored in *ptr.
    // the return is always the original value in *ptr
    ATOMIC_CMP_SWAP,

    // Val, OUTCHAIN = ATOMIC_SWAP(INCHAIN, ptr, amt)
    // this corresponds to the atomic.swap intrinsic.
    // amt is stored to *ptr atomically.
    // the return is always the original value in *ptr
    ATOMIC_SWAP,

    // Val, OUTCHAIN = ATOMIC_LOAD_[OpName](INCHAIN, ptr, amt)
    // this corresponds to the atomic.load.[OpName] intrinsic.
    // op(*ptr, amt) is stored to *ptr atomically.
    // the return is always the original value in *ptr
    ATOMIC_LOAD_ADD,
    ATOMIC_LOAD_SUB,
    ATOMIC_LOAD_AND,
    ATOMIC_LOAD_OR,
    ATOMIC_LOAD_XOR,
    ATOMIC_LOAD_NAND,
    ATOMIC_LOAD_MIN,
    ATOMIC_LOAD_MAX,
    ATOMIC_LOAD_UMIN,
    ATOMIC_LOAD_UMAX,

    /// BUILTIN_OP_END - This must be the last enum value in this list.
    /// The target-specific pre-isel opcode values start here.
    BUILTIN_OP_END
  };

  /// FIRST_TARGET_MEMORY_OPCODE - Target-specific pre-isel operations
  /// which do not reference a specific memory location should be less than
  /// this value. Those that do must not be less than this value, and can
  /// be used with SelectionDAG::getMemIntrinsicNode.
  static const int FIRST_TARGET_MEMORY_OPCODE = BUILTIN_OP_END+80;

  /// Node predicates

  /// isBuildVectorAllOnes - Return true if the specified node is a
  /// BUILD_VECTOR where all of the elements are ~0 or undef.
  bool isBuildVectorAllOnes(const SDNode *N);

  /// isBuildVectorAllZeros - Return true if the specified node is a
  /// BUILD_VECTOR where all of the elements are 0 or undef.
  bool isBuildVectorAllZeros(const SDNode *N);

  /// isScalarToVector - Return true if the specified node is a
  /// ISD::SCALAR_TO_VECTOR node or a BUILD_VECTOR node where only the low
  /// element is not an undef.
  bool isScalarToVector(const SDNode *N);

  //===--------------------------------------------------------------------===//
  /// MemIndexedMode enum - This enum defines the load / store indexed
  /// addressing modes.
  ///
  /// UNINDEXED    "Normal" load / store. The effective address is already
  ///              computed and is available in the base pointer. The offset
  ///              operand is always undefined. In addition to producing a
  ///              chain, an unindexed load produces one value (result of the
  ///              load); an unindexed store does not produce a value.
  ///
  /// PRE_INC      Similar to the unindexed mode where the effective address is
  /// PRE_DEC      the value of the base pointer add / subtract the offset.
  ///              It considers the computation as being folded into the load /
  ///              store operation (i.e. the load / store does the address
  ///              computation as well as performing the memory transaction).
  ///              The base operand is always undefined. In addition to
  ///              producing a chain, pre-indexed load produces two values
  ///              (result of the load and the result of the address
  ///              computation); a pre-indexed store produces one value (result
  ///              of the address computation).
  ///
  /// POST_INC     The effective address is the value of the base pointer. The
  /// POST_DEC     value of the offset operand is then added to / subtracted
  ///              from the base after memory transaction. In addition to
  ///              producing a chain, post-indexed load produces two values
  ///              (the result of the load and the result of the base +/- offset
  ///              computation); a post-indexed store produces one value (the
  ///              the result of the base +/- offset computation).
  ///
  enum MemIndexedMode {
    UNINDEXED = 0,
    PRE_INC,
    PRE_DEC,
    POST_INC,
    POST_DEC,
    LAST_INDEXED_MODE
  };

  //===--------------------------------------------------------------------===//
  /// LoadExtType enum - This enum defines the three variants of LOADEXT
  /// (load with extension).
  ///
  /// SEXTLOAD loads the integer operand and sign extends it to a larger
  ///          integer result type.
  /// ZEXTLOAD loads the integer operand and zero extends it to a larger
  ///          integer result type.
  /// EXTLOAD  is used for three things: floating point extending loads,
  ///          integer extending loads [the top bits are undefined], and vector
  ///          extending loads [load into low elt].
  ///
  enum LoadExtType {
    NON_EXTLOAD = 0,
    EXTLOAD,
    SEXTLOAD,
    ZEXTLOAD,
    LAST_LOADEXT_TYPE
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

  //===--------------------------------------------------------------------===//
  /// CvtCode enum - This enum defines the various converts CONVERT_RNDSAT
  /// supports.
  enum CvtCode {
    CVT_FF,     // Float from Float
    CVT_FS,     // Float from Signed
    CVT_FU,     // Float from Unsigned
    CVT_SF,     // Signed from Float
    CVT_UF,     // Unsigned from Float
    CVT_SS,     // Signed from Signed
    CVT_SU,     // Signed from Unsigned
    CVT_US,     // Unsigned from Signed
    CVT_UU,     // Unsigned from Unsigned
    CVT_INVALID // Marker - Invalid opcode
  };
}  // end llvm::ISD namespace


//===----------------------------------------------------------------------===//
/// SDValue - Unlike LLVM values, Selection DAG nodes may return multiple
/// values as the result of a computation.  Many nodes return multiple values,
/// from loads (which define a token and a return value) to ADDC (which returns
/// a result and a carry value), to calls (which may return an arbitrary number
/// of values).
///
/// As such, each use of a SelectionDAG computation must indicate the node that
/// computes it as well as which return value to use from that node.  This pair
/// of information is represented with the SDValue value type.
///
class SDValue {
  SDNode *Node;       // The node defining the value we are using.
  unsigned ResNo;     // Which return value of the node we are using.
public:
  SDValue() : Node(0), ResNo(0) {}
  SDValue(SDNode *node, unsigned resno) : Node(node), ResNo(resno) {}

  /// get the index which selects a specific result in the SDNode
  unsigned getResNo() const { return ResNo; }

  /// get the SDNode which holds the desired result
  SDNode *getNode() const { return Node; }

  /// set the SDNode
  void setNode(SDNode *N) { Node = N; }

  inline SDNode *operator->() const { return Node; }
  
  bool operator==(const SDValue &O) const {
    return Node == O.Node && ResNo == O.ResNo;
  }
  bool operator!=(const SDValue &O) const {
    return !operator==(O);
  }
  bool operator<(const SDValue &O) const {
    return Node < O.Node || (Node == O.Node && ResNo < O.ResNo);
  }

  SDValue getValue(unsigned R) const {
    return SDValue(Node, R);
  }

  // isOperandOf - Return true if this node is an operand of N.
  bool isOperandOf(SDNode *N) const;

  /// getValueType - Return the ValueType of the referenced return value.
  ///
  inline EVT getValueType() const;

  /// getValueSizeInBits - Returns the size of the value in bits.
  ///
  unsigned getValueSizeInBits() const {
    return getValueType().getSizeInBits();
  }

  // Forwarding methods - These forward to the corresponding methods in SDNode.
  inline unsigned getOpcode() const;
  inline unsigned getNumOperands() const;
  inline const SDValue &getOperand(unsigned i) const;
  inline uint64_t getConstantOperandVal(unsigned i) const;
  inline bool isTargetMemoryOpcode() const;
  inline bool isTargetOpcode() const;
  inline bool isMachineOpcode() const;
  inline unsigned getMachineOpcode() const;
  inline const DebugLoc getDebugLoc() const;


  /// reachesChainWithoutSideEffects - Return true if this operand (which must
  /// be a chain) reaches the specified operand without crossing any
  /// side-effecting instructions.  In practice, this looks through token
  /// factors and non-volatile loads.  In order to remain efficient, this only
  /// looks a couple of nodes in, it does not do an exhaustive search.
  bool reachesChainWithoutSideEffects(SDValue Dest,
                                      unsigned Depth = 2) const;

  /// use_empty - Return true if there are no nodes using value ResNo
  /// of Node.
  ///
  inline bool use_empty() const;

  /// hasOneUse - Return true if there is exactly one node using value
  /// ResNo of Node.
  ///
  inline bool hasOneUse() const;
};


template<> struct DenseMapInfo<SDValue> {
  static inline SDValue getEmptyKey() {
    return SDValue((SDNode*)-1, -1U);
  }
  static inline SDValue getTombstoneKey() {
    return SDValue((SDNode*)-1, 0);
  }
  static unsigned getHashValue(const SDValue &Val) {
    return ((unsigned)((uintptr_t)Val.getNode() >> 4) ^
            (unsigned)((uintptr_t)Val.getNode() >> 9)) + Val.getResNo();
  }
  static bool isEqual(const SDValue &LHS, const SDValue &RHS) {
    return LHS == RHS;
  }
};
template <> struct isPodLike<SDValue> { static const bool value = true; };


/// simplify_type specializations - Allow casting operators to work directly on
/// SDValues as if they were SDNode*'s.
template<> struct simplify_type<SDValue> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDValue &Val) {
    return static_cast<SimpleType>(Val.getNode());
  }
};
template<> struct simplify_type<const SDValue> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDValue &Val) {
    return static_cast<SimpleType>(Val.getNode());
  }
};

/// SDUse - Represents a use of a SDNode. This class holds an SDValue,
/// which records the SDNode being used and the result number, a
/// pointer to the SDNode using the value, and Next and Prev pointers,
/// which link together all the uses of an SDNode.
///
class SDUse {
  /// Val - The value being used.
  SDValue Val;
  /// User - The user of this value.
  SDNode *User;
  /// Prev, Next - Pointers to the uses list of the SDNode referred by
  /// this operand.
  SDUse **Prev, *Next;

  SDUse(const SDUse &U);          // Do not implement
  void operator=(const SDUse &U); // Do not implement

public:
  SDUse() : Val(), User(NULL), Prev(NULL), Next(NULL) {}

  /// Normally SDUse will just implicitly convert to an SDValue that it holds.
  operator const SDValue&() const { return Val; }

  /// If implicit conversion to SDValue doesn't work, the get() method returns
  /// the SDValue.
  const SDValue &get() const { return Val; }

  /// getUser - This returns the SDNode that contains this Use.
  SDNode *getUser() { return User; }

  /// getNext - Get the next SDUse in the use list.
  SDUse *getNext() const { return Next; }

  /// getNode - Convenience function for get().getNode().
  SDNode *getNode() const { return Val.getNode(); }
  /// getResNo - Convenience function for get().getResNo().
  unsigned getResNo() const { return Val.getResNo(); }
  /// getValueType - Convenience function for get().getValueType().
  EVT getValueType() const { return Val.getValueType(); }

  /// operator== - Convenience function for get().operator==
  bool operator==(const SDValue &V) const {
    return Val == V;
  }

  /// operator!= - Convenience function for get().operator!=
  bool operator!=(const SDValue &V) const {
    return Val != V;
  }

  /// operator< - Convenience function for get().operator<
  bool operator<(const SDValue &V) const {
    return Val < V;
  }

private:
  friend class SelectionDAG;
  friend class SDNode;

  void setUser(SDNode *p) { User = p; }

  /// set - Remove this use from its existing use list, assign it the
  /// given value, and add it to the new value's node's use list.
  inline void set(const SDValue &V);
  /// setInitial - like set, but only supports initializing a newly-allocated
  /// SDUse with a non-null value.
  inline void setInitial(const SDValue &V);
  /// setNode - like set, but only sets the Node portion of the value,
  /// leaving the ResNo portion unmodified.
  inline void setNode(SDNode *N);

  void addToList(SDUse **List) {
    Next = *List;
    if (Next) Next->Prev = &Next;
    Prev = List;
    *List = this;
  }

  void removeFromList() {
    *Prev = Next;
    if (Next) Next->Prev = Prev;
  }
};

/// simplify_type specializations - Allow casting operators to work directly on
/// SDValues as if they were SDNode*'s.
template<> struct simplify_type<SDUse> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDUse &Val) {
    return static_cast<SimpleType>(Val.getNode());
  }
};
template<> struct simplify_type<const SDUse> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDUse &Val) {
    return static_cast<SimpleType>(Val.getNode());
  }
};


/// SDNode - Represents one node in the SelectionDAG.
///
class SDNode : public FoldingSetNode, public ilist_node<SDNode> {
private:
  /// NodeType - The operation that this node performs.
  ///
  int16_t NodeType;

  /// OperandsNeedDelete - This is true if OperandList was new[]'d.  If true,
  /// then they will be delete[]'d when the node is destroyed.
  uint16_t OperandsNeedDelete : 1;

protected:
  /// SubclassData - This member is defined by this class, but is not used for
  /// anything.  Subclasses can use it to hold whatever state they find useful.
  /// This field is initialized to zero by the ctor.
  uint16_t SubclassData : 15;

private:
  /// NodeId - Unique id per SDNode in the DAG.
  int NodeId;

  /// OperandList - The values that are used by this operation.
  ///
  SDUse *OperandList;

  /// ValueList - The types of the values this node defines.  SDNode's may
  /// define multiple values simultaneously.
  const EVT *ValueList;

  /// UseList - List of uses for this SDNode.
  SDUse *UseList;

  /// NumOperands/NumValues - The number of entries in the Operand/Value list.
  unsigned short NumOperands, NumValues;

  /// debugLoc - source line information.
  DebugLoc debugLoc;

  /// getValueTypeList - Return a pointer to the specified value type.
  static const EVT *getValueTypeList(EVT VT);

  friend class SelectionDAG;
  friend struct ilist_traits<SDNode>;

public:
  //===--------------------------------------------------------------------===//
  //  Accessors
  //

  /// getOpcode - Return the SelectionDAG opcode value for this node. For
  /// pre-isel nodes (those for which isMachineOpcode returns false), these
  /// are the opcode values in the ISD and <target>ISD namespaces. For
  /// post-isel opcodes, see getMachineOpcode.
  unsigned getOpcode()  const { return (unsigned short)NodeType; }

  /// isTargetOpcode - Test if this node has a target-specific opcode (in the
  /// \<target\>ISD namespace).
  bool isTargetOpcode() const { return NodeType >= ISD::BUILTIN_OP_END; }

  /// isTargetMemoryOpcode - Test if this node has a target-specific 
  /// memory-referencing opcode (in the \<target\>ISD namespace and
  /// greater than FIRST_TARGET_MEMORY_OPCODE).
  bool isTargetMemoryOpcode() const {
    return NodeType >= ISD::FIRST_TARGET_MEMORY_OPCODE;
  }

  /// isMachineOpcode - Test if this node has a post-isel opcode, directly
  /// corresponding to a MachineInstr opcode.
  bool isMachineOpcode() const { return NodeType < 0; }

  /// getMachineOpcode - This may only be called if isMachineOpcode returns
  /// true. It returns the MachineInstr opcode value that the node's opcode
  /// corresponds to.
  unsigned getMachineOpcode() const {
    assert(isMachineOpcode() && "Not a MachineInstr opcode!");
    return ~NodeType;
  }

  /// use_empty - Return true if there are no uses of this node.
  ///
  bool use_empty() const { return UseList == NULL; }

  /// hasOneUse - Return true if there is exactly one use of this node.
  ///
  bool hasOneUse() const {
    return !use_empty() && llvm::next(use_begin()) == use_end();
  }

  /// use_size - Return the number of uses of this node. This method takes
  /// time proportional to the number of uses.
  ///
  size_t use_size() const { return std::distance(use_begin(), use_end()); }

  /// getNodeId - Return the unique node id.
  ///
  int getNodeId() const { return NodeId; }

  /// setNodeId - Set unique node id.
  void setNodeId(int Id) { NodeId = Id; }

  /// getDebugLoc - Return the source location info.
  const DebugLoc getDebugLoc() const { return debugLoc; }

  /// setDebugLoc - Set source location info.  Try to avoid this, putting
  /// it in the constructor is preferable.
  void setDebugLoc(const DebugLoc dl) { debugLoc = dl; }

  /// use_iterator - This class provides iterator support for SDUse
  /// operands that use a specific SDNode.
  class use_iterator
    : public std::iterator<std::forward_iterator_tag, SDUse, ptrdiff_t> {
    SDUse *Op;
    explicit use_iterator(SDUse *op) : Op(op) {
    }
    friend class SDNode;
  public:
    typedef std::iterator<std::forward_iterator_tag,
                          SDUse, ptrdiff_t>::reference reference;
    typedef std::iterator<std::forward_iterator_tag,
                          SDUse, ptrdiff_t>::pointer pointer;

    use_iterator(const use_iterator &I) : Op(I.Op) {}
    use_iterator() : Op(0) {}

    bool operator==(const use_iterator &x) const {
      return Op == x.Op;
    }
    bool operator!=(const use_iterator &x) const {
      return !operator==(x);
    }

    /// atEnd - return true if this iterator is at the end of uses list.
    bool atEnd() const { return Op == 0; }

    // Iterator traversal: forward iteration only.
    use_iterator &operator++() {          // Preincrement
      assert(Op && "Cannot increment end iterator!");
      Op = Op->getNext();
      return *this;
    }

    use_iterator operator++(int) {        // Postincrement
      use_iterator tmp = *this; ++*this; return tmp;
    }

    /// Retrieve a pointer to the current user node.
    SDNode *operator*() const {
      assert(Op && "Cannot dereference end iterator!");
      return Op->getUser();
    }

    SDNode *operator->() const { return operator*(); }

    SDUse &getUse() const { return *Op; }

    /// getOperandNo - Retrieve the operand # of this use in its user.
    ///
    unsigned getOperandNo() const {
      assert(Op && "Cannot dereference end iterator!");
      return (unsigned)(Op - Op->getUser()->OperandList);
    }
  };

  /// use_begin/use_end - Provide iteration support to walk over all uses
  /// of an SDNode.

  use_iterator use_begin() const {
    return use_iterator(UseList);
  }

  static use_iterator use_end() { return use_iterator(0); }


  /// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
  /// indicated value.  This method ignores uses of other values defined by this
  /// operation.
  bool hasNUsesOfValue(unsigned NUses, unsigned Value) const;

  /// hasAnyUseOfValue - Return true if there are any use of the indicated
  /// value. This method ignores uses of other values defined by this operation.
  bool hasAnyUseOfValue(unsigned Value) const;

  /// isOnlyUserOf - Return true if this node is the only use of N.
  ///
  bool isOnlyUserOf(SDNode *N) const;

  /// isOperandOf - Return true if this node is an operand of N.
  ///
  bool isOperandOf(SDNode *N) const;

  /// isPredecessorOf - Return true if this node is a predecessor of N. This
  /// node is either an operand of N or it can be reached by recursively
  /// traversing up the operands.
  /// NOTE: this is an expensive method. Use it carefully.
  bool isPredecessorOf(SDNode *N) const;

  /// getNumOperands - Return the number of values used by this operation.
  ///
  unsigned getNumOperands() const { return NumOperands; }

  /// getConstantOperandVal - Helper method returns the integer value of a
  /// ConstantSDNode operand.
  uint64_t getConstantOperandVal(unsigned Num) const;

  const SDValue &getOperand(unsigned Num) const {
    assert(Num < NumOperands && "Invalid child # of SDNode!");
    return OperandList[Num];
  }

  typedef SDUse* op_iterator;
  op_iterator op_begin() const { return OperandList; }
  op_iterator op_end() const { return OperandList+NumOperands; }

  SDVTList getVTList() const {
    SDVTList X = { ValueList, NumValues };
    return X;
  }

  /// getFlaggedNode - If this node has a flag operand, return the node
  /// to which the flag operand points. Otherwise return NULL.
  SDNode *getFlaggedNode() const {
    if (getNumOperands() != 0 &&
      getOperand(getNumOperands()-1).getValueType().getSimpleVT() == MVT::Flag)
      return getOperand(getNumOperands()-1).getNode();
    return 0;
  }

  // If this is a pseudo op, like copyfromreg, look to see if there is a
  // real target node flagged to it.  If so, return the target node.
  const SDNode *getFlaggedMachineNode() const {
    const SDNode *FoundNode = this;

    // Climb up flag edges until a machine-opcode node is found, or the
    // end of the chain is reached.
    while (!FoundNode->isMachineOpcode()) {
      const SDNode *N = FoundNode->getFlaggedNode();
      if (!N) break;
      FoundNode = N;
    }

    return FoundNode;
  }

  /// getNumValues - Return the number of values defined/returned by this
  /// operator.
  ///
  unsigned getNumValues() const { return NumValues; }

  /// getValueType - Return the type of a specified result.
  ///
  EVT getValueType(unsigned ResNo) const {
    assert(ResNo < NumValues && "Illegal result number!");
    return ValueList[ResNo];
  }

  /// getValueSizeInBits - Returns MVT::getSizeInBits(getValueType(ResNo)).
  ///
  unsigned getValueSizeInBits(unsigned ResNo) const {
    return getValueType(ResNo).getSizeInBits();
  }

  typedef const EVT* value_iterator;
  value_iterator value_begin() const { return ValueList; }
  value_iterator value_end() const { return ValueList+NumValues; }

  /// getOperationName - Return the opcode of this operation for printing.
  ///
  std::string getOperationName(const SelectionDAG *G = 0) const;
  static const char* getIndexedModeName(ISD::MemIndexedMode AM);
  void print_types(raw_ostream &OS, const SelectionDAG *G) const;
  void print_details(raw_ostream &OS, const SelectionDAG *G) const;
  void print(raw_ostream &OS, const SelectionDAG *G = 0) const;
  void printr(raw_ostream &OS, const SelectionDAG *G = 0) const;

  /// printrFull - Print a SelectionDAG node and all children down to
  /// the leaves.  The given SelectionDAG allows target-specific nodes
  /// to be printed in human-readable form.  Unlike printr, this will
  /// print the whole DAG, including children that appear multiple
  /// times.
  ///
  void printrFull(raw_ostream &O, const SelectionDAG *G = 0) const;

  /// printrWithDepth - Print a SelectionDAG node and children up to
  /// depth "depth."  The given SelectionDAG allows target-specific
  /// nodes to be printed in human-readable form.  Unlike printr, this
  /// will print children that appear multiple times wherever they are
  /// used.
  ///
  void printrWithDepth(raw_ostream &O, const SelectionDAG *G = 0,
                       unsigned depth = 100) const;


  /// dump - Dump this node, for debugging.
  void dump() const;

  /// dumpr - Dump (recursively) this node and its use-def subgraph.
  void dumpr() const;

  /// dump - Dump this node, for debugging.
  /// The given SelectionDAG allows target-specific nodes to be printed
  /// in human-readable form.
  void dump(const SelectionDAG *G) const;

  /// dumpr - Dump (recursively) this node and its use-def subgraph.
  /// The given SelectionDAG allows target-specific nodes to be printed
  /// in human-readable form.
  void dumpr(const SelectionDAG *G) const;

  /// dumprFull - printrFull to dbgs().  The given SelectionDAG allows
  /// target-specific nodes to be printed in human-readable form.
  /// Unlike dumpr, this will print the whole DAG, including children
  /// that appear multiple times.
  ///
  void dumprFull(const SelectionDAG *G = 0) const;

  /// dumprWithDepth - printrWithDepth to dbgs().  The given
  /// SelectionDAG allows target-specific nodes to be printed in
  /// human-readable form.  Unlike dumpr, this will print children
  /// that appear multiple times wherever they are used.
  ///
  void dumprWithDepth(const SelectionDAG *G = 0, unsigned depth = 100) const;


  static bool classof(const SDNode *) { return true; }

  /// Profile - Gather unique data for the node.
  ///
  void Profile(FoldingSetNodeID &ID) const;

  /// addUse - This method should only be used by the SDUse class.
  ///
  void addUse(SDUse &U) { U.addToList(&UseList); }

protected:
  static SDVTList getSDVTList(EVT VT) {
    SDVTList Ret = { getValueTypeList(VT), 1 };
    return Ret;
  }

  SDNode(unsigned Opc, const DebugLoc dl, SDVTList VTs, const SDValue *Ops,
         unsigned NumOps)
    : NodeType(Opc), OperandsNeedDelete(true), SubclassData(0),
      NodeId(-1),
      OperandList(NumOps ? new SDUse[NumOps] : 0),
      ValueList(VTs.VTs), UseList(NULL),
      NumOperands(NumOps), NumValues(VTs.NumVTs),
      debugLoc(dl) {
    for (unsigned i = 0; i != NumOps; ++i) {
      OperandList[i].setUser(this);
      OperandList[i].setInitial(Ops[i]);
    }
    checkForCycles(this);
  }

  /// This constructor adds no operands itself; operands can be
  /// set later with InitOperands.
  SDNode(unsigned Opc, const DebugLoc dl, SDVTList VTs)
    : NodeType(Opc), OperandsNeedDelete(false), SubclassData(0),
      NodeId(-1), OperandList(0), ValueList(VTs.VTs), UseList(NULL),
      NumOperands(0), NumValues(VTs.NumVTs),
      debugLoc(dl) {}

  /// InitOperands - Initialize the operands list of this with 1 operand.
  void InitOperands(SDUse *Ops, const SDValue &Op0) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    NumOperands = 1;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// InitOperands - Initialize the operands list of this with 2 operands.
  void InitOperands(SDUse *Ops, const SDValue &Op0, const SDValue &Op1) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    Ops[1].setUser(this);
    Ops[1].setInitial(Op1);
    NumOperands = 2;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// InitOperands - Initialize the operands list of this with 3 operands.
  void InitOperands(SDUse *Ops, const SDValue &Op0, const SDValue &Op1,
                    const SDValue &Op2) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    Ops[1].setUser(this);
    Ops[1].setInitial(Op1);
    Ops[2].setUser(this);
    Ops[2].setInitial(Op2);
    NumOperands = 3;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// InitOperands - Initialize the operands list of this with 4 operands.
  void InitOperands(SDUse *Ops, const SDValue &Op0, const SDValue &Op1,
                    const SDValue &Op2, const SDValue &Op3) {
    Ops[0].setUser(this);
    Ops[0].setInitial(Op0);
    Ops[1].setUser(this);
    Ops[1].setInitial(Op1);
    Ops[2].setUser(this);
    Ops[2].setInitial(Op2);
    Ops[3].setUser(this);
    Ops[3].setInitial(Op3);
    NumOperands = 4;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// InitOperands - Initialize the operands list of this with N operands.
  void InitOperands(SDUse *Ops, const SDValue *Vals, unsigned N) {
    for (unsigned i = 0; i != N; ++i) {
      Ops[i].setUser(this);
      Ops[i].setInitial(Vals[i]);
    }
    NumOperands = N;
    OperandList = Ops;
    checkForCycles(this);
  }

  /// DropOperands - Release the operands and set this node to have
  /// zero operands.
  void DropOperands();
};


// Define inline functions from the SDValue class.

inline unsigned SDValue::getOpcode() const {
  return Node->getOpcode();
}
inline EVT SDValue::getValueType() const {
  return Node->getValueType(ResNo);
}
inline unsigned SDValue::getNumOperands() const {
  return Node->getNumOperands();
}
inline const SDValue &SDValue::getOperand(unsigned i) const {
  return Node->getOperand(i);
}
inline uint64_t SDValue::getConstantOperandVal(unsigned i) const {
  return Node->getConstantOperandVal(i);
}
inline bool SDValue::isTargetOpcode() const {
  return Node->isTargetOpcode();
}
inline bool SDValue::isTargetMemoryOpcode() const {
  return Node->isTargetMemoryOpcode();
}
inline bool SDValue::isMachineOpcode() const {
  return Node->isMachineOpcode();
}
inline unsigned SDValue::getMachineOpcode() const {
  return Node->getMachineOpcode();
}
inline bool SDValue::use_empty() const {
  return !Node->hasAnyUseOfValue(ResNo);
}
inline bool SDValue::hasOneUse() const {
  return Node->hasNUsesOfValue(1, ResNo);
}
inline const DebugLoc SDValue::getDebugLoc() const {
  return Node->getDebugLoc();
}

// Define inline functions from the SDUse class.

inline void SDUse::set(const SDValue &V) {
  if (Val.getNode()) removeFromList();
  Val = V;
  if (V.getNode()) V.getNode()->addUse(*this);
}

inline void SDUse::setInitial(const SDValue &V) {
  Val = V;
  V.getNode()->addUse(*this);
}

inline void SDUse::setNode(SDNode *N) {
  if (Val.getNode()) removeFromList();
  Val.setNode(N);
  if (N) N->addUse(*this);
}

/// UnarySDNode - This class is used for single-operand SDNodes.  This is solely
/// to allow co-allocation of node operands with the node itself.
class UnarySDNode : public SDNode {
  SDUse Op;
public:
  UnarySDNode(unsigned Opc, DebugLoc dl, SDVTList VTs, SDValue X)
    : SDNode(Opc, dl, VTs) {
    InitOperands(&Op, X);
  }
};

/// BinarySDNode - This class is used for two-operand SDNodes.  This is solely
/// to allow co-allocation of node operands with the node itself.
class BinarySDNode : public SDNode {
  SDUse Ops[2];
public:
  BinarySDNode(unsigned Opc, DebugLoc dl, SDVTList VTs, SDValue X, SDValue Y)
    : SDNode(Opc, dl, VTs) {
    InitOperands(Ops, X, Y);
  }
};

/// TernarySDNode - This class is used for three-operand SDNodes. This is solely
/// to allow co-allocation of node operands with the node itself.
class TernarySDNode : public SDNode {
  SDUse Ops[3];
public:
  TernarySDNode(unsigned Opc, DebugLoc dl, SDVTList VTs, SDValue X, SDValue Y,
                SDValue Z)
    : SDNode(Opc, dl, VTs) {
    InitOperands(Ops, X, Y, Z);
  }
};


/// HandleSDNode - This class is used to form a handle around another node that
/// is persistant and is updated across invocations of replaceAllUsesWith on its
/// operand.  This node should be directly created by end-users and not added to
/// the AllNodes list.
class HandleSDNode : public SDNode {
  SDUse Op;
public:
  // FIXME: Remove the "noinline" attribute once <rdar://problem/5852746> is
  // fixed.
#ifdef __GNUC__
  explicit __attribute__((__noinline__)) HandleSDNode(SDValue X)
#else
  explicit HandleSDNode(SDValue X)
#endif
    : SDNode(ISD::HANDLENODE, DebugLoc::getUnknownLoc(),
             getSDVTList(MVT::Other)) {
    InitOperands(&Op, X);
  }
  ~HandleSDNode();
  const SDValue &getValue() const { return Op; }
};

/// Abstact virtual class for operations for memory operations
class MemSDNode : public SDNode {
private:
  // MemoryVT - VT of in-memory value.
  EVT MemoryVT;

protected:
  /// MMO - Memory reference information.
  MachineMemOperand *MMO;

public:
  MemSDNode(unsigned Opc, DebugLoc dl, SDVTList VTs, EVT MemoryVT,
            MachineMemOperand *MMO);

  MemSDNode(unsigned Opc, DebugLoc dl, SDVTList VTs, const SDValue *Ops,
            unsigned NumOps, EVT MemoryVT, MachineMemOperand *MMO);

  bool readMem() const { return MMO->isLoad(); }
  bool writeMem() const { return MMO->isStore(); }

  /// Returns alignment and volatility of the memory access
  unsigned getOriginalAlignment() const { 
    return MMO->getBaseAlignment();
  }
  unsigned getAlignment() const {
    return MMO->getAlignment();
  }

  /// getRawSubclassData - Return the SubclassData value, which contains an
  /// encoding of the volatile flag, as well as bits used by subclasses. This
  /// function should only be used to compute a FoldingSetNodeID value.
  unsigned getRawSubclassData() const {
    return SubclassData;
  }

  bool isVolatile() const { return (SubclassData >> 5) & 1; }

  /// Returns the SrcValue and offset that describes the location of the access
  const Value *getSrcValue() const { return MMO->getValue(); }
  int64_t getSrcValueOffset() const { return MMO->getOffset(); }

  /// getMemoryVT - Return the type of the in-memory value.
  EVT getMemoryVT() const { return MemoryVT; }

  /// getMemOperand - Return a MachineMemOperand object describing the memory
  /// reference performed by operation.
  MachineMemOperand *getMemOperand() const { return MMO; }

  /// refineAlignment - Update this MemSDNode's MachineMemOperand information
  /// to reflect the alignment of NewMMO, if it has a greater alignment.
  /// This must only be used when the new alignment applies to all users of
  /// this MachineMemOperand.
  void refineAlignment(const MachineMemOperand *NewMMO) {
    MMO->refineAlignment(NewMMO);
  }

  const SDValue &getChain() const { return getOperand(0); }
  const SDValue &getBasePtr() const {
    return getOperand(getOpcode() == ISD::STORE ? 2 : 1);
  }

  // Methods to support isa and dyn_cast
  static bool classof(const MemSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    // For some targets, we lower some target intrinsics to a MemIntrinsicNode
    // with either an intrinsic or a target opcode.
    return N->getOpcode() == ISD::LOAD                ||
           N->getOpcode() == ISD::STORE               ||
           N->getOpcode() == ISD::ATOMIC_CMP_SWAP     ||
           N->getOpcode() == ISD::ATOMIC_SWAP         ||
           N->getOpcode() == ISD::ATOMIC_LOAD_ADD     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_SUB     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_AND     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_OR      ||
           N->getOpcode() == ISD::ATOMIC_LOAD_XOR     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_NAND    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MIN     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MAX     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMIN    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMAX    ||
           N->isTargetMemoryOpcode();
  }
};

/// AtomicSDNode - A SDNode reprenting atomic operations.
///
class AtomicSDNode : public MemSDNode {
  SDUse Ops[4];

public:
  // Opc:   opcode for atomic
  // VTL:    value type list
  // Chain:  memory chain for operaand
  // Ptr:    address to update as a SDValue
  // Cmp:    compare value
  // Swp:    swap value
  // SrcVal: address to update as a Value (used for MemOperand)
  // Align:  alignment of memory
  AtomicSDNode(unsigned Opc, DebugLoc dl, SDVTList VTL, EVT MemVT,
               SDValue Chain, SDValue Ptr,
               SDValue Cmp, SDValue Swp, MachineMemOperand *MMO)
    : MemSDNode(Opc, dl, VTL, MemVT, MMO) {
    assert(readMem() && "Atomic MachineMemOperand is not a load!");
    assert(writeMem() && "Atomic MachineMemOperand is not a store!");
    InitOperands(Ops, Chain, Ptr, Cmp, Swp);
  }
  AtomicSDNode(unsigned Opc, DebugLoc dl, SDVTList VTL, EVT MemVT,
               SDValue Chain, SDValue Ptr,
               SDValue Val, MachineMemOperand *MMO)
    : MemSDNode(Opc, dl, VTL, MemVT, MMO) {
    assert(readMem() && "Atomic MachineMemOperand is not a load!");
    assert(writeMem() && "Atomic MachineMemOperand is not a store!");
    InitOperands(Ops, Chain, Ptr, Val);
  }

  const SDValue &getBasePtr() const { return getOperand(1); }
  const SDValue &getVal() const { return getOperand(2); }

  bool isCompareAndSwap() const {
    unsigned Op = getOpcode();
    return Op == ISD::ATOMIC_CMP_SWAP;
  }

  // Methods to support isa and dyn_cast
  static bool classof(const AtomicSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ATOMIC_CMP_SWAP     ||
           N->getOpcode() == ISD::ATOMIC_SWAP         ||
           N->getOpcode() == ISD::ATOMIC_LOAD_ADD     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_SUB     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_AND     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_OR      ||
           N->getOpcode() == ISD::ATOMIC_LOAD_XOR     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_NAND    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MIN     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_MAX     ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMIN    ||
           N->getOpcode() == ISD::ATOMIC_LOAD_UMAX;
  }
};

/// MemIntrinsicSDNode - This SDNode is used for target intrinsics that touch
/// memory and need an associated MachineMemOperand. Its opcode may be
/// INTRINSIC_VOID, INTRINSIC_W_CHAIN, or a target-specific opcode with a
/// value not less than FIRST_TARGET_MEMORY_OPCODE.
class MemIntrinsicSDNode : public MemSDNode {
public:
  MemIntrinsicSDNode(unsigned Opc, DebugLoc dl, SDVTList VTs,
                     const SDValue *Ops, unsigned NumOps,
                     EVT MemoryVT, MachineMemOperand *MMO)
    : MemSDNode(Opc, dl, VTs, Ops, NumOps, MemoryVT, MMO) {
  }

  // Methods to support isa and dyn_cast
  static bool classof(const MemIntrinsicSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    // We lower some target intrinsics to their target opcode
    // early a node with a target opcode can be of this class
    return N->getOpcode() == ISD::INTRINSIC_W_CHAIN ||
           N->getOpcode() == ISD::INTRINSIC_VOID ||
           N->isTargetMemoryOpcode();
  }
};

/// ShuffleVectorSDNode - This SDNode is used to implement the code generator
/// support for the llvm IR shufflevector instruction.  It combines elements
/// from two input vectors into a new input vector, with the selection and
/// ordering of elements determined by an array of integers, referred to as
/// the shuffle mask.  For input vectors of width N, mask indices of 0..N-1
/// refer to elements from the LHS input, and indices from N to 2N-1 the RHS.
/// An index of -1 is treated as undef, such that the code generator may put
/// any value in the corresponding element of the result.
class ShuffleVectorSDNode : public SDNode {
  SDUse Ops[2];

  // The memory for Mask is owned by the SelectionDAG's OperandAllocator, and
  // is freed when the SelectionDAG object is destroyed.
  const int *Mask;
protected:
  friend class SelectionDAG;
  ShuffleVectorSDNode(EVT VT, DebugLoc dl, SDValue N1, SDValue N2, 
                      const int *M)
    : SDNode(ISD::VECTOR_SHUFFLE, dl, getSDVTList(VT)), Mask(M) {
    InitOperands(Ops, N1, N2);
  }
public:

  void getMask(SmallVectorImpl<int> &M) const {
    EVT VT = getValueType(0);
    M.clear();
    for (unsigned i = 0, e = VT.getVectorNumElements(); i != e; ++i)
      M.push_back(Mask[i]);
  }
  int getMaskElt(unsigned Idx) const {
    assert(Idx < getValueType(0).getVectorNumElements() && "Idx out of range!");
    return Mask[Idx];
  }
  
  bool isSplat() const { return isSplatMask(Mask, getValueType(0)); }
  int  getSplatIndex() const { 
    assert(isSplat() && "Cannot get splat index for non-splat!");
    return Mask[0];
  }
  static bool isSplatMask(const int *Mask, EVT VT);

  static bool classof(const ShuffleVectorSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::VECTOR_SHUFFLE;
  }
};
  
class ConstantSDNode : public SDNode {
  const ConstantInt *Value;
  friend class SelectionDAG;
  ConstantSDNode(bool isTarget, const ConstantInt *val, EVT VT)
    : SDNode(isTarget ? ISD::TargetConstant : ISD::Constant,
             DebugLoc::getUnknownLoc(), getSDVTList(VT)), Value(val) {
  }
public:

  const ConstantInt *getConstantIntValue() const { return Value; }
  const APInt &getAPIntValue() const { return Value->getValue(); }
  uint64_t getZExtValue() const { return Value->getZExtValue(); }
  int64_t getSExtValue() const { return Value->getSExtValue(); }

  bool isNullValue() const { return Value->isNullValue(); }
  bool isAllOnesValue() const { return Value->isAllOnesValue(); }

  static bool classof(const ConstantSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Constant ||
           N->getOpcode() == ISD::TargetConstant;
  }
};

class ConstantFPSDNode : public SDNode {
  const ConstantFP *Value;
  friend class SelectionDAG;
  ConstantFPSDNode(bool isTarget, const ConstantFP *val, EVT VT)
    : SDNode(isTarget ? ISD::TargetConstantFP : ISD::ConstantFP,
             DebugLoc::getUnknownLoc(), getSDVTList(VT)), Value(val) {
  }
public:

  const APFloat& getValueAPF() const { return Value->getValueAPF(); }
  const ConstantFP *getConstantFPValue() const { return Value; }

  /// isExactlyValue - We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.

  /// We leave the version with the double argument here because it's just so
  /// convenient to write "2.0" and the like.  Without this function we'd
  /// have to duplicate its logic everywhere it's called.
  bool isExactlyValue(double V) const {
    bool ignored;
    // convert is not supported on this type
    if (&Value->getValueAPF().getSemantics() == &APFloat::PPCDoubleDouble)
      return false;
    APFloat Tmp(V);
    Tmp.convert(Value->getValueAPF().getSemantics(),
                APFloat::rmNearestTiesToEven, &ignored);
    return isExactlyValue(Tmp);
  }
  bool isExactlyValue(const APFloat& V) const;

  bool isValueValidForType(EVT VT, const APFloat& Val);

  static bool classof(const ConstantFPSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantFP ||
           N->getOpcode() == ISD::TargetConstantFP;
  }
};

class GlobalAddressSDNode : public SDNode {
  GlobalValue *TheGlobal;
  int64_t Offset;
  unsigned char TargetFlags;
  friend class SelectionDAG;
  GlobalAddressSDNode(unsigned Opc, const GlobalValue *GA, EVT VT,
                      int64_t o, unsigned char TargetFlags);
public:

  GlobalValue *getGlobal() const { return TheGlobal; }
  int64_t getOffset() const { return Offset; }
  unsigned char getTargetFlags() const { return TargetFlags; }
  // Return the address space this GlobalAddress belongs to.
  unsigned getAddressSpace() const;

  static bool classof(const GlobalAddressSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::GlobalAddress ||
           N->getOpcode() == ISD::TargetGlobalAddress ||
           N->getOpcode() == ISD::GlobalTLSAddress ||
           N->getOpcode() == ISD::TargetGlobalTLSAddress;
  }
};

class FrameIndexSDNode : public SDNode {
  int FI;
  friend class SelectionDAG;
  FrameIndexSDNode(int fi, EVT VT, bool isTarg)
    : SDNode(isTarg ? ISD::TargetFrameIndex : ISD::FrameIndex,
      DebugLoc::getUnknownLoc(), getSDVTList(VT)), FI(fi) {
  }
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
  unsigned char TargetFlags;
  friend class SelectionDAG;
  JumpTableSDNode(int jti, EVT VT, bool isTarg, unsigned char TF)
    : SDNode(isTarg ? ISD::TargetJumpTable : ISD::JumpTable,
      DebugLoc::getUnknownLoc(), getSDVTList(VT)), JTI(jti), TargetFlags(TF) {
  }
public:

  int getIndex() const { return JTI; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  static bool classof(const JumpTableSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::JumpTable ||
           N->getOpcode() == ISD::TargetJumpTable;
  }
};

class ConstantPoolSDNode : public SDNode {
  union {
    Constant *ConstVal;
    MachineConstantPoolValue *MachineCPVal;
  } Val;
  int Offset;  // It's a MachineConstantPoolValue if top bit is set.
  unsigned Alignment;  // Minimum alignment requirement of CP (not log2 value).
  unsigned char TargetFlags;
  friend class SelectionDAG;
  ConstantPoolSDNode(bool isTarget, Constant *c, EVT VT, int o, unsigned Align,
                     unsigned char TF)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool,
             DebugLoc::getUnknownLoc(),
             getSDVTList(VT)), Offset(o), Alignment(Align), TargetFlags(TF) {
    assert((int)Offset >= 0 && "Offset is too large");
    Val.ConstVal = c;
  }
  ConstantPoolSDNode(bool isTarget, MachineConstantPoolValue *v,
                     EVT VT, int o, unsigned Align, unsigned char TF)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool,
             DebugLoc::getUnknownLoc(),
             getSDVTList(VT)), Offset(o), Alignment(Align), TargetFlags(TF) {
    assert((int)Offset >= 0 && "Offset is too large");
    Val.MachineCPVal = v;
    Offset |= 1 << (sizeof(unsigned)*CHAR_BIT-1);
  }
public:
  

  bool isMachineConstantPoolEntry() const {
    return (int)Offset < 0;
  }

  Constant *getConstVal() const {
    assert(!isMachineConstantPoolEntry() && "Wrong constantpool type");
    return Val.ConstVal;
  }

  MachineConstantPoolValue *getMachineCPVal() const {
    assert(isMachineConstantPoolEntry() && "Wrong constantpool type");
    return Val.MachineCPVal;
  }

  int getOffset() const {
    return Offset & ~(1 << (sizeof(unsigned)*CHAR_BIT-1));
  }

  // Return the alignment of this constant pool object, which is either 0 (for
  // default alignment) or the desired value.
  unsigned getAlignment() const { return Alignment; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  const Type *getType() const;

  static bool classof(const ConstantPoolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantPool ||
           N->getOpcode() == ISD::TargetConstantPool;
  }
};

class BasicBlockSDNode : public SDNode {
  MachineBasicBlock *MBB;
  friend class SelectionDAG;
  /// Debug info is meaningful and potentially useful here, but we create
  /// blocks out of order when they're jumped to, which makes it a bit
  /// harder.  Let's see if we need it first.
  explicit BasicBlockSDNode(MachineBasicBlock *mbb)
    : SDNode(ISD::BasicBlock, DebugLoc::getUnknownLoc(),
             getSDVTList(MVT::Other)), MBB(mbb) {
  }
public:

  MachineBasicBlock *getBasicBlock() const { return MBB; }

  static bool classof(const BasicBlockSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BasicBlock;
  }
};

/// BuildVectorSDNode - A "pseudo-class" with methods for operating on
/// BUILD_VECTORs.
class BuildVectorSDNode : public SDNode {
  // These are constructed as SDNodes and then cast to BuildVectorSDNodes.
  explicit BuildVectorSDNode();        // Do not implement
public:
  /// isConstantSplat - Check if this is a constant splat, and if so, find the
  /// smallest element size that splats the vector.  If MinSplatBits is
  /// nonzero, the element size must be at least that large.  Note that the
  /// splat element may be the entire vector (i.e., a one element vector).
  /// Returns the splat element value in SplatValue.  Any undefined bits in
  /// that value are zero, and the corresponding bits in the SplatUndef mask
  /// are set.  The SplatBitSize value is set to the splat element size in
  /// bits.  HasAnyUndefs is set to true if any bits in the vector are
  /// undefined.  isBigEndian describes the endianness of the target.
  bool isConstantSplat(APInt &SplatValue, APInt &SplatUndef,
                       unsigned &SplatBitSize, bool &HasAnyUndefs,
                       unsigned MinSplatBits = 0, bool isBigEndian = false);

  static inline bool classof(const BuildVectorSDNode *) { return true; }
  static inline bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BUILD_VECTOR;
  }
};

/// SrcValueSDNode - An SDNode that holds an arbitrary LLVM IR Value. This is
/// used when the SelectionDAG needs to make a simple reference to something
/// in the LLVM IR representation.
///
class SrcValueSDNode : public SDNode {
  const Value *V;
  friend class SelectionDAG;
  /// Create a SrcValue for a general value.
  explicit SrcValueSDNode(const Value *v)
    : SDNode(ISD::SRCVALUE, DebugLoc::getUnknownLoc(),
             getSDVTList(MVT::Other)), V(v) {}

public:
  /// getValue - return the contained Value.
  const Value *getValue() const { return V; }

  static bool classof(const SrcValueSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::SRCVALUE;
  }
};


class RegisterSDNode : public SDNode {
  unsigned Reg;
  friend class SelectionDAG;
  RegisterSDNode(unsigned reg, EVT VT)
    : SDNode(ISD::Register, DebugLoc::getUnknownLoc(),
             getSDVTList(VT)), Reg(reg) {
  }
public:

  unsigned getReg() const { return Reg; }

  static bool classof(const RegisterSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Register;
  }
};

class BlockAddressSDNode : public SDNode {
  BlockAddress *BA;
  unsigned char TargetFlags;
  friend class SelectionDAG;
  BlockAddressSDNode(unsigned NodeTy, EVT VT, BlockAddress *ba,
                     unsigned char Flags)
    : SDNode(NodeTy, DebugLoc::getUnknownLoc(), getSDVTList(VT)),
             BA(ba), TargetFlags(Flags) {
  }
public:
  BlockAddress *getBlockAddress() const { return BA; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  static bool classof(const BlockAddressSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BlockAddress ||
           N->getOpcode() == ISD::TargetBlockAddress;
  }
};

class LabelSDNode : public SDNode {
  SDUse Chain;
  unsigned LabelID;
  friend class SelectionDAG;
  LabelSDNode(unsigned NodeTy, DebugLoc dl, SDValue ch, unsigned id)
    : SDNode(NodeTy, dl, getSDVTList(MVT::Other)), LabelID(id) {
    InitOperands(&Chain, ch);
  }
public:
  unsigned getLabelID() const { return LabelID; }

  static bool classof(const LabelSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::EH_LABEL;
  }
};

class ExternalSymbolSDNode : public SDNode {
  const char *Symbol;
  unsigned char TargetFlags;
  
  friend class SelectionDAG;
  ExternalSymbolSDNode(bool isTarget, const char *Sym, unsigned char TF, EVT VT)
    : SDNode(isTarget ? ISD::TargetExternalSymbol : ISD::ExternalSymbol,
             DebugLoc::getUnknownLoc(),
             getSDVTList(VT)), Symbol(Sym), TargetFlags(TF) {
  }
public:

  const char *getSymbol() const { return Symbol; }
  unsigned char getTargetFlags() const { return TargetFlags; }

  static bool classof(const ExternalSymbolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ExternalSymbol ||
           N->getOpcode() == ISD::TargetExternalSymbol;
  }
};

class CondCodeSDNode : public SDNode {
  ISD::CondCode Condition;
  friend class SelectionDAG;
  explicit CondCodeSDNode(ISD::CondCode Cond)
    : SDNode(ISD::CONDCODE, DebugLoc::getUnknownLoc(),
             getSDVTList(MVT::Other)), Condition(Cond) {
  }
public:

  ISD::CondCode get() const { return Condition; }

  static bool classof(const CondCodeSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CONDCODE;
  }
};
  
/// CvtRndSatSDNode - NOTE: avoid using this node as this may disappear in the
/// future and most targets don't support it.
class CvtRndSatSDNode : public SDNode {
  ISD::CvtCode CvtCode;
  friend class SelectionDAG;
  explicit CvtRndSatSDNode(EVT VT, DebugLoc dl, const SDValue *Ops,
                           unsigned NumOps, ISD::CvtCode Code)
    : SDNode(ISD::CONVERT_RNDSAT, dl, getSDVTList(VT), Ops, NumOps),
      CvtCode(Code) {
    assert(NumOps == 5 && "wrong number of operations");
  }
public:
  ISD::CvtCode getCvtCode() const { return CvtCode; }

  static bool classof(const CvtRndSatSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CONVERT_RNDSAT;
  }
};

namespace ISD {
  struct ArgFlagsTy {
  private:
    static const uint64_t NoFlagSet      = 0ULL;
    static const uint64_t ZExt           = 1ULL<<0;  ///< Zero extended
    static const uint64_t ZExtOffs       = 0;
    static const uint64_t SExt           = 1ULL<<1;  ///< Sign extended
    static const uint64_t SExtOffs       = 1;
    static const uint64_t InReg          = 1ULL<<2;  ///< Passed in register
    static const uint64_t InRegOffs      = 2;
    static const uint64_t SRet           = 1ULL<<3;  ///< Hidden struct-ret ptr
    static const uint64_t SRetOffs       = 3;
    static const uint64_t ByVal          = 1ULL<<4;  ///< Struct passed by value
    static const uint64_t ByValOffs      = 4;
    static const uint64_t Nest           = 1ULL<<5;  ///< Nested fn static chain
    static const uint64_t NestOffs       = 5;
    static const uint64_t ByValAlign     = 0xFULL << 6; //< Struct alignment
    static const uint64_t ByValAlignOffs = 6;
    static const uint64_t Split          = 1ULL << 10;
    static const uint64_t SplitOffs      = 10;
    static const uint64_t OrigAlign      = 0x1FULL<<27;
    static const uint64_t OrigAlignOffs  = 27;
    static const uint64_t ByValSize      = 0xffffffffULL << 32; //< Struct size
    static const uint64_t ByValSizeOffs  = 32;

    static const uint64_t One            = 1ULL; //< 1 of this type, for shifts

    uint64_t Flags;
  public:
    ArgFlagsTy() : Flags(0) { }

    bool isZExt()   const { return Flags & ZExt; }
    void setZExt()  { Flags |= One << ZExtOffs; }

    bool isSExt()   const { return Flags & SExt; }
    void setSExt()  { Flags |= One << SExtOffs; }

    bool isInReg()  const { return Flags & InReg; }
    void setInReg() { Flags |= One << InRegOffs; }

    bool isSRet()   const { return Flags & SRet; }
    void setSRet()  { Flags |= One << SRetOffs; }

    bool isByVal()  const { return Flags & ByVal; }
    void setByVal() { Flags |= One << ByValOffs; }

    bool isNest()   const { return Flags & Nest; }
    void setNest()  { Flags |= One << NestOffs; }

    unsigned getByValAlign() const {
      return (unsigned)
        ((One << ((Flags & ByValAlign) >> ByValAlignOffs)) / 2);
    }
    void setByValAlign(unsigned A) {
      Flags = (Flags & ~ByValAlign) |
        (uint64_t(Log2_32(A) + 1) << ByValAlignOffs);
    }

    bool isSplit()   const { return Flags & Split; }
    void setSplit()  { Flags |= One << SplitOffs; }

    unsigned getOrigAlign() const {
      return (unsigned)
        ((One << ((Flags & OrigAlign) >> OrigAlignOffs)) / 2);
    }
    void setOrigAlign(unsigned A) {
      Flags = (Flags & ~OrigAlign) |
        (uint64_t(Log2_32(A) + 1) << OrigAlignOffs);
    }

    unsigned getByValSize() const {
      return (unsigned)((Flags & ByValSize) >> ByValSizeOffs);
    }
    void setByValSize(unsigned S) {
      Flags = (Flags & ~ByValSize) | (uint64_t(S) << ByValSizeOffs);
    }

    /// getArgFlagsString - Returns the flags as a string, eg: "zext align:4".
    std::string getArgFlagsString();

    /// getRawBits - Represent the flags as a bunch of bits.
    uint64_t getRawBits() const { return Flags; }
  };

  /// InputArg - This struct carries flags and type information about a
  /// single incoming (formal) argument or incoming (from the perspective
  /// of the caller) return value virtual register.
  ///
  struct InputArg {
    ArgFlagsTy Flags;
    EVT VT;
    bool Used;

    InputArg() : VT(MVT::Other), Used(false) {}
    InputArg(ISD::ArgFlagsTy flags, EVT vt, bool used)
      : Flags(flags), VT(vt), Used(used) {
      assert(VT.isSimple() &&
             "InputArg value type must be Simple!");
    }
  };

  /// OutputArg - This struct carries flags and a value for a
  /// single outgoing (actual) argument or outgoing (from the perspective
  /// of the caller) return value virtual register.
  ///
  struct OutputArg {
    ArgFlagsTy Flags;
    SDValue Val;
    bool IsFixed;

    OutputArg() : IsFixed(false) {}
    OutputArg(ISD::ArgFlagsTy flags, SDValue val, bool isfixed)
      : Flags(flags), Val(val), IsFixed(isfixed) {
      assert(Val.getValueType().isSimple() &&
             "OutputArg value type must be Simple!");
    }
  };
}

/// VTSDNode - This class is used to represent EVT's, which are used
/// to parameterize some operations.
class VTSDNode : public SDNode {
  EVT ValueType;
  friend class SelectionDAG;
  explicit VTSDNode(EVT VT)
    : SDNode(ISD::VALUETYPE, DebugLoc::getUnknownLoc(),
             getSDVTList(MVT::Other)), ValueType(VT) {
  }
public:

  EVT getVT() const { return ValueType; }

  static bool classof(const VTSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::VALUETYPE;
  }
};

/// LSBaseSDNode - Base class for LoadSDNode and StoreSDNode
///
class LSBaseSDNode : public MemSDNode {
  //! Operand array for load and store
  /*!
    \note Moving this array to the base class captures more
    common functionality shared between LoadSDNode and
    StoreSDNode
   */
  SDUse Ops[4];
public:
  LSBaseSDNode(ISD::NodeType NodeTy, DebugLoc dl, SDValue *Operands,
               unsigned numOperands, SDVTList VTs, ISD::MemIndexedMode AM,
               EVT MemVT, MachineMemOperand *MMO)
    : MemSDNode(NodeTy, dl, VTs, MemVT, MMO) {
    SubclassData |= AM << 2;
    assert(getAddressingMode() == AM && "MemIndexedMode encoding error!");
    InitOperands(Ops, Operands, numOperands);
    assert((getOffset().getOpcode() == ISD::UNDEF || isIndexed()) &&
           "Only indexed loads and stores have a non-undef offset operand");
  }

  const SDValue &getOffset() const {
    return getOperand(getOpcode() == ISD::LOAD ? 2 : 3);
  }

  /// getAddressingMode - Return the addressing mode for this load or store:
  /// unindexed, pre-inc, pre-dec, post-inc, or post-dec.
  ISD::MemIndexedMode getAddressingMode() const {
    return ISD::MemIndexedMode((SubclassData >> 2) & 7);
  }

  /// isIndexed - Return true if this is a pre/post inc/dec load/store.
  bool isIndexed() const { return getAddressingMode() != ISD::UNINDEXED; }

  /// isUnindexed - Return true if this is NOT a pre/post inc/dec load/store.
  bool isUnindexed() const { return getAddressingMode() == ISD::UNINDEXED; }

  static bool classof(const LSBaseSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD ||
           N->getOpcode() == ISD::STORE;
  }
};

/// LoadSDNode - This class is used to represent ISD::LOAD nodes.
///
class LoadSDNode : public LSBaseSDNode {
  friend class SelectionDAG;
  LoadSDNode(SDValue *ChainPtrOff, DebugLoc dl, SDVTList VTs,
             ISD::MemIndexedMode AM, ISD::LoadExtType ETy, EVT MemVT,
             MachineMemOperand *MMO)
    : LSBaseSDNode(ISD::LOAD, dl, ChainPtrOff, 3,
                   VTs, AM, MemVT, MMO) {
    SubclassData |= (unsigned short)ETy;
    assert(getExtensionType() == ETy && "LoadExtType encoding error!");
    assert(readMem() && "Load MachineMemOperand is not a load!");
    assert(!writeMem() && "Load MachineMemOperand is a store!");
  }
public:

  /// getExtensionType - Return whether this is a plain node,
  /// or one of the varieties of value-extending loads.
  ISD::LoadExtType getExtensionType() const {
    return ISD::LoadExtType(SubclassData & 3);
  }

  const SDValue &getBasePtr() const { return getOperand(1); }
  const SDValue &getOffset() const { return getOperand(2); }

  static bool classof(const LoadSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD;
  }
};

/// StoreSDNode - This class is used to represent ISD::STORE nodes.
///
class StoreSDNode : public LSBaseSDNode {
  friend class SelectionDAG;
  StoreSDNode(SDValue *ChainValuePtrOff, DebugLoc dl, SDVTList VTs,
              ISD::MemIndexedMode AM, bool isTrunc, EVT MemVT,
              MachineMemOperand *MMO)
    : LSBaseSDNode(ISD::STORE, dl, ChainValuePtrOff, 4,
                   VTs, AM, MemVT, MMO) {
    SubclassData |= (unsigned short)isTrunc;
    assert(isTruncatingStore() == isTrunc && "isTrunc encoding error!");
    assert(!readMem() && "Store MachineMemOperand is a load!");
    assert(writeMem() && "Store MachineMemOperand is not a store!");
  }
public:

  /// isTruncatingStore - Return true if the op does a truncation before store.
  /// For integers this is the same as doing a TRUNCATE and storing the result.
  /// For floats, it is the same as doing an FP_ROUND and storing the result.
  bool isTruncatingStore() const { return SubclassData & 1; }

  const SDValue &getValue() const { return getOperand(1); }
  const SDValue &getBasePtr() const { return getOperand(2); }
  const SDValue &getOffset() const { return getOperand(3); }

  static bool classof(const StoreSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::STORE;
  }
};

/// MachineSDNode - An SDNode that represents everything that will be needed
/// to construct a MachineInstr. These nodes are created during the
/// instruction selection proper phase.
///
class MachineSDNode : public SDNode {
public:
  typedef MachineMemOperand **mmo_iterator;

private:
  friend class SelectionDAG;
  MachineSDNode(unsigned Opc, const DebugLoc DL, SDVTList VTs)
    : SDNode(Opc, DL, VTs), MemRefs(0), MemRefsEnd(0) {}

  /// LocalOperands - Operands for this instruction, if they fit here. If
  /// they don't, this field is unused.
  SDUse LocalOperands[4];

  /// MemRefs - Memory reference descriptions for this instruction.
  mmo_iterator MemRefs;
  mmo_iterator MemRefsEnd;

public:
  mmo_iterator memoperands_begin() const { return MemRefs; }
  mmo_iterator memoperands_end() const { return MemRefsEnd; }
  bool memoperands_empty() const { return MemRefsEnd == MemRefs; }

  /// setMemRefs - Assign this MachineSDNodes's memory reference descriptor
  /// list. This does not transfer ownership.
  void setMemRefs(mmo_iterator NewMemRefs, mmo_iterator NewMemRefsEnd) {
    MemRefs = NewMemRefs;
    MemRefsEnd = NewMemRefsEnd;
  }

  static bool classof(const MachineSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->isMachineOpcode();
  }
};

class SDNodeIterator : public std::iterator<std::forward_iterator_tag,
                                            SDNode, ptrdiff_t> {
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
    return Node->getOperand(Operand).getNode();
  }
  pointer operator->() const { return operator*(); }

  SDNodeIterator& operator++() {                // Preincrement
    ++Operand;
    return *this;
  }
  SDNodeIterator operator++(int) { // Postincrement
    SDNodeIterator tmp = *this; ++*this; return tmp;
  }
  size_t operator-(SDNodeIterator Other) const {
    assert(Node == Other.Node &&
           "Cannot compare iterators of two different nodes!");
    return Operand - Other.Operand;
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

/// LargestSDNode - The largest SDNode class.
///
typedef LoadSDNode LargestSDNode;

/// MostAlignedSDNode - The SDNode class with the greatest alignment
/// requirement.
///
typedef GlobalAddressSDNode MostAlignedSDNode;

namespace ISD {
  /// isNormalLoad - Returns true if the specified node is a non-extending
  /// and unindexed load.
  inline bool isNormalLoad(const SDNode *N) {
    const LoadSDNode *Ld = dyn_cast<LoadSDNode>(N);
    return Ld && Ld->getExtensionType() == ISD::NON_EXTLOAD &&
      Ld->getAddressingMode() == ISD::UNINDEXED;
  }

  /// isNON_EXTLoad - Returns true if the specified node is a non-extending
  /// load.
  inline bool isNON_EXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::NON_EXTLOAD;
  }

  /// isEXTLoad - Returns true if the specified node is a EXTLOAD.
  ///
  inline bool isEXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::EXTLOAD;
  }

  /// isSEXTLoad - Returns true if the specified node is a SEXTLOAD.
  ///
  inline bool isSEXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::SEXTLOAD;
  }

  /// isZEXTLoad - Returns true if the specified node is a ZEXTLOAD.
  ///
  inline bool isZEXTLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::ZEXTLOAD;
  }

  /// isUNINDEXEDLoad - Returns true if the specified node is an unindexed load.
  ///
  inline bool isUNINDEXEDLoad(const SDNode *N) {
    return isa<LoadSDNode>(N) &&
      cast<LoadSDNode>(N)->getAddressingMode() == ISD::UNINDEXED;
  }

  /// isNormalStore - Returns true if the specified node is a non-truncating
  /// and unindexed store.
  inline bool isNormalStore(const SDNode *N) {
    const StoreSDNode *St = dyn_cast<StoreSDNode>(N);
    return St && !St->isTruncatingStore() &&
      St->getAddressingMode() == ISD::UNINDEXED;
  }

  /// isNON_TRUNCStore - Returns true if the specified node is a non-truncating
  /// store.
  inline bool isNON_TRUNCStore(const SDNode *N) {
    return isa<StoreSDNode>(N) && !cast<StoreSDNode>(N)->isTruncatingStore();
  }

  /// isTRUNCStore - Returns true if the specified node is a truncating
  /// store.
  inline bool isTRUNCStore(const SDNode *N) {
    return isa<StoreSDNode>(N) && cast<StoreSDNode>(N)->isTruncatingStore();
  }

  /// isUNINDEXEDStore - Returns true if the specified node is an
  /// unindexed store.
  inline bool isUNINDEXEDStore(const SDNode *N) {
    return isa<StoreSDNode>(N) &&
      cast<StoreSDNode>(N)->getAddressingMode() == ISD::UNINDEXED;
  }
}


} // end llvm namespace

#endif
