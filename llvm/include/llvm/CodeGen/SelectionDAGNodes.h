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

#include "llvm/Value.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/iterator"
#include "llvm/ADT/APFloat.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {

class SelectionDAG;
class GlobalValue;
class MachineBasicBlock;
class MachineConstantPoolValue;
class SDNode;
template <typename T> struct DenseMapInfo;
template <typename T> struct simplify_type;
template <typename T> struct ilist_traits;
template<typename NodeTy, typename Traits> class iplist;
template<typename NodeTy> class ilist_iterator;

/// SDVTList - This represents a list of ValueType's that has been intern'd by
/// a SelectionDAG.  Instances of this simple value class are returned by
/// SelectionDAG::getVTList(...).
///
struct SDVTList {
  const MVT::ValueType *VTs;
  unsigned short NumVTs;
};

/// ISD namespace - This namespace contains an enum which represents all of the
/// SelectionDAG node types and value types.
///
namespace ISD {
  namespace ParamFlags {    
  enum Flags {
    NoFlagSet         = 0,
    ZExt              = 1<<0,  ///< Parameter should be zero extended
    ZExtOffs          = 0,
    SExt              = 1<<1,  ///< Parameter should be sign extended
    SExtOffs          = 1,
    InReg             = 1<<2,  ///< Parameter should be passed in register
    InRegOffs         = 2,
    StructReturn      = 1<<3,  ///< Hidden struct-return pointer
    StructReturnOffs  = 3,
    ByVal             = 1<<4,  ///< Struct passed by value
    ByValOffs         = 4,
    Nest              = 1<<5,  ///< Parameter is nested function static chain
    NestOffs          = 5,
    ByValAlign        = 0xF << 6, //< The alignment of the struct
    ByValAlignOffs    = 6,
    ByValSize         = 0x1ffff << 10, //< The size of the struct
    ByValSizeOffs     = 10,
    OrigAlignment     = 0x1F<<27,
    OrigAlignmentOffs = 27
  };
  }

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
    GlobalAddress, GlobalTLSAddress, FrameIndex,
    JumpTable, ConstantPool, ExternalSymbol,

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
    
    /// FORMAL_ARGUMENTS(CHAIN, CC#, ISVARARG, FLAG0, ..., FLAGn) - This node
    /// represents the formal arguments for a function.  CC# is a Constant value
    /// indicating the calling convention of the function, and ISVARARG is a
    /// flag that indicates whether the function is varargs or not. This node
    /// has one result value for each incoming argument, plus one for the output
    /// chain. It must be custom legalized. See description of CALL node for
    /// FLAG argument contents explanation.
    /// 
    FORMAL_ARGUMENTS,
    
    /// RV1, RV2...RVn, CHAIN = CALL(CHAIN, CC#, ISVARARG, ISTAILCALL, CALLEE,
    ///                              ARG0, FLAG0, ARG1, FLAG1, ... ARGn, FLAGn)
    /// This node represents a fully general function call, before the legalizer
    /// runs.  This has one result value for each argument / flag pair, plus
    /// a chain result. It must be custom legalized. Flag argument indicates
    /// misc. argument attributes. Currently:
    /// Bit 0 - signness
    /// Bit 1 - 'inreg' attribute
    /// Bit 2 - 'sret' attribute
    /// Bit 4 - 'byval' attribute
    /// Bit 5 - 'nest' attribute
    /// Bit 6-9 - alignment of byval structures
    /// Bit 10-26 - size of byval structures
    /// Bits 31:27 - argument ABI alignment in the first argument piece and
    /// alignment '1' in other argument pieces.
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
    
    /// BUILD_VECTOR(ELT0, ELT1, ELT2, ELT3,...) - Return a vector
    /// with the specified, possibly variable, elements.  The number of elements
    /// is required to be a power of two.
    BUILD_VECTOR,
    
    /// INSERT_VECTOR_ELT(VECTOR, VAL, IDX) - Returns VECTOR with the element
    /// at IDX replaced with VAL.
    INSERT_VECTOR_ELT,

    /// EXTRACT_VECTOR_ELT(VECTOR, IDX) - Returns a single element from VECTOR
    /// identified by the (potentially variable) element number IDX.
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
    
    /// VECTOR_SHUFFLE(VEC1, VEC2, SHUFFLEVEC) - Returns a vector, of the same
    /// type as VEC1/VEC2.  SHUFFLEVEC is a BUILD_VECTOR of constant int values
    /// (regardless of whether its datatype is legal or not) that indicate
    /// which value each result element will get.  The elements of VEC1/VEC2 are
    /// enumerated in order.  This is quite similar to the Altivec 'vperm'
    /// instruction, except that the indices must be constants and are in terms
    /// of the element size of VEC1/VEC2, not in terms of bytes.
    VECTOR_SHUFFLE,
    
    /// SCALAR_TO_VECTOR(VAL) - This represents the operation of loading a
    /// scalar value into element 0 of the resultant vector type.  The top
    /// elements 1 to N-1 of the N-element vector are undefined.
    SCALAR_TO_VECTOR,
    
    // EXTRACT_SUBREG - This node is used to extract a sub-register value. 
    // This node takes a superreg and a constant sub-register index as operands.
    EXTRACT_SUBREG,
    
    // INSERT_SUBREG - This node is used to insert a sub-register value. 
    // This node takes a superreg, a subreg value, and a constant sub-register
    // index as operands.
    INSERT_SUBREG,
    
    // MULHU/MULHS - Multiply high - Multiply two integers of type iN, producing
    // an unsigned/signed value of type i[2*N], then return the top part.
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

    // FLT_ROUNDS - Returns current rounding mode:
    // -1 Undefined
    //  0 Round to 0
    //  1 Round to nearest
    //  2 Round to +inf
    //  3 Round to -inf
    FLT_ROUNDS,

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
    
    // FNEG, FABS, FSQRT, FSIN, FCOS, FPOWI, FPOW - Perform unary floating point
    // negation, absolute value, square root, sine and cosine, powi, and pow
    // operations.
    FNEG, FABS, FSQRT, FSIN, FCOS, FPOWI, FPOW,
    
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
    
    // LABEL - Represents a label in mid basic block used to track
    // locations needed for debug and exception handling tables.  This node
    // returns a chain.
    //   Operand #0 : input chain.
    //   Operand #1 : module unique number use to identify the label.
    LABEL,
    
    // STACKSAVE - STACKSAVE has one operand, an input chain.  It produces a
    // value, the same type as the pointer type for the system, and an output
    // chain.
    STACKSAVE,
    
    // STACKRESTORE has two operands, an input chain and a pointer to restore to
    // it returns an output chain.
    STACKRESTORE,
    
    // MEMSET/MEMCPY/MEMMOVE - The first operand is the chain. The following
    // correspond to the operands of the LLVM intrinsic functions and the last
    // one is AlwaysInline.  The only result is a token chain.  The alignment
    // argument is guaranteed to be a Constant node.
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
    // number, then a column then a file id (provided by MachineModuleInfo.) It
    // produces a token chain as output.
    DEBUG_LOC,

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
  /// MemIndexedMode enum - This enum defines the load / store indexed 
  /// addressing modes.
  ///
  /// UNINDEXED    "Normal" load / store. The effective address is already
  ///              computed and is available in the base pointer. The offset
  ///              operand is always undefined. In addition to producing a
  ///              chain, an unindexed load produces one value (result of the
  ///              load); an unindexed store does not produces a value.
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
    LAST_LOADX_TYPE
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
  inline uint64_t getConstantOperandVal(unsigned i) const;
  inline bool isTargetOpcode() const;
  inline unsigned getTargetOpcode() const;

  
  /// reachesChainWithoutSideEffects - Return true if this operand (which must
  /// be a chain) reaches the specified operand without crossing any 
  /// side-effecting instructions.  In practice, this looks through token
  /// factors and non-volatile loads.  In order to remain efficient, this only
  /// looks a couple of nodes in, it does not do an exhaustive search.
  bool reachesChainWithoutSideEffects(SDOperand Dest, unsigned Depth = 2) const;
  
  /// hasOneUse - Return true if there is exactly one operation using this
  /// result value of the defining operator.
  inline bool hasOneUse() const;

  /// use_empty - Return true if there are no operations using this
  /// result value of the defining operator.
  inline bool use_empty() const;
};


template<> struct DenseMapInfo<SDOperand> {
  static inline SDOperand getEmptyKey() { return SDOperand((SDNode*)-1, -1U); }
  static inline SDOperand getTombstoneKey() { return SDOperand((SDNode*)-1, 0);}
  static unsigned getHashValue(const SDOperand &Val) {
    return (unsigned)((uintptr_t)Val.Val >> 4) ^
           (unsigned)((uintptr_t)Val.Val >> 9) + Val.ResNo;
  }
  static bool isEqual(const SDOperand &LHS, const SDOperand &RHS) {
    return LHS == RHS;
  }
  static bool isPod() { return true; }
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
class SDNode : public FoldingSetNode {
  /// NodeType - The operation that this node performs.
  ///
  unsigned short NodeType;
  
  /// OperandsNeedDelete - This is true if OperandList was new[]'d.  If true,
  /// then they will be delete[]'d when the node is destroyed.
  bool OperandsNeedDelete : 1;

  /// NodeId - Unique id per SDNode in the DAG.
  int NodeId;

  /// OperandList - The values that are used by this operation.
  ///
  SDOperand *OperandList;
  
  /// ValueList - The types of the values this node defines.  SDNode's may
  /// define multiple values simultaneously.
  const MVT::ValueType *ValueList;

  /// NumOperands/NumValues - The number of entries in the Operand/Value list.
  unsigned short NumOperands, NumValues;
  
  /// Prev/Next pointers - These pointers form the linked list of of the
  /// AllNodes list in the current DAG.
  SDNode *Prev, *Next;
  friend struct ilist_traits<SDNode>;

  /// Uses - These are all of the SDNode's that use a value produced by this
  /// node.
  SmallVector<SDNode*,3> Uses;
  
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

  /// setNodeId - Set unique node id.
  void setNodeId(int Id) { NodeId = Id; }

  typedef SmallVector<SDNode*,3>::const_iterator use_iterator;
  use_iterator use_begin() const { return Uses.begin(); }
  use_iterator use_end() const { return Uses.end(); }

  /// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
  /// indicated value.  This method ignores uses of other values defined by this
  /// operation.
  bool hasNUsesOfValue(unsigned NUses, unsigned Value) const;

  /// hasAnyUseOfValue - Return true if there are any use of the indicated
  /// value. This method ignores uses of other values defined by this operation.
  bool hasAnyUseOfValue(unsigned Value) const;

  /// isOnlyUse - Return true if this node is the only use of N.
  ///
  bool isOnlyUse(SDNode *N) const;

  /// isOperand - Return true if this node is an operand of N.
  ///
  bool isOperand(SDNode *N) const;

  /// isPredecessor - Return true if this node is a predecessor of N. This node
  /// is either an operand of N or it can be reached by recursively traversing
  /// up the operands.
  /// NOTE: this is an expensive method. Use it carefully.
  bool isPredecessor(SDNode *N) const;

  /// getNumOperands - Return the number of values used by this operation.
  ///
  unsigned getNumOperands() const { return NumOperands; }

  /// getConstantOperandVal - Helper method returns the integer value of a 
  /// ConstantSDNode operand.
  uint64_t getConstantOperandVal(unsigned Num) const;

  const SDOperand &getOperand(unsigned Num) const {
    assert(Num < NumOperands && "Invalid child # of SDNode!");
    return OperandList[Num];
  }

  typedef const SDOperand* op_iterator;
  op_iterator op_begin() const { return OperandList; }
  op_iterator op_end() const { return OperandList+NumOperands; }


  SDVTList getVTList() const {
    SDVTList X = { ValueList, NumValues };
    return X;
  };
  
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
  std::string getOperationName(const SelectionDAG *G = 0) const;
  static const char* getIndexedModeName(ISD::MemIndexedMode AM);
  void dump() const;
  void dump(const SelectionDAG *G) const;

  static bool classof(const SDNode *) { return true; }

  /// Profile - Gather unique data for the node.
  ///
  void Profile(FoldingSetNodeID &ID);

protected:
  friend class SelectionDAG;
  
  /// getValueTypeList - Return a pointer to the specified value type.
  ///
  static MVT::ValueType *getValueTypeList(MVT::ValueType VT);
  static SDVTList getSDVTList(MVT::ValueType VT) {
    SDVTList Ret = { getValueTypeList(VT), 1 };
    return Ret;
  }

  SDNode(unsigned Opc, SDVTList VTs, const SDOperand *Ops, unsigned NumOps)
    : NodeType(Opc), NodeId(-1) {
    OperandsNeedDelete = true;
    NumOperands = NumOps;
    OperandList = NumOps ? new SDOperand[NumOperands] : 0;
    
    for (unsigned i = 0; i != NumOps; ++i) {
      OperandList[i] = Ops[i];
      Ops[i].Val->Uses.push_back(this);
    }
    
    ValueList = VTs.VTs;
    NumValues = VTs.NumVTs;
    Prev = 0; Next = 0;
  }
  SDNode(unsigned Opc, SDVTList VTs) : NodeType(Opc), NodeId(-1) {
    OperandsNeedDelete = false;  // Operands set with InitOperands.
    NumOperands = 0;
    OperandList = 0;
    
    ValueList = VTs.VTs;
    NumValues = VTs.NumVTs;
    Prev = 0; Next = 0;
  }
  
  /// InitOperands - Initialize the operands list of this node with the
  /// specified values, which are part of the node (thus they don't need to be
  /// copied in or allocated).
  void InitOperands(SDOperand *Ops, unsigned NumOps) {
    assert(OperandList == 0 && "Operands already set!");
    NumOperands = NumOps;
    OperandList = Ops;
    
    for (unsigned i = 0; i != NumOps; ++i)
      Ops[i].Val->Uses.push_back(this);
  }
  
  /// MorphNodeTo - This frees the operands of the current node, resets the
  /// opcode, types, and operands to the specified value.  This should only be
  /// used by the SelectionDAG class.
  void MorphNodeTo(unsigned Opc, SDVTList L,
                   const SDOperand *Ops, unsigned NumOps);
  
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
inline uint64_t SDOperand::getConstantOperandVal(unsigned i) const {
  return Val->getConstantOperandVal(i);
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
inline bool SDOperand::use_empty() const {
  return !Val->hasAnyUseOfValue(ResNo);
}

/// UnarySDNode - This class is used for single-operand SDNodes.  This is solely
/// to allow co-allocation of node operands with the node itself.
class UnarySDNode : public SDNode {
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
  SDOperand Op;
public:
  UnarySDNode(unsigned Opc, SDVTList VTs, SDOperand X)
    : SDNode(Opc, VTs), Op(X) {
    InitOperands(&Op, 1);
  }
};

/// BinarySDNode - This class is used for two-operand SDNodes.  This is solely
/// to allow co-allocation of node operands with the node itself.
class BinarySDNode : public SDNode {
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
  SDOperand Ops[2];
public:
  BinarySDNode(unsigned Opc, SDVTList VTs, SDOperand X, SDOperand Y)
    : SDNode(Opc, VTs) {
    Ops[0] = X;
    Ops[1] = Y;
    InitOperands(Ops, 2);
  }
};

/// TernarySDNode - This class is used for three-operand SDNodes. This is solely
/// to allow co-allocation of node operands with the node itself.
class TernarySDNode : public SDNode {
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
  SDOperand Ops[3];
public:
  TernarySDNode(unsigned Opc, SDVTList VTs, SDOperand X, SDOperand Y,
                SDOperand Z)
    : SDNode(Opc, VTs) {
    Ops[0] = X;
    Ops[1] = Y;
    Ops[2] = Z;
    InitOperands(Ops, 3);
  }
};


/// HandleSDNode - This class is used to form a handle around another node that
/// is persistant and is updated across invocations of replaceAllUsesWith on its
/// operand.  This node should be directly created by end-users and not added to
/// the AllNodes list.
class HandleSDNode : public SDNode {
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
  SDOperand Op;
public:
  explicit HandleSDNode(SDOperand X)
    : SDNode(ISD::HANDLENODE, getSDVTList(MVT::Other)), Op(X) {
    InitOperands(&Op, 1);
  }
  ~HandleSDNode();  
  SDOperand getValue() const { return Op; }
};

class StringSDNode : public SDNode {
  std::string Value;
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  explicit StringSDNode(const std::string &val)
    : SDNode(ISD::STRING, getSDVTList(MVT::Other)), Value(val) {
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
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  ConstantSDNode(bool isTarget, uint64_t val, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetConstant : ISD::Constant, getSDVTList(VT)),
      Value(val) {
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
  APFloat Value;
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
  // Longterm plan: replace all uses of getValue with getValueAPF, remove
  // getValue, rename getValueAPF to getValue.
protected:
  friend class SelectionDAG;
  ConstantFPSDNode(bool isTarget, const APFloat& val, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetConstantFP : ISD::ConstantFP,
             getSDVTList(VT)), Value(val) {
  }
public:

  const APFloat& getValueAPF() const { return Value; }

  /// isExactlyValue - We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.

  /// We leave the version with the double argument here because it's just so
  /// convenient to write "2.0" and the like.  Without this function we'd 
  /// have to duplicate its logic everywhere it's called.
  bool isExactlyValue(double V) const { 
    if (getValueType(0)==MVT::f64)
      return isExactlyValue(APFloat(V));
    else
      return isExactlyValue(APFloat((float)V));
  }
  bool isExactlyValue(const APFloat& V) const;

  bool isValueValidForType(MVT::ValueType VT, const APFloat& Val);

  static bool classof(const ConstantFPSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantFP || 
           N->getOpcode() == ISD::TargetConstantFP;
  }
};

class GlobalAddressSDNode : public SDNode {
  GlobalValue *TheGlobal;
  int Offset;
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  GlobalAddressSDNode(bool isTarget, const GlobalValue *GA, MVT::ValueType VT,
                      int o = 0);
public:

  GlobalValue *getGlobal() const { return TheGlobal; }
  int getOffset() const { return Offset; }

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
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  FrameIndexSDNode(int fi, MVT::ValueType VT, bool isTarg)
    : SDNode(isTarg ? ISD::TargetFrameIndex : ISD::FrameIndex, getSDVTList(VT)),
      FI(fi) {
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
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  JumpTableSDNode(int jti, MVT::ValueType VT, bool isTarg)
    : SDNode(isTarg ? ISD::TargetJumpTable : ISD::JumpTable, getSDVTList(VT)),
      JTI(jti) {
  }
public:
    
    int getIndex() const { return JTI; }
  
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
  unsigned Alignment;
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  ConstantPoolSDNode(bool isTarget, Constant *c, MVT::ValueType VT,
                     int o=0)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool,
             getSDVTList(VT)), Offset(o), Alignment(0) {
    assert((int)Offset >= 0 && "Offset is too large");
    Val.ConstVal = c;
  }
  ConstantPoolSDNode(bool isTarget, Constant *c, MVT::ValueType VT, int o,
                     unsigned Align)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool, 
             getSDVTList(VT)), Offset(o), Alignment(Align) {
    assert((int)Offset >= 0 && "Offset is too large");
    Val.ConstVal = c;
  }
  ConstantPoolSDNode(bool isTarget, MachineConstantPoolValue *v,
                     MVT::ValueType VT, int o=0)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool, 
             getSDVTList(VT)), Offset(o), Alignment(0) {
    assert((int)Offset >= 0 && "Offset is too large");
    Val.MachineCPVal = v;
    Offset |= 1 << (sizeof(unsigned)*8-1);
  }
  ConstantPoolSDNode(bool isTarget, MachineConstantPoolValue *v,
                     MVT::ValueType VT, int o, unsigned Align)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool,
             getSDVTList(VT)), Offset(o), Alignment(Align) {
    assert((int)Offset >= 0 && "Offset is too large");
    Val.MachineCPVal = v;
    Offset |= 1 << (sizeof(unsigned)*8-1);
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
    return Offset & ~(1 << (sizeof(unsigned)*8-1));
  }
  
  // Return the alignment of this constant pool object, which is either 0 (for
  // default alignment) or log2 of the desired value.
  unsigned getAlignment() const { return Alignment; }

  const Type *getType() const;

  static bool classof(const ConstantPoolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantPool ||
           N->getOpcode() == ISD::TargetConstantPool;
  }
};

class BasicBlockSDNode : public SDNode {
  MachineBasicBlock *MBB;
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  explicit BasicBlockSDNode(MachineBasicBlock *mbb)
    : SDNode(ISD::BasicBlock, getSDVTList(MVT::Other)), MBB(mbb) {
  }
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
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  SrcValueSDNode(const Value* v, int o)
    : SDNode(ISD::SRCVALUE, getSDVTList(MVT::Other)), V(v), offset(o) {
  }

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
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  RegisterSDNode(unsigned reg, MVT::ValueType VT)
    : SDNode(ISD::Register, getSDVTList(VT)), Reg(reg) {
  }
public:

  unsigned getReg() const { return Reg; }

  static bool classof(const RegisterSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Register;
  }
};

class ExternalSymbolSDNode : public SDNode {
  const char *Symbol;
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  ExternalSymbolSDNode(bool isTarget, const char *Sym, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetExternalSymbol : ISD::ExternalSymbol,
             getSDVTList(VT)), Symbol(Sym) {
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
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  explicit CondCodeSDNode(ISD::CondCode Cond)
    : SDNode(ISD::CONDCODE, getSDVTList(MVT::Other)), Condition(Cond) {
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
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
protected:
  friend class SelectionDAG;
  explicit VTSDNode(MVT::ValueType VT)
    : SDNode(ISD::VALUETYPE, getSDVTList(MVT::Other)), ValueType(VT) {
  }
public:

  MVT::ValueType getVT() const { return ValueType; }

  static bool classof(const VTSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::VALUETYPE;
  }
};

/// LSBaseSDNode - Base class for LoadSDNode and StoreSDNode
///
class LSBaseSDNode : public SDNode {
private:
  //! SrcValue - Memory location for alias analysis.
  const Value *SrcValue;

  //! SVOffset - Memory location offset.
  int SVOffset;

  //! Alignment - Alignment of memory location in bytes.
  unsigned Alignment;

  //! IsVolatile - True if the store is volatile.
  bool IsVolatile;
protected:
  //! Operand array for load and store
  /*!
    \note Moving this array to the base class captures more
    common functionality shared between LoadSDNode and
    StoreSDNode
   */
  SDOperand Ops[4];
public:
  LSBaseSDNode(ISD::NodeType NodeTy, SDVTList VTs, const Value *SV, int SVO,
               unsigned Align, bool Vol)
    : SDNode(NodeTy, VTs),
      SrcValue(SV), SVOffset(SVO), Alignment(Align), IsVolatile(Vol)
  { }

  const SDOperand getChain() const {
    return getOperand(0);
  }
  const SDOperand getBasePtr() const {
    return getOperand(getOpcode() == ISD::LOAD ? 1 : 2);
  }
  const SDOperand getOffset() const {
    return getOperand(getOpcode() == ISD::LOAD ? 2 : 3);
  }
  const SDOperand getValue() const {
    assert(getOpcode() == ISD::STORE);
    return getOperand(1);
  }

  const Value *getSrcValue() const { return SrcValue; }
  int getSrcValueOffset() const { return SVOffset; }
  unsigned getAlignment() const { return Alignment; }
  bool isVolatile() const { return IsVolatile; }

  static bool classof(const LSBaseSDNode *N) { return true; }
  static bool classof(const SDNode *N) { return true; }
};

/// LoadSDNode - This class is used to represent ISD::LOAD nodes.
///
class LoadSDNode : public LSBaseSDNode {
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
  
  // AddrMode - unindexed, pre-indexed, post-indexed.
  ISD::MemIndexedMode AddrMode;

  // ExtType - non-ext, anyext, sext, zext.
  ISD::LoadExtType ExtType;

  // LoadedVT - VT of loaded value before extension.
  MVT::ValueType LoadedVT;
protected:
  friend class SelectionDAG;
  LoadSDNode(SDOperand *ChainPtrOff, SDVTList VTs,
             ISD::MemIndexedMode AM, ISD::LoadExtType ETy, MVT::ValueType LVT,
             const Value *SV, int O=0, unsigned Align=0, bool Vol=false)
    : LSBaseSDNode(ISD::LOAD, VTs, SV, O, Align, Vol),
      AddrMode(AM), ExtType(ETy), LoadedVT(LVT) {
    Ops[0] = ChainPtrOff[0]; // Chain
    Ops[1] = ChainPtrOff[1]; // Ptr
    Ops[2] = ChainPtrOff[2]; // Off
    InitOperands(Ops, 3);
    assert(Align != 0 && "Loads should have non-zero aligment");
    assert((getOffset().getOpcode() == ISD::UNDEF ||
            AddrMode != ISD::UNINDEXED) &&
           "Only indexed load has a non-undef offset operand");
  }
public:

  ISD::MemIndexedMode getAddressingMode() const { return AddrMode; }
  ISD::LoadExtType getExtensionType() const { return ExtType; }
  MVT::ValueType getLoadedVT() const { return LoadedVT; }

  static bool classof(const LoadSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD;
  }
};

/// StoreSDNode - This class is used to represent ISD::STORE nodes.
///
class StoreSDNode : public LSBaseSDNode {
  virtual void ANCHOR();  // Out-of-line virtual method to give class a home.
    
  // AddrMode - unindexed, pre-indexed, post-indexed.
  ISD::MemIndexedMode AddrMode;

  // IsTruncStore - True if the op does a truncation before store.
  bool IsTruncStore;

  // StoredVT - VT of the value after truncation.
  MVT::ValueType StoredVT;
protected:
  friend class SelectionDAG;
  StoreSDNode(SDOperand *ChainValuePtrOff, SDVTList VTs,
              ISD::MemIndexedMode AM, bool isTrunc, MVT::ValueType SVT,
              const Value *SV, int O=0, unsigned Align=0, bool Vol=false)
    : LSBaseSDNode(ISD::STORE, VTs, SV, O, Align, Vol),
      AddrMode(AM), IsTruncStore(isTrunc), StoredVT(SVT) {
    Ops[0] = ChainValuePtrOff[0]; // Chain
    Ops[1] = ChainValuePtrOff[1]; // Value
    Ops[2] = ChainValuePtrOff[2]; // Ptr
    Ops[3] = ChainValuePtrOff[3]; // Off
    InitOperands(Ops, 4);
    assert(Align != 0 && "Stores should have non-zero aligment");
    assert((getOffset().getOpcode() == ISD::UNDEF || 
            AddrMode != ISD::UNINDEXED) &&
           "Only indexed store has a non-undef offset operand");
  }
public:

  ISD::MemIndexedMode getAddressingMode() const { return AddrMode; }
  bool isTruncatingStore() const { return IsTruncStore; }
  MVT::ValueType getStoredVT() const { return StoredVT; }

  static bool classof(const StoreSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::STORE;
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
    return new SDNode(ISD::EntryToken, SDNode::getSDVTList(MVT::Other));
  }
  static void destroySentinel(SDNode *N) { delete N; }
  //static SDNode *createNode(const SDNode &V) { return new SDNode(V); }
  
  
  void addNodeToList(SDNode *NTy) {}
  void removeNodeFromList(SDNode *NTy) {}
  void transferNodesFromList(iplist<SDNode, ilist_traits> &L2,
                             const ilist_iterator<SDNode> &X,
                             const ilist_iterator<SDNode> &Y) {}
};

namespace ISD {
  /// isNormalLoad - Returns true if the specified node is a non-extending
  /// and unindexed load.
  inline bool isNormalLoad(const SDNode *N) {
    if (N->getOpcode() != ISD::LOAD)
      return false;
    const LoadSDNode *Ld = cast<LoadSDNode>(N);
    return Ld->getExtensionType() == ISD::NON_EXTLOAD &&
      Ld->getAddressingMode() == ISD::UNINDEXED;
  }

  /// isNON_EXTLoad - Returns true if the specified node is a non-extending
  /// load.
  inline bool isNON_EXTLoad(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::NON_EXTLOAD;
  }

  /// isEXTLoad - Returns true if the specified node is a EXTLOAD.
  ///
  inline bool isEXTLoad(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::EXTLOAD;
  }

  /// isSEXTLoad - Returns true if the specified node is a SEXTLOAD.
  ///
  inline bool isSEXTLoad(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::SEXTLOAD;
  }

  /// isZEXTLoad - Returns true if the specified node is a ZEXTLOAD.
  ///
  inline bool isZEXTLoad(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD &&
      cast<LoadSDNode>(N)->getExtensionType() == ISD::ZEXTLOAD;
  }

  /// isUNINDEXEDLoad - Returns true if the specified node is a unindexed load.
  ///
  inline bool isUNINDEXEDLoad(const SDNode *N) {
    return N->getOpcode() == ISD::LOAD &&
      cast<LoadSDNode>(N)->getAddressingMode() == ISD::UNINDEXED;
  }

  /// isNON_TRUNCStore - Returns true if the specified node is a non-truncating
  /// store.
  inline bool isNON_TRUNCStore(const SDNode *N) {
    return N->getOpcode() == ISD::STORE &&
      !cast<StoreSDNode>(N)->isTruncatingStore();
  }

  /// isTRUNCStore - Returns true if the specified node is a truncating
  /// store.
  inline bool isTRUNCStore(const SDNode *N) {
    return N->getOpcode() == ISD::STORE &&
      cast<StoreSDNode>(N)->isTruncatingStore();
  }
}


} // end llvm namespace

#endif
