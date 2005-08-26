//===- Target.td - Target Independent TableGen interface ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces which should be
// implemented by each target which is using a TableGen based code generator.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Value types - These values correspond to the register types defined in the
// ValueTypes.h file.  If you update anything here, you must update it there as
// well!
//
class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

def OtherVT: ValueType<0  ,  0>;   // "Other" value
def i1     : ValueType<1  ,  1>;   // One bit boolean value
def i8     : ValueType<8  ,  2>;   // 8-bit integer value
def i16    : ValueType<16 ,  3>;   // 16-bit integer value
def i32    : ValueType<32 ,  4>;   // 32-bit integer value
def i64    : ValueType<64 ,  5>;   // 64-bit integer value
def i128   : ValueType<128,  5>;   // 128-bit integer value
def f32    : ValueType<32 ,  7>;   // 32-bit floating point value
def f64    : ValueType<64 ,  8>;   // 64-bit floating point value
def f80    : ValueType<80 ,  9>;   // 80-bit floating point value
def f128   : ValueType<128, 10>;   // 128-bit floating point value
def FlagVT : ValueType<0  , 11>;   // Condition code or machine flag
def isVoid : ValueType<0  , 12>;   // Produces no value

//===----------------------------------------------------------------------===//
// Register file description - These classes are used to fill in the target
// description classes in llvm/Target/MRegisterInfo.h


// Register - You should define one instance of this class for each register
// in the target machine.  String n will become the "name" of the register.
class RegisterBase<string n> {
  string Namespace = "";
  string Name = n;

  // SpillSize - If this value is set to a non-zero value, it is the size in
  // bits of the spill slot required to hold this register.  If this value is
  // set to zero, the information is inferred from any register classes the
  // register belongs to.
  int SpillSize = 0;

  // SpillAlignment - This value is used to specify the alignment required for
  // spilling the register.  Like SpillSize, this should only be explicitly
  // specified if the register is not in a register class.
  int SpillAlignment = 0;
}

class Register<string n> : RegisterBase<n> {
  list<RegisterBase> Aliases = [];
}

// RegisterGroup - This can be used to define instances of Register which
// need to specify aliases.
// List "aliases" specifies which registers are aliased to this one.  This
// allows the code generator to be careful not to put two values with 
// overlapping live ranges into registers which alias.
class RegisterGroup<string n, list<Register> aliases> : Register<n> {
  let Aliases = aliases;
}

// RegisterClass - Now that all of the registers are defined, and aliases
// between registers are defined, specify which registers belong to which
// register classes.  This also defines the default allocation order of
// registers by register allocators.
//
class RegisterClass<string namespace, ValueType regType, int alignment,
                    list<Register> regList> {
  string Namespace = namespace;

  // RegType - Specify the ValueType of the registers in this register class.
  // Note that all registers in a register class must have the same ValueType.
  //
  ValueType RegType = regType;

  // Alignment - Specify the alignment required of the registers when they are
  // stored or loaded to memory.
  //
  int Size = RegType.Size;
  int Alignment = alignment;

  // MemberList - Specify which registers are in this class.  If the
  // allocation_order_* method are not specified, this also defines the order of
  // allocation used by the register allocator.
  //
  list<Register> MemberList = regList;

  // MethodProtos/MethodBodies - These members can be used to insert arbitrary
  // code into a generated register class.   The normal usage of this is to 
  // overload virtual methods.
  code MethodProtos = [{}];
  code MethodBodies = [{}];
}


//===----------------------------------------------------------------------===//
// Instruction set description - These classes correspond to the C++ classes in
// the Target/TargetInstrInfo.h file.
//
class Instruction {
  string Name = "";         // The opcode string for this instruction
  string Namespace = "";

  dag OperandList;          // An dag containing the MI operand list.
  string AsmString = "";    // The .s format to print the instruction with.

  // Pattern - Set to the DAG pattern for this instruction, if we know of one,
  // otherwise, uninitialized.
  list<dag> Pattern;

  // The follow state will eventually be inferred automatically from the
  // instruction pattern.

  list<Register> Uses = []; // Default to using no non-operand registers
  list<Register> Defs = []; // Default to modifying no non-operand registers

  // These bits capture information about the high-level semantics of the
  // instruction.
  bit isReturn     = 0;     // Is this instruction a return instruction?
  bit isBranch     = 0;     // Is this instruction a branch instruction?
  bit isBarrier    = 0;     // Can control flow fall through this instruction?
  bit isCall       = 0;     // Is this instruction a call instruction?
  bit isLoad       = 0;     // Is this instruction a load instruction?
  bit isStore      = 0;     // Is this instruction a store instruction?
  bit isTwoAddress = 0;     // Is this a two address instruction?
  bit isConvertibleToThreeAddress = 0;  // Can this 2-addr instruction promote?
  bit isCommutable = 0;     // Is this 3 operand instruction commutable?
  bit isTerminator = 0;     // Is this part of the terminator for a basic block?
  bit hasDelaySlot = 0;     // Does this instruction have an delay slot?
  bit usesCustomDAGSchedInserter = 0; // Pseudo instr needing special help.
}


/// ops definition - This is just a simple marker used to identify the operands
/// list for an instruction.  This should be used like this:
///     (ops R32:$dst, R32:$src) or something similar.
def ops;

/// variable_ops definition - Mark this instruction as taking a variable number
/// of operands.
def variable_ops;

/// Operand Types - These provide the built-in operand types that may be used
/// by a target.  Targets can optionally provide their own operand types as
/// needed, though this should not be needed for RISC targets.
class Operand<ValueType ty> {
  int NumMIOperands = 1;
  ValueType Type = ty;
  string PrintMethod = "printOperand";
}

def i1imm  : Operand<i1>;
def i8imm  : Operand<i8>;
def i16imm : Operand<i16>;
def i32imm : Operand<i32>;
def i64imm : Operand<i64>;

// InstrInfo - This class should only be instantiated once to provide parameters
// which are global to the the target machine.
//
class InstrInfo {
  Instruction PHIInst;

  // If the target wants to associate some target-specific information with each
  // instruction, it should provide these two lists to indicate how to assemble
  // the target specific information into the 32 bits available.
  //
  list<string> TSFlagsFields = [];
  list<int>    TSFlagsShifts = [];

  // Target can specify its instructions in either big or little-endian formats.
  // For instance, while both Sparc and PowerPC are big-endian platforms, the
  // Sparc manual specifies its instructions in the format [31..0] (big), while
  // PowerPC specifies them using the format [0..31] (little).
  bit isLittleEndianEncoding = 0;
}

//===----------------------------------------------------------------------===//
// AsmWriter - This class can be implemented by targets that need to customize
// the format of the .s file writer.
//
// Subtargets can have multiple different asmwriters (e.g. AT&T vs Intel syntax
// on X86 for example).
//
class AsmWriter {
  // AsmWriterClassName - This specifies the suffix to use for the asmwriter
  // class.  Generated AsmWriter classes are always prefixed with the target
  // name.
  string AsmWriterClassName  = "AsmPrinter";

  // InstFormatName - AsmWriters can specify the name of the format string to
  // print instructions with.
  string InstFormatName = "AsmString";

  // Variant - AsmWriters can be of multiple different variants.  Variants are
  // used to support targets that need to emit assembly code in ways that are
  // mostly the same for different targets, but have minor differences in
  // syntax.  If the asmstring contains {|} characters in them, this integer
  // will specify which alternative to use.  For example "{x|y|z}" with Variant
  // == 1, will expand to "y".
  int Variant = 0;
}
def DefaultAsmWriter : AsmWriter;


//===----------------------------------------------------------------------===//
// Target - This class contains the "global" target information
//
class Target {
  // CalleeSavedRegisters - As you might guess, this is a list of the callee
  // saved registers for a target.
  list<Register> CalleeSavedRegisters = [];
  
  // PointerType - Specify the value type to be used to represent pointers in
  // this target.  Typically this is an i32 or i64 type.
  ValueType PointerType;

  // InstructionSet - Instruction set description for this target.
  InstrInfo InstructionSet;

  // AssemblyWriters - The AsmWriter instances available for this target.
  list<AsmWriter> AssemblyWriters = [DefaultAsmWriter];
}


//===----------------------------------------------------------------------===//
// DAG node definitions used by the instruction selector.
//
// NOTE: all of this is a work-in-progress and should be ignored for now.
//
/*
class Expander<dag pattern, list<dag> result> {
  dag Pattern      = pattern;
  list<dag> Result = result;
}

class DagNodeValType;
def DNVT_any   : DagNodeValType;  // No constraint on tree node
def DNVT_void  : DagNodeValType;  // Tree node always returns void
def DNVT_val   : DagNodeValType;  // A non-void type
def DNVT_arg0  : DagNodeValType;  // Tree node returns same type as Arg0
def DNVT_arg1  : DagNodeValType;  // Tree node returns same type as Arg1
def DNVT_ptr   : DagNodeValType;  // The target pointer type
def DNVT_i8    : DagNodeValType;  // Always have an i8 value

class DagNode<DagNodeValType ret, list<DagNodeValType> args> {
  DagNodeValType RetType = ret;
  list<DagNodeValType> ArgTypes = args;
  string EnumName = ?;
}

// BuiltinDagNodes are built into the instruction selector and correspond to
// enum values.
class BuiltinDagNode<DagNodeValType Ret, list<DagNodeValType> Args,
                     string Ename> : DagNode<Ret, Args> {
  let EnumName = Ename;
}

// Magic nodes...
def Void       : RegisterClass<isVoid,0,[]> { let isDummyClass = 1; }
def set        : DagNode<DNVT_void, [DNVT_val, DNVT_arg0]>;
def chain      : BuiltinDagNode<DNVT_void, [DNVT_void, DNVT_void], "ChainNode">;
def blockchain : BuiltinDagNode<DNVT_void, [DNVT_void, DNVT_void],
                                "BlockChainNode">;
def ChainExpander      : Expander<(chain Void, Void), []>;
def BlockChainExpander : Expander<(blockchain Void, Void), []>;


// Terminals...
def imm        : BuiltinDagNode<DNVT_val, [], "Constant">;
def frameidx   : BuiltinDagNode<DNVT_ptr, [], "FrameIndex">;
def basicblock : BuiltinDagNode<DNVT_ptr, [], "BasicBlock">;

// Arithmetic...
def plus    : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "Plus">;
def minus   : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "Minus">;
def times   : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "Times">;
def sdiv    : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "SDiv">;
def udiv    : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "UDiv">;
def srem    : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "SRem">;
def urem    : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "URem">;
def and     : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "And">;
def or      : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "Or">;
def xor     : BuiltinDagNode<DNVT_arg0, [DNVT_arg1, DNVT_arg0], "Xor">;

// Comparisons...
def seteq   : BuiltinDagNode<DNVT_i8  , [DNVT_arg1, DNVT_arg0], "SetEQ">;
def setne   : BuiltinDagNode<DNVT_i8  , [DNVT_arg1, DNVT_arg0], "SetNE">;
def setlt   : BuiltinDagNode<DNVT_i8  , [DNVT_arg1, DNVT_arg0], "SetLT">;
def setle   : BuiltinDagNode<DNVT_i8  , [DNVT_arg1, DNVT_arg0], "SetLE">;
def setgt   : BuiltinDagNode<DNVT_i8  , [DNVT_arg1, DNVT_arg0], "SetGT">;
def setge   : BuiltinDagNode<DNVT_i8  , [DNVT_arg1, DNVT_arg0], "SetGE">;

def load    : BuiltinDagNode<DNVT_val, [DNVT_ptr], "Load">;
//def store   : BuiltinDagNode<DNVT_Void, [DNVT_ptr, DNVT_val]>;

// Other...
def ret     : BuiltinDagNode<DNVT_void, [DNVT_val], "Ret">;
def retvoid : BuiltinDagNode<DNVT_void, [], "RetVoid">;
def br      : BuiltinDagNode<DNVT_void, [DNVT_ptr], "Br">;
def brcond  : BuiltinDagNode<DNVT_void, [DNVT_i8, DNVT_ptr, DNVT_ptr],
                             "BrCond">;

def unspec1 : BuiltinDagNode<DNVT_any , [DNVT_val], "Unspec1">;
def unspec2 : BuiltinDagNode<DNVT_any , [DNVT_val, DNVT_val], "Unspec2">;

//===----------------------------------------------------------------------===//
// DAG nonterminals definitions used by the instruction selector...
//
class Nonterminal<dag pattern> {
  dag Pattern = pattern;
  bit BuiltIn = 0;
}

*/
