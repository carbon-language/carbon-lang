//===- Target.td - Target Independent TableGen interface --------*- C++ -*-===//
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

def i1     : ValueType<1  ,  1>;   // One bit boolean value
def i8     : ValueType<8  ,  2>;   // 8-bit integer value
def i16    : ValueType<16 ,  3>;   // 16-bit integer value
def i32    : ValueType<32 ,  4>;   // 32-bit integer value
def i64    : ValueType<64 ,  5>;   // 64-bit integer value
def i128   : ValueType<128,  5>;   // 128-bit integer value
def f32    : ValueType<32 ,  7>;   // 32-bit floating point value
def f64    : ValueType<64 ,  8>;   // 64-bit floating point value
def f80    : ValueType<80 ,  9>;   // 80-bit floating point value
def f128   : ValueType<128,  9>;   // 128-bit floating point value
def isVoid : ValueType<0  , 11>;   // Produces no value

//===----------------------------------------------------------------------===//
// Register file description - These classes are used to fill in the target
// description classes in llvm/Target/MRegisterInfo.h


// Register - You should define one instance of this class for each register in
// the target machine.
//
class Register {
  string Namespace = "";
  string Name = "";
}

// NamedReg - If the name for the 'def' of the register should not become the
// "name" of the register, you can use this to specify a custom name instead.
//
class NamedReg<string n> : Register {
  let Name = n;
}

// RegisterAliases - You should define instances of this class to indicate which
// registers in the register file are aliased together.  This allows the code
// generator to be careful not to put two values with overlapping live ranges
// into registers which alias.
//
class RegisterAliases<Register reg, list<Register> aliases> {
  Register Reg = reg;
  list<Register> Aliases = aliases;
}

// RegisterClass - Now that all of the registers are defined, and aliases
// between registers are defined, specify which registers belong to which
// register classes.  This also defines the default allocation order of
// registers by register allocators.
//
class RegisterClass<ValueType regType, int alignment, list<Register> regList> {
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

  // Methods - This member can be used to insert arbitrary code into a generated
  // register class.   The normal usage of this is to overload virtual methods.
  code Methods = [{}];
}


//===----------------------------------------------------------------------===//
// Instruction set description - These classes correspond to the C++ classes in
// the Target/TargetInstrInfo.h file.
//

class Instruction {
  string Name;          // The opcode string for this instruction
  string Namespace = "";

  list<Register> Uses = [];  // Default to using no non-operand registers
  list<Register> Defs = [];  // Default to modifying no non-operand registers

  // These bits capture information about the high-level semantics of the
  // instruction.
  bit isReturn     = 0;     // Is this instruction a return instruction?
  bit isBranch     = 0;     // Is this instruction a branch instruction?
  bit isCall       = 0;     // Is this instruction a call instruction?
  bit isTwoAddress = 0;     // Is this a two address instruction?
  bit isTerminator = 0;     // Is this part of the terminator for a basic block?

  // Pattern - Set to the DAG pattern for this instruction, if we know of one,
  // otherwise, uninitialized.
  dag Pattern;
}

class Expander<dag pattern, list<dag> result> {
  dag Pattern      = pattern;
  list<dag> Result = result;
}


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
}


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

  // InstructionSet - Instruction set description for this target
  InstrInfo InstructionSet;
}


//===----------------------------------------------------------------------===//
// DAG node definitions used by the instruction selector...
//
class DagNodeValType;
def DNVT_void  : DagNodeValType;  // Tree node always returns void
def DNVT_val   : DagNodeValType;  // A non-void type
def DNVT_arg0  : DagNodeValType;  // Tree node returns same type as Arg0
def DNVT_arg1  : DagNodeValType;  // Tree node returns same type as Arg1
def DNVT_ptr   : DagNodeValType;  // The target pointer type
def DNVT_bool  : DagNodeValType;  // Always returns bool

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
def set     : DagNode<DNVT_void, [DNVT_val, DNVT_arg0]>;

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
def seteq   : BuiltinDagNode<DNVT_bool, [DNVT_arg1, DNVT_arg0], "SetEQ">;
def setne   : BuiltinDagNode<DNVT_bool, [DNVT_arg1, DNVT_arg0], "SetNE">;
def setlt   : BuiltinDagNode<DNVT_bool, [DNVT_arg1, DNVT_arg0], "SetLT">;
def setle   : BuiltinDagNode<DNVT_bool, [DNVT_arg1, DNVT_arg0], "SetLE">;
def setgt   : BuiltinDagNode<DNVT_bool, [DNVT_arg1, DNVT_arg0], "SetGT">;
def setge   : BuiltinDagNode<DNVT_bool, [DNVT_arg1, DNVT_arg0], "SetGE">;

def load    : BuiltinDagNode<DNVT_val, [DNVT_ptr], "Load">;
//def store   : BuiltinDagNode<DNVT_Void, [DNVT_ptr, DNVT_val]>;

// Other...
def ret     : BuiltinDagNode<DNVT_void, [DNVT_val], "Ret">;
def retvoid : BuiltinDagNode<DNVT_void, [], "RetVoid">;
def br      : BuiltinDagNode<DNVT_void, [DNVT_ptr], "Br">;
def brcond  : BuiltinDagNode<DNVT_void, [DNVT_bool, DNVT_ptr, DNVT_ptr],
                             "BrCond">;

//===----------------------------------------------------------------------===//
// DAG nonterminals definitions used by the instruction selector...
//
class Nonterminal<dag pattern> {
  dag Pattern = pattern;
  bit BuiltIn = 0;
}

