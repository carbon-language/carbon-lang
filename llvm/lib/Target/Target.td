//===- Target.td - Target Independent TableGen interface --------*- C++ -*-===//
//
// This file defines the target-independent interfaces which should be
// implemented by each target which is using a TableGen based code generator.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//
// Value types - These values correspond to the register types defined in the
// ValueTypes.h file.
//
class ValueType<int size> { string Namespace = "MVT"; int Size = size; }

def i1   : ValueType<1>;      // One bit boolean value
def i8   : ValueType<8>;      // 8-bit integer value
def i16  : ValueType<16>;     // 16-bit integer value
def i32  : ValueType<32>;     // 32-bit integer value
def i64  : ValueType<64>;     // 64-bit integer value
def i128 : ValueType<128>;    // 128-bit integer value
def f32  : ValueType<32>;     // 32-bit floating point value
def f64  : ValueType<64>;     // 64-bit floating point value
def f80  : ValueType<80>;     // 80-bit floating point value
def f128 : ValueType<128>;    // 128-bit floating point value


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
def set;    // FIXME: these are subject to substantial change
def plus;
def minus;
def mult;
def div;
def udiv;
def mod;
def umod;
def imm8;
def imm16;
def imm32; 
