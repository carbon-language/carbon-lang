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

class ValueType { string Namespace = "MVT"; }

def i1   : ValueType;    // One bit boolean value
def i8   : ValueType;    // 8-bit integer value
def i16  : ValueType;    // 16-bit integer value
def i32  : ValueType;    // 32-bit integer value
def i64  : ValueType;    // 64-bit integer value
def i128 : ValueType;    // 128-bit integer value
def f32  : ValueType;    // 32-bit floating point value
def f64  : ValueType;    // 64-bit floating point value
def f80  : ValueType;    // 80-bit floating point value
def f128 : ValueType;    // 128-bit floating point value


//===----------------------------------------------------------------------===//
// Register file description - These classes are used to fill in the target
// description classes in llvm/Target/MRegisterInfo.h


// Register - You should define one instance of this class for each register in
// the target machine.
//
class Register {
  string Namespace = "";
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
  ValueType RegType = regType;
  int Alignment = alignment;
  list<Register> MemberList = regList;
}


//===----------------------------------------------------------------------===//
// Instruction set description - 
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
}
