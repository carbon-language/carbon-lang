//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
// vim:ft=cpp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Target-Independent interface
//===----------------------------------------------------------------------===//

class Register {
  string Namespace = "";
  int Size;
}

class Instruction {
  string Name;          // The opcode string for this instruction
  string Namespace = "";

  list<Register> Uses = [];  // Default to using no non-operand registers
  list<Register> Defs = [];  // Default to modifying no non-operand registers

  // These bits capture information about the high-level semantics of the
  // instruction.
  bit isReturn     = 0;      // Is this instruction a return instruction?
  bit isBranch     = 0;      // Is this instruction a branch instruction?
  bit isCall       = 0;      // Is this instruction a call instruction?
}
