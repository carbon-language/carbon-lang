//===-- llvm/CodeGen/MInstruction.h - Machine Instruction -------*- C++ -*-===//
//
// This class represents a single machine instruction for the LLVM backend.
// This instruction is represented in a completely generic way to allow all
// backends to share a common representation.  MInstructions are embedded into
// MBasicBlocks, and are maintained as a doubly linked list.
//
// Because there are a lot of machine instruction that may be in use at a time
// (being manipulated), we are sure to keep a very compact representation that
// is extremely light-weight.
//
// This class is used to represent an instruction when it is in SSA form as well
// as when it has been register allocated to use physical registers.
//
// FIXME: This should eventually be merged with the MachineInstr class.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_MINSTRUCTION_H
#define CODEGEN_MINSTRUCTION_H

#include <vector>
template<typename NodeTy> struct ilist_traits;
class MBasicBlock;

/// MOperand - This class represents a single operand in an instruction.
/// Interpretation of this operand is not really possible without information
/// from the machine instruction that it is embedded into.
///
class MOperand {
  union {
    unsigned uVal;
    int      iVal;
  };
public:
  MOperand(unsigned Value) : uVal(Value) {}
  MOperand(int Value) : iVal(Value) {}

  /// Interpretation - This enum is used by the MInstruction class to interpret
  /// the untyped value field of the operand.
  enum Interpretation {
    Register,               // This is some register number
    SignExtImmediate,       // This is a sign extended immediate
    ZeroExtImmediate,       // This is a zero extended immediate
    PCRelativeDisp          // This is a displacement relative to the PC
    // FIXME: We need a symbolic value here, like global variable address
  };

  unsigned getUnsignedValue() const { return uVal; }
  unsigned getSignedValue() const { return iVal; }
};

/// MInstruction - Represent a single machine instruction in the code generator.
/// This is meant to be a light weight representation that is completely
/// independant of the target machine being code generated for.
///
class MInstruction {
  MInstruction *Prev, *Next;      // Doubly linked list of instructions...

  unsigned Opcode;                // Opcode of the instruction
  unsigned Dest;                  // Destination register written (or 0 if none)

  std::vector<MOperand> Operands; // Operands of the instruction...

  /// OperandInterpretation - This array specifies how the operands of the
  /// instruction are to be interpreted (is it a register?, an immediate
  /// constant?, a PC relative displacement?, etc...).  Only four values are
  /// allowed, so any instruction with more than four operands (should be
  /// exceedingly rare, perhaps only PHI nodes) are assumed to have register
  /// operands beyond the fourth.
  ///
  unsigned char OperandInterpretation[4];
public:
  /// MInstruction ctor - Create a new machine instruction, with the specified
  /// opcode and destination register.  Operands are then added with the
  /// addOperand method.
  ///
  MInstruction(unsigned O = 0, unsigned D = 0) : Opcode(O), Dest(D) {}
  
  /// MInstruction ctor - Create a new instruction, and append it to the
  /// specified basic block.
  ///
  MInstruction(MBasicBlock *BB, unsigned Opcode = 0, unsigned Dest = 0);

  /// getOpcode - Return the opcode for this machine instruction.  The value of
  /// the opcode defines how to interpret the operands of the instruction.
  ///
  unsigned getOpcode() const { return Opcode; }

  /// getDestinationReg - This method returns the register written to by this
  /// instruction.  If this returns zero, the instruction does not produce a
  /// value, because register #0 is always the garbage marker.
  ///
  unsigned getDestinationReg() const { return Dest; }

  /// setDestinationReg - This method changes the register written to by this
  /// instruction.  Note that if SSA form is currently active then the SSA table
  /// needs to be updated to match this, thus this method shouldn't be used
  /// directly.
  ///
  void setDestinationReg(unsigned R) { Dest = R; }

  /// getNumOperands - Return the number of operands the instruction currently
  /// has.
  ///
  unsigned getNumOperands() const { return Operands.size(); }

  /// getOperandInterpretation - Return the interpretation of operand #Op
  ///
  MOperand::Interpretation getOperandInterpretation(unsigned Op) const {
    if (Op < 4) return (MOperand::Interpretation)OperandInterpretation[Op];
    return MOperand::Register;  // Operands >= 4 are all registers
  }

  unsigned getRegisterOperand(unsigned Op) const {
    assert(getOperandInterpretation(Op) == MOperand::Register &&
           "Operand isn't a register!");
    return Operands[Op].getUnsignedValue();
  }
  int getSignExtOperand(unsigned Op) const {
    assert(getOperandInterpretation(Op) == MOperand::SignExtImmediate &&
           "Operand isn't a sign extended immediate!");
    return Operands[Op].getSignedValue();
  }
  unsigned getZeroExtOperand(unsigned Op) const {
    assert(getOperandInterpretation(Op) == MOperand::ZeroExtImmediate &&
           "Operand isn't a zero extended immediate!");
    return Operands[Op].getUnsignedValue();
  }
  int getPCRelativeOperand(unsigned Op) const {
    assert(getOperandInterpretation(Op) == MOperand::PCRelativeDisp &&
           "Operand isn't a PC relative displacement!");
    return Operands[Op].getSignedValue();
  }

  /// addOperand - Add a new operand to the instruction with the specified value
  /// and interpretation.
  ///
  void addOperand(unsigned Value, MOperand::Interpretation Ty);

private:   // Methods used to maintain doubly linked list of instructions...
  friend class ilist_traits<MInstruction>;

  MInstruction *getPrev() const { return Prev; }
  MInstruction *getNext() const { return Next; }
  void setPrev(MInstruction *P) { Prev = P; }
  void setNext(MInstruction *N) { Next = N; }
};

#endif
