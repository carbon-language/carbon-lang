//===-- X86InstrBuilder.h - Functions to aid building x86 insts -*- C++ -*-===//
//
// This file exposes functions that may be used with BuildMI from the
// MachineInstrBuilder.h file to handle X86'isms in a clean way.
//
// The BuildMem function may be used with the BuildMI function to add entire
// memory references in a single, typed, function call.  X86 memory references
// can be very complex expressions (described in the README), so wrapping them
// up behind an easier to use interface makes sense.  Descriptions of the
// functions are included below.
//
// For reference, the order of operands for memory references is:
// (Operand), Base, Scale, Index, Displacement.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRBUILDER_H
#define X86INSTRBUILDER_H

#include "llvm/CodeGen/MachineInstrBuilder.h"

/// addDirectMem - This function is used to add a direct memory reference to the
/// current instruction -- that is, a dereference of an address in a register,
/// with no scale, index or displacement. An example is: DWORD PTR [EAX].
///
inline const MachineInstrBuilder &addDirectMem(const MachineInstrBuilder &MIB,
                                               unsigned Reg) {
  // Because memory references are always represented with four
  // values, this adds: Reg, [1, NoReg, 0] to the instruction.
  return MIB.addReg(Reg).addZImm(1).addReg(0).addSImm(0);
}


/// addRegOffset - This function is used to add a memory reference of the form
/// [Reg + Offset], i.e., one with no scale or index, but with a
/// displacement. An example is: DWORD PTR [EAX + 4].
///
inline const MachineInstrBuilder &addRegOffset(const MachineInstrBuilder &MIB,
                                               unsigned Reg, int Offset) {
  return MIB.addReg(Reg).addZImm(1).addReg(0).addSImm(Offset);
}

/// addFrameReference - This function is used to add a reference to the base of
/// an abstract object on the stack frame of the current function.  This
/// reference has base register as the FrameIndex offset until it is resolved.
/// This allows a constant offset to be specified as well...
///
inline const MachineInstrBuilder &
addFrameReference(const MachineInstrBuilder &MIB, int FI, int Offset = 0) {
  return MIB.addFrameIndex(FI).addZImm(1).addReg(0).addSImm(Offset);
}

/// addConstantPoolReference - This function is used to add a reference to the
/// base of a constant value spilled to the per-function constant pool.  The
/// reference has base register ConstantPoolIndex offset which is retained until
/// either machine code emission or assembly output.  This allows an optional
/// offset to be added as well.
///
inline const MachineInstrBuilder &
addConstantPoolReference(const MachineInstrBuilder &MIB, unsigned CPI,
			 int Offset = 0) {
  return MIB.addConstantPoolIndex(CPI).addZImm(1).addReg(0).addSImm(Offset);
}

#endif
