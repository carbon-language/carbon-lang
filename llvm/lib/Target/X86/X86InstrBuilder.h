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
//===----------------------------------------------------------------------===//

#ifndef X86INSTRBUILDER_H
#define X86INSTRBUILDER_H

#include "llvm/CodeGen/MachineInstrBuilder.h"

/// addDirectMem - This function is used to add a direct memory reference to the
/// current instruction.  Because memory references are always represented with
/// four values, this adds: Reg, [1, NoReg, 0] to the instruction
///
inline const MachineInstrBuilder &addDirectMem(const MachineInstrBuilder &MIB,
                                               unsigned Reg) {
  return MIB.addReg(Reg).addZImm(1).addMReg(0).addSImm(0);
}

#endif
