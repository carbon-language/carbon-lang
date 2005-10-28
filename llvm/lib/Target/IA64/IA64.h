//===-- IA64.h - Top-level interface for IA64 representation ------*- C++ -*-===//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the IA64
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_IA64_H
#define TARGET_IA64_H

#include <iosfwd>

namespace llvm {

class TargetMachine;
class FunctionPass;
class IntrinsicLowering;

/// createIA64DAGToDAGInstructionSelector - This pass converts an LLVM
/// function into IA64 machine code in a sane, DAG->DAG transform.
///
FunctionPass *createIA64DAGToDAGInstructionSelector(TargetMachine &TM);

/// createIA64PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation in a more aggressive way.
///
FunctionPass *createIA64PatternInstructionSelector(TargetMachine &TM);

/// createIA64CodePrinterPass - Returns a pass that prints the IA64
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *createIA64CodePrinterPass(std::ostream &o,TargetMachine &tm);

} // End llvm namespace

// Defines symbolic names for IA64 registers.  This defines a mapping from
// register name to register number.
//
#include "IA64GenRegisterNames.inc"

// Defines symbolic names for the IA64 instructions.
//
#include "IA64GenInstrNames.inc"

#endif


