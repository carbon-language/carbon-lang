//===- MIRPrinter.h - MIR serialization format printer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the functions that print out the LLVM IR and the machine
// functions using the MIR serialization format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MIRPRINTER_H
#define LLVM_CODEGEN_MIRPRINTER_H

namespace llvm {

class MachineBasicBlock;
class MachineFunction;
class Module;
class raw_ostream;
template <typename T> class SmallVectorImpl;

/// Print LLVM IR using the MIR serialization format to the given output stream.
void printMIR(raw_ostream &OS, const Module &M);

/// Print a machine function using the MIR serialization format to the given
/// output stream.
void printMIR(raw_ostream &OS, const MachineFunction &MF);

/// Determine a possible list of successors of a basic block based on the
/// basic block machine operand being used inside the block. This should give
/// you the correct list of successor blocks in most cases except for things
/// like jump tables where the basic block references can't easily be found.
/// The MIRPRinter will skip printing successors if they match the result of
/// this funciton and the parser will use this function to construct a list if
/// it is missing.
void guessSuccessors(const MachineBasicBlock &MBB,
                     SmallVectorImpl<MachineBasicBlock*> &Result,
                     bool &IsFallthrough);

} // end namespace llvm

#endif
