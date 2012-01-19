//===-- CodeGen/MachineInstBundle.h - MI bundle utilities -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provide utility functions to manipulate machine instruction
// bundles.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUNDLE_H
#define LLVM_CODEGEN_MACHINEINSTRBUNDLE_H

#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

/// finalizeBundle - Finalize a machine instruction bundle which includes
/// a sequence of instructions starting from FirstMI to LastMI (exclusive).
/// This routine adds a BUNDLE instruction to represent the bundle, it adds
/// IsInternalRead markers to MachineOperands which are defined inside the
/// bundle, and it copies externally visible defs and uses to the BUNDLE
/// instruction.
void finalizeBundle(MachineBasicBlock &MBB,
                    MachineBasicBlock::instr_iterator FirstMI,
                    MachineBasicBlock::instr_iterator LastMI);
  
/// finalizeBundle - Same functionality as the previous finalizeBundle except
/// the last instruction in the bundle is not provided as an input. This is
/// used in cases where bundles are pre-determined by marking instructions
/// with 'InsideBundle' marker.
void finalizeBundle(MachineBasicBlock &MBB,
                    MachineBasicBlock::instr_iterator FirstMI);

} // End llvm namespace

#endif
