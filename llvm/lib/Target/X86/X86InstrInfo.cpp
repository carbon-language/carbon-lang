//===- X86InstructionInfo.cpp - X86 Instruction Information ---------------===//
//
// This file contains the X86 implementation of the MInstructionInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstructionInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include <iostream>

// X86Insts - Turn the InstructionInfo.def file into a bunch of instruction
// descriptors
//
static const MachineInstrDescriptor X86Insts[] = {
#define I(ENUM, NAME, FLAGS, TSFLAGS) \
             { NAME, -1, -1, 0, false, 0, 0, TSFLAGS, FLAGS },
#include "X86InstructionInfo.def"
};

X86InstructionInfo::X86InstructionInfo()
  : MachineInstrInfo(X86Insts, sizeof(X86Insts)/sizeof(X86Insts[0]), 0) {
}


// print - Print out an x86 instruction in GAS syntax
void X86InstructionInfo::print(const MachineInstr *MI, std::ostream &O) const {
  // FIXME: This sucks.
  O << getName(MI->getOpCode()) << "\n";
}

