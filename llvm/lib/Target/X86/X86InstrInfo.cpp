//===- X86InstructionInfo.cpp - X86 Instruction Information ---------------===//
//
// This file contains the X86 implementation of the MInstructionInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstructionInfo.h"
#include "llvm/CodeGen/MInstruction.h"
#include <ostream>

// X86Insts - Turn the InstructionInfo.def file into a bunch of instruction
// descriptors
//
static const MInstructionDesc X86Insts[] = {
#define I(ENUM, NAME, FLAGS, TSFLAGS) { NAME, FLAGS, TSFLAGS },
#include "X86InstructionInfo.def"
};

X86InstructionInfo::X86InstructionInfo()
  : MInstructionInfo(X86Insts, sizeof(X86Insts)/sizeof(X86Insts[0])) {
}


// print - Print out an x86 instruction in GAS syntax
void X86InstructionInfo::print(const MInstruction *MI, std::ostream &O) const {
  // FIXME: This sucks.
  O << get(MI->getOpcode()).Name << "\n";
}

