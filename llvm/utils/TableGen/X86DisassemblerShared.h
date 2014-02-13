//===- X86DisassemblerShared.h - Emitter shared header ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86DISASSEMBLERSHARED_H
#define X86DISASSEMBLERSHARED_H

#include <string.h>
#include <string>

#define INSTRUCTION_SPECIFIER_FIELDS       \
  struct OperandSpecifier operands[X86_MAX_OPERANDS]; \
  InstructionContext      insnContext;     \
  std::string             name;            \
                                           \
  InstructionSpecifier() {                 \
    insnContext = IC;                      \
    name = "";                             \
    memset(operands, 0, sizeof(operands)); \
  }

#define INSTRUCTION_IDS           \
  InstrUID   instructionIDs[256];

#include "../../lib/Target/X86/Disassembler/X86DisassemblerDecoderCommon.h"

#undef INSTRUCTION_SPECIFIER_FIELDS
#undef INSTRUCTION_IDS

#endif
