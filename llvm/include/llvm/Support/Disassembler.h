//===- llvm/Support/Disassembler.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Anton Korobeynikov and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the necessary glue to call external disassembler
// libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DISASSEMBLER_H
#define LLVM_SUPPORT_DISASSEMBLER_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {

namespace Disassembler {
  enum Type {
    X86_32,
    X86_64,
    Undefined
  };
}
  

std::string disassembleBuffer(uint8_t* start, size_t length,
                              Disassembler::Type type, uint64_t pc);
}

#endif // LLVM_SUPPORT_DISASSEMBLER_H
