//===-EDMain.cpp - LLVM Enhanced Disassembly C API ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the enhanced disassembler's public C API.
//
//===----------------------------------------------------------------------===//

#include <llvm-c/EnhancedDisassembly.h>

int EDGetDisassembler(EDDisassemblerRef *disassembler,
                      const char *triple,
                      EDAssemblySyntax_t syntax) {
  return -1;
}
