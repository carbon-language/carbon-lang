//===-- LTODisassembler.cpp - LTO Disassembler interface ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This function provides utility methods used by clients of libLTO that want
// to use the disassembler.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/lto.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

void lto_initialize_disassembler() {
  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();
}
