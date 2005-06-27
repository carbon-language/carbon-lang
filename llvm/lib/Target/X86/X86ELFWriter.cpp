//===-- X86ELFWriter.cpp - Emit an ELF file for the X86 backend -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an ELF writer for the X86 backend.  The public interface
// to this file is the createX86ELFObjectWriterPass function.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/CodeGen/ELFWriter.h"
using namespace llvm;

namespace {
  class X86ELFWriter : public ELFWriter {
  public:
    X86ELFWriter(std::ostream &O, TargetMachine &TM) : ELFWriter(O, TM) {
      e_machine = 3;   // EM_386
    }
  };
}

/// createX86ELFObjectWriterPass - Returns a pass that outputs the generated
/// code as an ELF object file.
///
FunctionPass *llvm::createX86ELFObjectWriterPass(std::ostream &O,
                                                 TargetMachine &TM) {
  return new X86ELFWriter(O, TM);
}
