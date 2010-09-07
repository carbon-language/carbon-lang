//===-- PTXAsmPrinter.cpp - PTX LLVM assembly writer ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PTX assembly language.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

namespace {
  class PTXAsmPrinter : public AsmPrinter {
    public:
      explicit PTXAsmPrinter(TargetMachine &TM, MCStreamer &Streamer) :
        AsmPrinter(TM, Streamer) {}
      const char *getPassName() const { return "PTX Assembly Printer"; }
  };
} // namespace

// Force static initialization.
extern "C" void LLVMInitializePTXAsmPrinter()
{
  RegisterAsmPrinter<PTXAsmPrinter> X(ThePTXTarget);
}
