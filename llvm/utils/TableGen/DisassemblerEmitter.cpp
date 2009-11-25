//===- DisassemblerEmitter.cpp - Generate a disassembler ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DisassemblerEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
using namespace llvm;

void DisassemblerEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target;

  OS << "/*===- TableGen'erated file "
     << "---------------------------------------*- C -*-===*\n"
     << " *\n"
     << " * " << Target.getName() << " Disassembler\n"
     << " *\n"
     << " * Automatically generated file, do not edit!\n"
     << " *\n"
     << " *===---------------------------------------------------------------"
     << "-------===*/\n";

  throw TGError(Target.getTargetRecord()->getLoc(),
                "Unable to generate disassembler for this target");
}
