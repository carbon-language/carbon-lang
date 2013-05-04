//===- lib/Linker/Linker.cpp - Basic Linker functionality  ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains basic Linker functionality that all usages will need.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

Linker::Linker(StringRef progname, StringRef modname,
               LLVMContext& C, unsigned flags):
  Context(C),
  Composite(new Module(modname, C)),
  Flags(flags),
  Error(),
  ProgramName(progname) { }

Linker::Linker(StringRef progname, Module* aModule, unsigned flags) :
  Context(aModule->getContext()),
  Composite(aModule),
  Flags(flags),
  Error(),
  ProgramName(progname) { }

Linker::~Linker() {
  delete Composite;
}
