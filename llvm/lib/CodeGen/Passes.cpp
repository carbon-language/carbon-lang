//===-- Passes.cpp - Target independent code generation passes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
using namespace llvm;

namespace {
  enum RegAllocName { simple, local, linearscan };

  static cl::opt<RegAllocName>
  RegAlloc(
    "regalloc",
    cl::desc("Register allocator to use: (default = linearscan)"),
    cl::Prefix,
    cl::values(
       clEnumVal(simple,        "  simple register allocator"),
       clEnumVal(local,         "  local register allocator"),
       clEnumVal(linearscan,    "  linear scan register allocator"),
       clEnumValEnd),
    cl::init(linearscan));
}


RegisterRegAlloc *RegisterRegAlloc::List = NULL;

/// Find - Finds a register allocator in registration list.
///
RegisterRegAlloc::FunctionPassCtor RegisterRegAlloc::Find(const char *N) {
  for (RegisterRegAlloc *RA = List; RA; RA = RA->Next) {
    if (strcmp(N, RA->Name) == 0) return RA->Ctor;
  }
  return NULL;
}


#ifndef NDEBUG  
void RegisterRegAlloc::print() {
  for (RegisterRegAlloc *RA = List; RA; RA = RA->Next) {
    std::cerr << "RegAlloc:" << RA->Name << "\n";
  }
}
#endif


static RegisterRegAlloc
  simpleRegAlloc("simple", "  simple register allocator",
                 createSimpleRegisterAllocator);

static RegisterRegAlloc
  localRegAlloc("local", "  local register allocator",
                createLocalRegisterAllocator);

static RegisterRegAlloc
  linearscanRegAlloc("linearscan", "linear scan register allocator",
                     createLinearScanRegisterAllocator);


FunctionPass *llvm::createRegisterAllocator() {
  const char *Names[] = {"simple", "local", "linearscan"};
  const char *DefltName = "linearscan";
  
  RegisterRegAlloc::FunctionPassCtor Ctor =
                    RegisterRegAlloc::Find(Names[RegAlloc]);
  if (!Ctor) Ctor = RegisterRegAlloc::Find(DefltName);

  assert(Ctor && "No register allocator found");
  
  return Ctor();
}


