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

#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>

using namespace llvm;

namespace {
  cl::opt<const char *, false, RegisterPassParser<RegisterRegAlloc> >
  RegAlloc("regalloc",
           cl::init("linearscan"),
           cl::desc("Register allocator to use: (default = linearscan)")); 
}

FunctionPass *llvm::createRegisterAllocator() {
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getCache();
  
  if (!Ctor) {
    Ctor = RegisterRegAlloc::FindCtor(RegAlloc);
    assert(Ctor && "No register allocator found");
    if (!Ctor) Ctor = RegisterRegAlloc::FirstCtor();
    RegisterRegAlloc::setCache(Ctor);
  }
  
  assert(Ctor && "No register allocator found");
  
  return Ctor();
}
