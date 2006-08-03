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

#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include <iostream>

using namespace llvm;

//===---------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterRegAlloc::Registry;


//===---------------------------------------------------------------------===//
///
/// RegAlloc command line options.
///
//===---------------------------------------------------------------------===//
namespace {
  cl::opt<RegisterRegAlloc::FunctionPassCtor, false,
          RegisterPassParser<RegisterRegAlloc> >
  RegAlloc("regalloc",
           cl::init(&createLinearScanRegisterAllocator),
           cl::desc("Register allocator to use: (default = linearscan)")); 
}


//===---------------------------------------------------------------------===//
///
/// createRegisterAllocator - choose the appropriate register allocator.
///
//===---------------------------------------------------------------------===//
FunctionPass *llvm::createRegisterAllocator() {
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();
  
  if (!Ctor) {
    Ctor = RegAlloc;
    RegisterRegAlloc::setDefault(RegAlloc);
  }
  
  return Ctor();
}
