//===-- Passes.cpp - Target independent code generation passes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Passes.h"

using namespace llvm;

//===---------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterRegAlloc::Registry;

static FunctionPass *createDefaultRegisterAllocator() { return 0; }
static RegisterRegAlloc
defaultRegAlloc("default",
                "pick register allocator based on -O option",
                createDefaultRegisterAllocator);

//===---------------------------------------------------------------------===//
///
/// RegAlloc command line options.
///
//===---------------------------------------------------------------------===//
static cl::opt<RegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RegisterRegAlloc> >
RegAlloc("regalloc",
         cl::init(&createDefaultRegisterAllocator),
         cl::desc("Register allocator to use"));


//===---------------------------------------------------------------------===//
///
/// createRegisterAllocator - choose the appropriate register allocator.
///
//===---------------------------------------------------------------------===//
FunctionPass *llvm::createRegisterAllocator(CodeGenOpt::Level OptLevel) {
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = RegAlloc;
    RegisterRegAlloc::setDefault(RegAlloc);
  }

  if (Ctor != createDefaultRegisterAllocator)
    return Ctor();

  // When the 'default' allocator is requested, pick one based on OptLevel.
  switch (OptLevel) {
  case CodeGenOpt::None:
    return createFastRegisterAllocator();
  default:
    return createGreedyRegisterAllocator();
  }
}
