//===-- Passes.cpp - Target independent code generation passes -*- C++ -*-===//
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
#include "Support/CommandLine.h"

namespace llvm {

namespace {
  enum RegAllocName { simple, local };

  cl::opt<RegAllocName>
  RegAlloc("regalloc",
           cl::desc("Register allocator to use: (default = simple)"),
           cl::Prefix,
           cl::values(clEnumVal(simple, "  simple register allocator"),
                      clEnumVal(local,  "  local register allocator"),
                      0),
           cl::init(local));
}

FunctionPass *createRegisterAllocator()
{
  switch (RegAlloc) {
  case simple:
    return createSimpleRegisterAllocator();
  case local:
    return createLocalRegisterAllocator();
  default:
    assert(0 && "no register allocator selected");
    return 0; // not reached
  }
}

} // End llvm namespace
