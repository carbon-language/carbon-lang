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

FunctionPass *llvm::createRegisterAllocator() {
  switch (RegAlloc) {
  default:
    std::cerr << "no register allocator selected";
    abort();
  case simple:
    return createSimpleRegisterAllocator();
  case local:
    return createLocalRegisterAllocator();
  case linearscan:
    return createLinearScanRegisterAllocator();
  }
}

