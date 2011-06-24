//===-- X86TargetDesc.h - X86 Target Descriptions ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides X86 specific target descriptions.
//
//===----------------------------------------------------------------------===//

namespace llvm {
class Target;

extern Target TheX86_32Target, TheX86_64Target;
} // End llvm namespace

// Defines symbolic names for X86 registers.  This defines a mapping from
// register name to register number.
//
#include "X86GenRegisterNames.inc"
