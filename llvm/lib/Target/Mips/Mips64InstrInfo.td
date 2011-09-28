//===- Mips64InstrInfo.td - Mips64 Instruction Information -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips64 instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips64 Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasMips64    : Predicate<"Subtarget.hasMips64()">;
def HasMips64r2  : Predicate<"Subtarget.hasMips64r2()">;

