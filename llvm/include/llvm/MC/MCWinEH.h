//===- MCWinEH.h - Windows Unwinding Support --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWINEH_H
#define LLVM_MC_MCWINEH_H

namespace llvm {
class MCSymbol;

namespace WinEH {
struct Instruction {
  const MCSymbol *Label;
  const unsigned Offset;
  const unsigned Register;
  const unsigned Operation;

  Instruction(unsigned Op, MCSymbol *L, unsigned Reg, unsigned Off)
    : Label(L), Offset(Off), Register(Reg), Operation(Op) {}
};
}
}

#endif
