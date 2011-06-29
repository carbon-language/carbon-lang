//===-- llvm/MC/MCInstrInfo.h - Target Instruction Info ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the target machine instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTRINFO_H
#define LLVM_MC_MCINSTRINFO_H

#include "llvm/MC/MCInstrDesc.h"
#include <cassert>

namespace llvm {

//---------------------------------------------------------------------------
///
/// MCInstrInfo - Interface to description of machine instruction set
///
class MCInstrInfo {
  const MCInstrDesc *Desc;  // Raw array to allow static init'n
  unsigned NumOpcodes;      // Number of entries in the desc array

public:
  /// InitMCInstrInfo - Initialize MCInstrInfo, called by TableGen
  /// auto-generated routines. *DO NOT USE*.
  void InitMCInstrInfo(const MCInstrDesc *D, unsigned NO) {
    Desc = D;
    NumOpcodes = NO;
  }

  unsigned getNumOpcodes() const { return NumOpcodes; }

  /// get - Return the machine instruction descriptor that corresponds to the
  /// specified instruction opcode.
  ///
  const MCInstrDesc &get(unsigned Opcode) const {
    assert(Opcode < NumOpcodes && "Invalid opcode!");
    return Desc[Opcode];
  }
};

} // End llvm namespace

#endif
