//===-- SystemZMCInstLower.h - Lower MachineInstr to MCInst ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEMZMCINSTLOWER_H
#define LLVM_SYSTEMZMCINSTLOWER_H

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCContext;
class MCInst;
class MCOperand;
class MCSymbol;
class MachineInstr;
class MachineOperand;
class Mangler;
class SystemZAsmPrinter;

class LLVM_LIBRARY_VISIBILITY SystemZMCInstLower {
  Mangler *Mang;
  MCContext &Ctx;
  SystemZAsmPrinter &AsmPrinter;

public:
  SystemZMCInstLower(Mangler *mang, MCContext &ctx,
                     SystemZAsmPrinter &asmPrinter);

  // Lower MachineInstr MI to MCInst OutMI.
  void lower(const MachineInstr *MI, MCInst &OutMI) const;

  // Return an MCOperand for MO.  Return an empty operand if MO is implicit.
  MCOperand lowerOperand(const MachineOperand& MO) const;

  // Return an MCOperand for MO, given that it equals Symbol + Offset.
  MCOperand lowerSymbolOperand(const MachineOperand &MO,
                               const MCSymbol *Symbol, int64_t Offset) const;
};
} // end namespace llvm

#endif
