//===-- MBlazeMCInstLower.h - Lower MachineInstr to MCInst ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZE_MCINSTLOWER_H
#define MBLAZE_MCINSTLOWER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
  class AsmPrinter;
  class MCAsmInfo;
  class MCContext;
  class MCInst;
  class MCOperand;
  class MCSymbol;
  class MachineInstr;
  class MachineModuleInfoMachO;
  class MachineOperand;
  class Mangler;

  /// MBlazeMCInstLower - This class is used to lower an MachineInstr
  /// into an MCInst.
class LLVM_LIBRARY_VISIBILITY MBlazeMCInstLower {
  MCContext &Ctx;
  Mangler &Mang;

  AsmPrinter &Printer;
public:
  MBlazeMCInstLower(MCContext &ctx, Mangler &mang, AsmPrinter &printer)
    : Ctx(ctx), Mang(mang), Printer(printer) {}
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;

  MCSymbol *GetGlobalAddressSymbol(const MachineOperand &MO) const;
  MCSymbol *GetExternalSymbolSymbol(const MachineOperand &MO) const;
  MCSymbol *GetJumpTableSymbol(const MachineOperand &MO) const;
  MCSymbol *GetConstantPoolIndexSymbol(const MachineOperand &MO) const;
  MCSymbol *GetBlockAddressSymbol(const MachineOperand &MO) const;
};

}

#endif
