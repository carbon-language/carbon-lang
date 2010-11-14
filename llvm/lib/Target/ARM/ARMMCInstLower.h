//===-- ARMMCInstLower.h - Lower MachineInstr to MCInst -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ARM_MCINSTLOWER_H
#define ARM_MCINSTLOWER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
  class AsmPrinter;
  class GlobalValue;
  class MCAsmInfo;
  class MCContext;
  class MCInst;
  class MCOperand;
  class MCSymbol;
  class MCSymbolRefExpr;
  class MachineInstr;
  class MachineModuleInfoMachO;
  class MachineOperand;
  class Mangler;
  //class ARMSubtarget;

/// ARMMCInstLower - This class is used to lower an MachineInstr into an MCInst.
class LLVM_LIBRARY_VISIBILITY ARMMCInstLower {
  MCContext &Ctx;
  Mangler &Mang;
  AsmPrinter &Printer;
public:
  ARMMCInstLower(MCContext &ctx, Mangler &mang, AsmPrinter &printer)
    : Ctx(ctx), Mang(mang), Printer(printer) {}

  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

private:
  MCSymbol *GetGlobalAddressSymbol(const GlobalValue *GV) const;
  MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Sym) const;
};

} // end namespace llvm

#endif
