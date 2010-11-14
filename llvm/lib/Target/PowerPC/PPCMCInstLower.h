//===-- PPCMCInstLower.h - Lower MachineInstr to MCInst -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_MCINSTLOWER_H
#define PPC_MCINSTLOWER_H

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
  
/// PPCMCInstLower - This class is used to lower an MachineInstr into an MCInst.
class LLVM_LIBRARY_VISIBILITY PPCMCInstLower {
  MCContext &Ctx;
  Mangler &Mang;
  AsmPrinter &Printer;
public:
  PPCMCInstLower(MCContext &ctx, Mangler &mang, AsmPrinter &printer)
  : Ctx(ctx), Mang(mang), Printer(printer) {}
  
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;
  
private:
  MCSymbol *GetGlobalAddressSymbol(const GlobalValue *GV) const;
  const MCSymbolRefExpr *GetSymbolRef(const MachineOperand &MO) const;
  const MCSymbolRefExpr *GetExternalSymbolSymbol(const MachineOperand &MO)
  const;
  MCSymbol *GetJumpTableSymbol(const MachineOperand &MO) const;
  MCSymbol *GetConstantPoolIndexSymbol(const MachineOperand &MO) const;
  MCOperand LowerSymbolRefOperand(const MachineOperand &MO,
                                  const MCSymbolRefExpr *Expr) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;
  
};
} // end namespace llvm

#endif
