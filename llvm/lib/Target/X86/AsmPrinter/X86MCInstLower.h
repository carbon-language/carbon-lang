//===-- X86MCInstLower.h - Lower MachineInstr to MCInst -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_MCINSTLOWER_H
#define X86_MCINSTLOWER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
  class MCContext;
  class MCInst;
  class MCOperand;
  class MCSymbol;
  class MachineInstr;
  class MachineModuleInfoMachO;
  class MachineOperand;
  class Mangler;
  class X86AsmPrinter;
  class X86Subtarget;
  
/// X86MCInstLower - This class is used to lower an MachineInstr into an MCInst.
class VISIBILITY_HIDDEN X86MCInstLower {
  MCContext &Ctx;
  Mangler &Mang;
  X86AsmPrinter &AsmPrinter;

  const X86Subtarget &getSubtarget() const;
public:
  X86MCInstLower(MCContext &ctx, Mangler &mang, X86AsmPrinter &asmprinter)
    : Ctx(ctx), Mang(mang), AsmPrinter(asmprinter) {}
  
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCSymbol *GetPICBaseSymbol() const;
  
  MCSymbol *GetGlobalAddressSymbol(const MachineOperand &MO) const;
  MCSymbol *GetExternalSymbolSymbol(const MachineOperand &MO) const;
  MCSymbol *GetJumpTableSymbol(const MachineOperand &MO) const;
  MCSymbol *GetConstantPoolIndexSymbol(const MachineOperand &MO) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;
  
private:
  MachineModuleInfoMachO &getMachOMMI() const;
};

}

#endif
