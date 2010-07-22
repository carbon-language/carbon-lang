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
  class MCAsmInfo;
  class MCContext;
  class MCInst;
  class MCOperand;
  class MCSymbol;
  class MachineInstr;
  class MachineFunction;
  class MachineModuleInfoMachO;
  class MachineOperand;
  class Mangler;
  class TargetMachine;
  class X86AsmPrinter;
  
/// X86MCInstLower - This class is used to lower an MachineInstr into an MCInst.
class LLVM_LIBRARY_VISIBILITY X86MCInstLower {
  MCContext &Ctx;
  Mangler *Mang;
  const MachineFunction &MF;
  const TargetMachine &TM;
  const MCAsmInfo &MAI;
  X86AsmPrinter &AsmPrinter;
public:
  X86MCInstLower(Mangler *mang, const MachineFunction &MF,
                 X86AsmPrinter &asmprinter);
  
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCSymbol *GetPICBaseSymbol() const;
  
  MCSymbol *GetSymbolFromOperand(const MachineOperand &MO) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;
  
private:
  MachineModuleInfoMachO &getMachOMMI() const;
};

}

#endif
