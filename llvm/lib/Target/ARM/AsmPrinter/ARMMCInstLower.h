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
  class MCContext;
  class MCInst;
  class MCOperand;
  class MCSymbol;
  class MachineInstr;
  class MachineModuleInfoMachO;
  class MachineOperand;
  class Mangler;
  //class ARMSubtarget;
  
/// ARMMCInstLower - This class is used to lower an MachineInstr into an MCInst.
class VISIBILITY_HIDDEN ARMMCInstLower {
  MCContext &Ctx;
  Mangler *Mang;

  //const ARMSubtarget &getSubtarget() const;
public:
  ARMMCInstLower(MCContext &ctx, Mangler *mang)
    : Ctx(ctx), Mang(mang) {}
  
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

/*
 MCSymbol *GetPICBaseSymbol() const;
  
  MCSymbol *GetGlobalAddressSymbol(const MachineOperand &MO) const;
  MCSymbol *GetExternalSymbolSymbol(const MachineOperand &MO) const;
  MCSymbol *GetJumpTableSymbol(const MachineOperand &MO) const;
  MCSymbol *GetConstantPoolIndexSymbol(const MachineOperand &MO) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;
  
private:
  MachineModuleInfoMachO &getMachOMMI() const;
 */
};

}

#endif
