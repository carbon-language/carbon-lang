//===-- MSP430MCInstLower.h - Lower MachineInstr to MCInst ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MSP430_MCINSTLOWER_H
#define MSP430_MCINSTLOWER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
  class MCAsmInfo;
  class MCContext;
  class MCInst;
  class MCOperand;
  class MCSymbol;
  class MachineInstr;
  class MachineModuleInfoMachO;
  class MachineOperand;
  class Mangler;

  /// MSP430MCInstLower - This class is used to lower an MachineInstr
  /// into an MCInst.
class VISIBILITY_HIDDEN MSP430MCInstLower {
  MCContext &Ctx;
  Mangler *Mang;

  #if 0
  const unsigned CurFunctionNumber;
  const MCAsmInfo &MAI;
  #endif

public:
  MSP430MCInstLower(MCContext &ctx, Mangler *mang) : Ctx(ctx), Mang(mang) {}

  void Lower(const MachineInstr *MI, MCInst &OutMI) const;
};

}

#endif
