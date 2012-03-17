//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// PTX back-end.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_H
#define PTX_H

#include "MCTargetDesc/PTXBaseInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class MachineInstr;
  class MCInst;
  class PTXAsmPrinter;
  class PTXTargetMachine;
  class FunctionPass;

  FunctionPass *createPTXISelDag(PTXTargetMachine &TM,
                                 CodeGenOpt::Level OptLevel);

  FunctionPass *createPTXMFInfoExtract(PTXTargetMachine &TM,
                                       CodeGenOpt::Level OptLevel);

  FunctionPass *createPTXFPRoundingModePass(PTXTargetMachine &TM,
                                            CodeGenOpt::Level OptLevel);

  FunctionPass *createPTXRegisterAllocator();

  void LowerPTXMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                    PTXAsmPrinter &AP);

} // namespace llvm;

#endif // PTX_H
