//===-- ARM.h - Top-level interface for ARM representation---- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// ARM back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ARM_H
#define TARGET_ARM_H

#include "ARMBaseInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>

namespace llvm {

class ARMBaseTargetMachine;
class FunctionPass;
class JITCodeEmitter;
class formatted_raw_ostream;
class MCCodeEmitter;
class TargetAsmBackend;
class MachineInstr;
class ARMAsmPrinter;
class MCInst;

MCCodeEmitter *createARMMCCodeEmitter(const Target &,
                                      TargetMachine &TM,
                                      MCContext &Ctx);

TargetAsmBackend *createARMAsmBackend(const Target &, const std::string &);

FunctionPass *createARMISelDag(ARMBaseTargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

FunctionPass *createARMJITCodeEmitterPass(ARMBaseTargetMachine &TM,
                                          JITCodeEmitter &JCE);

FunctionPass *createARMLoadStoreOptimizationPass(bool PreAlloc = false);
FunctionPass *createARMExpandPseudoPass();
FunctionPass *createARMGlobalMergePass(const TargetLowering* tli);
FunctionPass *createARMConstantIslandPass();
FunctionPass *createNEONMoveFixPass();
FunctionPass *createMLxExpansionPass();
FunctionPass *createThumb2ITBlockPass();
FunctionPass *createThumb2SizeReductionPass();

extern Target TheARMTarget, TheThumbTarget;

void LowerARMMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                  ARMAsmPrinter &AP);

} // end namespace llvm;

#endif
