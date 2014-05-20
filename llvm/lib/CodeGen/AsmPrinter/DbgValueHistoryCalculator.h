//===-- llvm/CodeGen/AsmPrinter/DbgValueHistoryCalculator.h ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DBGVALUEHISTORYCALCULATOR_H_
#define CODEGEN_ASMPRINTER_DBGVALUEHISTORYCALCULATOR_H_

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class MachineFunction;
class MachineInstr;
class MDNode;
class TargetRegisterInfo;

// For each user variable, keep a list of DBG_VALUE instructions for it
// in the order of appearance. The list can also contain another
// instructions, which are assumed to clobber the previous DBG_VALUE.
// The variables are listed in order of appearance.
typedef MapVector<const MDNode *, SmallVector<const MachineInstr *, 4>>
DbgValueHistoryMap;

void calculateDbgValueHistory(const MachineFunction *MF,
                              const TargetRegisterInfo *TRI,
                              DbgValueHistoryMap &Result);
}

#endif
