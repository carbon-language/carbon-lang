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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class MachineFunction;
class MachineInstr;
class MDNode;
class TargetRegisterInfo;

// For each user variable, keep a list of DBG_VALUE instructions in order.
// The list can also contain normal instructions that clobber the previous
// DBG_VALUE.
typedef DenseMap<const MDNode *, SmallVector<const MachineInstr *, 4>>
DbgValueHistoryMap;

void calculateDbgValueHistory(const MachineFunction *MF,
                              const TargetRegisterInfo *TRI,
                              DbgValueHistoryMap &Result);
}

#endif
