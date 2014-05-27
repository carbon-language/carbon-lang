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

// For each user variable, keep a list of instruction ranges where this variable
// is accessible. The variables are listed in order of appearance.
class DbgValueHistoryMap {
  // Each instruction range starts with a DBG_VALUE instruction, specifying the
  // location of a variable, which is assumed to be valid until the end of the
  // range. If end is not specified, location is valid until the start
  // instruction of the next instruction range, or until the end of the
  // function.
  typedef std::pair<const MachineInstr *, const MachineInstr *> InstrRange;
  typedef SmallVector<InstrRange, 4> InstrRanges;
  typedef MapVector<const MDNode *, InstrRanges> InstrRangesMap;
  InstrRangesMap VarInstrRanges;

public:
  void startInstrRange(const MDNode *Var, const MachineInstr &MI);
  void endInstrRange(const MDNode *Var, const MachineInstr &MI);
  // Returns register currently describing @Var. If @Var is currently
  // unaccessible or is not described by a register, returns 0.
  unsigned getRegisterForVar(const MDNode *Var) const;

  bool empty() const { return VarInstrRanges.empty(); }
  void clear() { VarInstrRanges.clear(); }
  InstrRangesMap::const_iterator begin() const { return VarInstrRanges.begin(); }
  InstrRangesMap::const_iterator end() const { return VarInstrRanges.end(); }
};

void calculateDbgValueHistory(const MachineFunction *MF,
                              const TargetRegisterInfo *TRI,
                              DbgValueHistoryMap &Result);
}

#endif
