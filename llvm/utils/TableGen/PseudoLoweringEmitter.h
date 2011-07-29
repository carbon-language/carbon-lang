//===- PseudoLoweringEmitter.h - PseudoLowering Generator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PSEUDOLOWERINGEMITTER_H
#define PSEUDOLOWERINGEMITTER_H

#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "TableGenBackend.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class PseudoLoweringEmitter : public TableGenBackend {
  struct OpData {
    enum MapKind { Operand, Imm, Reg };
    MapKind Kind;
    union {
      unsigned Operand;   // Operand number mapped to.
      uint64_t Imm;       // Integer immedate value.
      Record *Reg;        // Physical register.
    } Data;
  };
  struct PseudoExpansion {
    CodeGenInstruction Source;  // The source pseudo instruction definition.
    CodeGenInstruction Dest;    // The destination instruction to lower to.
    IndexedMap<OpData> OperandMap;

    PseudoExpansion(CodeGenInstruction &s, CodeGenInstruction &d,
                    IndexedMap<OpData> &m) :
      Source(s), Dest(d), OperandMap(m) {}
  };

  RecordKeeper &Records;

  // It's overkill to have an instance of the full CodeGenTarget object,
  // but it loads everything on demand, not in the constructor, so it's
  // lightweight in performance, so it works out OK.
  CodeGenTarget Target;

  SmallVector<PseudoExpansion, 64> Expansions;

  unsigned addDagOperandMapping(Record *Rec, DagInit *Dag,
                                CodeGenInstruction &Insn,
                                IndexedMap<OpData> &OperandMap,
                                unsigned BaseIdx);
  void evaluateExpansion(Record *Pseudo);
  void emitLoweringEmitter(raw_ostream &o);
public:
  PseudoLoweringEmitter(RecordKeeper &R) : Records(R), Target(R) {}

  /// run - Output the pseudo-lowerings.
  void run(raw_ostream &o);
};

} // end llvm namespace

#endif
