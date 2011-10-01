//===- CodeEmitterGen.h - Code Emitter Generator ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FIXME: document
//
//===----------------------------------------------------------------------===//

#ifndef CODEMITTERGEN_H
#define CODEMITTERGEN_H

#include "llvm/TableGen/TableGenBackend.h"
#include <vector>
#include <string>

namespace llvm {

class RecordVal;
class BitsInit;
class CodeGenTarget;

class CodeEmitterGen : public TableGenBackend {
  RecordKeeper &Records;
public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}

  // run - Output the code emitter
  void run(raw_ostream &o);
private:
  void emitMachineOpEmitter(raw_ostream &o, const std::string &Namespace);
  void emitGetValueBit(raw_ostream &o, const std::string &Namespace);
  void reverseBits(std::vector<Record*> &Insts);
  int getVariableBit(const std::string &VarName, BitsInit *BI, int bit);
  std::string getInstructionCase(Record *R, CodeGenTarget &Target);
  void
  AddCodeToMergeInOperand(Record *R, BitsInit *BI, const std::string &VarName,
                          unsigned &NumberedOp,
                          std::string &Case, CodeGenTarget &Target);
    
};

} // End llvm namespace

#endif
