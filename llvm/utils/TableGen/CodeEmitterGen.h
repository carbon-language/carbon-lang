//===- CodeEmitterGen.h - Code Emitter Generator ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FIXME: document
//
//===----------------------------------------------------------------------===//

#ifndef CODEMITTERGEN_H
#define CODEMITTERGEN_H

#include "TableGenBackend.h"
#include <map>
#include <vector>

namespace llvm {

class RecordVal;

class CodeEmitterGen : public TableGenBackend {
  RecordKeeper &Records;
public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}

  // run - Output the code emitter
  void run(std::ostream &o);
private:
  void emitMachineOpEmitter(std::ostream &o, const std::string &Namespace);
  void emitGetValueBit(std::ostream &o, const std::string &Namespace);
  void emitInstrOpBits(std::ostream &o,
                       const std::vector<RecordVal> &Vals,
                       std::map<std::string, unsigned> &OpOrder,
                       std::map<std::string, bool> &OpContinuous);
};

} // End llvm namespace

#endif
