//===- CodeEmitterGen.h - Code Emitter Generator ----------------*- C++ -*-===//
//
// FIXME: document
//
//===----------------------------------------------------------------------===//

#ifndef CODEMITTERGEN_H
#define CODEMITTERGEN_H

#include "Record.h"

class CodeEmitterGen {
  RecordKeeper &Records;
public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}
  
  int createEmitter(std::ostream &o);
private:
  void emitMachineOpEmitter(std::ostream &o, const std::string &Namespace);
  void emitGetValueBit(std::ostream &o, const std::string &Namespace);
};

#endif
