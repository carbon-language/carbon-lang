//===- CodeEmitterGen.h - Code Emitter Generator ----------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef CODEMITTERGEN_H
#define CODEMITTERGEN_H

#include "Record.h"
#include <ostream>

struct CodeEmitterGen {
  RecordKeeper &Records;
  
public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}
  
  void createEmitter(std::ostream &o);
  void emitMachineOpEmitter(std::ostream &o, const std::string &Namespace);
  void emitGetValueBit(std::ostream &o, const std::string &Namespace);
};

#endif
