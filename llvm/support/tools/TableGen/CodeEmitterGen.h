//===- CodeEmitterGen.h - Code Emitter Generator ----------------*- C++ -*-===//
//
// FIXME: document
//
//===----------------------------------------------------------------------===//

#ifndef CODEMITTERGEN_H
#define CODEMITTERGEN_H

#include <string>
#include <iosfwd>
class RecordKeeper;

class CodeEmitterGen {
  RecordKeeper &Records;
public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}
  
  // run - Output the code emitter, returning true on failure.
  bool run(std::ostream &o);
private:
  void emitMachineOpEmitter(std::ostream &o, const std::string &Namespace);
  void emitGetValueBit(std::ostream &o, const std::string &Namespace);
};

#endif
