//===- SimpleInstrSelEmitter.h - Generate a Simple Instruction Selector ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a simple instruction selector
// 
//
//===----------------------------------------------------------------------===//

#ifndef SIMPLE_INSTR_SELECTOR_EMITTER_H
#define SIMPLE_INSTR_SELECTOR_EMITTER_H

#include "TableGenBackend.h"
#include "CodeGenWrappers.h"
#include <vector>
#include <map>
#include <cassert>

namespace llvm {

class Init;
class InstrSelectorEmitter;


/// InstrSelectorEmitter - The top-level class which coordinates construction
/// and emission of the instruction selector.
///
class SimpleInstrSelEmitter : public TableGenBackend {
  RecordKeeper &Records;
  std::string globalSpacing;

public:
  SimpleInstrSelEmitter(RecordKeeper &R) : Records(R) {globalSpacing = "  ";}
  
  // run - Output the instruction set description, returning true on failure.
  void run(std::ostream &OS);

  Record* SimpleInstrSelEmitter::findInstruction(std::ostream &OS, std::string cl, std::vector<std::string>& vi);

  Record* SimpleInstrSelEmitter::findRegister(std::ostream &OS, std::string regname);

  std::string SimpleInstrSelEmitter::formatRegister(std::ostream &OS, std::string regname);

  void SimpleInstrSelEmitter::generateBMIcall(std::ostream &OS, std::string MBB, std::string IP, std::string Opcode, int NumOperands, ListInit &l, ListInit &r);

  void SimpleInstrSelEmitter::InstrSubclasses(std::ostream &OS, std::string prefix, std::string InstrClassName, ListInit* SupportedSubclasses, std::vector<std::string>& vi, unsigned depth);

  std::string spacing();
  std::string addspacing();
  std::string remspacing();

};

} // End llvm namespace

#endif
