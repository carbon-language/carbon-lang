//===- InstrInfoEmitter.h - Generate a Instruction Set Desc. ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef INSTRINFO_EMITTER_H
#define INSTRINFO_EMITTER_H

#include "TableGenBackend.h"
#include <vector>
#include <map>

namespace llvm {

class StringInit;
class IntInit;
class ListInit;
struct CodeGenInstruction;

class InstrInfoEmitter : public TableGenBackend {
  RecordKeeper &Records;
  bool IsItineraries;
  std::map<std::string, unsigned> ItinClassMap;
  
public:
  InstrInfoEmitter(RecordKeeper &R) : Records(R), IsItineraries(false) {}

  // run - Output the instruction set description, returning true on failure.
  void run(std::ostream &OS);

  // runEnums - Print out enum values for all of the instructions.
  void runEnums(std::ostream &OS);
private:
  void printDefList(const std::vector<Record*> &Uses, unsigned Num,
                    std::ostream &OS) const;
  void emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                  Record *InstrInfo, 
                  std::map<std::vector<Record*>, unsigned> &EL,
                  std::map<std::vector<Record*>, unsigned> &OpInfo,
                  std::ostream &OS);
  void GatherItinClasses();
  unsigned ItinClassNumber(std::string ItinName);
  void emitShiftedValue(Record *R, StringInit *Val, IntInit *Shift,
                        std::ostream &OS);
};

} // End llvm namespace

#endif
