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
class StringInit;
class IntInit;
class ListInit;

class InstrInfoEmitter : public TableGenBackend {
  RecordKeeper &Records;
public:
  InstrInfoEmitter(RecordKeeper &R) : Records(R) {}
  
  // run - Output the instruction set description, returning true on failure.
  void run(std::ostream &OS);

  // runEnums - Print out enum values for all of the instructions.
  void runEnums(std::ostream &OS);
private:
  void printDefList(ListInit *LI, const std::string &Name,
                    std::ostream &OS) const;
  void emitRecord(Record *R, unsigned Num, Record *InstrInfo, std::ostream &OS);
  void emitShiftedValue(Record *R, StringInit *Val, IntInit *Shift,
                        std::ostream &OS);
};

#endif
