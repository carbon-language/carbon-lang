//===- InstrInfoEmitter.h - Generate a Instruction Set Desc. ----*- C++ -*-===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef INSTRINFO_EMITTER_H
#define INSTRINFO_EMITTER_H

#include <iosfwd>
class RecordKeeper;
class Record;
class StringInit;
class IntInit;

class InstrInfoEmitter {
  RecordKeeper &Records;
public:
  InstrInfoEmitter(RecordKeeper &R) : Records(R) {}
  
  // run - Output the instruction set description, returning true on failure.
  void run(std::ostream &OS);

  // runEnums - Print out enum values for all of the instructions.
  void runEnums(std::ostream &OS);
private:
  void emitRecord(Record *R, unsigned Num, Record *InstrInfo, std::ostream &OS);
  void emitShiftedValue(Record *R, StringInit *Val, IntInit *Shift,
                        std::ostream &OS);
};

#endif
