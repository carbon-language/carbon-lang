//===- CodeGenWrappers.h - Code Generation Class Wrappers -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// These classes wrap target description classes used by the various code
// generation TableGen backends.  This makes it easier to access the data and
// provides a single place that needs to check it for validity.  All of these
// classes throw exceptions on error conditions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGENWRAPPERS_H
#define CODEGENWRAPPERS_H

#include "llvm/CodeGen/ValueTypes.h"
#include <iosfwd>
#include <string>
#include <vector>

namespace llvm {

class Record;
class RecordKeeper;

/// getValueType - Return the MVT::ValueType that the specified TableGen record
/// corresponds to.
MVT::ValueType getValueType(Record *Rec);

std::ostream &operator<<(std::ostream &OS, MVT::ValueType T);
std::string getName(MVT::ValueType T);
std::string getEnumName(MVT::ValueType T);


/// CodeGenTarget - This class corresponds to the Target class in the .td files.
///
class CodeGenTarget {
  Record *TargetRec;
  std::vector<Record*> CalleeSavedRegisters;
  MVT::ValueType PointerType;

public:
  CodeGenTarget();

  Record *getTargetRecord() const { return TargetRec; }
  const std::string &getName() const;

  const std::vector<Record*> &getCalleeSavedRegisters() const {
    return CalleeSavedRegisters;
  }

  MVT::ValueType getPointerType() const { return PointerType; }

  // getInstructionSet - Return the InstructionSet object...
  Record *getInstructionSet() const;

  // getInstructionSet - Return the CodeGenInstructionSet object for this
  // target, lazily reading it from the record keeper as needed.
  // CodeGenInstructionSet *getInstructionSet -
};

} // End llvm namespace

#endif
