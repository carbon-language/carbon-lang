//===- CodeGenWrappers.cpp - Code Generation Class Wrappers -----*- C++ -*-===//
//
// These classes wrap target description classes used by the various code
// generation TableGen backends.  This makes it easier to access the data and
// provides a single place that needs to check it for validity.  All of these
// classes throw exceptions on error conditions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenWrappers.h"
#include "Record.h"

/// getValueType - Return the MCV::ValueType that the specified TableGen record
/// corresponds to.
MVT::ValueType getValueType(Record *Rec) {
  return (MVT::ValueType)Rec->getValueAsInt("Value");
}

std::ostream &operator<<(std::ostream &OS, MVT::ValueType T) {
  switch (T) {
  case MVT::Other: return OS << "UNKNOWN";
  case MVT::i1:    return OS << "i1";
  case MVT::i8:    return OS << "i8";
  case MVT::i16:   return OS << "i16";
  case MVT::i32:   return OS << "i32";
  case MVT::i64:   return OS << "i64";
  case MVT::i128:  return OS << "i128";
  case MVT::f32:   return OS << "f32";
  case MVT::f64:   return OS << "f64";
  case MVT::f80:   return OS << "f80";
  case MVT::f128:  return OS << "f128";
  case MVT::isVoid:return OS << "void";
  }
  return OS;
}



/// getTarget - Return the current instance of the Target class.
///
CodeGenTarget::CodeGenTarget() {
  std::vector<Record*> Targets = Records.getAllDerivedDefinitions("Target");
  if (Targets.size() != 1)
    throw std::string("ERROR: Multiple subclasses of Target defined!");
  TargetRec = Targets[0];

  // Read in all of the CalleeSavedRegisters...
  ListInit *LI = TargetRec->getValueAsListInit("CalleeSavedRegisters");
  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    if (DefInit *DI = dynamic_cast<DefInit*>(LI->getElement(i)))
      CalleeSavedRegisters.push_back(DI->getDef());
    else
      throw "Target: " + TargetRec->getName() +
            " expected register definition in CalleeSavedRegisters list!";

  PointerType = getValueType(TargetRec->getValueAsDef("PointerType"));
}


const std::string &CodeGenTarget::getName() const {
  return TargetRec->getName();
}

Record *CodeGenTarget::getInstructionSet() const {
  return TargetRec->getValueAsDef("InstructionSet");
}
