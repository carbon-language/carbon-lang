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

std::string getName(MVT::ValueType T) {
  switch (T) {
  case MVT::Other: return "UNKNOWN";
  case MVT::i1:    return "i1";
  case MVT::i8:    return "i8";
  case MVT::i16:   return "i16";
  case MVT::i32:   return "i32";
  case MVT::i64:   return "i64";
  case MVT::i128:  return "i128";
  case MVT::f32:   return "f32";
  case MVT::f64:   return "f64";
  case MVT::f80:   return "f80";
  case MVT::f128:  return "f128";
  case MVT::isVoid:return "void";
  default: assert(0 && "ILLEGAL VALUE TYPE!"); return "";
  }
}

std::string getEnumName(MVT::ValueType T) {
  switch (T) {
  case MVT::Other: return "Other";
  case MVT::i1:    return "i1";
  case MVT::i8:    return "i8";
  case MVT::i16:   return "i16";
  case MVT::i32:   return "i32";
  case MVT::i64:   return "i64";
  case MVT::i128:  return "i128";
  case MVT::f32:   return "f32";
  case MVT::f64:   return "f64";
  case MVT::f80:   return "f80";
  case MVT::f128:  return "f128";
  case MVT::isVoid:return "isVoid";
  default: assert(0 && "ILLEGAL VALUE TYPE!"); return "";
  }
}


std::ostream &operator<<(std::ostream &OS, MVT::ValueType T) {
  return OS << getName(T);
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
