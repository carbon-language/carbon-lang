//===- CodeGenTarget.cpp - CodeGen Target Class Wrapper ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class wrap target description classes used by the various code
// generation TableGen backends.  This makes it easier to access the data and
// provides a single place that needs to check it for validity.  All of these
// classes throw exceptions on error conditions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "Record.h"
using namespace llvm;

/// getValueType - Return the MCV::ValueType that the specified TableGen record
/// corresponds to.
MVT::ValueType llvm::getValueType(Record *Rec) {
  return (MVT::ValueType)Rec->getValueAsInt("Value");
}

std::string llvm::getName(MVT::ValueType T) {
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

std::string llvm::getEnumName(MVT::ValueType T) {
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


std::ostream &llvm::operator<<(std::ostream &OS, MVT::ValueType T) {
  return OS << getName(T);
}


/// getTarget - Return the current instance of the Target class.
///
CodeGenTarget::CodeGenTarget() : PointerType(MVT::Other) {
  std::vector<Record*> Targets = Records.getAllDerivedDefinitions("Target");
  if (Targets.size() == 0)
    throw std::string("ERROR: No 'Target' subclasses defined!");  
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

/// getAsmWriter - Return the AssemblyWriter definition for this target.
///
Record *CodeGenTarget::getAsmWriter() const {
  return TargetRec->getValueAsDef("AssemblyWriter");
}


void CodeGenTarget::ReadInstructions() const {
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");

  if (Insts.size() == 0)
    throw std::string("No 'Instruction' subclasses defined!");

  std::string InstFormatName =
    getAsmWriter()->getValueAsString("InstFormatName");

  for (unsigned i = 0, e = Insts.size(); i != e; ++i) {
    std::string AsmStr = Insts[i]->getValueAsString(InstFormatName);
    Instructions.insert(std::make_pair(Insts[i]->getName(),
                                       CodeGenInstruction(Insts[i], AsmStr)));
  }
}

/// getPHIInstruction - Return the designated PHI instruction.
const CodeGenInstruction &CodeGenTarget::getPHIInstruction() const {
  Record *PHI = getInstructionSet()->getValueAsDef("PHIInst");
  std::map<std::string, CodeGenInstruction>::const_iterator I =
    getInstructions().find(PHI->getName());
  if (I == Instructions.end())
    throw "Could not find PHI instruction named '" + PHI->getName() + "'!";
  return I->second;
}

CodeGenInstruction::CodeGenInstruction(Record *R, const std::string &AsmStr)
  : TheDef(R), AsmString(AsmStr) {
  Name      = R->getValueAsString("Name");
  Namespace = R->getValueAsString("Namespace");

  isReturn     = R->getValueAsBit("isReturn");
  isBranch     = R->getValueAsBit("isBranch");
  isBarrier    = R->getValueAsBit("isBarrier");
  isCall       = R->getValueAsBit("isCall");
  isTwoAddress = R->getValueAsBit("isTwoAddress");
  isTerminator = R->getValueAsBit("isTerminator");

  try {
    DagInit *DI = R->getValueAsDag("OperandList");

    unsigned MIOperandNo = 0;
    for (unsigned i = 0, e = DI->getNumArgs(); i != e; ++i)
      if (DefInit *Arg = dynamic_cast<DefInit*>(DI->getArg(i))) {
        Record *Rec = Arg->getDef();
        MVT::ValueType Ty;
        std::string PrintMethod = "printOperand";
        unsigned NumOps = 1;
        if (Rec->isSubClassOf("RegisterClass"))
          Ty = getValueType(Rec->getValueAsDef("RegType"));
        else if (Rec->isSubClassOf("Operand")) {
          Ty = getValueType(Rec->getValueAsDef("Type"));
          PrintMethod = Rec->getValueAsString("PrintMethod");
          NumOps = Rec->getValueAsInt("NumMIOperands");
        } else
          throw "Unknown operand class '" + Rec->getName() +
                "' in instruction '" + R->getName() + "' instruction!";
        
        OperandList.push_back(OperandInfo(Rec, Ty, DI->getArgName(i),
                                          PrintMethod, MIOperandNo));
        MIOperandNo += NumOps;
      } else {
        throw "Illegal operand for the '" + R->getName() + "' instruction!";
      }
  } catch (...) {
    // Error parsing operands list, just ignore it.
    AsmString.clear();
    OperandList.clear();
  }
}



/// getOperandNamed - Return the index of the operand with the specified
/// non-empty name.  If the instruction does not have an operand with the
/// specified name, throw an exception.
unsigned CodeGenInstruction::getOperandNamed(const std::string &Name) const {
  assert(!Name.empty() && "Cannot search for operand with no name!");
  for (unsigned i = 0, e = OperandList.size(); i != e; ++i)
    if (OperandList[i].Name == Name) return i;
  throw "Instruction '" + TheDef->getName() +
        "' does not have an operand named '$" + Name + "'!";
}
