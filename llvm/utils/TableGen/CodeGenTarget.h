//===- CodeGenTarget.h - Target Class Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines wrappers for the Target class and related global
// functionality.  This makes it easier to access the data and provides a single
// place that needs to check it for validity.  All of these classes throw
// exceptions on error conditions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_TARGET_H
#define CODEGEN_TARGET_H

#include "CodeGenRegisters.h"
#include "CodeGenInstruction.h"
#include <iosfwd>
#include <map>

namespace llvm {

class Record;
class RecordKeeper;
struct CodeGenRegister;
class CodeGenTarget;

/// getValueType - Return the MVT::ValueType that the specified TableGen record
/// corresponds to.
MVT::ValueType getValueType(Record *Rec, const CodeGenTarget *CGT = 0);

std::ostream &operator<<(std::ostream &OS, MVT::ValueType T);
std::string getName(MVT::ValueType T);
std::string getEnumName(MVT::ValueType T);


/// CodeGenTarget - This class corresponds to the Target class in the .td files.
///
class CodeGenTarget {
  Record *TargetRec;
  std::vector<Record*> CalleeSavedRegisters;
  MVT::ValueType PointerType;

  mutable std::map<std::string, CodeGenInstruction> Instructions;
  mutable std::vector<CodeGenRegister> Registers;
  mutable std::vector<CodeGenRegisterClass> RegisterClasses;
  mutable std::vector<MVT::ValueType> LegalValueTypes;
  void ReadRegisters() const;
  void ReadRegisterClasses() const;
  void ReadInstructions() const;
  void ReadLegalValueTypes() const;
public:
  CodeGenTarget();

  Record *getTargetRecord() const { return TargetRec; }
  const std::string &getName() const;

  const std::vector<Record*> &getCalleeSavedRegisters() const {
    return CalleeSavedRegisters;
  }

  MVT::ValueType getPointerType() const { return PointerType; }

  /// getInstructionSet - Return the InstructionSet object.
  ///
  Record *getInstructionSet() const;

  /// getAsmWriter - Return the AssemblyWriter definition for this target.
  ///
  Record *getAsmWriter() const;

  const std::vector<CodeGenRegister> &getRegisters() const {
    if (Registers.empty()) ReadRegisters();
    return Registers;
  }

  const std::vector<CodeGenRegisterClass> &getRegisterClasses() const {
    if (RegisterClasses.empty()) ReadRegisterClasses();
    return RegisterClasses;
  }
  
  const CodeGenRegisterClass &getRegisterClass(Record *R) const {
    const std::vector<CodeGenRegisterClass> &RC = getRegisterClasses();
    for (unsigned i = 0, e = RC.size(); i != e; ++i)
      if (RC[i].TheDef == R)
        return RC[i];
    assert(0 && "Didn't find the register class");
    abort();
  }
  
  /// getRegisterClassForRegister - Find the register class that contains the
  /// specified physical register.  If there register exists in multiple
  /// register classes or is not in a register class, return null.
  const CodeGenRegisterClass *getRegisterClassForRegister(Record *R) const {
    const std::vector<CodeGenRegisterClass> &RCs = getRegisterClasses();
    const CodeGenRegisterClass *FoundRC = 0;
    for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = RegisterClasses[i];
      for (unsigned ei = 0, ee = RC.Elements.size(); ei != ee; ++ei) {
        if (R == RC.Elements[ei]) {
          if (FoundRC) return 0;  // In multiple RC's
          FoundRC = &RC;
          break;
        }
      }
    }
    return FoundRC;
  }
  
  const std::vector<MVT::ValueType> &getLegalValueTypes() const {
    if (LegalValueTypes.empty()) ReadLegalValueTypes();
    return LegalValueTypes;
  }
  
  /// isLegalValueType - Return true if the specified value type is natively
  /// supported by the target (i.e. there are registers that directly hold it).
  bool isLegalValueType(MVT::ValueType VT) const {
    const std::vector<MVT::ValueType> &LegalVTs = getLegalValueTypes();
    for (unsigned i = 0, e = LegalVTs.size(); i != e; ++i)
      if (LegalVTs[i] == VT) return true;
    return false;    
  }

  /// getInstructions - Return all of the instructions defined for this target.
  ///
  const std::map<std::string, CodeGenInstruction> &getInstructions() const {
    if (Instructions.empty()) ReadInstructions();
    return Instructions;
  }

  CodeGenInstruction &getInstruction(const std::string &Name) const {
    const std::map<std::string, CodeGenInstruction> &Insts = getInstructions();
    assert(Insts.count(Name) && "Not an instruction!");
    return const_cast<CodeGenInstruction&>(Insts.find(Name)->second);
  }

  typedef std::map<std::string,
                   CodeGenInstruction>::const_iterator inst_iterator;
  inst_iterator inst_begin() const { return getInstructions().begin(); }
  inst_iterator inst_end() const { return Instructions.end(); }

  /// getInstructionsByEnumValue - Return all of the instructions defined by the
  /// target, ordered by their enum value.
  void getInstructionsByEnumValue(std::vector<const CodeGenInstruction*>
                                                &NumberedInstructions);


  /// isLittleEndianEncoding - are instruction bit patterns defined as  [0..n]?
  ///
  bool isLittleEndianEncoding() const;
};

/// ComplexPattern - ComplexPattern info, corresponding to the ComplexPattern
/// tablegen class in TargetSelectionDAG.td
class ComplexPattern {
  MVT::ValueType Ty;
  unsigned NumOperands;
  std::string SelectFunc;
  std::vector<Record*> RootNodes;
public:
  ComplexPattern() : NumOperands(0) {};
  ComplexPattern(Record *R);

  MVT::ValueType getValueType() const { return Ty; }
  unsigned getNumOperands() const { return NumOperands; }
  const std::string &getSelectFunc() const { return SelectFunc; }
  const std::vector<Record*> &getRootNodes() const {
    return RootNodes;
  }
};

} // End llvm namespace

#endif
