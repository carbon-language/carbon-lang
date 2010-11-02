//===- CodeGenTarget.h - Target Class Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "Record.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {

struct CodeGenRegister;
class CodeGenTarget;

// SelectionDAG node properties.
//  SDNPMemOperand: indicates that a node touches memory and therefore must
//                  have an associated memory operand that describes the access.
enum SDNP {
  SDNPCommutative, 
  SDNPAssociative, 
  SDNPHasChain,
  SDNPOutFlag,
  SDNPInFlag,
  SDNPOptInFlag,
  SDNPMayLoad,
  SDNPMayStore,
  SDNPSideEffect,
  SDNPMemOperand,
  SDNPVariadic,
  SDNPWantRoot,
  SDNPWantParent
};

/// getValueType - Return the MVT::SimpleValueType that the specified TableGen
/// record corresponds to.
MVT::SimpleValueType getValueType(Record *Rec);

std::string getName(MVT::SimpleValueType T);
std::string getEnumName(MVT::SimpleValueType T);

/// getQualifiedName - Return the name of the specified record, with a
/// namespace qualifier if the record contains one.
std::string getQualifiedName(const Record *R);
  
/// CodeGenTarget - This class corresponds to the Target class in the .td files.
///
class CodeGenTarget {
  Record *TargetRec;

  mutable DenseMap<const Record*, CodeGenInstruction*> Instructions;
  mutable std::vector<CodeGenRegister> Registers;
  mutable std::vector<Record*> SubRegIndices;
  mutable std::vector<CodeGenRegisterClass> RegisterClasses;
  mutable std::vector<MVT::SimpleValueType> LegalValueTypes;
  void ReadRegisters() const;
  void ReadSubRegIndices() const;
  void ReadRegisterClasses() const;
  void ReadInstructions() const;
  void ReadLegalValueTypes() const;
  
  mutable std::vector<const CodeGenInstruction*> InstrsByEnum;
public:
  CodeGenTarget();

  Record *getTargetRecord() const { return TargetRec; }
  const std::string &getName() const;

  /// getInstNamespace - Return the target-specific instruction namespace.
  ///
  std::string getInstNamespace() const;

  /// getInstructionSet - Return the InstructionSet object.
  ///
  Record *getInstructionSet() const;

  /// getAsmParser - Return the AssemblyParser definition for this target.
  ///
  Record *getAsmParser() const;

  /// getAsmWriter - Return the AssemblyWriter definition for this target.
  ///
  Record *getAsmWriter() const;

  const std::vector<CodeGenRegister> &getRegisters() const {
    if (Registers.empty()) ReadRegisters();
    return Registers;
  }
  
  /// getRegisterByName - If there is a register with the specific AsmName,
  /// return it.
  const CodeGenRegister *getRegisterByName(StringRef Name) const;

  const std::vector<Record*> &getSubRegIndices() const {
    if (SubRegIndices.empty()) ReadSubRegIndices();
    return SubRegIndices;
  }

  // Map a SubRegIndex Record to its number.
  unsigned getSubRegIndexNo(Record *idx) const {
    if (SubRegIndices.empty()) ReadSubRegIndices();
    std::vector<Record*>::const_iterator i =
      std::find(SubRegIndices.begin(), SubRegIndices.end(), idx);
    assert(i != SubRegIndices.end() && "Not a SubRegIndex");
    return (i - SubRegIndices.begin()) + 1;
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
  /// specified physical register.  If the register is not in a register
  /// class, return null. If the register is in multiple classes, and the
  /// classes have a superset-subset relationship and the same set of
  /// types, return the superclass.  Otherwise return null.
  const CodeGenRegisterClass *getRegisterClassForRegister(Record *R) const {
    const std::vector<CodeGenRegisterClass> &RCs = getRegisterClasses();
    const CodeGenRegisterClass *FoundRC = 0;
    for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = RegisterClasses[i];
      for (unsigned ei = 0, ee = RC.Elements.size(); ei != ee; ++ei) {
        if (R != RC.Elements[ei])
          continue;

        // If a register's classes have different types, return null.
        if (FoundRC && RC.getValueTypes() != FoundRC->getValueTypes())
          return 0;

        // If this is the first class that contains the register,
        // make a note of it and go on to the next class.
        if (!FoundRC) {
          FoundRC = &RC;
          break;
        }

        std::vector<Record *> Elements(RC.Elements);
        std::vector<Record *> FoundElements(FoundRC->Elements);
        std::sort(Elements.begin(), Elements.end());
        std::sort(FoundElements.begin(), FoundElements.end());

        // Check to see if the previously found class that contains
        // the register is a subclass of the current class. If so,
        // prefer the superclass.
        if (std::includes(Elements.begin(), Elements.end(),
                          FoundElements.begin(), FoundElements.end())) {
          FoundRC = &RC;
          break;
        }

        // Check to see if the previously found class that contains
        // the register is a superclass of the current class. If so,
        // prefer the superclass.
        if (std::includes(FoundElements.begin(), FoundElements.end(),
                          Elements.begin(), Elements.end()))
          break;

        // Multiple classes, and neither is a superclass of the other.
        // Return null.
        return 0;
      }
    }
    return FoundRC;
  }

  /// getRegisterVTs - Find the union of all possible SimpleValueTypes for the
  /// specified physical register.
  std::vector<MVT::SimpleValueType> getRegisterVTs(Record *R) const;
  
  const std::vector<MVT::SimpleValueType> &getLegalValueTypes() const {
    if (LegalValueTypes.empty()) ReadLegalValueTypes();
    return LegalValueTypes;
  }
  
  /// isLegalValueType - Return true if the specified value type is natively
  /// supported by the target (i.e. there are registers that directly hold it).
  bool isLegalValueType(MVT::SimpleValueType VT) const {
    const std::vector<MVT::SimpleValueType> &LegalVTs = getLegalValueTypes();
    for (unsigned i = 0, e = LegalVTs.size(); i != e; ++i)
      if (LegalVTs[i] == VT) return true;
    return false;    
  }

private:
  DenseMap<const Record*, CodeGenInstruction*> &getInstructions() const {
    if (Instructions.empty()) ReadInstructions();
    return Instructions;
  }
public:
  
  CodeGenInstruction &getInstruction(const Record *InstRec) const {
    if (Instructions.empty()) ReadInstructions();
    DenseMap<const Record*, CodeGenInstruction*>::iterator I =
      Instructions.find(InstRec);
    assert(I != Instructions.end() && "Not an instruction");
    return *I->second;
  }

  /// getInstructionsByEnumValue - Return all of the instructions defined by the
  /// target, ordered by their enum value.
  const std::vector<const CodeGenInstruction*> &
  getInstructionsByEnumValue() const {
    if (InstrsByEnum.empty()) ComputeInstrsByEnum();
    return InstrsByEnum;
  }

  typedef std::vector<const CodeGenInstruction*>::const_iterator inst_iterator;
  inst_iterator inst_begin() const{return getInstructionsByEnumValue().begin();}
  inst_iterator inst_end() const { return getInstructionsByEnumValue().end(); }
  
  
  /// isLittleEndianEncoding - are instruction bit patterns defined as  [0..n]?
  ///
  bool isLittleEndianEncoding() const;
  
private:
  void ComputeInstrsByEnum() const;
};

/// ComplexPattern - ComplexPattern info, corresponding to the ComplexPattern
/// tablegen class in TargetSelectionDAG.td
class ComplexPattern {
  MVT::SimpleValueType Ty;
  unsigned NumOperands;
  std::string SelectFunc;
  std::vector<Record*> RootNodes;
  unsigned Properties; // Node properties
public:
  ComplexPattern() : NumOperands(0) {}
  ComplexPattern(Record *R);

  MVT::SimpleValueType getValueType() const { return Ty; }
  unsigned getNumOperands() const { return NumOperands; }
  const std::string &getSelectFunc() const { return SelectFunc; }
  const std::vector<Record*> &getRootNodes() const {
    return RootNodes;
  }
  bool hasProperty(enum SDNP Prop) const { return Properties & (1 << Prop); }
};

} // End llvm namespace

#endif
