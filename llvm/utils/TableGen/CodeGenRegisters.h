//===- CodeGenRegisters.h - Register and RegisterClass Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_REGISTERS_H
#define CODEGEN_REGISTERS_H

#include "SetTheory.h"
#include "llvm/TableGen/Record.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <cstdlib>
#include <map>
#include <string>
#include <set>
#include <vector>

namespace llvm {
  class CodeGenRegBank;

  /// CodeGenRegister - Represents a register definition.
  struct CodeGenRegister {
    Record *TheDef;
    unsigned EnumValue;
    unsigned CostPerUse;

    // Map SubRegIndex -> Register.
    typedef std::map<Record*, CodeGenRegister*, LessRecord> SubRegMap;

    CodeGenRegister(Record *R, unsigned Enum);

    const std::string &getName() const;

    // Get a map of sub-registers computed lazily.
    // This includes unique entries for all sub-sub-registers.
    const SubRegMap &getSubRegs(CodeGenRegBank&);

    const SubRegMap &getSubRegs() const {
      assert(SubRegsComplete && "Must precompute sub-registers");
      return SubRegs;
    }

    // Add sub-registers to OSet following a pre-order defined by the .td file.
    void addSubRegsPreOrder(SetVector<CodeGenRegister*> &OSet) const;

    // List of super-registers in topological order, small to large.
    typedef std::vector<CodeGenRegister*> SuperRegList;

    // Get the list of super-registers.
    // This is only valid after computeDerivedInfo has visited all registers.
    const SuperRegList &getSuperRegs() const {
      assert(SubRegsComplete && "Must precompute sub-registers");
      return SuperRegs;
    }

    // Order CodeGenRegister pointers by EnumValue.
    struct Less {
      bool operator()(const CodeGenRegister *A,
                      const CodeGenRegister *B) const {
        assert(A && B);
        return A->EnumValue < B->EnumValue;
      }
    };

    // Canonically ordered set.
    typedef std::set<const CodeGenRegister*, Less> Set;

  private:
    bool SubRegsComplete;
    SubRegMap SubRegs;
    SuperRegList SuperRegs;
  };


  class CodeGenRegisterClass {
    CodeGenRegister::Set Members;
    // Allocation orders. Order[0] always contains all registers in Members.
    std::vector<SmallVector<Record*, 16> > Orders;
    // Bit mask of sub-classes including this, indexed by their EnumValue.
    BitVector SubClasses;
    // List of super-classes, topologocally ordered to have the larger classes
    // first.  This is the same as sorting by EnumValue.
    SmallVector<CodeGenRegisterClass*, 4> SuperClasses;
    Record *TheDef;
    std::string Name;

    // For a synthesized class, inherit missing properties from the nearest
    // super-class.
    void inheritProperties(CodeGenRegBank&);

    // Map SubRegIndex -> sub-class
    DenseMap<Record*, CodeGenRegisterClass*> SubClassWithSubReg;

  public:
    unsigned EnumValue;
    std::string Namespace;
    std::vector<MVT::SimpleValueType> VTs;
    unsigned SpillSize;
    unsigned SpillAlignment;
    int CopyCost;
    bool Allocatable;
    // Map SubRegIndex -> RegisterClass
    DenseMap<Record*,Record*> SubRegClasses;
    std::string AltOrderSelect;

    // Return the Record that defined this class, or NULL if the class was
    // created by TableGen.
    Record *getDef() const { return TheDef; }

    const std::string &getName() const { return Name; }
    std::string getQualifiedName() const;
    const std::vector<MVT::SimpleValueType> &getValueTypes() const {return VTs;}
    unsigned getNumValueTypes() const { return VTs.size(); }

    MVT::SimpleValueType getValueTypeNum(unsigned VTNum) const {
      if (VTNum < VTs.size())
        return VTs[VTNum];
      assert(0 && "VTNum greater than number of ValueTypes in RegClass!");
      abort();
    }

    // Return true if this this class contains the register.
    bool contains(const CodeGenRegister*) const;

    // Returns true if RC is a subclass.
    // RC is a sub-class of this class if it is a valid replacement for any
    // instruction operand where a register of this classis required. It must
    // satisfy these conditions:
    //
    // 1. All RC registers are also in this.
    // 2. The RC spill size must not be smaller than our spill size.
    // 3. RC spill alignment must be compatible with ours.
    //
    bool hasSubClass(const CodeGenRegisterClass *RC) const {
      return SubClasses.test(RC->EnumValue);
    }

    // getSubClassWithSubReg - Returns the largest sub-class where all
    // registers have a SubIdx sub-register.
    CodeGenRegisterClass *getSubClassWithSubReg(Record *SubIdx) const {
      return SubClassWithSubReg.lookup(SubIdx);
    }

    void setSubClassWithSubReg(Record *SubIdx, CodeGenRegisterClass *SubRC) {
      SubClassWithSubReg[SubIdx] = SubRC;
    }

    // getSubClasses - Returns a constant BitVector of subclasses indexed by
    // EnumValue.
    // The SubClasses vector includs an entry for this class.
    const BitVector &getSubClasses() const { return SubClasses; }

    // getSuperClasses - Returns a list of super classes ordered by EnumValue.
    // The array does not include an entry for this class.
    ArrayRef<CodeGenRegisterClass*> getSuperClasses() const {
      return SuperClasses;
    }

    // Returns an ordered list of class members.
    // The order of registers is the same as in the .td file.
    // No = 0 is the default allocation order, No = 1 is the first alternative.
    ArrayRef<Record*> getOrder(unsigned No = 0) const {
        return Orders[No];
    }

    // Return the total number of allocation orders available.
    unsigned getNumOrders() const { return Orders.size(); }

    // Get the set of registers.  This set contains the same registers as
    // getOrder(0).
    const CodeGenRegister::Set &getMembers() const { return Members; }

    CodeGenRegisterClass(CodeGenRegBank&, Record *R);

    // A key representing the parts of a register class used for forming
    // sub-classes.  Note the ordering provided by this key is not the same as
    // the topological order used for the EnumValues.
    struct Key {
      const CodeGenRegister::Set *Members;
      unsigned SpillSize;
      unsigned SpillAlignment;

      Key(const Key &O)
        : Members(O.Members),
          SpillSize(O.SpillSize),
          SpillAlignment(O.SpillAlignment) {}

      Key(const CodeGenRegister::Set *M, unsigned S = 0, unsigned A = 0)
        : Members(M), SpillSize(S), SpillAlignment(A) {}

      Key(const CodeGenRegisterClass &RC)
        : Members(&RC.getMembers()),
          SpillSize(RC.SpillSize),
          SpillAlignment(RC.SpillAlignment) {}

      // Lexicographical order of (Members, SpillSize, SpillAlignment).
      bool operator<(const Key&) const;
    };

    // Create a non-user defined register class.
    CodeGenRegisterClass(StringRef Name, Key Props);

    // Called by CodeGenRegBank::CodeGenRegBank().
    static void computeSubClasses(CodeGenRegBank&);
  };

  // CodeGenRegBank - Represent a target's registers and the relations between
  // them.
  class CodeGenRegBank {
    RecordKeeper &Records;
    SetTheory Sets;

    std::vector<Record*> SubRegIndices;
    unsigned NumNamedIndices;
    std::vector<CodeGenRegister*> Registers;
    DenseMap<Record*, CodeGenRegister*> Def2Reg;

    // Register classes.
    std::vector<CodeGenRegisterClass*> RegClasses;
    DenseMap<Record*, CodeGenRegisterClass*> Def2RC;
    typedef std::map<CodeGenRegisterClass::Key, CodeGenRegisterClass*> RCKeyMap;
    RCKeyMap Key2RC;

    // Add RC to *2RC maps.
    void addToMaps(CodeGenRegisterClass*);

    // Create a synthetic sub-class if it is missing.
    CodeGenRegisterClass *getOrCreateSubClass(const CodeGenRegisterClass *RC,
                                              const CodeGenRegister::Set *Membs,
                                              StringRef Name);

    // Infer missing register classes.
    void computeInferredRegisterClasses();
    void inferCommonSubClass(CodeGenRegisterClass *RC);

    // Composite SubRegIndex instances.
    // Map (SubRegIndex, SubRegIndex) -> SubRegIndex.
    typedef DenseMap<std::pair<Record*, Record*>, Record*> CompositeMap;
    CompositeMap Composite;

    // Populate the Composite map from sub-register relationships.
    void computeComposites();

  public:
    CodeGenRegBank(RecordKeeper&);

    SetTheory &getSets() { return Sets; }

    // Sub-register indices. The first NumNamedIndices are defined by the user
    // in the .td files. The rest are synthesized such that all sub-registers
    // have a unique name.
    const std::vector<Record*> &getSubRegIndices() { return SubRegIndices; }
    unsigned getNumNamedIndices() { return NumNamedIndices; }

    // Map a SubRegIndex Record to its enum value.
    unsigned getSubRegIndexNo(Record *idx);

    // Find or create a sub-register index representing the A+B composition.
    Record *getCompositeSubRegIndex(Record *A, Record *B, bool create = false);

    const std::vector<CodeGenRegister*> &getRegisters() { return Registers; }

    // Find a register from its Record def.
    CodeGenRegister *getReg(Record*);

    ArrayRef<CodeGenRegisterClass*> getRegClasses() const {
      return RegClasses;
    }

    // Find a register class from its def.
    CodeGenRegisterClass *getRegClass(Record*);

    /// getRegisterClassForRegister - Find the register class that contains the
    /// specified physical register.  If the register is not in a register
    /// class, return null. If the register is in multiple classes, and the
    /// classes have a superset-subset relationship and the same set of types,
    /// return the superclass.  Otherwise return null.
    const CodeGenRegisterClass* getRegClassForRegister(Record *R);

    // Computed derived records such as missing sub-register indices.
    void computeDerivedInfo();

    // Compute full overlap sets for every register. These sets include the
    // rarely used aliases that are neither sub nor super-registers.
    //
    // Map[R1].count(R2) is reflexive and symmetric, but not transitive.
    //
    // If R1 is a sub-register of R2, Map[R1] is a subset of Map[R2].
    void computeOverlaps(std::map<const CodeGenRegister*,
                                  CodeGenRegister::Set> &Map);
  };
}

#endif
