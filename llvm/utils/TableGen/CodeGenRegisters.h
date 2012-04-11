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
#include "llvm/Support/ErrorHandling.h"
#include <cstdlib>
#include <map>
#include <string>
#include <set>
#include <vector>

namespace llvm {
  class CodeGenRegBank;

  /// CodeGenSubRegIndex - Represents a sub-register index.
  class CodeGenSubRegIndex {
    Record *const TheDef;
    const unsigned EnumValue;

  public:
    CodeGenSubRegIndex(Record *R, unsigned Enum);

    const std::string &getName() const;
    std::string getNamespace() const;
    std::string getQualifiedName() const;

    // Order CodeGenSubRegIndex pointers by EnumValue.
    struct Less {
      bool operator()(const CodeGenSubRegIndex *A,
                      const CodeGenSubRegIndex *B) const {
        assert(A && B);
        return A->EnumValue < B->EnumValue;
      }
    };

    // Map of composite subreg indices.
    typedef std::map<CodeGenSubRegIndex*, CodeGenSubRegIndex*, Less> CompMap;

    // Returns the subreg index that results from composing this with Idx.
    // Returns NULL if this and Idx don't compose.
    CodeGenSubRegIndex *compose(CodeGenSubRegIndex *Idx) const {
      CompMap::const_iterator I = Composed.find(Idx);
      return I == Composed.end() ? 0 : I->second;
    }

    // Add a composite subreg index: this+A = B.
    // Return a conflicting composite, or NULL
    CodeGenSubRegIndex *addComposite(CodeGenSubRegIndex *A,
                                     CodeGenSubRegIndex *B) {
      std::pair<CompMap::iterator, bool> Ins =
        Composed.insert(std::make_pair(A, B));
      return (Ins.second || Ins.first->second == B) ? 0 : Ins.first->second;
    }

    // Update the composite maps of components specified in 'ComposedOf'.
    void updateComponents(CodeGenRegBank&);

    // Clean out redundant composite mappings.
    void cleanComposites();

    // Return the map of composites.
    const CompMap &getComposites() const { return Composed; }

  private:
    CompMap Composed;
  };

  /// CodeGenRegister - Represents a register definition.
  struct CodeGenRegister {
    Record *TheDef;
    unsigned EnumValue;
    unsigned CostPerUse;
    bool CoveredBySubRegs;

    // Map SubRegIndex -> Register.
    typedef std::map<CodeGenSubRegIndex*, CodeGenRegister*,
                     CodeGenSubRegIndex::Less> SubRegMap;

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
    void addSubRegsPreOrder(SetVector<const CodeGenRegister*> &OSet,
                            CodeGenRegBank&) const;

    // List of super-registers in topological order, small to large.
    typedef std::vector<const CodeGenRegister*> SuperRegList;

    // Get the list of super-registers. This is valid after getSubReg
    // visits all registers during RegBank construction.
    const SuperRegList &getSuperRegs() const {
      assert(SubRegsComplete && "Must precompute sub-registers");
      return SuperRegs;
    }

    // List of register units in ascending order.
    typedef SmallVector<unsigned, 16> RegUnitList;

    // Get the list of register units.
    // This is only valid after getSubRegs() completes.
    const RegUnitList &getRegUnits() const { return RegUnits; }

    // Inherit register units from subregisters.
    // Return true if the RegUnits changed.
    bool inheritRegUnits(CodeGenRegBank &RegBank);

    // Adopt a register unit for pressure tracking.
    // A unit is adopted iff its unit number is >= NumNativeRegUnits.
    void adoptRegUnit(unsigned RUID) { RegUnits.push_back(RUID); }

    // Get the sum of this register's register unit weights.
    unsigned getWeight(const CodeGenRegBank &RegBank) const;

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
    RegUnitList RegUnits;
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

    // Map SubRegIndex -> sub-class.  This is the largest sub-class where all
    // registers have a SubRegIndex sub-register.
    DenseMap<CodeGenSubRegIndex*, CodeGenRegisterClass*> SubClassWithSubReg;

    // Map SubRegIndex -> set of super-reg classes.  This is all register
    // classes SuperRC such that:
    //
    //   R:SubRegIndex in this RC for all R in SuperRC.
    //
    DenseMap<CodeGenSubRegIndex*,
             SmallPtrSet<CodeGenRegisterClass*, 8> > SuperRegClasses;

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
      llvm_unreachable("VTNum greater than number of ValueTypes in RegClass!");
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
    CodeGenRegisterClass*
    getSubClassWithSubReg(CodeGenSubRegIndex *SubIdx) const {
      return SubClassWithSubReg.lookup(SubIdx);
    }

    void setSubClassWithSubReg(CodeGenSubRegIndex *SubIdx,
                               CodeGenRegisterClass *SubRC) {
      SubClassWithSubReg[SubIdx] = SubRC;
    }

    // getSuperRegClasses - Returns a bit vector of all register classes
    // containing only SubIdx super-registers of this class.
    void getSuperRegClasses(CodeGenSubRegIndex *SubIdx, BitVector &Out) const;

    // addSuperRegClass - Add a class containing only SudIdx super-registers.
    void addSuperRegClass(CodeGenSubRegIndex *SubIdx,
                          CodeGenRegisterClass *SuperRC) {
      SuperRegClasses[SubIdx].insert(SuperRC);
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

    // Populate a unique sorted list of units from a register set.
    void buildRegUnitSet(std::vector<unsigned> &RegUnits) const;

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

  // Each RegUnitSet is a sorted vector with a name.
  struct RegUnitSet {
    typedef std::vector<unsigned>::const_iterator iterator;

    std::string Name;
    std::vector<unsigned> Units;
  };

  // CodeGenRegBank - Represent a target's registers and the relations between
  // them.
  class CodeGenRegBank {
    RecordKeeper &Records;
    SetTheory Sets;

    // SubRegIndices.
    std::vector<CodeGenSubRegIndex*> SubRegIndices;
    DenseMap<Record*, CodeGenSubRegIndex*> Def2SubRegIdx;
    unsigned NumNamedIndices;

    // Registers.
    std::vector<CodeGenRegister*> Registers;
    DenseMap<Record*, CodeGenRegister*> Def2Reg;
    unsigned NumNativeRegUnits;
    unsigned NumRegUnits; // # native + adopted register units.

    // Map each register unit to a weight (for register pressure).
    // Includes native and adopted register units.
    std::vector<unsigned> RegUnitWeights;

    // Register classes.
    std::vector<CodeGenRegisterClass*> RegClasses;
    DenseMap<Record*, CodeGenRegisterClass*> Def2RC;
    typedef std::map<CodeGenRegisterClass::Key, CodeGenRegisterClass*> RCKeyMap;
    RCKeyMap Key2RC;

    // Remember each unique set of register units. Initially, this contains a
    // unique set for each register class. Simliar sets are coalesced with
    // pruneUnitSets and new supersets are inferred during computeRegUnitSets.
    std::vector<RegUnitSet> RegUnitSets;

    // Map RegisterClass index to the index of the RegUnitSet that contains the
    // class's units and any inferred RegUnit supersets.
    std::vector<std::vector<unsigned> > RegClassUnitSets;

    // Add RC to *2RC maps.
    void addToMaps(CodeGenRegisterClass*);

    // Create a synthetic sub-class if it is missing.
    CodeGenRegisterClass *getOrCreateSubClass(const CodeGenRegisterClass *RC,
                                              const CodeGenRegister::Set *Membs,
                                              StringRef Name);

    // Infer missing register classes.
    void computeInferredRegisterClasses();
    void inferCommonSubClass(CodeGenRegisterClass *RC);
    void inferSubClassWithSubReg(CodeGenRegisterClass *RC);
    void inferMatchingSuperRegClass(CodeGenRegisterClass *RC,
                                    unsigned FirstSubRegRC = 0);

    // Iteratively prune unit sets.
    void pruneUnitSets();

    // Compute a weight for each register unit created during getSubRegs.
    void computeRegUnitWeights();

    // Create a RegUnitSet for each RegClass and infer superclasses.
    void computeRegUnitSets();

    // Populate the Composite map from sub-register relationships.
    void computeComposites();

  public:
    CodeGenRegBank(RecordKeeper&);

    SetTheory &getSets() { return Sets; }

    // Sub-register indices. The first NumNamedIndices are defined by the user
    // in the .td files. The rest are synthesized such that all sub-registers
    // have a unique name.
    ArrayRef<CodeGenSubRegIndex*> getSubRegIndices() { return SubRegIndices; }
    unsigned getNumNamedIndices() { return NumNamedIndices; }

    // Find a SubRegIndex form its Record def.
    CodeGenSubRegIndex *getSubRegIdx(Record*);

    // Find or create a sub-register index representing the A+B composition.
    CodeGenSubRegIndex *getCompositeSubRegIndex(CodeGenSubRegIndex *A,
                                                CodeGenSubRegIndex *B);

    const std::vector<CodeGenRegister*> &getRegisters() { return Registers; }

    // Find a register from its Record def.
    CodeGenRegister *getReg(Record*);

    // Get a Register's index into the Registers array.
    unsigned getRegIndex(const CodeGenRegister *Reg) const {
      return Reg->EnumValue - 1;
    }

    // Create a new non-native register unit that can be adopted by a register
    // to increase its pressure. Note that NumNativeRegUnits is not increased.
    unsigned newRegUnit(unsigned Weight) {
      if (!RegUnitWeights.empty()) {
        assert(Weight && "should only add allocatable units");
        RegUnitWeights.resize(NumRegUnits+1);
        RegUnitWeights[NumRegUnits] = Weight;
      }
      return NumRegUnits++;
    }

    // Native units are the singular unit of a leaf register. Register aliasing
    // is completely characterized by native units. Adopted units exist to give
    // register additional weight but don't affect aliasing.
    bool isNativeUnit(unsigned RUID) {
      return RUID < NumNativeRegUnits;
    }

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

    // Get a register unit's weight. Zero for unallocatable registers.
    unsigned getRegUnitWeight(unsigned RUID) const {
      return RegUnitWeights[RUID];
    }

    // Get the sum of unit weights.
    unsigned getRegUnitSetWeight(const std::vector<unsigned> &Units) const {
      unsigned Weight = 0;
      for (std::vector<unsigned>::const_iterator
             I = Units.begin(), E = Units.end(); I != E; ++I)
        Weight += getRegUnitWeight(*I);
      return Weight;
    }

    // Increase a RegUnitWeight.
    void increaseRegUnitWeight(unsigned RUID, unsigned Inc) {
      RegUnitWeights[RUID] += Inc;
    }

    // Get the number of register pressure dimensions.
    unsigned getNumRegPressureSets() const { return RegUnitSets.size(); }

    // Get a set of register unit IDs for a given dimension of pressure.
    RegUnitSet getRegPressureSet(unsigned Idx) const {
      return RegUnitSets[Idx];
    }

    // Get a list of pressure set IDs for a register class. Liveness of a
    // register in this class impacts each pressure set in this list by the
    // weight of the register. An exact solution requires all registers in a
    // class to have the same class, but it is not strictly guaranteed.
    ArrayRef<unsigned> getRCPressureSetIDs(unsigned RCIdx) const {
      return RegClassUnitSets[RCIdx];
    }

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

    // Compute the set of registers completely covered by the registers in Regs.
    // The returned BitVector will have a bit set for each register in Regs,
    // all sub-registers, and all super-registers that are covered by the
    // registers in Regs.
    //
    // This is used to compute the mask of call-preserved registers from a list
    // of callee-saves.
    BitVector computeCoveredRegisters(ArrayRef<Record*> Regs);
  };
}

#endif
