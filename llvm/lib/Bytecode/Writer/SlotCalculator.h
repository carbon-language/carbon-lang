//===-- Analysis/SlotCalculator.h - Calculate value slots -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class calculates the slots that values will land in.  This is useful for
// when writing bytecode or assembly out, because you have to know these things.
//
// Specifically, this class calculates the "type plane numbering" that you see
// for a function if you strip out all of the symbols in it.  For assembly
// writing, this is used when a symbol does not have a name.  For bytecode
// writing, this is always used, and the symbol table is added on later.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SLOTCALCULATOR_H
#define LLVM_ANALYSIS_SLOTCALCULATOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm {

class Value;
class Type;
class Module;
class Function;
class SymbolTable;
class TypeSymbolTable;
class ValueSymbolTable;
class ConstantArray;

struct ModuleLevelDenseMapKeyInfo {
  static inline unsigned getEmptyKey() { return ~0U; }
  static inline unsigned getTombstoneKey() { return ~1U; }
  static unsigned getHashValue(unsigned Val) { return Val ^ Val >> 4; }
  static bool isPod() { return true; }
};


class SlotCalculator {
  const Module *TheModule;
public:
  typedef std::vector<const Type*> TypeList;
  typedef SmallVector<const Value*, 16> TypePlane;
private:
  std::vector<TypePlane> Table;
  TypeList Types;
  typedef DenseMap<const Value*, unsigned> NodeMapType;
  NodeMapType NodeMap;

  typedef DenseMap<const Type*, unsigned> TypeMapType;
  TypeMapType TypeMap;

  /// ConstantStrings - If we are indexing for a bytecode file, this keeps track
  /// of all of the constants strings that need to be emitted.
  std::vector<const ConstantArray*> ConstantStrings;

  /// ModuleLevel - Used to keep track of which values belong to the module,
  /// and which values belong to the currently incorporated function.
  ///
  DenseMap<unsigned,unsigned,ModuleLevelDenseMapKeyInfo> ModuleLevel;
  unsigned NumModuleTypes;

  SlotCalculator(const SlotCalculator &);  // DO NOT IMPLEMENT
  void operator=(const SlotCalculator &);  // DO NOT IMPLEMENT
public:
  SlotCalculator(const Module *M);

  /// getSlot - Return the slot number of the specified value in it's type
  /// plane.
  ///
  unsigned getSlot(const Value *V) const {
    NodeMapType::const_iterator I = NodeMap.find(V);
    assert(I != NodeMap.end() && "Value not in slotcalculator!");
    return I->second;
  }
  
  unsigned getTypeSlot(const Type* T) const {
    TypeMapType::const_iterator I = TypeMap.find(T);
    assert(I != TypeMap.end() && "Type not in slotcalc!");
    return I->second;
  }

  inline unsigned getNumPlanes() const { return Table.size(); }
  inline unsigned getNumTypes() const { return Types.size(); }

  TypePlane &getPlane(unsigned Plane) {
    // Okay we are just returning an entry out of the main Table.  Make sure the
    // plane exists and return it.
    if (Plane >= Table.size())
      Table.resize(Plane+1);
    return Table[Plane];
  }

  TypeList& getTypes() { return Types; }

  /// incorporateFunction/purgeFunction - If you'd like to deal with a function,
  /// use these two methods to get its data into the SlotCalculator!
  ///
  void incorporateFunction(const Function *F);
  void purgeFunction();

  /// string_iterator/string_begin/end - Access the list of module-level
  /// constant strings that have been incorporated.  This is only applicable to
  /// bytecode files.
  typedef std::vector<const ConstantArray*>::const_iterator string_iterator;
  string_iterator string_begin() const { return ConstantStrings.begin(); }
  string_iterator string_end() const   { return ConstantStrings.end(); }

private:
  void CreateSlotIfNeeded(const Value *V);
  void CreateFunctionValueSlot(const Value *V);
  unsigned getOrCreateTypeSlot(const Type *T);

  // processModule - Process all of the module level function declarations and
  // types that are available.
  //
  void processModule();

  // processSymbolTable - Insert all of the values in the specified symbol table
  // into the values table...
  //
  void processTypeSymbolTable(const TypeSymbolTable *ST);
  void processValueSymbolTable(const ValueSymbolTable *ST);

  // insertPrimitives - helper for constructors to insert primitive types.
  void insertPrimitives();
};

} // End llvm namespace

#endif
