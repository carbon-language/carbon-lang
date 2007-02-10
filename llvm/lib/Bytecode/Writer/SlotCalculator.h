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

#include <vector>
#include <map>

namespace llvm {

class Value;
class Type;
class Module;
class Function;
class SymbolTable;
class TypeSymbolTable;
class ValueSymbolTable;
class ConstantArray;

class SlotCalculator {
  const Module *TheModule;

  typedef std::vector<const Type*> TypeList;
  typedef std::vector<const Value*> TypePlane;
  std::vector<TypePlane> Table;
  TypeList Types;
  typedef std::map<const Value*, unsigned> NodeMapType;
  NodeMapType NodeMap;

  typedef std::map<const Type*, unsigned> TypeMapType;
  TypeMapType TypeMap;

  /// ConstantStrings - If we are indexing for a bytecode file, this keeps track
  /// of all of the constants strings that need to be emitted.
  std::vector<const ConstantArray*> ConstantStrings;

  /// ModuleLevel - Used to keep track of which values belong to the module,
  /// and which values belong to the currently incorporated function.
  ///
  std::vector<unsigned> ModuleLevel;
  unsigned ModuleTypeLevel;

  SlotCalculator(const SlotCalculator &);  // DO NOT IMPLEMENT
  void operator=(const SlotCalculator &);  // DO NOT IMPLEMENT
public:
  SlotCalculator(const Module *M);

  /// getSlot - Return the slot number of the specified value in it's type
  /// plane.  This returns < 0 on error!
  ///
  int getSlot(const Value *V) const;
  int getTypeSlot(const Type* T) const;

  inline unsigned getNumPlanes() const { return Table.size(); }
  inline unsigned getNumTypes() const { return Types.size(); }

  inline unsigned getModuleLevel(unsigned Plane) const {
    return Plane < ModuleLevel.size() ? ModuleLevel[Plane] : 0;
  }

  /// Returns the number of types in the type list that are at module level
  inline unsigned getModuleTypeLevel() const {
    return ModuleTypeLevel;
  }

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
