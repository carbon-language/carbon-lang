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

  /// ModuleContainsAllFunctionConstants - This flag is set to true if all
  /// function constants are incorporated into the module constant table.  This
  /// is only possible if building information for a bytecode file.
  bool ModuleContainsAllFunctionConstants;

  /// CompactionTable/NodeMap - When function compaction has been performed,
  /// these entries provide a compacted view of the namespace needed to emit
  /// instructions in a function body.  The 'getSlot()' method automatically
  /// returns these entries if applicable, or the global entries if not.
  std::vector<TypePlane> CompactionTable;
  TypeList CompactionTypes;
  typedef std::map<const Value*, unsigned> CompactionNodeMapType;
  CompactionNodeMapType CompactionNodeMap;
  typedef std::map<const Type*, unsigned> CompactionTypeMapType;
  CompactionTypeMapType CompactionTypeMap;

  SlotCalculator(const SlotCalculator &);  // DO NOT IMPLEMENT
  void operator=(const SlotCalculator &);  // DO NOT IMPLEMENT
public:
  SlotCalculator(const Module *M);
  // Start out in incorp state
  SlotCalculator(const Function *F);

  /// getSlot - Return the slot number of the specified value in it's type
  /// plane.  This returns < 0 on error!
  ///
  int getSlot(const Value *V) const;
  int getSlot(const Type* T) const;

  /// getGlobalSlot - Return a slot number from the global table.  This can only
  /// be used when a compaction table is active.
  unsigned getGlobalSlot(const Value *V) const;
  unsigned getGlobalSlot(const Type *V) const;

  inline unsigned getNumPlanes() const {
    if (CompactionTable.empty())
      return Table.size();
    else
      return CompactionTable.size();
  }

  inline unsigned getNumTypes() const {
    if (CompactionTypes.empty())
      return Types.size();
    else
      return CompactionTypes.size();
  }

  inline unsigned getModuleLevel(unsigned Plane) const {
    return Plane < ModuleLevel.size() ? ModuleLevel[Plane] : 0;
  }

  /// Returns the number of types in the type list that are at module level
  inline unsigned getModuleTypeLevel() const {
    return ModuleTypeLevel;
  }

  TypePlane &getPlane(unsigned Plane);
  TypeList& getTypes() {
    if (!CompactionTypes.empty())
      return CompactionTypes;
    return Types;
  }

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

  const std::vector<TypePlane> &getCompactionTable() const {
    return CompactionTable;
  }

  const TypeList& getCompactionTypes() const { return CompactionTypes; }

  /// @brief Determine if the compaction table (not types) is empty
  bool CompactionTableIsEmpty() const;

private:
  // getOrCreateSlot - Values can be crammed into here at will... if
  // they haven't been inserted already, they get inserted, otherwise
  // they are ignored.
  //
  int getOrCreateSlot(const Value *V);
  int getOrCreateSlot(const Type *T);

  // insertValue - Insert a value into the value table... Return the
  // slot that it occupies, or -1 if the declaration is to be ignored
  // because of the IgnoreNamedNodes flag.
  //
  int insertValue(const Value *D, bool dontIgnore = false);
  int insertType(const Type *T, bool dontIgnore = false);

  // doInsertValue - Small helper function to be called only be insertVal.
  int doInsertValue(const Value *V);
  int doInsertType(const Type *T);

  // processModule - Process all of the module level function declarations and
  // types that are available.
  //
  void processModule();

  // processSymbolTable - Insert all of the values in the specified symbol table
  // into the values table...
  //
  void processSymbolTable(const SymbolTable *ST);
  void processSymbolTableConstants(const SymbolTable *ST);

  void buildCompactionTable(const Function *F);
  unsigned getOrCreateCompactionTableSlot(const Value *V);
  unsigned getOrCreateCompactionTableSlot(const Type *V);
  void pruneCompactionTable();
};

} // End llvm namespace

#endif
