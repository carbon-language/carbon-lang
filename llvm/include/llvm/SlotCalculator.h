//===-- llvm/SlotCalculator.h - Calculate value slots -----------*- C++ -*-===//
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

#ifndef LLVM_SLOTCALCULATOR_H
#define LLVM_SLOTCALCULATOR_H

#include <vector>
#include <map>
class Value;
class Module;
class Function;
class SymbolTable;

class SlotCalculator {
  const Module *TheModule;
  bool IgnoreNamedNodes;     // Shall we not count named nodes?

  typedef std::vector<const Value*> TypePlane;
  std::vector<TypePlane> Table;
  std::map<const Value *, unsigned> NodeMap;

  // ModuleLevel - Used to keep track of which values belong to the module,
  // and which values belong to the currently incorporated function.
  //
  std::vector<unsigned> ModuleLevel;

public:
  SlotCalculator(const Module *M, bool IgnoreNamed);
  // Start out in incorp state
  SlotCalculator(const Function *M, bool IgnoreNamed);
  inline ~SlotCalculator() {}
  
  // getSlot returns < 0 on error!
  int getSlot(const Value *D) const;

  inline unsigned getNumPlanes() const { return Table.size(); }
  inline unsigned getModuleLevel(unsigned Plane) const { 
    return Plane < ModuleLevel.size() ? ModuleLevel[Plane] : 0; 
  }

  inline const TypePlane &getPlane(unsigned Plane) const { 
    return Table[Plane]; 
  }

  // If you'd like to deal with a function, use these two methods to get its
  // data into the SlotCalculator!
  //
  void incorporateFunction(const Function *F);
  void purgeFunction();

protected:
  // getOrCreateSlot - Values can be crammed into here at will... if
  // they haven't been inserted already, they get inserted, otherwise
  // they are ignored.
  //
  int getOrCreateSlot(const Value *D);

  // insertValue - Insert a value into the value table... Return the
  // slot that it occupies, or -1 if the declaration is to be ignored
  // because of the IgnoreNamedNodes flag.
  //
  int insertValue(const Value *D, bool dontIgnore = false);

  // doInsertValue - Small helper function to be called only be insertVal.
  int doInsertValue(const Value *D);

  // processModule - Process all of the module level function declarations and
  // types that are available.
  //
  void processModule();

  // processSymbolTable - Insert all of the values in the specified symbol table
  // into the values table...
  //
  void processSymbolTable(const SymbolTable *ST);
  void processSymbolTableConstants(const SymbolTable *ST);
};

#endif
