//===-- llvm/Analysis/SlotCalculator.h - Calculate value slots ---*- C++ -*-==//
//
// This ModuleAnalyzer subclass calculates the slots that values will land in.
// This is useful for when writing bytecode or assembly out, because you have 
// to know these things.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SLOTCALCULATOR_H
#define LLVM_ANALYSIS_SLOTCALCULATOR_H

#include "llvm/SymTabValue.h"
#include <vector>
#include <map>
class Value;
class Module;
class Method;
class MethodArgument;
class BasicBlock;
class Instruction;

class SlotCalculator {
  const Module *TheModule;
  bool IgnoreNamedNodes;     // Shall we not count named nodes?

  typedef vector<const Value*> TypePlane;
  vector<TypePlane> Table;
  map<const Value *, unsigned> NodeMap;

  // ModuleLevel - Used to keep track of which values belong to the module,
  // and which values belong to the currently incorporated method.
  //
  vector<unsigned> ModuleLevel;

public:
  SlotCalculator(const Module *M, bool IgnoreNamed);
  SlotCalculator(const Method *M, bool IgnoreNamed);// Start out in incorp state
  inline ~SlotCalculator() {}
  
  // getValSlot returns < 0 on error!
  int getValSlot(const Value *D) const;

  inline unsigned getNumPlanes() const { return Table.size(); }
  inline unsigned getModuleLevel(unsigned Plane) const { 
    return Plane < ModuleLevel.size() ? ModuleLevel[Plane] : 0; 
  }

  inline const TypePlane &getPlane(unsigned Plane) const { 
    return Table[Plane]; 
  }

  // If you'd like to deal with a method, use these two methods to get its data
  // into the SlotCalculator!
  //
  void incorporateMethod(const Method *M);
  void purgeMethod();

protected:
  // insertVal - Insert a value into the value table... Return the slot that it
  // occupies, or -1 if the declaration is to be ignored because of the
  // IgnoreNamedNodes flag.
  //
  int insertVal(const Value *D, bool dontIgnore = false);

  // insertValue - Values can be crammed into here at will... if they haven't
  // been inserted already, they get inserted, otherwise they are ignored.
  //
  int insertValue(const Value *D);

  // doInsertVal - Small helper function to be called only be insertVal.
  int doInsertVal(const Value *D);

  // processModule - Process all of the module level method declarations and
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
