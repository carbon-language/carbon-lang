//===-- llvm/Analysis/SlotCalculator.h - Calculate value slots ---*- C++ -*-==//
//
// This ModuleAnalyzer subclass calculates the slots that values will land in.
// This is useful for when writing bytecode or assembly out, because you have 
// to know these things.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SLOTCALCULATOR_H
#define LLVM_ANALYSIS_SLOTCALCULATOR_H

#include "llvm/Analysis/ModuleAnalyzer.h"
#include "llvm/SymTabValue.h"
#include <vector>
#include <map>
class Value;

class SlotCalculator : public ModuleAnalyzer {
  const Module *TheModule;
  bool IgnoreNamedNodes;     // Shall we not count named nodes?

  typedef vector<const Value*> TypePlane;
  vector <TypePlane> Table;
  map<const Value *, unsigned> NodeMap;

  // ModuleLevel - Used to keep track of which values belong to the module,
  // and which values belong to the currently incorporated method.
  //
  vector <unsigned> ModuleLevel;

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
  // insertVal - Insert a value into the value table...
  //
  void insertVal(const Value *D);

  // visitMethod - This member is called after the constant pool has been 
  // processed.  The default implementation of this is a noop.
  //
  virtual bool visitMethod(const Method *M);

  // processConstant is called once per each constant in the constant pool.  It
  // traverses the constant pool such that it visits each constant in the
  // order of its type.  Thus, all 'int' typed constants shall be visited 
  // sequentially, etc...
  //
  virtual bool processConstant(const ConstPoolVal *CPV);

  // processType - This callback occurs when an derived type is discovered
  // at the class level. This activity occurs when processing a constant pool.
  //
  virtual bool processType(const Type *Ty);

  // processMethods - The default implementation of this method loops through 
  // all of the methods in the module and processModule's them.  We don't want
  // this (we want to explicitly visit them with incorporateMethod), so we 
  // disable it.
  //
  virtual bool processMethods(const Module *M) { return false; }

  // processMethodArgument - This member is called for every argument that 
  // is passed into the method.
  //
  virtual bool processMethodArgument(const MethodArgument *MA);

  // processBasicBlock - This member is called for each basic block in a methd.
  //
  virtual bool processBasicBlock(const BasicBlock *BB);

  // processInstruction - This member is called for each Instruction in a methd.
  //
  virtual bool processInstruction(const Instruction *I);
};

#endif
