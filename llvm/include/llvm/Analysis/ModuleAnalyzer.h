//===-- llvm/Analysis/ModuleAnalyzer.h - Module analysis driver --*- C++ -*-==//
//
// This class provides a nice interface to traverse a module in a predictable
// way.  This is used by the AssemblyWriter, BytecodeWriter, and SlotCalculator
// to do analysis of a module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MODULEANALYZER_H
#define LLVM_ANALYSIS_MODULEANALYZER_H

#include <set>

class Type;
class Module;
class Method;
class BasicBlock;
class Instruction;
class MethodType;
class MethodArgument;

class ModuleAnalyzer {
  ModuleAnalyzer(const ModuleAnalyzer &);                   // do not impl
  const ModuleAnalyzer &operator=(const ModuleAnalyzer &);  // do not impl
public:
  ModuleAnalyzer() {}
  virtual ~ModuleAnalyzer() {}
  
protected:
  // processModule - Driver function to call all of my subclasses virtual 
  // methods.  Commonly called by derived type's constructor.
  //
  bool processModule(const Module *M);

  // processType - This callback occurs when an derived type is discovered
  // at the class level. This activity occurs when processing a constant pool.
  //
  virtual bool processType(const Type *Ty) { return false; }

  // processMethods - The default implementation of this method loops through 
  // all of the methods in the module and processModule's them.
  //
  virtual bool processMethods(const Module *M);

  // visitMethod - This member is called after the constant pool has been 
  // processed.  The default implementation of this is a noop.
  //
  virtual bool visitMethod(const Method *M) { return false; }

  //===--------------------------------------------------------------------===//
  //  Stages of processing Method level information
  //

  // processMethod - Process all aspects of a method.
  //
  virtual bool processMethod(const Method *M);

  // processMethodArgument - This member is called for every argument that 
  // is passed into the method.
  //
  virtual bool processMethodArgument(const MethodArgument *MA) { return false; }

  // processBasicBlock - This member is called for each basic block in a methd.
  //
  virtual bool processBasicBlock(const BasicBlock *BB);

  //===--------------------------------------------------------------------===//
  //  Stages of processing BasicBlock level information
  //

  // preProcessInstruction - This member is called for each Instruction in a 
  // method before processInstruction.
  //
  virtual bool preProcessInstruction(const Instruction *I);

  // processInstruction - This member is called for each Instruction in a method
  //
  virtual bool processInstruction(const Instruction *I) { return false; }

private:
  bool handleType(std::set<const Type *> &TypeSet, const Type *T);
};

#endif
