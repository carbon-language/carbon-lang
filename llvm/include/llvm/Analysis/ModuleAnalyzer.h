//===-- llvm/Analysis/ModuleAnalyzer.h - Module analysis driver --*- C++ -*-==//
//
// This class provides a nice interface to traverse a module in a predictable
// way.  This is used by the AssemblyWriter, BytecodeWriter, and SlotCalculator
// to do analysis of a module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MODULEANALYZER_H
#define LLVM_ANALYSIS_MODULEANALYZER_H

#include "llvm/ConstantPool.h"
#include <set>

class Module;
class Method;
class BasicBlock;
class Instruction;
class ConstPoolVal;
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

  //===--------------------------------------------------------------------===//
  //  Stages of processing Module level information
  //
  virtual bool processConstPool(const ConstantPool &CP, bool isMethod);

  // processType - This callback occurs when an derived type is discovered
  // at the class level. This activity occurs when processing a constant pool.
  //
  virtual bool processType(const Type *Ty) { return false; }

  // processMethods - The default implementation of this method loops through 
  // all of the methods in the module and processModule's them.
  //
  virtual bool processMethods(const Module *M);

  //===--------------------------------------------------------------------===//
  //  Stages of processing a constant pool
  //

  // processConstPoolPlane - Called once for every populated plane in the
  // constant pool.  The default action is to do nothing.  The processConstPool
  // method does the iteration over constants.
  //
  virtual bool processConstPoolPlane(const ConstantPool &CP,
				     const ConstantPool::PlaneType &Pl, 
				     bool isMethod) {
    return false;
  }

  // processConstant is called once per each constant in the constant pool.  It
  // traverses the constant pool such that it visits each constant in the
  // order of its type.  Thus, all 'int' typed constants shall be visited 
  // sequentially, etc...
  //
  virtual bool processConstant(const ConstPoolVal *CPV) { return false; }

  // visitMethod - This member is called after the constant pool has been 
  // processed.  The default implementation of this is a noop.
  //
  virtual bool visitMethod(const Method *M) { return false; }

  //===--------------------------------------------------------------------===//
  //  Stages of processing Method level information
  //
  // (processConstPool is also used above, with the isMethod flag set to true)
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
  bool handleType(set<const Type *> &TypeSet, const Type *T);
};

#endif
