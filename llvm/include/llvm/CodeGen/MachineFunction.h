//===-- llvm/CodeGen/MachineFunction.h --------------------------*- C++ -*-===//
// 
// Collect native machine code information for a method.  This allows
// target-specific information about the generated code to be stored with each
// method.
//   
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFUNCTION_H
#define LLVM_CODEGEN_MACHINEFUNCTION_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "Support/NonCopyable.h"
#include "Support/HashExtras.h"
#include <Support/hash_set>
class Value;
class Function;
class Constant;
class Type;
class TargetMachine;
class Pass;

Pass *createMachineCodeConstructionPass(TargetMachine &Target);
Pass *createMachineCodeDestructionPass();

class MachineFunction : private Annotation {
  hash_set<const Constant*> constantsForConstPool;
  hash_map<const Value*, int> offsets;
  const         Function* method;

  // FIXME: State should be held elsewhere...
  unsigned	staticStackSize;
  unsigned	automaticVarsSize;
  unsigned	regSpillsSize;
  unsigned	maxOptionalArgsSize;
  unsigned	maxOptionalNumArgs;
  unsigned	currentTmpValuesSize;
  unsigned	maxTmpValuesSize;
  bool          compiledAsLeaf;
  bool          spillsAreaFrozen;
  bool          automaticVarsAreaFrozen;
  
public:
  MachineFunction(const Function* function,
                  const TargetMachine& target);
  
  // The next two methods are used to construct and to retrieve
  // the MachineFunction object for the given method.
  // construct() -- Allocates and initializes for a given method and target
  // get()       -- Returns a handle to the object.
  //                This should not be called before "construct()"
  //                for a given Method.
  // 
  static MachineFunction& construct(const Function *method,
                                         const TargetMachine &target);
  static void destruct(const Function *F);
  static MachineFunction& get(const Function* function);
  


  //===--------------------------------------------------------------------===//
  //
  // FIXME: Most of the following state should be moved out to passes that use
  // it, instead of being put here.
  //

  //
  // Accessors for global information about generated code for a method.
  // 
  inline bool     isCompiledAsLeafMethod() const { return compiledAsLeaf; }
  inline unsigned getStaticStackSize()     const { return staticStackSize; }
  inline unsigned getAutomaticVarsSize()   const { return automaticVarsSize; }
  inline unsigned getRegSpillsSize()       const { return regSpillsSize; }
  inline unsigned getMaxOptionalArgsSize() const { return maxOptionalArgsSize;}
  inline unsigned getMaxOptionalNumArgs()  const { return maxOptionalNumArgs;}
  inline const hash_set<const Constant*>&
                  getConstantPoolValues() const {return constantsForConstPool;}
  
  //
  // Modifiers used during code generation
  // 
  void            initializeFrameLayout    (const TargetMachine& target);
  
  void            addToConstantPool        (const Constant* constVal)
                                    { constantsForConstPool.insert(constVal); }
  
  inline void     markAsLeafMethod()              { compiledAsLeaf = true; }
  
  int             computeOffsetforLocalVar (const TargetMachine& target,
                                            const Value*  local,
                                            unsigned int& getPaddedSize,
                                            unsigned int  sizeToUse = 0);
  int             allocateLocalVar         (const TargetMachine& target,
                                            const Value* local,
                                            unsigned int sizeToUse = 0);
  
  int             allocateSpilledValue     (const TargetMachine& target,
                                            const Type* type);
  
  int             pushTempValue            (const TargetMachine& target,
                                            unsigned int size);
  
  void            popAllTempValues         (const TargetMachine& target);
  
  void            freezeSpillsArea         () { spillsAreaFrozen = true; } 
  void            freezeAutomaticVarsArea  () { automaticVarsAreaFrozen=true; }
  
  int             getOffset                (const Value* val) const;
  
  // int          getOffsetFromFP       (const Value* val) const;
  
  void            dump                     () const;

private:
  inline void     incrementAutomaticVarsSize(int incr) {
    automaticVarsSize+= incr;
    staticStackSize += incr;
  }
  inline void     incrementRegSpillsSize(int incr) {
    regSpillsSize+= incr;
    staticStackSize += incr;
  }
  inline void     incrementTmpAreaSize(int incr) {
    currentTmpValuesSize += incr;
    if (maxTmpValuesSize < currentTmpValuesSize)
      {
        staticStackSize += currentTmpValuesSize - maxTmpValuesSize;
        maxTmpValuesSize = currentTmpValuesSize;
      }
  }
  inline void     resetTmpAreaSize() {
    currentTmpValuesSize = 0;
  }
  int             allocateOptionalArg      (const TargetMachine& target,
                                            const Type* type);
};

#endif
