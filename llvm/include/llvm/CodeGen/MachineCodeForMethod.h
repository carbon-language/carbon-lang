//===-- llvm/CodeGen/MachineCodeForMethod.h ----------------------*- C++ -*--=//
// 
// Purpose:
//   Collect native machine code information for a method.
//   This allows target-specific information about the generated code
//   to be stored with each method.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECODEFORMETHOD_H
#define LLVM_CODEGEN_MACHINECODEFORMETHOD_H

#include "llvm/Annotation.h"
#include "Support/NonCopyable.h"
#include "Support/HashExtras.h"
#include <ext/hash_set>
class Value;
class Method;
class Constant;
class Type;
class TargetMachine;


class MachineCodeForMethod : private Annotation {
  const Method* method;
  bool          compiledAsLeaf;
  unsigned	staticStackSize;
  unsigned	automaticVarsSize;
  unsigned	regSpillsSize;
  unsigned	currentOptionalArgsSize;
  unsigned	maxOptionalArgsSize;
  unsigned	currentTmpValuesSize;
  std::hash_set<const Constant*> constantsForConstPool;
  std::hash_map<const Value*, int> offsets;
  // hash_map<const Value*, int> offsetsFromSP;
  
public:
  /*ctor*/      MachineCodeForMethod(const Method* method,
                                     const TargetMachine& target);
  
  // The next two methods are used to construct and to retrieve
  // the MachineCodeForMethod object for the given method.
  // construct() -- Allocates and initializes for a given method and target
  // get()       -- Returns a handle to the object.
  //                This should not be called before "construct()"
  //                for a given Method.
  // 
  static MachineCodeForMethod& construct(const Method *method,
                                         const TargetMachine &target);
  static void destruct(const Method *M);
  static MachineCodeForMethod& get(const Method* method);
  
  //
  // Accessors for global information about generated code for a method.
  // 
  inline bool     isCompiledAsLeafMethod() const { return compiledAsLeaf; }
  inline unsigned getStaticStackSize()     const { return staticStackSize; }
  inline unsigned getAutomaticVarsSize()   const { return automaticVarsSize; }
  inline unsigned getRegSpillsSize()       const { return regSpillsSize; }
  inline unsigned getMaxOptionalArgsSize() const { return maxOptionalArgsSize;}
  inline unsigned getCurrentOptionalArgsSize() const
                                             { return currentOptionalArgsSize;}
  inline const std::hash_set<const Constant*>&
                  getConstantPoolValues() const {return constantsForConstPool;}
  
  //
  // Modifiers used during code generation
  // 
  void            initializeFrameLayout    (const TargetMachine& target);
  
  void            addToConstantPool        (const Constant* constVal)
                                    { constantsForConstPool.insert(constVal); }
  
  inline void     markAsLeafMethod()              { compiledAsLeaf = true; }
  
  int             allocateLocalVar         (const TargetMachine& target,
                                            const Value* local,
                                            unsigned int size = 0);
  
  int             allocateSpilledValue     (const TargetMachine& target,
                                            const Type* type);
  
  int             allocateOptionalArg      (const TargetMachine& target,
                                            const Type* type);
  
  void            resetOptionalArgs        (const TargetMachine& target);
  
  int             pushTempValue            (const TargetMachine& target,
                                            unsigned int size);
  
  void            popAllTempValues         (const TargetMachine& target);
  
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
  inline void     incrementCurrentOptionalArgsSize(int incr) {
    currentOptionalArgsSize+= incr;     // stack size already includes this!
  }
};

#endif
