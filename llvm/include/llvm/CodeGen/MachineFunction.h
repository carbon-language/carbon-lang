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
#include "llvm/Annotation.h"
#include "Support/HashExtras.h"
#include "Support/hash_set"
#include "Support/ilist"

class Value;
class Function;
class Constant;
class Type;
class TargetMachine;
class Pass;

Pass *createMachineCodeConstructionPass(TargetMachine &Target);
Pass *createMachineCodeDestructionPass();

class MachineFunction : private Annotation {
  const Function *Fn;
  const TargetMachine &Target;

  // List of machine basic blocks in function
  iplist<MachineBasicBlock> BasicBlocks;

  // FIXME: State should be held elsewhere...
  hash_set<const Constant*> constantsForConstPool;
  hash_map<const Value*, int> offsets;
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
  MachineFunction(const Function *Fn, const TargetMachine& target);


  /// CalculateArgSize - Call this method to fill in the maxOptionalArgsSize &
  /// staticStackSize fields...
  ///
  void CalculateArgSize();

  /// getFunction - Return the LLVM function that this machine code represents
  ///
  const Function *getFunction() const { return Fn; }

  /// getTarget - Return the target machine this machine code is compiled with
  ///
  const TargetMachine &getTarget() const { return Target; }
  
  // The next two methods are used to construct and to retrieve
  // the MachineFunction object for the given method.
  // construct() -- Allocates and initializes for a given method and target
  // get()       -- Returns a handle to the object.
  //                This should not be called before "construct()"
  //                for a given Method.
  // 
  static MachineFunction& construct(const Function *Fn,
                                    const TargetMachine &target);
  static void destruct(const Function *F);
  static MachineFunction& get(const Function *F);

  // Provide accessors for the MachineBasicBlock list...
  typedef iplist<MachineBasicBlock> BasicBlockListType;
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  // Provide accessors for basic blocks...
  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }
 
  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  iterator                 begin()       { return BasicBlocks.begin(); }
  const_iterator           begin() const { return BasicBlocks.begin(); }
  iterator                 end  ()       { return BasicBlocks.end();   }
  const_iterator           end  () const { return BasicBlocks.end();   }

  reverse_iterator        rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator  rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator        rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator  rend  () const { return BasicBlocks.rend();   }

  unsigned                  size() const { return BasicBlocks.size(); }
  bool                     empty() const { return BasicBlocks.empty(); }
  const MachineBasicBlock &front() const { return BasicBlocks.front(); }
        MachineBasicBlock &front()       { return BasicBlocks.front(); }
  const MachineBasicBlock & back() const { return BasicBlocks.back(); }
        MachineBasicBlock & back()       { return BasicBlocks.back(); }

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
