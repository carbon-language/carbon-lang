//===-- llvm/AllOpts.h - Header file to get all opt passes -------*- C++ -*--=//
//
// This file #include's all of the small optimization header files.
//
// Note that all optimizations return true if they modified the program, false
// if not.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_ALLOPTS_H
#define LLVM_OPT_ALLOPTS_H

#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/Tools/STLExtras.h"
class Method;
class CallInst;

//===----------------------------------------------------------------------===//
// Helper functions
//

static inline bool ApplyOptToAllMethods(Module *C, bool (*Opt)(Method*)) {
  return reduce_apply_bool(C->begin(), C->end(), ptr_fun(Opt));
}

//===----------------------------------------------------------------------===//
// Dead Code Elimination Pass
//

bool DoDeadCodeElimination(Method *M);         // DCE a method
bool DoRemoveUnusedConstants(SymTabValue *S);  // RUC a method or module
bool DoDeadCodeElimination(Module *C);         // DCE & RUC a whole module

//===----------------------------------------------------------------------===//
// Constant Propogation Pass
//

bool DoConstantPropogation(Method *M);

static inline bool DoConstantPropogation(Module *C) { 
  return ApplyOptToAllMethods(C, DoConstantPropogation); 
}

//===----------------------------------------------------------------------===//
// Constant Pool Merging Pass
//
// This function merges all constants in the specified constant pool that have
// identical types and values.  This is useful for passes that generate lots of
// constants as a side effect of running.
//
bool DoConstantPoolMerging(ConstantPool &CP);
bool DoConstantPoolMerging(Method *M);
static inline bool DoConstantPoolMerging(Module *M) {
  return ApplyOptToAllMethods(M, DoConstantPoolMerging) |
         DoConstantPoolMerging(M->getConstantPool());
}


//===----------------------------------------------------------------------===//
// Sparse Conditional Constant Propogation Pass
//

bool DoSparseConditionalConstantProp(Method *M);

static inline bool DoSparseConditionalConstantProp(Module *M) {
  return ApplyOptToAllMethods(M, DoSparseConditionalConstantProp);
}

// Define a shorter version of the name...
template <class Unit> bool DoSCCP(Unit *M) { 
  return DoSparseConditionalConstantProp(M); 
}

//===----------------------------------------------------------------------===//
// Method Inlining Pass
//

// DoMethodInlining - Use a heuristic based approach to inline methods that seem
// to look good.
//
bool DoMethodInlining(Method *M);

static inline bool DoMethodInlining(Module *C) { 
  return ApplyOptToAllMethods(C, DoMethodInlining); 
}

// InlineMethod - This function forcibly inlines the called method into the
// basic block of the caller.  This returns true if it is not possible to inline
// this call.  The program is still in a well defined state if this occurs 
// though.
//
// Note that this only does one level of inlining.  For example, if the 
// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now 
// exists in the instruction stream.  Similiarly this will inline a recursive
// method by one level.
//
bool InlineMethod(CallInst *C);
bool InlineMethod(BasicBlock::iterator CI);  // *CI must be CallInst


//===----------------------------------------------------------------------===//
// Symbol Stripping Pass
//

// DoSymbolStripping - Remove all symbolic information from a method
//
bool DoSymbolStripping(Method *M);

// DoSymbolStripping - Remove all symbolic information from all methods in a 
// module
//
static inline bool DoSymbolStripping(Module *M) { 
  return ApplyOptToAllMethods(M, DoSymbolStripping); 
}

// DoFullSymbolStripping - Remove all symbolic information from all methods 
// in a module, and all module level symbols. (method names, etc...)
//
bool DoFullSymbolStripping(Module *M);


//===----------------------------------------------------------------------===//
// Induction Variable Cannonicalization
//

// DoInductionVariableCannonicalize - Simplify induction variables in loops
//
bool DoInductionVariableCannonicalize(Method *M);
static inline bool DoInductionVariableCannonicalize(Module *M) { 
  return ApplyOptToAllMethods(M, DoInductionVariableCannonicalize); 
}

#endif
