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
class Method;
class CallInst;

//===----------------------------------------------------------------------===//
// Helper functions
//

static inline bool ApplyOptToAllMethods(Module *C, bool (*Opt)(Method*)) {
  bool Modified = false;
  for (Module::MethodListType::iterator I = C->getMethodList().begin(); 
       I != C->getMethodList().end(); I++)
    Modified |= Opt(*I);
  return Modified;
}

//===----------------------------------------------------------------------===//
// Dead Code Elimination Pass
//

bool DoDeadCodeElimination(Method *M);         // DCE a method
bool DoRemoveUnusedConstants(SymTabValue *S);  // RUC a method or class
bool DoDeadCodeElimination(Module *C);         // DCE & RUC a whole class

//===----------------------------------------------------------------------===//
// Constant Propogation Pass
//

bool DoConstantPropogation(Method *M);

static inline bool DoConstantPropogation(Module *C) { 
  return ApplyOptToAllMethods(C, DoConstantPropogation); 
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
bool InlineMethod(BasicBlock::InstListType::iterator CI);// *CI must be CallInst


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

#endif
