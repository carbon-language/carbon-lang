//===- llvm/Transforms/IPO.h - Interprocedural Optimiations -----*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_H
#define LLVM_TRANSFORMS_IPO_H

class Pass;

//===----------------------------------------------------------------------===//
// createDeadTypeEliminationPass - Return a new pass that eliminates symbol
// table entries for types that are never used.
//
Pass *createDeadTypeEliminationPass();


//===----------------------------------------------------------------------===//
// FunctionResolvingPass - Go over the functions that are in the module and
// look for functions that have the same name.  More often than not, there will
// be things like:
//    void "foo"(...)
//    void "foo"(int, int)
// because of the way things are declared in C.  If this is the case, patch
// things up.
//
// This is an interprocedural pass.
//
Pass *createFunctionResolvingPass();

#endif
