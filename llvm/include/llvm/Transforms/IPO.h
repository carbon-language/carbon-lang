//===- llvm/Transforms/IPO.h - Interprocedural Transformations --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the IPO transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_H
#define LLVM_TRANSFORMS_IPO_H

class Pass;
class Function;

//===----------------------------------------------------------------------===//
// createLowerSetJmpPass - This function lowers the setjmp/longjmp intrinsics to
// invoke/unwind instructions.  This should really be part of the C/C++
// front-end, but it's so much easier to write transformations in LLVM proper.
//
Pass* createLowerSetJmpPass();

//===----------------------------------------------------------------------===//
// createConstantMergePass - This function returns a new pass that merges
// duplicate global constants together into a single constant that is shared.
// This is useful because some passes (ie TraceValues) insert a lot of string
// constants into the program, regardless of whether or not they duplicate an
// existing string.
//
Pass *createConstantMergePass();


//===----------------------------------------------------------------------===//
// createRaiseAllocationsPass - Return a new pass that transforms malloc and
// free function calls into malloc and free instructions.
//
Pass *createRaiseAllocationsPass();


//===----------------------------------------------------------------------===//
// createDeadTypeEliminationPass - Return a new pass that eliminates symbol
// table entries for types that are never used.
//
Pass *createDeadTypeEliminationPass();


//===----------------------------------------------------------------------===//
// createGlobalDCEPass - This transform is designed to eliminate unreachable
// internal globals (functions or global variables)
//
Pass *createGlobalDCEPass();


//===----------------------------------------------------------------------===//
// createFunctionExtractionPass - This pass deletes as much of the module as
// possible, except for the function specified.
//
Pass *createFunctionExtractionPass(Function *F);


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

//===----------------------------------------------------------------------===//
// createFunctionInliningPass - Return a new pass object that uses a heuristic
// to inline direct function calls to small functions.
//
Pass *createFunctionInliningPass();

//===----------------------------------------------------------------------===//
// createPruneEHPass - Return a new pass object which transforms invoke
// instructions into calls, if the callee can _not_ unwind the stack.
//
Pass *createPruneEHPass();

//===----------------------------------------------------------------------===//
// createInternalizePass - This pass loops over all of the functions in the
// input module, looking for a main function.  If a main function is found, all
// other functions are marked as internal.
//
Pass *createInternalizePass();

//===----------------------------------------------------------------------===//
// createDeadArgEliminationPass - This pass removes arguments from functions
// which are not used by the body of the function.
//
Pass *createDeadArgEliminationPass();

// DeadArgHacking pass - Same as DAE, but delete arguments of external functions
// as well.  This is definately not safe, and should only be used by bugpoint.
Pass *createDeadArgHackingPass();

//===----------------------------------------------------------------------===//
// createIPConstantPropagationPass - This pass propagates constants from call
// sites into the bodies of functions.
//
Pass *createIPConstantPropagationPass();


//===----------------------------------------------------------------------===//
// These passes are wrappers that can do a few simple structure mutation
// transformations.
//
Pass *createSwapElementsPass();
Pass *createSortElementsPass();

#endif
