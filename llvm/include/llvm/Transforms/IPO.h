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

#include <vector>

namespace llvm {

class FunctionPass;
class ModulePass;
class Function;
class BasicBlock;

//===----------------------------------------------------------------------===//
//
// These functions removes symbols from functions and modules.  If OnlyDebugInfo
// is true, only debugging information is removed from the module.
//
ModulePass *createStripSymbolsPass(bool OnlyDebugInfo = false);

//===----------------------------------------------------------------------===//
/// createLowerSetJmpPass - This function lowers the setjmp/longjmp intrinsics
/// to invoke/unwind instructions.  This should really be part of the C/C++
/// front-end, but it's so much easier to write transformations in LLVM proper.
///
ModulePass* createLowerSetJmpPass();

//===----------------------------------------------------------------------===//
/// createConstantMergePass - This function returns a new pass that merges
/// duplicate global constants together into a single constant that is shared.
/// This is useful because some passes (ie TraceValues) insert a lot of string
/// constants into the program, regardless of whether or not they duplicate an
/// existing string.
///
ModulePass *createConstantMergePass();


//===----------------------------------------------------------------------===//
/// createGlobalOptimizerPass - This function returns a new pass that optimizes
/// non-address taken internal globals.
///
ModulePass *createGlobalOptimizerPass();


//===----------------------------------------------------------------------===//
/// createRaiseAllocationsPass - Return a new pass that transforms malloc and
/// free function calls into malloc and free instructions.
///
ModulePass *createRaiseAllocationsPass();


//===----------------------------------------------------------------------===//
/// createDeadTypeEliminationPass - Return a new pass that eliminates symbol
/// table entries for types that are never used.
///
ModulePass *createDeadTypeEliminationPass();


//===----------------------------------------------------------------------===//
/// createGlobalDCEPass - This transform is designed to eliminate unreachable
/// internal globals (functions or global variables)
///
ModulePass *createGlobalDCEPass();


//===----------------------------------------------------------------------===//
/// createFunctionExtractionPass - If deleteFn is true, this pass deletes as
/// the specified function. Otherwise, it deletes as much of the module as
/// possible, except for the function specified.
///
ModulePass *createFunctionExtractionPass(Function *F, bool deleteFn = false);


//===----------------------------------------------------------------------===//
/// FunctionResolvingPass - Go over the functions that are in the module and
/// look for functions that have the same name.  More often than not, there will
/// be things like:
///    void "foo"(...)
///    void "foo"(int, int)
/// because of the way things are declared in C.  If this is the case, patch
/// things up.
///
/// This is an interprocedural pass.
///
ModulePass *createFunctionResolvingPass();

//===----------------------------------------------------------------------===//
/// createFunctionInliningPass - Return a new pass object that uses a heuristic
/// to inline direct function calls to small functions.
///
ModulePass *createFunctionInliningPass();

//===----------------------------------------------------------------------===//
/// createPruneEHPass - Return a new pass object which transforms invoke
/// instructions into calls, if the callee can _not_ unwind the stack.
///
ModulePass *createPruneEHPass();

//===----------------------------------------------------------------------===//
/// createInternalizePass - This pass loops over all of the functions in the
/// input module, looking for a main function.  If a list of symbols is
/// specified with the -internalize-public-api-* command line options, those
/// symbols are internalized.  Otherwise if InternalizeEverything is set and
/// the main function is found, all other globals are marked as internal.
///
ModulePass *createInternalizePass(bool InternalizeEverything);
ModulePass *createInternalizePass(const std::vector<const char *> &exportList);

//===----------------------------------------------------------------------===//
/// createDeadArgEliminationPass - This pass removes arguments from functions
/// which are not used by the body of the function.
///
ModulePass *createDeadArgEliminationPass();

/// DeadArgHacking pass - Same as DAE, but delete arguments of external
/// functions as well.  This is definitely not safe, and should only be used by
/// bugpoint.
ModulePass *createDeadArgHackingPass();

//===----------------------------------------------------------------------===//
/// createArgumentPromotionPass - This pass promotes "by reference" arguments to
/// be passed by value.
///
ModulePass *createArgumentPromotionPass();

//===----------------------------------------------------------------------===//
/// createIPConstantPropagationPass - This pass propagates constants from call
/// sites into the bodies of functions.
///
ModulePass *createIPConstantPropagationPass();

//===----------------------------------------------------------------------===//
/// createIPSCCPPass - This pass propagates constants from call sites into the
/// bodies of functions, and keeps track of whether basic blocks are executable
/// in the process.
///
ModulePass *createIPSCCPPass();

//===----------------------------------------------------------------------===//
//
/// createLoopExtractorPass - This pass extracts all natural loops from the
/// program into a function if it can.
///
FunctionPass *createLoopExtractorPass();

/// createSingleLoopExtractorPass - This pass extracts one natural loop from the
/// program into a function if it can.  This is used by bugpoint.
///
FunctionPass *createSingleLoopExtractorPass();

// createBlockExtractorPass - This pass extracts all blocks (except those
// specified in the argument list) from the functions in the module.
//
ModulePass *createBlockExtractorPass(std::vector<BasicBlock*> &BTNE);

// createOptimizeWellKnownCallsPass - This pass optimizes specific calls to
// specific well-known (library) functions.
ModulePass *createSimplifyLibCallsPass();


// createIndMemRemPass - This pass removes potential indirect calls of
// malloc and free
ModulePass *createIndMemRemPass();

} // End llvm namespace

#endif
