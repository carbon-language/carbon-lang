//===- Transforms/Instrumentation.h - Instrumentation passes ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This files defines constructor functions for instrumentation passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_H

namespace llvm {

class ModulePass;
class FunctionPass;

// Reoptimizer support pass: add instrumentation calls to back-edges of loops
FunctionPass *createLoopInstrumentationPass ();

// Reoptimizer support pass: combine multiple back-edges w/ same target into one
FunctionPass *createCombineBranchesPass();

// Reoptimizer support pass: emit table of global functions
ModulePass *createEmitFunctionTablePass ();


//===----------------------------------------------------------------------===//
// Support for inserting LLVM code to print values at basic block and function
// exits.
//

// Just trace function entry/exit
FunctionPass *createTraceValuesPassForBasicBlocks();

// Trace BB's and methods
FunctionPass *createTraceValuesPassForFunction();

} // End llvm namespace

#endif
