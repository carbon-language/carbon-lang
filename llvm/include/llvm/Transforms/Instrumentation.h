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

class Pass;

//===----------------------------------------------------------------------===//
// Support for inserting LLVM code to print values at basic block and function
// exits.
//
Pass *createTraceValuesPassForFunction();     // Just trace function entry/exit
Pass *createTraceValuesPassForBasicBlocks();  // Trace BB's and methods

} // End llvm namespace

#endif
