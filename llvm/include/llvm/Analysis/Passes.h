//===-- llvm/Analysis/Passes.h - Constructors for analyses ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the analysis libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PASSES_H
#define LLVM_ANALYSIS_PASSES_H

namespace llvm {
  class Pass;

  //===--------------------------------------------------------------------===//
  //
  // createGlobalsModRefPass - This function creates and returns an instance of
  // the GlobalsModRef alias analysis pass.
  //
  Pass *createGlobalsModRefPass();
}

#endif
