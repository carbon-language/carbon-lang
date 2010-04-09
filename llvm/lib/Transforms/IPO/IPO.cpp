//===-- Scalar.cpp --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C bindings for libLLVMIPO.a, which implements
// several transformations over the LLVM intermediate representation.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Transforms/IPO.h"
#include "llvm/PassManager.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

void LLVMAddArgumentPromotionPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createArgumentPromotionPass());
}

void LLVMAddConstantMergePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createConstantMergePass());
}

void LLVMAddDeadArgEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDeadArgEliminationPass());
}

void LLVMAddDeadTypeEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDeadTypeEliminationPass());
}

void LLVMAddFunctionAttrsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFunctionAttrsPass());
}

void LLVMAddFunctionInliningPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFunctionInliningPass());
}

void LLVMAddGlobalDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGlobalDCEPass());
}

void LLVMAddGlobalOptimizerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGlobalOptimizerPass());
}

void LLVMAddIPConstantPropagationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIPConstantPropagationPass());
}

void LLVMAddLowerSetJmpPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerSetJmpPass());
}

void LLVMAddPruneEHPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPruneEHPass());
}

void LLVMAddIPSCCPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIPSCCPPass());
}

void LLVMAddInternalizePass(LLVMPassManagerRef PM, unsigned AllButMain) {
  unwrap(PM)->add(createInternalizePass(AllButMain != 0));
}


void LLVMAddRaiseAllocationsPass(LLVMPassManagerRef PM) {
  // FIXME: Remove in LLVM 3.0.
}

void LLVMAddStripDeadPrototypesPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStripDeadPrototypesPass());
}

void LLVMAddStripSymbolsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStripSymbolsPass());
}
