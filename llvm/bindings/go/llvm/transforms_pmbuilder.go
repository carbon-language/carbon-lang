//===- transforms_pmbuilder.go - Bindings for PassManagerBuilder ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the PassManagerBuilder class.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Transforms/PassManagerBuilder.h"
*/
import "C"

type PassManagerBuilder struct {
	C C.LLVMPassManagerBuilderRef
}

func NewPassManagerBuilder() (pmb PassManagerBuilder) {
	pmb.C = C.LLVMPassManagerBuilderCreate()
	return
}

func (pmb PassManagerBuilder) SetOptLevel(level int) {
	C.LLVMPassManagerBuilderSetOptLevel(pmb.C, C.uint(level))
}

func (pmb PassManagerBuilder) SetSizeLevel(level int) {
	C.LLVMPassManagerBuilderSetSizeLevel(pmb.C, C.uint(level))
}

func (pmb PassManagerBuilder) Populate(pm PassManager) {
	C.LLVMPassManagerBuilderPopulateModulePassManager(pmb.C, pm.C)
}

func (pmb PassManagerBuilder) PopulateFunc(pm PassManager) {
	C.LLVMPassManagerBuilderPopulateFunctionPassManager(pmb.C, pm.C)
}

func (pmb PassManagerBuilder) Dispose() {
	C.LLVMPassManagerBuilderDispose(pmb.C)
}
