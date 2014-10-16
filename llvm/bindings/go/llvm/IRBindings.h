//===- IRBindings.h - Additional bindings for IR ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines additional C bindings for the IR component.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINDINGS_GO_LLVM_IRBINDINGS_H
#define LLVM_BINDINGS_GO_LLVM_IRBINDINGS_H

#include "llvm-c/Core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// These functions duplicate the LLVM*FunctionAttr functions in the stable C
// API. We cannot use the existing functions because they take 32-bit attribute
// values, and the Go bindings expose all of the LLVM attributes, some of which
// have values >= 1<<32.

void LLVMAddFunctionAttr2(LLVMValueRef Fn, uint64_t PA);
uint64_t LLVMGetFunctionAttr2(LLVMValueRef Fn);
void LLVMRemoveFunctionAttr2(LLVMValueRef Fn, uint64_t PA);

#ifdef __cplusplus
}
#endif

#endif
