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
#ifdef __cplusplus
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CBindingWrapping.h"
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLVMOpaqueMetadata *LLVMMetadataRef;

// These functions duplicate the LLVM*FunctionAttr functions in the stable C
// API. We cannot use the existing functions because they take 32-bit attribute
// values, and the Go bindings expose all of the LLVM attributes, some of which
// have values >= 1<<32.

void LLVMAddFunctionAttr2(LLVMValueRef Fn, uint64_t PA);
uint64_t LLVMGetFunctionAttr2(LLVMValueRef Fn);
void LLVMRemoveFunctionAttr2(LLVMValueRef Fn, uint64_t PA);

LLVMMetadataRef LLVMConstantAsMetadata(LLVMValueRef Val);

LLVMMetadataRef LLVMMDString2(LLVMContextRef C, const char *Str, unsigned SLen);
LLVMMetadataRef LLVMMDNode2(LLVMContextRef C, LLVMMetadataRef *MDs,
                            unsigned Count);
LLVMMetadataRef LLVMTemporaryMDNode(LLVMContextRef C, LLVMMetadataRef *MDs,
                                    unsigned Count);

void LLVMAddNamedMetadataOperand2(LLVMModuleRef M, const char *name,
                                  LLVMMetadataRef Val);
void LLVMSetMetadata2(LLVMValueRef Inst, unsigned KindID, LLVMMetadataRef MD);

void LLVMMetadataReplaceAllUsesWith(LLVMMetadataRef MD, LLVMMetadataRef New);

void LLVMSetCurrentDebugLocation2(LLVMBuilderRef Bref, unsigned Line,
                                  unsigned Col, LLVMMetadataRef Scope,
                                  LLVMMetadataRef InlinedAt);

void LLVMSetSubprogram(LLVMValueRef Fn, LLVMMetadataRef SP);

#ifdef __cplusplus
}

namespace llvm {

DEFINE_ISA_CONVERSION_FUNCTIONS(Metadata, LLVMMetadataRef)

inline Metadata **unwrap(LLVMMetadataRef *Vals) {
  return reinterpret_cast<Metadata**>(Vals);
}

}

#endif

#endif
