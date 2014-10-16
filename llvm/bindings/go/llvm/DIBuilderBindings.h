//===- DIBuilderBindings.h - Bindings for DIBuilder -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines C bindings for the DIBuilder class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINDINGS_GO_LLVM_DIBUILDERBINDINGS_H
#define LLVM_BINDINGS_GO_LLVM_DIBUILDERBINDINGS_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: These bindings shouldn't be Go-specific and should eventually move to
// a (somewhat) less stable collection of C APIs for use in creating bindings of
// LLVM in other languages.

typedef struct LLVMOpaqueDIBuilder *LLVMDIBuilderRef;

LLVMDIBuilderRef LLVMNewDIBuilder(LLVMModuleRef m);

void LLVMDIBuilderDestroy(LLVMDIBuilderRef d);
void LLVMDIBuilderFinalize(LLVMDIBuilderRef d);

LLVMValueRef LLVMDIBuilderCreateCompileUnit(LLVMDIBuilderRef D,
                                            unsigned Language, const char *File,
                                            const char *Dir,
                                            const char *Producer, int Optimized,
                                            const char *Flags,
                                            unsigned RuntimeVersion);

LLVMValueRef LLVMDIBuilderCreateFile(LLVMDIBuilderRef D, const char *File,
                                     const char *Dir);

LLVMValueRef LLVMDIBuilderCreateLexicalBlock(LLVMDIBuilderRef D,
                                             LLVMValueRef Scope,
                                             LLVMValueRef File, unsigned Line,
                                             unsigned Column);

LLVMValueRef LLVMDIBuilderCreateLexicalBlockFile(LLVMDIBuilderRef D,
                                                 LLVMValueRef Scope,
                                                 LLVMValueRef File,
                                                 unsigned Discriminator);

LLVMValueRef LLVMDIBuilderCreateFunction(
    LLVMDIBuilderRef D, LLVMValueRef Scope, const char *Name,
    const char *LinkageName, LLVMValueRef File, unsigned Line,
    LLVMValueRef CompositeType, int IsLocalToUnit, int IsDefinition,
    unsigned ScopeLine, unsigned Flags, int IsOptimized, LLVMValueRef Function);

LLVMValueRef LLVMDIBuilderCreateLocalVariable(
    LLVMDIBuilderRef D, unsigned Tag, LLVMValueRef Scope, const char *Name,
    LLVMValueRef File, unsigned Line, LLVMValueRef Ty, int AlwaysPreserve,
    unsigned Flags, unsigned ArgNo);

LLVMValueRef LLVMDIBuilderCreateBasicType(LLVMDIBuilderRef D, const char *Name,
                                          uint64_t SizeInBits,
                                          uint64_t AlignInBits,
                                          unsigned Encoding);

LLVMValueRef LLVMDIBuilderCreatePointerType(LLVMDIBuilderRef D,
                                            LLVMValueRef PointeeType,
                                            uint64_t SizeInBits,
                                            uint64_t AlignInBits,
                                            const char *Name);

LLVMValueRef LLVMDIBuilderCreateSubroutineType(LLVMDIBuilderRef D,
                                               LLVMValueRef File,
                                               LLVMValueRef ParameterTypes);

LLVMValueRef LLVMDIBuilderCreateStructType(
    LLVMDIBuilderRef D, LLVMValueRef Scope, const char *Name, LLVMValueRef File,
    unsigned Line, uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags,
    LLVMValueRef DerivedFrom, LLVMValueRef ElementTypes);

LLVMValueRef LLVMDIBuilderCreateMemberType(
    LLVMDIBuilderRef D, LLVMValueRef Scope, const char *Name, LLVMValueRef File,
    unsigned Line, uint64_t SizeInBits, uint64_t AlignInBits,
    uint64_t OffsetInBits, unsigned Flags, LLVMValueRef Ty);

LLVMValueRef LLVMDIBuilderCreateArrayType(LLVMDIBuilderRef D,
                                          uint64_t SizeInBits,
                                          uint64_t AlignInBits,
                                          LLVMValueRef ElementType,
                                          LLVMValueRef Subscripts);

LLVMValueRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef D, LLVMValueRef Ty,
                                        const char *Name, LLVMValueRef File,
                                        unsigned Line, LLVMValueRef Context);

LLVMValueRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef D, int64_t Lo,
                                              int64_t Count);

LLVMValueRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef D,
                                           LLVMValueRef *Data, size_t Length);

LLVMValueRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef D,
                                               LLVMValueRef *Data,
                                               size_t Length);

LLVMValueRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Dref, int64_t *Addr,
                                           size_t Length);

LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(LLVMDIBuilderRef D,
                                             LLVMValueRef Storage,
                                             LLVMValueRef VarInfo,
                                             LLVMValueRef Expr,
                                             LLVMBasicBlockRef Block);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
