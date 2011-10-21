/*===-- llvm-c/Object.h - Object Lib C Iface --------------------*- C++ -*-===*/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is distributed under the University of Illinois Open Source      */
/* License. See LICENSE.TXT for details.                                      */
/*                                                                            */
/*===----------------------------------------------------------------------===*/
/*                                                                            */
/* This header declares the C interface to libLLVMObject.a, which             */
/* implements object file reading and writing.                                */
/*                                                                            */
/* Many exotic languages can interoperate with C code but have a harder time  */
/* with C++ due to name mangling. So in addition to C, this interface enables */
/* tools written in such languages.                                           */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_OBJECT_H
#define LLVM_C_OBJECT_H

#include "llvm-c/Core.h"
#include "llvm/Config/llvm-config.h"

#ifdef __cplusplus
#include "llvm/Object/ObjectFile.h"

extern "C" {
#endif

// Opaque type wrappers
typedef struct LLVMOpaqueObjectFile *LLVMObjectFileRef;
typedef struct LLVMOpaqueSectionIterator *LLVMSectionIteratorRef;
typedef struct LLVMOpaqueSymbolIterator *LLVMSymbolIteratorRef;

// ObjectFile creation
LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf);
void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile);

// ObjectFile Section iterators
LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile);
void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI);
LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSectionIteratorRef SI);
void LLVMMoveToNextSection(LLVMSectionIteratorRef SI);
void LLVMMoveToContainingSection(LLVMSectionIteratorRef Sect,
                                 LLVMSymbolIteratorRef Sym);

// ObjectFile Symbol iterators
LLVMSymbolIteratorRef LLVMGetSymbols(LLVMObjectFileRef ObjectFile);
void LLVMDisposeSymbolIterator(LLVMSymbolIteratorRef SI);
LLVMBool LLVMIsSymbolIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSymbolIteratorRef SI);
void LLVMMoveToNextSymbol(LLVMSymbolIteratorRef SI);

// SectionRef accessors
const char *LLVMGetSectionName(LLVMSectionIteratorRef SI);
uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI);
const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI);
uint64_t LLVMGetSectionAddress(LLVMSectionIteratorRef SI);
int LLVMGetSectionContainsSymbol(LLVMSectionIteratorRef SI,
                                 LLVMSymbolIteratorRef Sym);

// SymbolRef accessors
const char *LLVMGetSymbolName(LLVMSymbolIteratorRef SI);
uint64_t LLVMGetSymbolAddress(LLVMSymbolIteratorRef SI);
uint64_t LLVMGetSymbolOffset(LLVMSymbolIteratorRef SI);
uint64_t LLVMGetSymbolSize(LLVMSymbolIteratorRef SI);

#ifdef __cplusplus
}

namespace llvm {
  namespace object {
    inline ObjectFile *unwrap(LLVMObjectFileRef OF) {
      return reinterpret_cast<ObjectFile*>(OF);
    }

    inline LLVMObjectFileRef wrap(const ObjectFile *OF) {
      return reinterpret_cast<LLVMObjectFileRef>(const_cast<ObjectFile*>(OF));
    }

    inline section_iterator *unwrap(LLVMSectionIteratorRef SI) {
      return reinterpret_cast<section_iterator*>(SI);
    }

    inline LLVMSectionIteratorRef
    wrap(const section_iterator *SI) {
      return reinterpret_cast<LLVMSectionIteratorRef>
        (const_cast<section_iterator*>(SI));
    }

    inline symbol_iterator *unwrap(LLVMSymbolIteratorRef SI) {
      return reinterpret_cast<symbol_iterator*>(SI);
    }

    inline LLVMSymbolIteratorRef
    wrap(const symbol_iterator *SI) {
      return reinterpret_cast<LLVMSymbolIteratorRef>
        (const_cast<symbol_iterator*>(SI));
    }
  }
}

#endif /* defined(__cplusplus) */

#endif

