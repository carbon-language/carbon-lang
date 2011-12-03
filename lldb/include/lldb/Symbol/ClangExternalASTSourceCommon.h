//===-- ClangExternalASTSourceCommon.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExternalASTSourceCommon_h
#define liblldb_ClangExternalASTSourceCommon_h

// Clang headers like to use NDEBUG inside of them to enable/disable debug 
// releated features using "#ifndef NDEBUG" preprocessor blocks to do one thing
// or another. This is bad because it means that if clang was built in release
// mode, it assumes that you are building in release mode which is not always
// the case. You can end up with functions that are defined as empty in header
// files when NDEBUG is not defined, and this can cause link errors with the
// clang .a files that you have since you might be missing functions in the .a
// file. So we have to define NDEBUG when including clang headers to avoid any
// mismatches. This is covered by rdar://problem/8691220

#if !defined(NDEBUG) && !defined(LLVM_NDEBUG_OFF)
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#include "clang/AST/ExternalASTSource.h"

#ifdef LLDB_DEFINED_NDEBUG_FOR_CLANG
#undef NDEBUG
#undef LLDB_DEFINED_NDEBUG_FOR_CLANG
// Need to re-include assert.h so it is as _we_ would expect it to be (enabled)
#include <assert.h>
#endif

namespace lldb_private {

class ClangExternalASTSourceCommon : public clang::ExternalASTSource 
{
public:
    ClangExternalASTSourceCommon();
    
    virtual uint64_t GetMetadata(uintptr_t object);
    virtual void SetMetadata(uintptr_t object, uint64_t metadata);
    virtual bool HasMetadata(uintptr_t object);
private:
    typedef llvm::DenseMap<uintptr_t, uint64_t> MetadataMap;
    
    MetadataMap m_metadata;
    uint64_t    m_magic;        ///< Because we don't have RTTI, we must take it
                                ///< on faith that any valid ExternalASTSource that
                                ///< we try to use the *Metadata APIs on inherits
                                ///< from ClangExternalASTSourceCommon.  This magic
                                ///< number exists to enforce that.
};
    
};

#endif
