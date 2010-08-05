//===-- SBType.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBType_h_
#define LLDB_SBType_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBTypeMember;

class SBType
{
public:

    SBType (void *ast = NULL, void *clang_type = NULL);
    
    ~SBType ();

    bool
    IsValid();

    const char *
    GetName();

    uint64_t
    GetByteSize();

    Encoding
    GetEncoding (uint32_t &count);

    uint64_t
    GetNumberChildren (bool omit_empty_base_classes);

    bool
    GetChildAtIndex (bool omit_empty_base_classes, uint32_t idx, SBTypeMember &member);

    uint32_t
    GetChildIndexForName (bool omit_empty_base_classes, const char *name);

    bool
    IsPointerType ();

    SBType
    GetPointeeType ();

    static bool
    IsPointerType (void *opaque_type);

protected:
    void *m_ast;
    void *m_type;
};

class SBTypeMember
{
public:

    SBTypeMember ();
    
    ~SBTypeMember ();

    bool
    IsValid ();

    void
    Clear();

    bool
    IsBitfield ();
    
    size_t
    GetBitfieldWidth ();
    
    size_t
    GetBitfieldOffset ();

    size_t
    GetOffset ();

    const char *
    GetName ();

    SBType
    GetType();

    SBType
    GetParentType();

    void
    SetName (const char *name);

protected:
    friend class SBType;
        
    void *m_ast;
    void *m_parent_type;
    void *m_member_type;
    char *m_member_name;
    int32_t m_offset;
    uint32_t m_bit_size;
    uint32_t m_bit_offset;

};


} // namespace lldb

#endif // LLDB_SBType_h_
