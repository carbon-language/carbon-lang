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
    
    SBType (const SBType &rhs);

    ~SBType ();

#ifndef SWIG
    const SBType &
    operator =(const SBType &rhs);
#endif

    bool
    IsValid();

    const char *
    GetName();

    uint64_t
    GetByteSize();

#ifndef SWIG
    lldb::Encoding
    GetEncoding (uint32_t &count);
#endif

    uint64_t
    GetNumberChildren (bool omit_empty_base_classes);

    bool
    GetChildAtIndex (bool omit_empty_base_classes, uint32_t idx, SBTypeMember &member);

    uint32_t
    GetChildIndexForName (bool omit_empty_base_classes, const char *name);

    bool
    IsAPointerType ();

    SBType
    GetPointeeType ();

    static bool
    IsPointerType (void *opaque_type);

    bool
    GetDescription (lldb::SBStream &description);

protected:
    void *m_ast;
    void *m_type;
};

class SBTypeMember
{
public:

    SBTypeMember ();
    
    SBTypeMember (const SBTypeMember &rhs);

#ifndef SWIG
    const SBTypeMember&
    operator =(const SBTypeMember &rhs);
#endif

    ~SBTypeMember ();

    bool
    IsBaseClass ();

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
    bool m_is_base_class;
    bool m_is_deref_of_paremt;
};


} // namespace lldb

#endif // LLDB_SBType_h_
