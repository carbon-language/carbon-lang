//===-- SWIG Interface for SBType -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBTypeMember;

class SBType
{
public:

    SBType (void *ast = NULL, void *clang_type = NULL);
    
    SBType (const SBType &rhs);

    ~SBType ();

    bool
    IsValid();

    const char *
    GetName();

    uint64_t
    GetByteSize();

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
};

class SBTypeMember
{
public:

    SBTypeMember ();
    
    SBTypeMember (const SBTypeMember &rhs);

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
};

} // namespace lldb
