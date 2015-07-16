//===-- ValueObjectConstResultCast.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectConstResultCast_h_
#define liblldb_ValueObjectConstResultCast_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObjectCast.h"
#include "lldb/Core/ValueObjectConstResultImpl.h"

namespace lldb_private {

class ValueObjectConstResultCast : public ValueObjectCast
{
public:
    ValueObjectConstResultCast (
        ValueObject &parent,
        const ConstString &name,
        const ClangASTType &cast_type,
        lldb::addr_t live_address = LLDB_INVALID_ADDRESS);

    virtual
    ~ValueObjectConstResultCast ();

    virtual lldb::ValueObjectSP
    Dereference (Error &error);

    virtual ValueObject *
    CreateChildAtIndex (size_t idx,
                        bool synthetic_array_member,
                        int32_t synthetic_index);

    virtual ClangASTType
    GetClangType ()
    {
        return ValueObjectCast::GetClangType();
    }

    virtual lldb::ValueObjectSP
    GetSyntheticChildAtOffset(uint32_t offset,
                              const ClangASTType& type,
                              bool can_create);

    virtual lldb::ValueObjectSP
    AddressOf (Error &error);

    virtual size_t
    GetPointeeData (DataExtractor& data,
                    uint32_t item_idx = 0,
                    uint32_t item_count = 1);

    virtual lldb::ValueObjectSP
    Cast (const ClangASTType &clang_ast_type);

protected:
    ValueObjectConstResultImpl m_impl;

private:
    friend class ValueObject;
    friend class ValueObjectConstResult;
    friend class ValueObjectConstResultImpl;

    DISALLOW_COPY_AND_ASSIGN (ValueObjectConstResultCast);
};

} // namespace lldb_private

#endif // liblldb_ValueObjectConstResultCast_h_
