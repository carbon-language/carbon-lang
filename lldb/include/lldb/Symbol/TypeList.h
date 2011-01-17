//===-- TypeList.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TypeList_h_
#define liblldb_TypeList_h_

#include "lldb/lldb-private.h"
#include "lldb/Symbol/Type.h"
#include <map>

namespace lldb_private {

class TypeList
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    TypeList();

    virtual
    ~TypeList();

    void
    Clear();

    void
    Dump(Stream *s, bool show_context);

    lldb::TypeSP
    FindType(lldb::user_id_t uid);

    TypeList
    FindTypes(const ConstString &name);

    void
    Insert (lldb::TypeSP& type);

    bool
    InsertUnique (lldb::TypeSP& type);

    uint32_t
    GetSize() const;

    lldb::TypeSP
    GetTypeAtIndex(uint32_t idx);

private:
    typedef std::multimap<lldb::user_id_t, lldb::TypeSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    collection m_types;

    DISALLOW_COPY_AND_ASSIGN (TypeList);
};

} // namespace lldb_private

#endif  // liblldb_TypeList_h_
