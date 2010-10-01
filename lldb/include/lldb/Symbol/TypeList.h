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
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"
#include <vector>

namespace lldb_private {

class TypeList
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    TypeList(const char *target_triple = NULL);

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

    lldb::TypeSP
    InsertUnique(lldb::TypeSP& type);

    uint32_t
    GetSize() const;

    lldb::TypeSP
    GetTypeAtIndex(uint32_t idx);

    //------------------------------------------------------------------
    // Classes that inherit from TypeList can see and modify these
    //------------------------------------------------------------------
    ClangASTContext &
    GetClangASTContext ();

    void *
    CreateClangPointerType (Type *type);

    void *
    CreateClangTypedefType (Type *typedef_type, Type *base_type);

    // For C++98 references (&)
    void *
    CreateClangLValueReferenceType (Type *type);

    // For C++0x references (&&)
    void *
    CreateClangRValueReferenceType (Type *type);

private:
    typedef std::vector<lldb::TypeSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;
    ClangASTContext m_ast; ///< The type abtract syntax tree.

    collection m_types;

    DISALLOW_COPY_AND_ASSIGN (TypeList);
};

} // namespace lldb_private

#endif  // liblldb_TypeList_h_
