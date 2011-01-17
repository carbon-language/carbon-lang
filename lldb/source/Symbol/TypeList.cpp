//===-- TypeList.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

// Project includes
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"

using namespace lldb;
using namespace lldb_private;
using namespace clang;

TypeList::TypeList() :
    m_types ()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
TypeList::~TypeList()
{
}

void
TypeList::Insert (TypeSP& type_sp)
{
    // Just push each type on the back for now. We will worry about uniquing later
    if (type_sp)
        m_types.insert(std::make_pair(type_sp->GetID(), type_sp));
}


bool
TypeList::InsertUnique (TypeSP& type_sp)
{
    if (type_sp)
    {
        user_id_t type_uid = type_sp->GetID();
        iterator pos, end = m_types.end();
        
        for (pos = m_types.find(type_uid); pos != end && pos->second->GetID() == type_uid; ++pos)
        {
            if (pos->second.get() == type_sp.get())
                return false;
        }
    }
    Insert (type_sp);
    return true;
}

//----------------------------------------------------------------------
// Find a base type by its unique ID.
//----------------------------------------------------------------------
TypeSP
TypeList::FindType(lldb::user_id_t uid)
{
    iterator pos = m_types.find(uid);
    if (pos != m_types.end())
        return pos->second;
    return TypeSP();
}

//----------------------------------------------------------------------
// Find a type by name.
//----------------------------------------------------------------------
//TypeList
//TypeList::FindTypes (const ConstString &name)
//{
//    // Do we ever need to make a lookup by name map? Here we are doing
//    // a linear search which isn't going to be fast.
//    TypeList types(m_ast.getTargetInfo()->getTriple().getTriple().c_str());
//    iterator pos, end;
//    for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
//        if (pos->second->GetName() == name)
//            types.Insert (pos->second);
//    return types;
//}

void
TypeList::Clear()
{
    m_types.clear();
}

uint32_t
TypeList::GetSize() const
{
    return m_types.size();
}

// GetTypeAtIndex isn't used a lot for large type lists, currently only for
// type lists that are returned for "image dump -t TYPENAME" commands and other
// simple symbol queries that grab the first result...

TypeSP
TypeList::GetTypeAtIndex(uint32_t idx)
{
    iterator pos, end;
    uint32_t i = idx;
    for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
    {
        if (i == 0)
            return pos->second;
        --i;
    }
    return TypeSP();
}

void
TypeList::Dump(Stream *s, bool show_context)
{
    for (iterator pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
    {
        pos->second->Dump(s, show_context);
    }
}

//void *
//TypeList::CreateClangPointerType (Type *type)
//{
//    assert(type);
//    return m_ast.CreatePointerType(type->GetClangForwardType());
//}
//
//void *
//TypeList::CreateClangTypedefType (Type *typedef_type, Type *base_type)
//{
//    assert(typedef_type && base_type);
//    return m_ast.CreateTypedefType (typedef_type->GetName().AsCString(), 
//                                    base_type->GetClangForwardType(), 
//                                    typedef_type->GetSymbolFile()->GetClangDeclContextForTypeUID(typedef_type->GetID()));
//}
//
//void *
//TypeList::CreateClangLValueReferenceType (Type *type)
//{
//    assert(type);
//    return m_ast.CreateLValueReferenceType(type->GetClangForwardType());
//}
//
//void *
//TypeList::CreateClangRValueReferenceType (Type *type)
//{
//    assert(type);
//    return m_ast.CreateRValueReferenceType (type->GetClangForwardType());
//}
//


