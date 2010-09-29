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
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"

using namespace lldb;
using namespace lldb_private;
using namespace clang;

TypeList::TypeList(const char *target_triple) :
    m_ast (target_triple),
    m_types ()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
TypeList::~TypeList()
{
}

//----------------------------------------------------------------------
// Add a base type to the type list
//----------------------------------------------------------------------

//struct CampareDCTypeBaton
//{
//  CampareDCTypeBaton(const std::vector<TypeSP>& _types, const Type* _search_type) :
//      types(_types),
//      search_type(_search_type)
//  {
//  }
//  const std::vector<TypeSP>& types;
//  const Type* search_type;
//};
//
//static int
//compare_dc_type (const void *key, const void *arrmem)
//{
//  const Type* search_type = ((CampareDCTypeBaton*) key)->search_type;
//  uint32_t curr_index = *(uint32_t *)arrmem;
//  const Type* curr_type = ((CampareDCTypeBaton*) key)->types[curr_index].get();
//  Type::CompareState state;
//  return Type::Compare(*search_type, *curr_type, state);
//}
//
//struct LessThanBinaryPredicate
//{
//  LessThanBinaryPredicate(const CampareDCTypeBaton& _compare_baton) :
//      compare_baton(_compare_baton)
//  {
//  }
//
//  bool operator() (uint32_t a, uint32_t b) const
//  {
//      Type::CompareState state;
//      return Type::Compare(*compare_baton.search_type, *compare_baton.types[b].get(), state) < 0;
//  }
//  const CampareDCTypeBaton& compare_baton;
//};

TypeSP
TypeList::InsertUnique(TypeSP& type_sp)
{
#if 0
//  Stream s(stdout);
//  s << "TypeList::InsertUnique for type ";
//  type_sp->Dump(s);
//  s << "Current list:\n";
//  Dump(s);

    CampareDCTypeBaton compare_baton(m_types, type_sp.get());
    uint32_t* match_index_ptr = (uint32_t*)bsearch(&compare_baton, &m_sorted_indexes[0], m_sorted_indexes.size(), sizeof(uint32_t), compare_dc_type);
    if (match_index_ptr)
    {
//      s << "returning existing type: " << (void *)m_types[*match_index_ptr].get() << "\n";
        return m_types[*match_index_ptr];
    }

    // Get the new index within the m_types array before we add the new type
    uint32_t uniqued_type_index = m_types.size();
    // Add the new shared pointer to our type by appending it to the end of the types array
    m_types.push_back(type_sp);
    // Figure out what the sorted index of this new type should be
    uint32_t fake_index = 0;
    LessThanBinaryPredicate compare_func_obj(compare_baton);
    std::vector<uint32_t>::iterator insert_pos = std::upper_bound(m_sorted_indexes.begin(), m_sorted_indexes.end(), fake_index, compare_func_obj);
    // Insert the sorted index into our sorted index array
    m_sorted_indexes.insert(insert_pos, uniqued_type_index);
#else
    // Just push each type on the back for now. We will worry about uniquing later
    m_types.push_back (type_sp);
#endif
//  s << "New list:\n";
//  Dump(s);

    return type_sp;
}

//----------------------------------------------------------------------
// Find a base type by its unique ID.
//----------------------------------------------------------------------
TypeSP
TypeList::FindType(lldb::user_id_t uid)
{
    TypeSP type_sp;
    iterator pos, end;
    for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
        if ((*pos)->GetID() == uid)
            return *pos;

    return type_sp;
}

//----------------------------------------------------------------------
// Find a type by name.
//----------------------------------------------------------------------
TypeList
TypeList::FindTypes(const ConstString &name)
{
    TypeList types(m_ast.getTargetInfo()->getTriple().getTriple().c_str());
    iterator pos, end;
    for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
        if ((*pos)->GetName() == name)
            types.InsertUnique(*pos);
    return types;
}

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

TypeSP
TypeList::GetTypeAtIndex(uint32_t idx)
{
    TypeSP type_sp;
    if (idx < m_types.size())
        type_sp = m_types[idx];
    return type_sp;
}

void
TypeList::Dump(Stream *s, bool show_context)
{
//  std::vector<uint32_t>::const_iterator pos, end;
//  for (pos = end = m_sorted_indexes.begin(), end = m_sorted_indexes.end(); pos != end; ++pos)
//  {
//      m_types[*pos]->Dump(s, show_context);
//  }

    m_ast.getASTContext()->getTranslationUnitDecl()->print(llvm::fouts(), 0);
    const size_t num_types = m_types.size();
    for (size_t i=0; i<num_types; ++i)
    {
        m_types[i]->Dump(s, show_context);
    }
//  ASTContext *ast_context = GetClangASTContext ().getASTContext();
//  if (ast_context)
//      ast_context->PrintStats();
}


ClangASTContext &
TypeList::GetClangASTContext ()
{
    return m_ast;
}

void *
TypeList::CreateClangPointerType (Type *type, bool forward_decl_is_ok)
{
    assert(type);
    return m_ast.CreatePointerType(type->GetClangType(forward_decl_is_ok));
}

void *
TypeList::CreateClangTypedefType (Type *typedef_type, Type *base_type, bool forward_decl_is_ok)
{
    assert(typedef_type && base_type);
    return m_ast.CreateTypedefType(typedef_type->GetName().AsCString(), base_type->GetClangType(forward_decl_is_ok), typedef_type->GetSymbolFile()->GetClangDeclContextForTypeUID(typedef_type->GetID()));
}

void *
TypeList::CreateClangLValueReferenceType (Type *type, bool forward_decl_is_ok)
{
    assert(type);
    return m_ast.CreateLValueReferenceType(type->GetClangType(forward_decl_is_ok));
}

void *
TypeList::CreateClangRValueReferenceType (Type *type, bool forward_decl_is_ok)
{
    assert(type);
    return m_ast.CreateRValueReferenceType (type->GetClangType(forward_decl_is_ok));
}



