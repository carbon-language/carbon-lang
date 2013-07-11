//===-- SBType.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include "clang/AST/ASTContext.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/Type.h"

using namespace lldb;
using namespace lldb_private;
using namespace clang;

SBType::SBType() :
    m_opaque_sp()
{
}

SBType::SBType (const ClangASTType &type) :
    m_opaque_sp(new TypeImpl(ClangASTType(type.GetASTContext(),
                                          type.GetOpaqueQualType())))
{
}

SBType::SBType (const lldb::TypeSP &type_sp) :
    m_opaque_sp(new TypeImpl(type_sp))
{
}

SBType::SBType (const lldb::TypeImplSP &type_impl_sp) :
    m_opaque_sp(type_impl_sp)
{
}
    

SBType::SBType (const SBType &rhs) :
    m_opaque_sp()
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
}


//SBType::SBType (TypeImpl* impl) :
//    m_opaque_ap(impl)
//{}
//
bool
SBType::operator == (SBType &rhs)
{
    if (IsValid() == false)
        return !rhs.IsValid();
    
    return  (rhs.m_opaque_sp->GetASTContext() == m_opaque_sp->GetASTContext()) &&
            (rhs.m_opaque_sp->GetOpaqueQualType() == m_opaque_sp->GetOpaqueQualType());
}

bool
SBType::operator != (SBType &rhs)
{    
    if (IsValid() == false)
        return rhs.IsValid();

    return  (rhs.m_opaque_sp->GetASTContext() != m_opaque_sp->GetASTContext()) ||
            (rhs.m_opaque_sp->GetOpaqueQualType() != m_opaque_sp->GetOpaqueQualType());
}

lldb::TypeImplSP
SBType::GetSP ()
{
    return m_opaque_sp;
}


void
SBType::SetSP (const lldb::TypeImplSP &type_impl_sp)
{
    m_opaque_sp = type_impl_sp;
}

SBType &
SBType::operator = (const SBType &rhs)
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
    return *this;
}

SBType::~SBType ()
{}

TypeImpl &
SBType::ref ()
{
    if (m_opaque_sp.get() == NULL)
        m_opaque_sp.reset (new TypeImpl());
        return *m_opaque_sp;
}

const TypeImpl &
SBType::ref () const
{
    // "const SBAddress &addr" should already have checked "addr.IsValid()" 
    // prior to calling this function. In case you didn't we will assert
    // and die to let you know.
    assert (m_opaque_sp.get());
    return *m_opaque_sp;
}

bool
SBType::IsValid() const
{
    if (m_opaque_sp.get() == NULL)
        return false;
    
    return m_opaque_sp->IsValid();
}

uint64_t
SBType::GetByteSize()
{
    if (!IsValid())
        return 0;
    
    return m_opaque_sp->GetClangASTType().GetByteSize();
    
}

bool
SBType::IsPointerType()
{
    if (!IsValid())
        return false;
    return m_opaque_sp->GetClangASTType().IsPointerType();
}

bool
SBType::IsReferenceType()
{
    if (!IsValid())
        return false;
    return m_opaque_sp->GetClangASTType().IsReferenceType();
}

SBType
SBType::GetPointerType()
{
    if (!IsValid())
        return SBType();

    return SBType(ClangASTType(m_opaque_sp->GetClangASTType().GetPointerType()));
}

SBType
SBType::GetPointeeType()
{
    if (!IsValid())
        return SBType();
    return SBType(ClangASTType(m_opaque_sp->GetClangASTType().GetPointeeType()));
}

SBType
SBType::GetReferenceType()
{
    if (!IsValid())
        return SBType();
    return SBType(ClangASTType(m_opaque_sp->GetClangASTType().GetLValueReferenceType()));
}

SBType
SBType::GetDereferencedType()
{
    if (!IsValid())
        return SBType();
    return SBType(ClangASTType(m_opaque_sp->GetClangASTType().GetNonReferenceType()));
}

bool 
SBType::IsFunctionType ()
{
    if (!IsValid())
        return false;
    return m_opaque_sp->GetClangASTType().IsFunctionType();
}

bool
SBType::IsPolymorphicClass ()
{
    if (!IsValid())
        return false;
    return m_opaque_sp->GetClangASTType().IsPolymorphicClass();
}



lldb::SBType
SBType::GetFunctionReturnType ()
{
    if (IsValid())
    {
        ClangASTType return_clang_type (m_opaque_sp->GetClangASTType().GetFunctionReturnType());
        if (return_clang_type.IsValid())
            return SBType(return_clang_type);
    }
    return lldb::SBType();
}

lldb::SBTypeList
SBType::GetFunctionArgumentTypes ()
{
    SBTypeList sb_type_list;
    if (IsValid())
    {
        QualType qual_type(QualType::getFromOpaquePtr(m_opaque_sp->GetOpaqueQualType()));
        const FunctionProtoType* func = dyn_cast<FunctionProtoType>(qual_type.getTypePtr());
        if (func)
        {
            const uint32_t num_args = func->getNumArgs();
            for (uint32_t i=0; i<num_args; ++i)
                sb_type_list.Append (SBType(ClangASTType(m_opaque_sp->GetASTContext(), func->getArgType(i).getAsOpaquePtr())));
        }
    }
    return sb_type_list;
}

lldb::SBType
SBType::GetUnqualifiedType()
{
    if (!IsValid())
        return SBType();
    return SBType(m_opaque_sp->GetClangASTType().GetFullyUnqualifiedType());
}

lldb::SBType
SBType::GetCanonicalType()
{
    if (IsValid())
        return SBType(m_opaque_sp->GetClangASTType().GetCanonicalType());
    return SBType();
}


lldb::BasicType
SBType::GetBasicType()
{
    if (IsValid())
        return m_opaque_sp->GetClangASTType().GetBasicTypeEnumeration ();
    return eBasicTypeInvalid;
}

SBType
SBType::GetBasicType(lldb::BasicType basic_type)
{
    if (IsValid())
        return SBType (ClangASTContext::GetBasicType (m_opaque_sp->GetASTContext(), basic_type));
    return SBType();
}

uint32_t
SBType::GetNumberOfDirectBaseClasses ()
{
    if (IsValid())
        return m_opaque_sp->GetClangASTType().GetNumDirectBaseClasses();
    return 0;
}

uint32_t
SBType::GetNumberOfVirtualBaseClasses ()
{
    if (IsValid())
        return m_opaque_sp->GetClangASTType().GetNumVirtualBaseClasses();
    return 0;
}

uint32_t
SBType::GetNumberOfFields ()
{
    if (IsValid())
        return m_opaque_sp->GetClangASTType().GetNumFields();
    return 0;
}

bool
SBType::GetDescription (SBStream &description, lldb::DescriptionLevel description_level)
{
    Stream &strm = description.ref();

    if (m_opaque_sp)
    {
        m_opaque_sp->GetDescription (strm, description_level);
    }
    else
        strm.PutCString ("No value");
    
    return true;
}



SBTypeMember
SBType::GetDirectBaseClassAtIndex (uint32_t idx)
{
    SBTypeMember sb_type_member;
    if (IsValid())
    {
        ClangASTType this_type (m_opaque_sp->GetClangASTType ());
        if (this_type.IsValid())
        {
            uint32_t bit_offset = 0;
            ClangASTType base_class_type (this_type.GetDirectBaseClassAtIndex(idx, &bit_offset));
            if (base_class_type.IsValid())
            {
                sb_type_member.reset (new TypeMemberImpl (TypeImplSP(new TypeImpl(base_class_type)), bit_offset));
            }
        }
    }
    return sb_type_member;

}

SBTypeMember
SBType::GetVirtualBaseClassAtIndex (uint32_t idx)
{
    SBTypeMember sb_type_member;
    if (IsValid())
    {
        ClangASTType this_type (m_opaque_sp->GetClangASTType ());
        if (this_type.IsValid())
        {
            uint32_t bit_offset = 0;
            ClangASTType base_class_type (this_type.GetVirtualBaseClassAtIndex(idx, &bit_offset));
            if (base_class_type.IsValid())
            {
                sb_type_member.reset (new TypeMemberImpl (TypeImplSP(new TypeImpl(base_class_type)), bit_offset));
            }
        }
    }
    return sb_type_member;
}

SBTypeMember
SBType::GetFieldAtIndex (uint32_t idx)
{
    SBTypeMember sb_type_member;
    if (IsValid())
    {
        ClangASTType this_type (m_opaque_sp->GetClangASTType ());
        if (this_type.IsValid())
        {
            uint64_t bit_offset = 0;
            uint32_t bitfield_bit_size = 0;
            bool is_bitfield = false;
            std::string name_sstr;
            ClangASTType field_type (this_type.GetFieldAtIndex (idx,
                                                                name_sstr,
                                                                &bit_offset,
                                                                &bitfield_bit_size,
                                                                &is_bitfield));
            if (field_type.IsValid())
            {
                ConstString name;
                if (!name_sstr.empty())
                    name.SetCString(name_sstr.c_str());
                sb_type_member.reset (new TypeMemberImpl (TypeImplSP (new TypeImpl(field_type)),
                                                          bit_offset,
                                                          name,
                                                          bitfield_bit_size,
                                                          is_bitfield));
            }
        }
    }
    return sb_type_member;
}

bool
SBType::IsTypeComplete()
{
    if (!IsValid())
        return false;    
    return m_opaque_sp->GetClangASTType().IsCompleteType();
}

const char*
SBType::GetName()
{
    if (!IsValid())
        return "";
    return m_opaque_sp->GetClangASTType().GetConstTypeName().GetCString();
}

lldb::TypeClass
SBType::GetTypeClass ()
{
    if (IsValid())
        return m_opaque_sp->GetClangASTType().GetTypeClass();
    return lldb::eTypeClassInvalid;
}

uint32_t
SBType::GetNumberOfTemplateArguments ()
{
    if (IsValid())
        return m_opaque_sp->GetClangASTType().GetNumTemplateArguments();
    return 0;
}

lldb::SBType
SBType::GetTemplateArgumentType (uint32_t idx)
{
    if (IsValid())
    {
        TemplateArgumentKind kind = eTemplateArgumentKindNull;
        ClangASTType template_arg_type = m_opaque_sp->GetClangASTType().GetTemplateArgument (idx, kind);
        if (template_arg_type.IsValid())
            return SBType(template_arg_type);
    }
    return SBType();
}


lldb::TemplateArgumentKind
SBType::GetTemplateArgumentKind (uint32_t idx)
{
    TemplateArgumentKind kind = eTemplateArgumentKindNull;
    if (IsValid())
        m_opaque_sp->GetClangASTType().GetTemplateArgument (idx, kind);
    return kind;
}


SBTypeList::SBTypeList() :
    m_opaque_ap(new TypeListImpl())
{
}

SBTypeList::SBTypeList(const SBTypeList& rhs) :
    m_opaque_ap(new TypeListImpl())
{
    for (uint32_t i = 0, rhs_size = const_cast<SBTypeList&>(rhs).GetSize(); i < rhs_size; i++)
        Append(const_cast<SBTypeList&>(rhs).GetTypeAtIndex(i));
}

bool
SBTypeList::IsValid ()
{
    return (m_opaque_ap.get() != NULL);
}

SBTypeList&
SBTypeList::operator = (const SBTypeList& rhs)
{
    if (this != &rhs)
    {
        m_opaque_ap.reset (new TypeListImpl());
        for (uint32_t i = 0, rhs_size = const_cast<SBTypeList&>(rhs).GetSize(); i < rhs_size; i++)
            Append(const_cast<SBTypeList&>(rhs).GetTypeAtIndex(i));
    }
    return *this;
}

void
SBTypeList::Append (SBType type)
{
    if (type.IsValid())
        m_opaque_ap->Append (type.m_opaque_sp);
}

SBType
SBTypeList::GetTypeAtIndex(uint32_t index)
{
    if (m_opaque_ap.get())
        return SBType(m_opaque_ap->GetTypeAtIndex(index));
    return SBType();
}

uint32_t
SBTypeList::GetSize()
{
    return m_opaque_ap->GetSize();
}

SBTypeList::~SBTypeList()
{
}

SBTypeMember::SBTypeMember() :
    m_opaque_ap()
{
}

SBTypeMember::~SBTypeMember()
{
}

SBTypeMember::SBTypeMember (const SBTypeMember& rhs) :
    m_opaque_ap()
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset(new TypeMemberImpl(rhs.ref()));
    }
}

lldb::SBTypeMember&
SBTypeMember::operator = (const lldb::SBTypeMember& rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset(new TypeMemberImpl(rhs.ref()));
    }
    return *this;
}

bool
SBTypeMember::IsValid() const
{
    return m_opaque_ap.get();
}

const char *
SBTypeMember::GetName ()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetName().GetCString();
    return NULL;
}

SBType
SBTypeMember::GetType ()
{
    SBType sb_type;
    if (m_opaque_ap.get())
    {
        sb_type.SetSP (m_opaque_ap->GetTypeImpl());
    }
    return sb_type;

}

uint64_t
SBTypeMember::GetOffsetInBytes()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetBitOffset() / 8u;
    return 0;
}

uint64_t
SBTypeMember::GetOffsetInBits()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetBitOffset();
    return 0;
}

bool
SBTypeMember::IsBitfield()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetIsBitfield();
    return false;
}

uint32_t
SBTypeMember::GetBitfieldSizeInBits()
{
    if (m_opaque_ap.get())
        return m_opaque_ap->GetBitfieldBitSize();
    return 0;
}


bool
SBTypeMember::GetDescription (lldb::SBStream &description, lldb::DescriptionLevel description_level)
{
    Stream &strm = description.ref();

    if (m_opaque_ap.get())
    {
        const uint32_t bit_offset = m_opaque_ap->GetBitOffset();
        const uint32_t byte_offset = bit_offset / 8u;
        const uint32_t byte_bit_offset = bit_offset % 8u;
        const char *name = m_opaque_ap->GetName().GetCString();
        if (byte_bit_offset)
            strm.Printf ("+%u + %u bits: (", byte_offset, byte_bit_offset);
        else
            strm.Printf ("+%u: (", byte_offset);
        
        TypeImplSP type_impl_sp (m_opaque_ap->GetTypeImpl());
        if (type_impl_sp)
            type_impl_sp->GetDescription(strm, description_level);
        
        strm.Printf (") %s", name);
        if (m_opaque_ap->GetIsBitfield())
        {
            const uint32_t bitfield_bit_size = m_opaque_ap->GetBitfieldBitSize();
            strm.Printf (" : %u", bitfield_bit_size);
        }
    }
    else
    {
        strm.PutCString ("No value");
    }
    return true;   
}


void
SBTypeMember::reset(TypeMemberImpl *type_member_impl)
{
    m_opaque_ap.reset(type_member_impl);
}

TypeMemberImpl &
SBTypeMember::ref ()
{
    if (m_opaque_ap.get() == NULL)
        m_opaque_ap.reset (new TypeMemberImpl());
    return *m_opaque_ap.get();
}

const TypeMemberImpl &
SBTypeMember::ref () const
{
    return *m_opaque_ap.get();
}
