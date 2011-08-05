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

SBType::SBType (const lldb_private::ClangASTType &type) :
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
SBType::operator == (const lldb::SBType &rhs) const
{
    if (IsValid() == false)
        return !rhs.IsValid();
    
    return  (rhs.m_opaque_sp->GetASTContext() == m_opaque_sp->GetASTContext()) &&
            (rhs.m_opaque_sp->GetOpaqueQualType() == m_opaque_sp->GetOpaqueQualType());
}

bool
SBType::operator != (const lldb::SBType &rhs) const
{    
    if (IsValid() == false)
        return rhs.IsValid();

    return  (rhs.m_opaque_sp->GetASTContext() != m_opaque_sp->GetASTContext()) ||
            (rhs.m_opaque_sp->GetOpaqueQualType() != m_opaque_sp->GetOpaqueQualType());
}


const lldb::SBType &
SBType::operator = (const lldb::SBType &rhs)
{
    if (*this != rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
    return *this;
}

SBType::~SBType ()
{}

lldb_private::TypeImpl &
SBType::ref ()
{
    if (m_opaque_sp.get() == NULL)
        m_opaque_sp.reset (new lldb_private::TypeImpl());
        return *m_opaque_sp;
}

const lldb_private::TypeImpl &
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

size_t
SBType::GetByteSize() const
{
    if (!IsValid())
        return 0;
    
    return ClangASTType::GetTypeByteSize(m_opaque_sp->GetASTContext(), m_opaque_sp->GetOpaqueQualType());
    
}

bool
SBType::IsPointerType() const
{
    if (!IsValid())
        return false;
    
    QualType qt = QualType::getFromOpaquePtr(m_opaque_sp->GetOpaqueQualType());
    const clang::Type* typePtr = qt.getTypePtrOrNull();
    
    if (typePtr)
        return typePtr->isAnyPointerType();
    return false;
}

bool
SBType::IsReferenceType() const
{
    if (!IsValid())
        return false;

    QualType qt = QualType::getFromOpaquePtr(m_opaque_sp->GetOpaqueQualType());
    const clang::Type* typePtr = qt.getTypePtrOrNull();
    
    if (typePtr)
        return typePtr->isReferenceType();
    return false;
}

SBType
SBType::GetPointerType() const
{
    if (!IsValid())
        return SBType();

    return SBType(ClangASTType(m_opaque_sp->GetASTContext(),
                               ClangASTContext::CreatePointerType(m_opaque_sp->GetASTContext(), m_opaque_sp->GetOpaqueQualType())));
}

SBType
SBType::GetPointeeType() const
{
    if (!IsValid())
        return SBType();

    QualType qt = QualType::getFromOpaquePtr(m_opaque_sp->GetOpaqueQualType());
    const clang::Type* typePtr = qt.getTypePtrOrNull();
    
    if (typePtr)
        return SBType(ClangASTType(m_opaque_sp->GetASTContext(),typePtr->getPointeeType().getAsOpaquePtr()));
    return SBType();
}

SBType
SBType::GetReferenceType() const
{
    if (!IsValid())
        return SBType();
    
    return SBType(ClangASTType(m_opaque_sp->GetASTContext(),
                               ClangASTContext::CreateLValueReferenceType(m_opaque_sp->GetASTContext(), m_opaque_sp->GetOpaqueQualType())));
}

SBType
SBType::GetDereferencedType() const
{
    if (!IsValid())
        return SBType();

    QualType qt = QualType::getFromOpaquePtr(m_opaque_sp->GetOpaqueQualType());
    
    return SBType(ClangASTType(m_opaque_sp->GetASTContext(),qt.getNonReferenceType().getAsOpaquePtr()));
}

SBType
SBType::GetBasicType(lldb::BasicType type) const
{
    
    if (!IsValid())
        return SBType();
    
    clang::CanQualType base_type_qual;
    
    switch (type)
    {
        case eBasicTypeChar:
            base_type_qual = m_opaque_sp->GetASTContext()->CharTy;
            break;
        case eBasicTypeSignedChar:
            base_type_qual = m_opaque_sp->GetASTContext()->SignedCharTy;
            break;
        case eBasicTypeShort:
            base_type_qual = m_opaque_sp->GetASTContext()->ShortTy;
            break;
        case eBasicTypeUnsignedShort:
            base_type_qual = m_opaque_sp->GetASTContext()->UnsignedShortTy;
            break;
        case eBasicTypeInt:
            base_type_qual = m_opaque_sp->GetASTContext()->IntTy;
            break;
        case eBasicTypeUnsignedInt:
            base_type_qual = m_opaque_sp->GetASTContext()->UnsignedIntTy;
            break;
        case eBasicTypeLong:
            base_type_qual = m_opaque_sp->GetASTContext()->LongTy;
            break;
        case eBasicTypeUnsignedLong:
            base_type_qual = m_opaque_sp->GetASTContext()->UnsignedLongTy;
            break;
        case eBasicTypeBool:
            base_type_qual = m_opaque_sp->GetASTContext()->BoolTy;
            break;
        case eBasicTypeFloat:
            base_type_qual = m_opaque_sp->GetASTContext()->FloatTy;
            break;
        case eBasicTypeDouble:
            base_type_qual = m_opaque_sp->GetASTContext()->DoubleTy;
            break;
        case eBasicTypeObjCID:
            base_type_qual = m_opaque_sp->GetASTContext()->ObjCBuiltinIdTy;
            break;
        case eBasicTypeVoid:
            base_type_qual = m_opaque_sp->GetASTContext()->VoidTy;
            break;
        case eBasicTypeWChar:
            base_type_qual = m_opaque_sp->GetASTContext()->WCharTy;
            break;
        case eBasicTypeChar16:
            base_type_qual = m_opaque_sp->GetASTContext()->Char16Ty;
            break;
        case eBasicTypeChar32:
            base_type_qual = m_opaque_sp->GetASTContext()->Char32Ty;
            break;
        case eBasicTypeLongLong:
            base_type_qual = m_opaque_sp->GetASTContext()->LongLongTy;
            break;
        case eBasicTypeUnsignedLongLong:
            base_type_qual = m_opaque_sp->GetASTContext()->UnsignedLongLongTy;
            break;
        case eBasicTypeInt128:
            base_type_qual = m_opaque_sp->GetASTContext()->Int128Ty;
            break;
        case eBasicTypeUnsignedInt128:
            base_type_qual = m_opaque_sp->GetASTContext()->UnsignedInt128Ty;
            break;
        case eBasicTypeLongDouble:
            base_type_qual = m_opaque_sp->GetASTContext()->LongDoubleTy;
            break;
        case eBasicTypeFloatComplex:
            base_type_qual = m_opaque_sp->GetASTContext()->FloatComplexTy;
            break;
        case eBasicTypeDoubleComplex:
            base_type_qual = m_opaque_sp->GetASTContext()->DoubleComplexTy;
            break;
        case eBasicTypeLongDoubleComplex:
            base_type_qual = m_opaque_sp->GetASTContext()->LongDoubleComplexTy;
            break;
        case eBasicTypeObjCClass:
            base_type_qual = m_opaque_sp->GetASTContext()->ObjCBuiltinClassTy;
            break;
        case eBasicTypeObjCSel:
            base_type_qual = m_opaque_sp->GetASTContext()->ObjCBuiltinSelTy;
            break;
        default:
            return SBType();
    }
    
    return SBType(ClangASTType(m_opaque_sp->GetASTContext(), base_type_qual.getAsOpaquePtr()));
}

const char*
SBType::GetName()
{
    if (!IsValid())
        return "";

    return ClangASTType::GetConstTypeName(m_opaque_sp->GetOpaqueQualType()).GetCString();
}

SBTypeList::SBTypeList() :
    m_opaque_ap(new TypeListImpl())
{
}

SBTypeList::SBTypeList(const SBTypeList& rhs) :
    m_opaque_ap(new TypeListImpl())
{
    for (uint32_t i = 0, rhs_size = rhs.GetSize(); i < rhs_size; i++)
        Append(rhs.GetTypeAtIndex(i));
}

SBTypeList&
SBTypeList::operator = (const SBTypeList& rhs)
{
    if (this != &rhs && m_opaque_ap.get() != rhs.m_opaque_ap.get())
    {
        m_opaque_ap.reset(new TypeListImpl());
        for (uint32_t i = 0, rhs_size = rhs.GetSize(); i < rhs_size; i++)
            Append(rhs.GetTypeAtIndex(i));
    }
    return *this;
}

void
SBTypeList::Append (const SBType& type)
{
    if (type.IsValid())
        m_opaque_ap->Append (type.m_opaque_sp);
}

SBType
SBTypeList::GetTypeAtIndex(int index) const
{
    return SBType(m_opaque_ap->GetTypeAtIndex(index));
}

int
SBTypeList::GetSize() const
{
    return m_opaque_ap->GetSize();
}

SBTypeList::~SBTypeList()
{
}

bool
SBType::IsPointerType (void *opaque_type)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    bool ret_value = ClangASTContext::IsPointerType (opaque_type);
    
    if (log)
        log->Printf ("SBType::IsPointerType (opaque_type=%p) ==> '%s'", opaque_type, (ret_value ? "true" : "false"));
    
    return ret_value;
}
