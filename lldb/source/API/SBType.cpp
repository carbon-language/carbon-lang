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

using namespace lldb;
using namespace lldb_private;
using namespace clang;

SBType::SBType (lldb_private::ClangASTType type) :
m_opaque_ap(new TypeImpl(ClangASTType(type.GetASTContext(),
                                      type.GetOpaqueQualType())))
{
}

SBType::SBType (lldb::TypeSP type) :
m_opaque_ap(new TypeImpl(type))
{}

SBType::SBType (const SBType &rhs)
{
    if (rhs.m_opaque_ap.get() != NULL)
    {
        m_opaque_ap = std::auto_ptr<TypeImpl>(new TypeImpl(ClangASTType(rhs.m_opaque_ap->GetASTContext(),
                                                                          rhs.m_opaque_ap->GetOpaqueQualType())));
    }
}

SBType::SBType (clang::ASTContext *ctx, clang_type_t ty) :
m_opaque_ap(new TypeImpl(ClangASTType(ctx, ty)))
{
}

SBType::SBType() :
m_opaque_ap(NULL)
{
}

SBType::SBType (TypeImpl impl) :
m_opaque_ap(&impl)
{}

bool
SBType::operator == (const lldb::SBType &rhs) const
{
    if (IsValid() == false)
        return !rhs.IsValid();
    
    return  (rhs.m_opaque_ap->GetASTContext() == m_opaque_ap->GetASTContext())
            &&
            (rhs.m_opaque_ap->GetOpaqueQualType() == m_opaque_ap->GetOpaqueQualType());
}

bool
SBType::operator != (const lldb::SBType &rhs) const
{    
    if (IsValid() == false)
        return rhs.IsValid();

    return  (rhs.m_opaque_ap->GetASTContext() != m_opaque_ap->GetASTContext())
            ||
            (rhs.m_opaque_ap->GetOpaqueQualType() != m_opaque_ap->GetOpaqueQualType());
}


const lldb::SBType &
SBType::operator = (const lldb::SBType &rhs)
{
    if (*this != rhs)
    {
        if (!rhs.IsValid())
            m_opaque_ap.reset(NULL);
        else
            m_opaque_ap = std::auto_ptr<TypeImpl>(new TypeImpl(ClangASTType(rhs.m_opaque_ap->GetASTContext(),
                                                                            rhs.m_opaque_ap->GetOpaqueQualType())));
    }
    return *this;
}

SBType::~SBType ()
{}

lldb_private::TypeImpl &
SBType::ref ()
{
    if (m_opaque_ap.get() == NULL)
        m_opaque_ap.reset (new lldb_private::TypeImpl());
        return *m_opaque_ap;
}

const lldb_private::TypeImpl &
SBType::ref () const
{
    // "const SBAddress &addr" should already have checked "addr.IsValid()" 
    // prior to calling this function. In case you didn't we will assert
    // and die to let you know.
    assert (m_opaque_ap.get());
    return *m_opaque_ap;
}

bool
SBType::IsValid() const
{
    if (m_opaque_ap.get() == NULL)
        return false;
    
    return m_opaque_ap->IsValid();
}

size_t
SBType::GetByteSize() const
{
    if (!IsValid())
        return 0;
    
    return ClangASTType::GetTypeByteSize(m_opaque_ap->GetASTContext(), m_opaque_ap->GetOpaqueQualType());
    
}

bool
SBType::IsPointerType() const
{
    if (!IsValid())
        return false;
    
    QualType qt = QualType::getFromOpaquePtr(m_opaque_ap->GetOpaqueQualType());
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

    QualType qt = QualType::getFromOpaquePtr(m_opaque_ap->GetOpaqueQualType());
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

    return SBType(m_opaque_ap->GetASTContext(),
                  ClangASTContext::CreatePointerType(m_opaque_ap->GetASTContext(), m_opaque_ap->GetOpaqueQualType()));
}

SBType
SBType::GetPointeeType() const
{
    if (!IsValid())
        return SBType();

    QualType qt = QualType::getFromOpaquePtr(m_opaque_ap->GetOpaqueQualType());
    const clang::Type* typePtr = qt.getTypePtrOrNull();
    
    if (typePtr)
        return SBType(m_opaque_ap->GetASTContext(),typePtr->getPointeeType().getAsOpaquePtr());
    return SBType();
}

SBType
SBType::GetReferenceType() const
{
    if (!IsValid())
        return SBType();
    
    return SBType(m_opaque_ap->GetASTContext(),
                  ClangASTContext::CreateLValueReferenceType(m_opaque_ap->GetASTContext(), m_opaque_ap->GetOpaqueQualType()));
}

SBType
SBType::GetDereferencedType() const
{
    if (!IsValid())
        return SBType();

    QualType qt = QualType::getFromOpaquePtr(m_opaque_ap->GetOpaqueQualType());
    
    return SBType(m_opaque_ap->GetASTContext(),qt.getNonReferenceType().getAsOpaquePtr());
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
            base_type_qual = m_opaque_ap->GetASTContext()->CharTy;
            break;
        case eBasicTypeSignedChar:
            base_type_qual = m_opaque_ap->GetASTContext()->SignedCharTy;
            break;
        case eBasicTypeShort:
            base_type_qual = m_opaque_ap->GetASTContext()->ShortTy;
            break;
        case eBasicTypeUnsignedShort:
            base_type_qual = m_opaque_ap->GetASTContext()->UnsignedShortTy;
            break;
        case eBasicTypeInt:
            base_type_qual = m_opaque_ap->GetASTContext()->IntTy;
            break;
        case eBasicTypeUnsignedInt:
            base_type_qual = m_opaque_ap->GetASTContext()->UnsignedIntTy;
            break;
        case eBasicTypeLong:
            base_type_qual = m_opaque_ap->GetASTContext()->LongTy;
            break;
        case eBasicTypeUnsignedLong:
            base_type_qual = m_opaque_ap->GetASTContext()->UnsignedLongTy;
            break;
        case eBasicTypeBool:
            base_type_qual = m_opaque_ap->GetASTContext()->BoolTy;
            break;
        case eBasicTypeFloat:
            base_type_qual = m_opaque_ap->GetASTContext()->FloatTy;
            break;
        case eBasicTypeDouble:
            base_type_qual = m_opaque_ap->GetASTContext()->DoubleTy;
            break;
        case eBasicTypeObjCID:
            base_type_qual = m_opaque_ap->GetASTContext()->ObjCBuiltinIdTy;
            break;
        case eBasicTypeVoid:
            base_type_qual = m_opaque_ap->GetASTContext()->VoidTy;
            break;
        case eBasicTypeWChar:
            base_type_qual = m_opaque_ap->GetASTContext()->WCharTy;
            break;
        case eBasicTypeChar16:
            base_type_qual = m_opaque_ap->GetASTContext()->Char16Ty;
            break;
        case eBasicTypeChar32:
            base_type_qual = m_opaque_ap->GetASTContext()->Char32Ty;
            break;
        case eBasicTypeLongLong:
            base_type_qual = m_opaque_ap->GetASTContext()->LongLongTy;
            break;
        case eBasicTypeUnsignedLongLong:
            base_type_qual = m_opaque_ap->GetASTContext()->UnsignedLongLongTy;
            break;
        case eBasicTypeInt128:
            base_type_qual = m_opaque_ap->GetASTContext()->Int128Ty;
            break;
        case eBasicTypeUnsignedInt128:
            base_type_qual = m_opaque_ap->GetASTContext()->UnsignedInt128Ty;
            break;
        case eBasicTypeLongDouble:
            base_type_qual = m_opaque_ap->GetASTContext()->LongDoubleTy;
            break;
        case eBasicTypeFloatComplex:
            base_type_qual = m_opaque_ap->GetASTContext()->FloatComplexTy;
            break;
        case eBasicTypeDoubleComplex:
            base_type_qual = m_opaque_ap->GetASTContext()->DoubleComplexTy;
            break;
        case eBasicTypeLongDoubleComplex:
            base_type_qual = m_opaque_ap->GetASTContext()->LongDoubleComplexTy;
            break;
        case eBasicTypeObjCClass:
            base_type_qual = m_opaque_ap->GetASTContext()->ObjCBuiltinClassTy;
            break;
        case eBasicTypeObjCSel:
            base_type_qual = m_opaque_ap->GetASTContext()->ObjCBuiltinSelTy;
            break;
        default:
            return SBType();
    }
    
    return SBType(m_opaque_ap->GetASTContext(),
                  base_type_qual.getAsOpaquePtr());
}

const char*
SBType::GetName()
{
    if (!IsValid())
        return "";

    return ClangASTType::GetConstTypeName(m_opaque_ap->GetOpaqueQualType()).GetCString();
}

SBTypeList::SBTypeList() :
m_opaque_ap(new TypeListImpl())
{
}

SBTypeList::SBTypeList(const SBTypeList& rhs) :
m_opaque_ap(new TypeListImpl())
{
    for (int j = 0; j < rhs.GetSize(); j++)
        AppendType(rhs.GetTypeAtIndex(j));
}

SBTypeList&
SBTypeList::operator = (const SBTypeList& rhs)
{
    if (m_opaque_ap.get() != rhs.m_opaque_ap.get())
    {
        m_opaque_ap.reset(new TypeListImpl());
        for (int j = 0; j < rhs.GetSize(); j++)
            AppendType(rhs.GetTypeAtIndex(j));
    }
    return *this;
}

void
SBTypeList::AppendType(SBType type)
{
    if (type.IsValid())
        m_opaque_ap->AppendType(*type.m_opaque_ap.get());
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