//===-- ASTDumper.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ASTDumper.h"

using namespace lldb_private;
using namespace clang;

// MARK: Utility functions

static const char* SfB (bool b)
{
    return b ? "True" : "False";
}

// MARK: DeclVisitor

void ASTDumper::VisitDecl (clang::Decl *decl)
{
    m_stream.Indent();  m_stream.Printf("class : Decl\n");
    m_stream.Indent();  m_stream.Printf("getDeclKindName() : %s\n", decl->getDeclKindName());
    m_stream.Indent();  m_stream.Printf("getTranslationUnitDecl() : ");
    
    TranslationUnitDecl *translation_unit_decl = decl->getTranslationUnitDecl();
    
    if (translation_unit_decl)
    {
        if (KeepDumping() && !Visiting(translation_unit_decl))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(translation_unit_decl);
            VisitTranslationUnitDecl(translation_unit_decl);
            DidVisit(translation_unit_decl);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    
    m_stream.Indent();  m_stream.Printf("getAccess() : ");
    switch (decl->getAccess())
    {
        default:            m_stream.Printf("~\n");
        case AS_public:     m_stream.Printf("AS_public\n");
        case AS_protected:  m_stream.Printf("AS_protected\n");
        case AS_private:    m_stream.Printf("AS_private\n");
        case AS_none:       m_stream.Printf("AS_none\n");
    }
    m_stream.Indent();  m_stream.Printf("getMaxAlignment() : %d\n", decl->getMaxAlignment());
    m_stream.Indent();  m_stream.Printf("isInvalidDecl() : %s\n", SfB(decl->isInvalidDecl()));
    m_stream.Indent();  m_stream.Printf("isImplicit() : %s\n", SfB(decl->isImplicit()));
    m_stream.Indent();  m_stream.Printf("isUsed() : %s\n", SfB(decl->isUsed()));
    m_stream.Indent();  m_stream.Printf("isOutOfLine() : %s\n", SfB(decl->isOutOfLine()));
    m_stream.Indent();  m_stream.Printf("isCanonicalDecl() : %s\n", SfB(decl->isCanonicalDecl()));
    m_stream.Indent();  m_stream.Printf("hasBody() : %s\n", SfB(decl->hasBody()));
    m_stream.Indent();  m_stream.Printf("isTemplateParameter() : %s\n", SfB(decl->isTemplateParameter()));
    m_stream.Indent();  m_stream.Printf("isTemplateParameterPack() : %s\n", SfB(decl->isTemplateParameterPack()));
    m_stream.Indent();  m_stream.Printf("isParameterPack() : %s\n", SfB(decl->isParameterPack()));
    m_stream.Indent();  m_stream.Printf("isFunctionOrFunctionTemplate() : %s\n", SfB(decl->isFunctionOrFunctionTemplate()));
    m_stream.Indent();  m_stream.Printf("getFriendObjectKind() : ");
    switch (decl->getFriendObjectKind())
    {
        default:                    m_stream.Printf("~\n");                 break;
        case Decl::FOK_None:        m_stream.Printf("FOK_None\n");          break;
        case Decl::FOK_Declared:    m_stream.Printf("FOK_Declared\n");      break;
        case Decl::FOK_Undeclared:  m_stream.Printf("FOK_Undeclared\n");    break;
    }
}

void ASTDumper::VisitTranslationUnitDecl (clang::TranslationUnitDecl *translation_unit_decl)
{
    m_stream.Indent();  m_stream.Printf("class : TranslationUnitDecl\n");
    m_stream.Indent();  m_stream.Printf("getAnonymousNamespace() : ");
    
    NamespaceDecl *anonymous_namespace = translation_unit_decl->getAnonymousNamespace();
    
    if (anonymous_namespace)
    {
        if (KeepDumping() && !Visiting(anonymous_namespace))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(anonymous_namespace);
            VisitNamespaceDecl(anonymous_namespace);
            DidVisit(anonymous_namespace);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    
    VisitDecl (translation_unit_decl);
}

void ASTDumper::VisitNamedDecl (clang::NamedDecl *named_decl)
{
    m_stream.Indent();  m_stream.Printf("class : NamedDecl\n");
    m_stream.Indent();  m_stream.Printf("getNameAsString() : %s\n", named_decl->getNameAsString().c_str());
    m_stream.Indent();  m_stream.Printf("hasLinkage() : %s\n", SfB(named_decl->hasLinkage()));
    m_stream.Indent();  m_stream.Printf("isCXXClassMember() : %s\n", SfB(named_decl->isCXXClassMember()));
    m_stream.Indent();  m_stream.Printf("isCXXInstanceMember() : %s\n", SfB(named_decl->isCXXClassMember()));
    m_stream.Indent();  m_stream.Printf("getVisibility() : ");
    switch (named_decl->getVisibility())
    {
        default:                    m_stream.Printf("~\n"); break;
        case HiddenVisibility:      m_stream.Printf("HiddenVisibility\n"); break;
        case ProtectedVisibility:   m_stream.Printf("ProtectedVisibility\n"); break;
        case DefaultVisibility:     m_stream.Printf("DefaultVisibility\n"); break;
    }
    m_stream.Indent();  m_stream.Printf("getUnderlyingDecl() : ");
    
    NamedDecl *underlying_decl = named_decl->getUnderlyingDecl();
    
    if (underlying_decl)
    {
        if (KeepDumping() && !Visiting(underlying_decl))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(underlying_decl);
            ::clang::DeclVisitor<ASTDumper, void>::Visit(underlying_decl);
            DidVisit(underlying_decl);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    
    VisitDecl (named_decl);
}

void ASTDumper::VisitNamespaceDecl (clang::NamespaceDecl *namespace_decl)
{
    m_stream.Indent();  m_stream.Printf("class : NamespaceDecl\n");
    m_stream.Indent();  m_stream.Printf("isAnonymousNamespace() : %s\n", SfB(namespace_decl->isAnonymousNamespace()));
    m_stream.Indent();  m_stream.Printf("isInline() : %s\n", SfB(namespace_decl->isInline()));
    m_stream.Indent();  m_stream.Printf("isOriginalNamespace() : %s\n", SfB(namespace_decl->isOriginalNamespace()));
    
    VisitNamedDecl (namespace_decl);
}

void ASTDumper::VisitValueDecl (clang::ValueDecl *value_decl)
{
    m_stream.Indent();  m_stream.Printf("class : ValueDecl\n");
    m_stream.Indent();  m_stream.Printf("getType() : ");
    if (value_decl->getType().getTypePtrOrNull())
    {
        const clang::Type *type_ptr = value_decl->getType().getTypePtr();
        
        if (KeepDumping() && !Visiting(type_ptr))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(type_ptr);
            ::clang::TypeVisitor<ASTDumper, void>::Visit(type_ptr);
            DidVisit(type_ptr);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    
    VisitNamedDecl (value_decl);
}

void ASTDumper::VisitDeclaratorDecl (clang::DeclaratorDecl *declarator_decl)
{
    m_stream.Indent();  m_stream.Printf("class : DeclaratorDecl\n");
    VisitValueDecl (declarator_decl);
}

void ASTDumper::VisitVarDecl (clang::VarDecl *var_decl)
{
    m_stream.Indent();  m_stream.Printf("class : VarDecl\n");
    VisitDeclaratorDecl (var_decl);
}

void ASTDumper::VisitTypeDecl (clang::TypeDecl *type_decl)
{
    m_stream.Indent();  m_stream.Printf("class : TypeDecl\n");
    m_stream.Indent();  m_stream.Printf("getTypeForDecl() : ");
    
    const clang::Type *type_for_decl = type_decl->getTypeForDecl();
    
    if (type_for_decl)
    {
        if (KeepDumping() && !Visiting(type_for_decl))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(type_for_decl);
            ::clang::TypeVisitor<ASTDumper, void>::Visit(type_for_decl);
            DidVisit(type_for_decl);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    
    VisitNamedDecl (type_decl);
}

void ASTDumper::VisitTagDecl (clang::TagDecl *tag_decl)
{
    m_stream.Indent();  m_stream.Printf("class : TagDecl\n");
    m_stream.Indent();  m_stream.Printf("isDefinition() : %s\n", SfB(tag_decl->isDefinition()));
    m_stream.Indent();  m_stream.Printf("isBeingDefined() : %s\n", SfB(tag_decl->isBeingDefined()));
    m_stream.Indent();  m_stream.Printf("isEmbeddedInDeclarator() : %s\n", SfB(tag_decl->isEmbeddedInDeclarator()));
    m_stream.Indent();  m_stream.Printf("isDependentType() : %s\n", SfB(tag_decl->isDependentType()));
    m_stream.Indent();  m_stream.Printf("getDefinition() : ");
    
    TagDecl *definition = tag_decl->getDefinition();
    
    if (definition)
    {
        if (KeepDumping() && !Visiting(definition))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(definition);
            ::clang::DeclVisitor<ASTDumper, void>::Visit(tag_decl->getDefinition());
            DidVisit(definition);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    m_stream.Indent();  m_stream.Printf("getKindName() : %s\n", tag_decl->getKindName());
    
    VisitTypeDecl(tag_decl);
}

void ASTDumper::VisitRecordDecl (clang::RecordDecl *record_decl)
{
    m_stream.Indent();  m_stream.Printf("class : RecordDecl\n");
    m_stream.Indent();  m_stream.Printf("hasFlexibleArrayMember() : %s\n", SfB(record_decl->hasFlexibleArrayMember()));
    m_stream.Indent();  m_stream.Printf("isAnonymousStructOrUnion() : %s\n", SfB(record_decl->isAnonymousStructOrUnion()));
    m_stream.Indent();  m_stream.Printf("hasObjectMember() : %s\n", SfB(record_decl->hasObjectMember()));
    m_stream.Indent();  m_stream.Printf("isInjectedClassName() : %s\n", SfB(record_decl->isInjectedClassName()));
    m_stream.Indent();  m_stream.Printf("field_begin() ... field_end() : ");
    if (KeepDumping())
    {
        if (record_decl->field_empty())
        {
            m_stream.Printf("~\n");
        }
        else
        {
            m_stream.Printf("\n");
            PushIndent();
            for (RecordDecl::field_iterator iter = record_decl->field_begin(), end_iter = record_decl->field_end();
                 iter != end_iter;
                 ++iter)
            {
                m_stream.Indent();  m_stream.Printf("- field:\n");
                PushIndent();
                if (Visiting (*iter))
                {
                    m_stream.Indent();  m_stream.Printf("capped\n");
                }
                else
                {
                    WillVisit(*iter);
                    ::clang::DeclVisitor<ASTDumper, void>::Visit(*iter);
                    DidVisit(*iter);
                }
                PopIndent();
            }
            PopIndent();
        }
    }
    else
    {
        m_stream.Printf("capped\n");
    }
    
    VisitTagDecl (record_decl);
}

void ASTDumper::VisitCXXRecordDecl (clang::CXXRecordDecl *cxx_record_decl)
{
    m_stream.Indent();  m_stream.Printf("class : CXXRecordDecl\n");
    m_stream.Indent();  m_stream.Printf("isDynamicClass() : %s\n", SfB(cxx_record_decl->isDynamicClass()));
    m_stream.Indent();  m_stream.Printf("bases_begin() ... bases_end() : ");
    if (KeepDumping())
    {
        if (cxx_record_decl->bases_begin() == cxx_record_decl->bases_end())
        {
            m_stream.Printf("~\n");
        }
        else
        {
            m_stream.Printf("\n");
            PushIndent();
            for (CXXRecordDecl::base_class_iterator iter = cxx_record_decl->bases_begin(), end_iter = cxx_record_decl->bases_end();
                 iter != end_iter;
                 ++iter)
            {
                m_stream.Indent();  m_stream.Printf("- CXXBaseSpecifier:\n");
                PushIndent();
                m_stream.Indent();  m_stream.Printf("isVirtual() : %s\n", SfB(iter->isVirtual()));
                m_stream.Indent();  m_stream.Printf("isBaseOfClass() : %s\n", SfB(iter->isBaseOfClass()));
                m_stream.Indent();  m_stream.Printf("isPackExpansion() : %s\n", SfB(iter->isPackExpansion()));
                m_stream.Indent();  m_stream.Printf("getAccessSpecifier() : ");
                switch (iter->getAccessSpecifier())
                {
                    default:                    m_stream.Printf("~\n"); break;
                    case clang::AS_none:        m_stream.Printf("AS_none\n"); break;
                    case clang::AS_private:     m_stream.Printf("AS_private\n"); break;
                    case clang::AS_protected:   m_stream.Printf("AS_protected\n"); break;
                    case clang::AS_public:      m_stream.Printf("AS_public\n"); break;
                }
                m_stream.Indent();  m_stream.Printf("getType() : ");
                const clang::Type *base_type = iter->getType().getTypePtr();
                
                if (Visiting(base_type))
                {
                    m_stream.Printf("capped\n");
                }
                else
                {
                    m_stream.Printf("\n");
                    PushIndent();
                    WillVisit(base_type);
                    ::clang::TypeVisitor<ASTDumper, void>::Visit(base_type);
                    DidVisit(base_type);
                    PopIndent();
                }
                PopIndent();
            }
            PopIndent();
        }
    }
    else
    {
        m_stream.Printf("capped\n");
    }
    
    VisitRecordDecl(cxx_record_decl);
}

// MARK: TypeVisitor

void ASTDumper::VisitType (const clang::Type *type)
{
    m_stream.Indent();  m_stream.Printf("class : Type\n");
    m_stream.Indent();  m_stream.Printf("getTypeClass() : ");
    switch (type->getTypeClass())
    {
        default:    m_stream.Printf("~\n"); break;
#define TYPE(Class, Base) case clang::Type::Class: m_stream.Printf("%s\n", #Class); break;
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
    }
    m_stream.Indent();  m_stream.Printf("isFromAST() : %s\n", SfB(type->isFromAST()));
    m_stream.Indent();  m_stream.Printf("containsUnexpandedParameterPack() : %s\n", SfB(type->containsUnexpandedParameterPack()));
    m_stream.Indent();  m_stream.Printf("isCanonicalUnqualified() : %s\n", SfB(type->isCanonicalUnqualified()));
    m_stream.Indent();  m_stream.Printf("isIncompleteType() : %s\n", SfB(type->isIncompleteType()));
    m_stream.Indent();  m_stream.Printf("isObjectType() : %s\n", SfB(type->isObjectType()));
    m_stream.Indent();  m_stream.Printf("isLiteralType() : %s\n", SfB(type->isLiteralType()));
    m_stream.Indent();  m_stream.Printf("isBuiltinType() : %s\n", SfB(type->isBuiltinType()));
    m_stream.Indent();  m_stream.Printf("isPlaceholderType() : %s\n", SfB(type->isPlaceholderType()));
    m_stream.Indent();  m_stream.Printf("isScalarType() : %s\n", SfB(type->isScalarType()));
    m_stream.Indent();  m_stream.Printf("getScalarTypeKind() : ");
    if (type->isScalarType())
    {
        switch (type->getScalarTypeKind())
        {
            default:                                m_stream.Printf("~\n"); break;
            case clang::Type::STK_Pointer:          m_stream.Printf("STK_Pointer\n"); break;
            case clang::Type::STK_MemberPointer:    m_stream.Printf("STK_MemberPointer\n"); break;
            case clang::Type::STK_Bool:             m_stream.Printf("STK_Bool\n"); break;
            case clang::Type::STK_Integral:         m_stream.Printf("STK_Integral\n"); break;
            case clang::Type::STK_Floating:         m_stream.Printf("STK_Floating\n"); break;
            case clang::Type::STK_IntegralComplex:  m_stream.Printf("STK_IntegralComplex\n"); break;
            case clang::Type::STK_FloatingComplex:  m_stream.Printf("STK_FloatingComplex\n"); break;
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    // ...
}

void ASTDumper::VisitReferenceType(const clang::ReferenceType *reference_type)
{
    m_stream.Indent();  m_stream.Printf("class : ReferenceType\n");
    m_stream.Indent();  m_stream.Printf("isSpelledAsLValue() : %s\n", SfB(reference_type->isSpelledAsLValue()));
    m_stream.Indent();  m_stream.Printf("isInnerRef() : %s\n", SfB(reference_type->isInnerRef()));
    m_stream.Indent();  m_stream.Printf("getPointeeType() : ");
    
    const clang::Type *pointee_type = reference_type->getPointeeType().getTypePtrOrNull();
    
    if (pointee_type)
    {
        if (KeepDumping() && !Visiting(pointee_type))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(pointee_type);
            ::clang::TypeVisitor<ASTDumper, void>::Visit(pointee_type);
            DidVisit(pointee_type);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    VisitType(reference_type);
}

void ASTDumper::VisitLValueReferenceType(const clang::LValueReferenceType *lvalue_reference_type)
{
    m_stream.Indent();  m_stream.Printf("class : LValueReferenceType\n");
    m_stream.Indent();  m_stream.Printf("isSugared() : %s\n", SfB(lvalue_reference_type->isSugared()));
    VisitReferenceType(lvalue_reference_type);
}

void ASTDumper::VisitPointerType(const clang::PointerType *pointer_type)
{
    m_stream.Indent();  m_stream.Printf("class : PointerType\n");
    m_stream.Indent();  m_stream.Printf("getPointeeType() : ");
    
    const clang::Type *pointee_type = pointer_type->getPointeeType().getTypePtrOrNull();
    
    if (pointee_type)
    {
        if (KeepDumping() && !Visiting(pointee_type))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(pointee_type);
            ::clang::TypeVisitor<ASTDumper, void>::Visit(pointee_type);
            DidVisit(pointee_type);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    m_stream.Indent();  m_stream.Printf("isSugared() : %s\n", SfB (pointer_type->isSugared()));
    VisitType(pointer_type);
}

void ASTDumper::VisitTagType(const clang::TagType *tag_type)
{
    m_stream.Indent();  m_stream.Printf("class : TagType\n");
    m_stream.Indent();  m_stream.Printf("getDecl() : ");
    
    Decl *decl = tag_type->getDecl();
    
    if (decl)
    {
        if (KeepDumping() && !Visiting(decl))
        {
            m_stream.Printf("\n");
            
            PushIndent();
            WillVisit(decl);
            ::clang::DeclVisitor<ASTDumper, void>::Visit(decl);
            DidVisit(decl);
            PopIndent();
        }
        else
        {
            m_stream.Printf("capped\n");
        }
    }
    else
    {
        m_stream.Printf("~\n");
    }
    m_stream.Indent();  m_stream.Printf("isBeingDefined() : %s\n", SfB(tag_type->isBeingDefined()));
    VisitType(tag_type);
}

void ASTDumper::VisitRecordType(const clang::RecordType *record_type)
{
    m_stream.Indent();  m_stream.Printf("class : RecordType\n");
    m_stream.Indent();  m_stream.Printf("hasConstFields() : %s\n", SfB(record_type->hasConstFields()));
    VisitTagType(record_type);
}
