//===-- ClangASTType.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangASTType.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/AST/VTableBuilder.h"

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VerifyDecl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"

#include <iterator>
#include <mutex>

using namespace lldb;
using namespace lldb_private;

static bool
GetCompleteQualType (clang::ASTContext *ast, clang::QualType qual_type, bool allow_completion = true)
{
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::ConstantArray:
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
        {
            const clang::ArrayType *array_type = llvm::dyn_cast<clang::ArrayType>(qual_type.getTypePtr());
            
            if (array_type)
                return GetCompleteQualType (ast, array_type->getElementType(), allow_completion);
        }
            break;
            
        case clang::Type::Record:
        case clang::Type::Enum:
        {
            const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
            if (tag_type)
            {
                clang::TagDecl *tag_decl = tag_type->getDecl();
                if (tag_decl)
                {
                    if (tag_decl->isCompleteDefinition())
                        return true;
                    
                    if (!allow_completion)
                        return false;
                    
                    if (tag_decl->hasExternalLexicalStorage())
                    {
                        if (ast)
                        {
                            clang::ExternalASTSource *external_ast_source = ast->getExternalSource();
                            if (external_ast_source)
                            {
                                external_ast_source->CompleteType(tag_decl);
                                return !tag_type->isIncompleteType();
                            }
                        }
                    }
                    return false;
                }
            }
            
        }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
        {
            const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
            if (objc_class_type)
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                // We currently can't complete objective C types through the newly added ASTContext
                // because it only supports TagDecl objects right now...
                if (class_interface_decl)
                {
                    if (class_interface_decl->getDefinition())
                        return true;
                    
                    if (!allow_completion)
                        return false;
                    
                    if (class_interface_decl->hasExternalLexicalStorage())
                    {
                        if (ast)
                        {
                            clang::ExternalASTSource *external_ast_source = ast->getExternalSource();
                            if (external_ast_source)
                            {
                                external_ast_source->CompleteType (class_interface_decl);
                                return !objc_class_type->isIncompleteType();
                            }
                        }
                    }
                    return false;
                }
            }
        }
            break;
            
        case clang::Type::Typedef:
            return GetCompleteQualType (ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType(), allow_completion);
            
        case clang::Type::Elaborated:
            return GetCompleteQualType (ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType(), allow_completion);
            
        case clang::Type::Paren:
            return GetCompleteQualType (ast, llvm::cast<clang::ParenType>(qual_type)->desugar(), allow_completion);
            
        default:
            break;
    }
    
    return true;
}

static clang::ObjCIvarDecl::AccessControl
ConvertAccessTypeToObjCIvarAccessControl (AccessType access)
{
    switch (access)
    {
        case eAccessNone:      return clang::ObjCIvarDecl::None;
        case eAccessPublic:    return clang::ObjCIvarDecl::Public;
        case eAccessPrivate:   return clang::ObjCIvarDecl::Private;
        case eAccessProtected: return clang::ObjCIvarDecl::Protected;
        case eAccessPackage:   return clang::ObjCIvarDecl::Package;
    }
    return clang::ObjCIvarDecl::None;
}

//----------------------------------------------------------------------
// Tests
//----------------------------------------------------------------------

ClangASTType::ClangASTType (clang::ASTContext *ast,
                            clang::QualType qual_type) :
    m_type (qual_type.getAsOpaquePtr()),
    m_ast (ast)
{
}

ClangASTType::~ClangASTType()
{
}

//----------------------------------------------------------------------
// Tests
//----------------------------------------------------------------------

bool
ClangASTType::IsAggregateType () const
{
    if (!IsValid())
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType());
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
        case clang::Type::ConstantArray:
        case clang::Type::ExtVector:
        case clang::Type::Vector:
        case clang::Type::Record:
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            return true;
        case clang::Type::Elaborated:
            return ClangASTType(m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsAggregateType();
        case clang::Type::Typedef:
            return ClangASTType(m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsAggregateType();
        case clang::Type::Paren:
            return ClangASTType(m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).IsAggregateType();
        default:
            break;
    }
    // The clang type does have a value
    return false;
}

bool
ClangASTType::IsArrayType (ClangASTType *element_type_ptr,
                           uint64_t *size,
                           bool *is_incomplete) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            default:
                break;
                
            case clang::Type::ConstantArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, llvm::cast<clang::ConstantArrayType>(qual_type)->getElementType());
                if (size)
                    *size = llvm::cast<clang::ConstantArrayType>(qual_type)->getSize().getLimitedValue(ULLONG_MAX);
                return true;
                
            case clang::Type::IncompleteArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, llvm::cast<clang::IncompleteArrayType>(qual_type)->getElementType());
                if (size)
                    *size = 0;
                if (is_incomplete)
                    *is_incomplete = true;
                return true;
                
            case clang::Type::VariableArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, llvm::cast<clang::VariableArrayType>(qual_type)->getElementType());
                if (size)
                    *size = 0;
                return true;
                
            case clang::Type::DependentSizedArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, llvm::cast<clang::DependentSizedArrayType>(qual_type)->getElementType());
                if (size)
                    *size = 0;
                return true;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsArrayType (element_type_ptr,
                                                                                                                       size,
                                                                                                                       is_incomplete);
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsArrayType (element_type_ptr,
                                                                                                          size,
                                                                                                          is_incomplete);
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).IsArrayType (element_type_ptr,
                                                                                                       size,
                                                                                                       is_incomplete);
        }
    }
    if (element_type_ptr)
        element_type_ptr->Clear();
    if (size)
        *size = 0;
    if (is_incomplete)
        *is_incomplete = false;
    return 0;
}

bool
ClangASTType::IsVectorType (ClangASTType *element_type,
                            uint64_t *size) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Vector:
            {
                const clang::VectorType *vector_type = qual_type->getAs<clang::VectorType>();
                if (vector_type)
                {
                    if (size)
                        *size = vector_type->getNumElements();
                    if (element_type)
                        *element_type = ClangASTType(m_ast, vector_type->getElementType().getAsOpaquePtr());
                }
                return true;
            }
                break;
            case clang::Type::ExtVector:
            {
                const clang::ExtVectorType *ext_vector_type = qual_type->getAs<clang::ExtVectorType>();
                if (ext_vector_type)
                {
                    if (size)
                        *size = ext_vector_type->getNumElements();
                    if (element_type)
                        *element_type = ClangASTType(m_ast, ext_vector_type->getElementType().getAsOpaquePtr());
                }
                return true;
            }
            default:
                break;
        }
    }
    return false;
}

bool
ClangASTType::IsRuntimeGeneratedType () const
{
    if (!IsValid())
        return false;
    
    clang::DeclContext* decl_ctx = GetDeclContextForType();
    if (!decl_ctx)
        return false;

    if (!llvm::isa<clang::ObjCInterfaceDecl>(decl_ctx))
        return false;
    
    clang::ObjCInterfaceDecl *result_iface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl_ctx);
    
    ClangASTMetadata* ast_metadata = ClangASTContext::GetMetadata(m_ast, result_iface_decl);
    if (!ast_metadata)
        return false;
    return (ast_metadata->GetISAPtr() != 0);
}

bool
ClangASTType::IsCharType () const
{
    if (!IsValid())
        return false;
    return GetQualType().getUnqualifiedType()->isCharType();
}


bool
ClangASTType::IsCompleteType () const
{
    if (!IsValid())
        return false;
    const bool allow_completion = false;
    return GetCompleteQualType (m_ast, GetQualType(), allow_completion);
}

bool
ClangASTType::IsConst() const
{
    return GetQualType().isConstQualified();
}

bool
ClangASTType::IsCStringType (uint32_t &length) const
{
    ClangASTType pointee_or_element_clang_type;
    length = 0;
    Flags type_flags (GetTypeInfo (&pointee_or_element_clang_type));
    
    if (!pointee_or_element_clang_type.IsValid())
        return false;
    
    if (type_flags.AnySet (eTypeIsArray | eTypeIsPointer))
    {        
        if (pointee_or_element_clang_type.IsCharType())
        {
            if (type_flags.Test (eTypeIsArray))
            {
                // We know the size of the array and it could be a C string
                // since it is an array of characters
                length = llvm::cast<clang::ConstantArrayType>(GetCanonicalQualType().getTypePtr())->getSize().getLimitedValue();
            }
            return true;
            
        }
    }
    return false;
}

bool
ClangASTType::IsFunctionType (bool *is_variadic_ptr) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        if (qual_type->isFunctionType())
        {
            if (is_variadic_ptr)
            {
                const clang::FunctionProtoType *function_proto_type = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
                if (function_proto_type)
                    *is_variadic_ptr = function_proto_type->isVariadic();
                else
                    *is_variadic_ptr = false;
            }
            return true;
        }
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            default:
                break;
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsFunctionType();
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsFunctionType();
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).IsFunctionType();
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                    if (reference_type)
                        return ClangASTType (m_ast, reference_type->getPointeeType()).IsFunctionType();
                }
                break;
        }
    }
    return false;
}

// Used to detect "Homogeneous Floating-point Aggregates"
uint32_t
ClangASTType::IsHomogeneousAggregate (ClangASTType* base_type_ptr) const
{
    if (!IsValid())
        return 0;
    
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType ())
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    if (cxx_record_decl->getNumBases() ||
                        cxx_record_decl->isDynamicClass())
                        return 0;
                }
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                if (record_type)
                {
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        // We are looking for a structure that contains only floating point types
                        clang::RecordDecl::field_iterator field_pos, field_end = record_decl->field_end();
                        uint32_t num_fields = 0;
                        bool is_hva = false;
                        bool is_hfa = false;
                        clang::QualType base_qual_type;
                        for (field_pos = record_decl->field_begin(); field_pos != field_end; ++field_pos)
                        {
                            clang::QualType field_qual_type = field_pos->getType();
                            if (field_qual_type->isFloatingType())
                            {
                                if (field_qual_type->isComplexType())
                                    return 0;
                                else
                                {
                                    if (num_fields == 0)
                                        base_qual_type = field_qual_type;
                                    else
                                    {
                                        if (is_hva)
                                            return 0;
                                        is_hfa = true;
                                        if (field_qual_type.getTypePtr() != base_qual_type.getTypePtr())
                                            return 0;
                                    }
                                }
                            }
                            else if (field_qual_type->isVectorType() || field_qual_type->isExtVectorType())
                            {
                                const clang::VectorType *array = field_qual_type.getTypePtr()->getAs<clang::VectorType>();
                                if (array && array->getNumElements() <= 4)
                                {
                                    if (num_fields == 0)
                                        base_qual_type = array->getElementType();
                                    else
                                    {
                                        if (is_hfa)
                                            return 0;
                                        is_hva = true;
                                        if (field_qual_type.getTypePtr() != base_qual_type.getTypePtr())
                                            return 0;
                                    }
                                }
                                else
                                    return 0;
                            }
                            else
                                return 0;
                            ++num_fields;
                        }
                        if (base_type_ptr)
                            *base_type_ptr = ClangASTType (m_ast, base_qual_type);
                        return num_fields;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return ClangASTType(m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsHomogeneousAggregate (base_type_ptr);
            
        case clang::Type::Elaborated:
            return  ClangASTType(m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsHomogeneousAggregate (base_type_ptr);
        default:
            break;
    }
    return 0;
}

size_t
ClangASTType::GetNumberOfFunctionArguments () const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
            return func->getNumParams();
    }
    return 0;
}

ClangASTType
ClangASTType::GetFunctionArgumentAtIndex (const size_t index) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
        {
            if (index < func->getNumParams())
                return ClangASTType(m_ast, func->getParamType(index).getAsOpaquePtr());
        }
    }
    return ClangASTType();
}

bool
ClangASTType::IsFunctionPointerType () const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        if (qual_type->isFunctionPointerType())
            return true;
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        default:
            break;
        case clang::Type::Typedef:
            return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsFunctionPointerType();
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsFunctionPointerType();
        case clang::Type::Paren:
            return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).IsFunctionPointerType();
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            {
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                if (reference_type)
                    return ClangASTType (m_ast, reference_type->getPointeeType()).IsFunctionPointerType();
            }
            break;
        }
    }
    return false;

}

bool
ClangASTType::IsIntegerType (bool &is_signed) const
{
    if (!IsValid())
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType());
    const clang::BuiltinType *builtin_type = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal());
    
    if (builtin_type)
    {
        if (builtin_type->isInteger())
        {
            is_signed = builtin_type->isSignedInteger();
            return true;
        }
    }
    
    return false;
}

bool
ClangASTType::IsPointerType (ClangASTType *pointee_type) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
                {
                    default:
                        break;
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                        return true;
                }
                return false;
            case clang::Type::ObjCObjectPointer:
                if (pointee_type)
                    pointee_type->SetClangType (m_ast, llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetClangType (m_ast, llvm::cast<clang::BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetClangType (m_ast, llvm::cast<clang::PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetClangType (m_ast, llvm::cast<clang::MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsPointerType(pointee_type);
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsPointerType(pointee_type);
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).IsPointerType(pointee_type);
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}


bool
ClangASTType::IsPointerOrReferenceType (ClangASTType *pointee_type) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                default:
                    break;
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                    return true;
            }
                return false;
            case clang::Type::ObjCObjectPointer:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::LValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::RValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsPointerOrReferenceType(pointee_type);
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsPointerOrReferenceType(pointee_type);
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).IsPointerOrReferenceType(pointee_type);
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}


bool
ClangASTType::IsReferenceType (ClangASTType *pointee_type, bool* is_rvalue) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::LValueReferenceType>(qual_type)->desugar());
                if (is_rvalue)
                    *is_rvalue = false;
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, llvm::cast<clang::RValueReferenceType>(qual_type)->desugar());
                if (is_rvalue)
                    *is_rvalue = true;
                return true;
            case clang::Type::Typedef:
                return ClangASTType(m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsReferenceType(pointee_type, is_rvalue);
            case clang::Type::Elaborated:
                return ClangASTType(m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsReferenceType(pointee_type, is_rvalue);
            case clang::Type::Paren:
                return ClangASTType(m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).IsReferenceType(pointee_type, is_rvalue);
                
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}

bool
ClangASTType::IsFloatingPointType (uint32_t &count, bool &is_complex) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        if (const clang::BuiltinType *BT = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal()))
        {
            clang::BuiltinType::Kind kind = BT->getKind();
            if (kind >= clang::BuiltinType::Float && kind <= clang::BuiltinType::LongDouble)
            {
                count = 1;
                is_complex = false;
                return true;
            }
        }
        else if (const clang::ComplexType *CT = llvm::dyn_cast<clang::ComplexType>(qual_type->getCanonicalTypeInternal()))
        {
            if (ClangASTType (m_ast, CT->getElementType()).IsFloatingPointType (count, is_complex))
            {
                count = 2;
                is_complex = true;
                return true;
            }
        }
        else if (const clang::VectorType *VT = llvm::dyn_cast<clang::VectorType>(qual_type->getCanonicalTypeInternal()))
        {
            if (ClangASTType (m_ast, VT->getElementType()).IsFloatingPointType (count, is_complex))
            {
                count = VT->getNumElements();
                is_complex = false;
                return true;
            }
        }
    }
    count = 0;
    is_complex = false;
    return false;
}


bool
ClangASTType::IsDefined() const
{
    if (!IsValid())
        return false;

    clang::QualType qual_type(GetQualType());
    const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
    if (tag_type)
    {
        clang::TagDecl *tag_decl = tag_type->getDecl();
        if (tag_decl)
            return tag_decl->isCompleteDefinition();
        return false;
    }
    else
    {
        const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
        if (objc_class_type)
        {
            clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            if (class_interface_decl)
                return class_interface_decl->getDefinition() != nullptr;
            return false;
        }
    }
    return true;
}

bool
ClangASTType::IsObjCClassType () const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
    
        const clang::ObjCObjectPointerType *obj_pointer_type = llvm::dyn_cast<clang::ObjCObjectPointerType>(qual_type);

        if (obj_pointer_type)
            return obj_pointer_type->isObjCClassType();
    }
    return false;
}

bool
ClangASTType::IsObjCObjectOrInterfaceType () const
{
    if (IsValid())
        return GetCanonicalQualType()->isObjCObjectOrInterfaceType();
    return false;
}

bool
ClangASTType::IsPolymorphicClass () const
{
    if (IsValid())
    {
        clang::QualType qual_type(GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType())
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                        if (cxx_record_decl)
                            return cxx_record_decl->isPolymorphic();
                    }
                }
                break;
                
            default:
                break;
        }
    }
    return false;
}

bool
ClangASTType::IsPossibleDynamicType (ClangASTType *dynamic_pointee_type,
                                     bool check_cplusplus,
                                     bool check_objc) const
{
    clang::QualType pointee_qual_type;
    if (m_type)
    {
        clang::QualType qual_type (GetCanonicalQualType());
        bool success = false;
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                if (check_objc && llvm::cast<clang::BuiltinType>(qual_type)->getKind() == clang::BuiltinType::ObjCId)
                {
                    if (dynamic_pointee_type)
                        dynamic_pointee_type->SetClangType(m_ast, m_type);
                    return true;
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (check_objc)
                {
                    if (dynamic_pointee_type)
                        dynamic_pointee_type->SetClangType(m_ast, llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                    return true;
                }
                break;
                
            case clang::Type::Pointer:
                pointee_qual_type = llvm::cast<clang::PointerType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                pointee_qual_type = llvm::cast<clang::ReferenceType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast,
                                     llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsPossibleDynamicType (dynamic_pointee_type,
                                                                                                                          check_cplusplus,
                                                                                                                          check_objc);
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast,
                                     llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).IsPossibleDynamicType (dynamic_pointee_type,
                                                                                                             check_cplusplus,
                                                                                                             check_objc);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast,
                                     llvm::cast<clang::ParenType>(qual_type)->desugar()).IsPossibleDynamicType (dynamic_pointee_type,
                                                                                                   check_cplusplus,
                                                                                                   check_objc);
            default:
                break;
        }
        
        if (success)
        {
            // Check to make sure what we are pointing too is a possible dynamic C++ type
            // We currently accept any "void *" (in case we have a class that has been
            // watered down to an opaque pointer) and virtual C++ classes.
            const clang::Type::TypeClass pointee_type_class = pointee_qual_type.getCanonicalType()->getTypeClass();
            switch (pointee_type_class)
            {
                case clang::Type::Builtin:
                    switch (llvm::cast<clang::BuiltinType>(pointee_qual_type)->getKind())
                {
                    case clang::BuiltinType::UnknownAny:
                    case clang::BuiltinType::Void:
                        if (dynamic_pointee_type)
                            dynamic_pointee_type->SetClangType(m_ast, pointee_qual_type);
                        return true;
                        
                    case clang::BuiltinType::NullPtr:
                    case clang::BuiltinType::Bool:
                    case clang::BuiltinType::Char_U:
                    case clang::BuiltinType::UChar:
                    case clang::BuiltinType::WChar_U:
                    case clang::BuiltinType::Char16:
                    case clang::BuiltinType::Char32:
                    case clang::BuiltinType::UShort:
                    case clang::BuiltinType::UInt:
                    case clang::BuiltinType::ULong:
                    case clang::BuiltinType::ULongLong:
                    case clang::BuiltinType::UInt128:
                    case clang::BuiltinType::Char_S:
                    case clang::BuiltinType::SChar:
                    case clang::BuiltinType::WChar_S:
                    case clang::BuiltinType::Short:
                    case clang::BuiltinType::Int:
                    case clang::BuiltinType::Long:
                    case clang::BuiltinType::LongLong:
                    case clang::BuiltinType::Int128:
                    case clang::BuiltinType::Float:
                    case clang::BuiltinType::Double:
                    case clang::BuiltinType::LongDouble:
                    case clang::BuiltinType::Dependent:
                    case clang::BuiltinType::Overload:
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                    case clang::BuiltinType::ObjCSel:
                    case clang::BuiltinType::BoundMember:
                    case clang::BuiltinType::Half:
                    case clang::BuiltinType::ARCUnbridgedCast:
                    case clang::BuiltinType::PseudoObject:
                    case clang::BuiltinType::BuiltinFn:
                    case clang::BuiltinType::OCLEvent:
                    case clang::BuiltinType::OCLImage1d:
                    case clang::BuiltinType::OCLImage1dArray:
                    case clang::BuiltinType::OCLImage1dBuffer:
                    case clang::BuiltinType::OCLImage2d:
                    case clang::BuiltinType::OCLImage2dArray:
                    case clang::BuiltinType::OCLImage3d:
                    case clang::BuiltinType::OCLSampler:
                        break;
                }
                    break;
                    
                case clang::Type::Record:
                    if (check_cplusplus)
                    {
                        clang::CXXRecordDecl *cxx_record_decl = pointee_qual_type->getAsCXXRecordDecl();
                        if (cxx_record_decl)
                        {
                            bool is_complete = cxx_record_decl->isCompleteDefinition();
                            
                            if (is_complete)
                                success = cxx_record_decl->isDynamicClass();
                            else
                            {
                                ClangASTMetadata *metadata = ClangASTContext::GetMetadata (m_ast, cxx_record_decl);
                                if (metadata)
                                    success = metadata->GetIsDynamicCXXType();
                                else
                                {
                                    is_complete = ClangASTType(m_ast, pointee_qual_type).GetCompleteType();
                                    if (is_complete)
                                        success = cxx_record_decl->isDynamicClass();
                                    else
                                        success = false;
                                }
                            }
                            
                            if (success)
                            {
                                if (dynamic_pointee_type)
                                    dynamic_pointee_type->SetClangType(m_ast, pointee_qual_type);
                                return true;
                            }
                        }
                    }
                    break;
                    
                case clang::Type::ObjCObject:
                case clang::Type::ObjCInterface:
                    if (check_objc)
                    {
                        if (dynamic_pointee_type)
                            dynamic_pointee_type->SetClangType(m_ast, pointee_qual_type);
                        return true;
                    }
                    break;
                    
                default:
                    break;
            }
        }
    }
    if (dynamic_pointee_type)
        dynamic_pointee_type->Clear();
    return false;
}


bool
ClangASTType::IsScalarType () const
{
    if (!IsValid())
        return false;

    return (GetTypeInfo (nullptr) & eTypeIsScalar) != 0;
}

bool
ClangASTType::IsTypedefType () const
{
    if (!IsValid())
        return false;
    return GetQualType()->getTypeClass() == clang::Type::Typedef;
}

bool
ClangASTType::IsVoidType () const
{
    if (!IsValid())
        return false;
    return GetCanonicalQualType()->isVoidType();
}

bool
ClangASTType::IsPointerToScalarType () const
{
    if (!IsValid())
        return false;
    
    return IsPointerType() && GetPointeeType().IsScalarType();
}

bool
ClangASTType::IsArrayOfScalarType () const
{
    ClangASTType element_type;
    if (IsArrayType(&element_type, nullptr, nullptr))
        return element_type.IsScalarType();
    return false;
}


bool
ClangASTType::GetCXXClassName (std::string &class_name) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
        if (cxx_record_decl)
        {
            class_name.assign (cxx_record_decl->getIdentifier()->getNameStart());
            return true;
        }
    }
    class_name.clear();
    return false;
}


bool
ClangASTType::IsCXXClassType () const
{
    if (!IsValid())
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType());
    if (qual_type->getAsCXXRecordDecl() != nullptr)
        return true;
    return false;
}

bool
ClangASTType::IsBeingDefined () const
{
    if (!IsValid())
        return false;
    clang::QualType qual_type (GetCanonicalQualType());
    const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type);
    if (tag_type)
        return tag_type->isBeingDefined();
    return false;
}

bool
ClangASTType::IsObjCObjectPointerType (ClangASTType *class_type_ptr)
{
    if (!IsValid())
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType());

    if (qual_type->isObjCObjectPointerType())
    {
        if (class_type_ptr)
        {
            if (!qual_type->isObjCClassType() &&
                !qual_type->isObjCIdType())
            {
                const clang::ObjCObjectPointerType *obj_pointer_type = llvm::dyn_cast<clang::ObjCObjectPointerType>(qual_type);
                if (obj_pointer_type == nullptr)
                    class_type_ptr->Clear();
                else
                    class_type_ptr->SetClangType (m_ast, clang::QualType(obj_pointer_type->getInterfaceType(), 0));
            }
        }
        return true;
    }
    if (class_type_ptr)
        class_type_ptr->Clear();
    return false;
}

bool
ClangASTType::GetObjCClassName (std::string &class_name)
{
    if (!IsValid())
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType());

    const clang::ObjCObjectType *object_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
    if (object_type)
    {
        const clang::ObjCInterfaceDecl *interface = object_type->getInterface();
        if (interface)
        {
            class_name = interface->getNameAsString();
            return true;
        }
    }
    return false;
}


//----------------------------------------------------------------------
// Type Completion
//----------------------------------------------------------------------

bool
ClangASTType::GetCompleteType () const
{
    if (!IsValid())
        return false;
    const bool allow_completion = true;
    return GetCompleteQualType (m_ast, GetQualType(), allow_completion);
}

//----------------------------------------------------------------------
// AST related queries
//----------------------------------------------------------------------
size_t
ClangASTType::GetPointerByteSize () const
{
    if (m_ast)
        return m_ast->getTypeSize(m_ast->VoidPtrTy) / 8;
    return 0;
}

ConstString
ClangASTType::GetConstQualifiedTypeName () const
{
    return GetConstTypeName ();
}

ConstString
ClangASTType::GetConstTypeName () const
{
    if (IsValid())
    {
        ConstString type_name (GetTypeName());
        if (type_name)
            return type_name;
    }
    return ConstString("<invalid>");
}

ConstString
ClangASTType::GetTypeName () const
{
    std::string type_name;
    if (IsValid())
    {
        clang::PrintingPolicy printing_policy (m_ast->getPrintingPolicy());
        clang::QualType qual_type(GetQualType());
        printing_policy.SuppressTagKeyword = true;
        printing_policy.LangOpts.WChar = true;
        const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
        if (typedef_type)
        {
            const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
            type_name = typedef_decl->getQualifiedNameAsString();
        }
        else
        {
            type_name = qual_type.getAsString(printing_policy);
        }
    }
    return ConstString(type_name);
}

ConstString
ClangASTType::GetDisplayTypeName () const
{
    return GetTypeName();
}

uint32_t
ClangASTType::GetTypeInfo (ClangASTType *pointee_or_element_clang_type) const
{
    if (!IsValid())
        return 0;
    
    if (pointee_or_element_clang_type)
        pointee_or_element_clang_type->Clear();
    
    clang::QualType qual_type (GetQualType());
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
        {
            const clang::BuiltinType *builtin_type = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal());
            
            uint32_t builtin_type_flags = eTypeIsBuiltIn | eTypeHasValue;
            switch (builtin_type->getKind())
            {
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                    if (pointee_or_element_clang_type)
                        pointee_or_element_clang_type->SetClangType(m_ast, m_ast->ObjCBuiltinClassTy);
                    builtin_type_flags |= eTypeIsPointer | eTypeIsObjC;
                    break;
                    
                case clang::BuiltinType::ObjCSel:
                    if (pointee_or_element_clang_type)
                        pointee_or_element_clang_type->SetClangType(m_ast, m_ast->CharTy);
                    builtin_type_flags |= eTypeIsPointer | eTypeIsObjC;
                    break;
                    
                case clang::BuiltinType::Bool:
                case clang::BuiltinType::Char_U:
                case clang::BuiltinType::UChar:
                case clang::BuiltinType::WChar_U:
                case clang::BuiltinType::Char16:
                case clang::BuiltinType::Char32:
                case clang::BuiltinType::UShort:
                case clang::BuiltinType::UInt:
                case clang::BuiltinType::ULong:
                case clang::BuiltinType::ULongLong:
                case clang::BuiltinType::UInt128:
                case clang::BuiltinType::Char_S:
                case clang::BuiltinType::SChar:
                case clang::BuiltinType::WChar_S:
                case clang::BuiltinType::Short:
                case clang::BuiltinType::Int:
                case clang::BuiltinType::Long:
                case clang::BuiltinType::LongLong:
                case clang::BuiltinType::Int128:
                case clang::BuiltinType::Float:
                case clang::BuiltinType::Double:
                case clang::BuiltinType::LongDouble:
                    builtin_type_flags |= eTypeIsScalar;
                    if (builtin_type->isInteger())
                    {
                        builtin_type_flags |= eTypeIsInteger;
                        if (builtin_type->isSignedInteger())
                            builtin_type_flags |= eTypeIsSigned;
                    }
                    else if (builtin_type->isFloatingPoint())
                        builtin_type_flags |= eTypeIsFloat;
                    break;
                default:
                    break;
            }
            return builtin_type_flags;
        }
            
        case clang::Type::BlockPointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, qual_type->getPointeeType());
            return eTypeIsPointer | eTypeHasChildren | eTypeIsBlock;
            
        case clang::Type::Complex:
        {
            uint32_t complex_type_flags = eTypeIsBuiltIn | eTypeHasValue | eTypeIsComplex;
            const clang::ComplexType *complex_type = llvm::dyn_cast<clang::ComplexType>(qual_type->getCanonicalTypeInternal());
            if (complex_type)
            {
                clang::QualType complex_element_type (complex_type->getElementType());
                if (complex_element_type->isIntegerType())
                    complex_type_flags |= eTypeIsFloat;
                else if (complex_element_type->isFloatingType())
                    complex_type_flags |= eTypeIsInteger;
            }
            return complex_type_flags;
        }
            break;
            
        case clang::Type::ConstantArray:
        case clang::Type::DependentSizedArray:
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, llvm::cast<clang::ArrayType>(qual_type.getTypePtr())->getElementType());
            return eTypeHasChildren | eTypeIsArray;
            
        case clang::Type::DependentName:                    return 0;
        case clang::Type::DependentSizedExtVector:          return eTypeHasChildren | eTypeIsVector;
        case clang::Type::DependentTemplateSpecialization:  return eTypeIsTemplate;
        case clang::Type::Decltype:                         return 0;
            
        case clang::Type::Enum:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, llvm::cast<clang::EnumType>(qual_type)->getDecl()->getIntegerType());
            return eTypeIsEnumeration | eTypeHasValue;
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::Paren:
            return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetTypeInfo (pointee_or_element_clang_type);
            
        case clang::Type::FunctionProto:                    return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::FunctionNoProto:                  return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::InjectedClassName:                return 0;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, llvm::cast<clang::ReferenceType>(qual_type.getTypePtr())->getPointeeType());
            return eTypeHasChildren | eTypeIsReference | eTypeHasValue;
            
        case clang::Type::MemberPointer:                    return eTypeIsPointer   | eTypeIsMember | eTypeHasValue;
            
        case clang::Type::ObjCObjectPointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, qual_type->getPointeeType());
            return eTypeHasChildren | eTypeIsObjC | eTypeIsClass | eTypeIsPointer | eTypeHasValue;
            
        case clang::Type::ObjCObject:                       return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;
        case clang::Type::ObjCInterface:                    return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;
            
        case clang::Type::Pointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, qual_type->getPointeeType());
            return eTypeHasChildren | eTypeIsPointer | eTypeHasValue;
            
        case clang::Type::Record:
            if (qual_type->getAsCXXRecordDecl())
                return eTypeHasChildren | eTypeIsClass | eTypeIsCPlusPlus;
            else
                return eTypeHasChildren | eTypeIsStructUnion;
            break;
        case clang::Type::SubstTemplateTypeParm:            return eTypeIsTemplate;
        case clang::Type::TemplateTypeParm:                 return eTypeIsTemplate;
        case clang::Type::TemplateSpecialization:           return eTypeIsTemplate;
            
        case clang::Type::Typedef:
            return eTypeIsTypedef | ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::TypeOfExpr:                       return 0;
        case clang::Type::TypeOf:                           return 0;
        case clang::Type::UnresolvedUsing:                  return 0;
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
        {
            uint32_t vector_type_flags = eTypeHasChildren | eTypeIsVector;
            const clang::VectorType *vector_type = llvm::dyn_cast<clang::VectorType>(qual_type->getCanonicalTypeInternal());
            if (vector_type)
            {
                if (vector_type->isIntegerType())
                    vector_type_flags |= eTypeIsFloat;
                else if (vector_type->isFloatingType())
                    vector_type_flags |= eTypeIsInteger;
            }
            return vector_type_flags;
        }
        default:                                            return 0;
    }
    return 0;
}



lldb::LanguageType
ClangASTType::GetMinimumLanguage ()
{
    if (!IsValid())
        return lldb::eLanguageTypeC;
    
    // If the type is a reference, then resolve it to what it refers to first:
    clang::QualType qual_type (GetCanonicalQualType().getNonReferenceType());
    if (qual_type->isAnyPointerType())
    {
        if (qual_type->isObjCObjectPointerType())
            return lldb::eLanguageTypeObjC;
        
        clang::QualType pointee_type (qual_type->getPointeeType());
        if (pointee_type->getPointeeCXXRecordDecl() != nullptr)
            return lldb::eLanguageTypeC_plus_plus;
        if (pointee_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type->isObjCClassType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type.getTypePtr() == m_ast->ObjCBuiltinIdTy.getTypePtr())
            return lldb::eLanguageTypeObjC;
    }
    else
    {
        if (qual_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (qual_type->getAsCXXRecordDecl())
            return lldb::eLanguageTypeC_plus_plus;
        switch (qual_type->getTypeClass())
        {
            default:
                break;
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                default:
                case clang::BuiltinType::Void:
                case clang::BuiltinType::Bool:
                case clang::BuiltinType::Char_U:
                case clang::BuiltinType::UChar:
                case clang::BuiltinType::WChar_U:
                case clang::BuiltinType::Char16:
                case clang::BuiltinType::Char32:
                case clang::BuiltinType::UShort:
                case clang::BuiltinType::UInt:
                case clang::BuiltinType::ULong:
                case clang::BuiltinType::ULongLong:
                case clang::BuiltinType::UInt128:
                case clang::BuiltinType::Char_S:
                case clang::BuiltinType::SChar:
                case clang::BuiltinType::WChar_S:
                case clang::BuiltinType::Short:
                case clang::BuiltinType::Int:
                case clang::BuiltinType::Long:
                case clang::BuiltinType::LongLong:
                case clang::BuiltinType::Int128:
                case clang::BuiltinType::Float:
                case clang::BuiltinType::Double:
                case clang::BuiltinType::LongDouble:
                    break;
                    
                case clang::BuiltinType::NullPtr:
                    return eLanguageTypeC_plus_plus;
                    
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                case clang::BuiltinType::ObjCSel:
                    return eLanguageTypeObjC;
                    
                case clang::BuiltinType::Dependent:
                case clang::BuiltinType::Overload:
                case clang::BuiltinType::BoundMember:
                case clang::BuiltinType::UnknownAny:
                    break;
            }
                break;
            case clang::Type::Typedef:
                return ClangASTType(m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetMinimumLanguage();
        }
    }
    return lldb::eLanguageTypeC;
}

lldb::TypeClass
ClangASTType::GetTypeClass () const
{
    if (!IsValid())
        return lldb::eTypeClassInvalid;
    
    clang::QualType qual_type(GetQualType());
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:           break;
        case clang::Type::FunctionNoProto:          return lldb::eTypeClassFunction;
        case clang::Type::FunctionProto:            return lldb::eTypeClassFunction;
        case clang::Type::IncompleteArray:          return lldb::eTypeClassArray;
        case clang::Type::VariableArray:            return lldb::eTypeClassArray;
        case clang::Type::ConstantArray:            return lldb::eTypeClassArray;
        case clang::Type::DependentSizedArray:      return lldb::eTypeClassArray;
        case clang::Type::DependentSizedExtVector:  return lldb::eTypeClassVector;
        case clang::Type::ExtVector:                return lldb::eTypeClassVector;
        case clang::Type::Vector:                   return lldb::eTypeClassVector;
        case clang::Type::Builtin:                  return lldb::eTypeClassBuiltin;
        case clang::Type::ObjCObjectPointer:        return lldb::eTypeClassObjCObjectPointer;
        case clang::Type::BlockPointer:             return lldb::eTypeClassBlockPointer;
        case clang::Type::Pointer:                  return lldb::eTypeClassPointer;
        case clang::Type::LValueReference:          return lldb::eTypeClassReference;
        case clang::Type::RValueReference:          return lldb::eTypeClassReference;
        case clang::Type::MemberPointer:            return lldb::eTypeClassMemberPointer;
        case clang::Type::Complex:
            if (qual_type->isComplexType())
                return lldb::eTypeClassComplexFloat;
            else
                return lldb::eTypeClassComplexInteger;
        case clang::Type::ObjCObject:               return lldb::eTypeClassObjCObject;
        case clang::Type::ObjCInterface:            return lldb::eTypeClassObjCInterface;
        case clang::Type::Record:
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                if (record_decl->isUnion())
                    return lldb::eTypeClassUnion;
                else if (record_decl->isStruct())
                    return lldb::eTypeClassStruct;
                else
                    return lldb::eTypeClassClass;
            }
            break;
        case clang::Type::Enum:                     return lldb::eTypeClassEnumeration;
        case clang::Type::Typedef:                  return lldb::eTypeClassTypedef;
        case clang::Type::UnresolvedUsing:          break;
        case clang::Type::Paren:
            return ClangASTType(m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetTypeClass();
        case clang::Type::Elaborated:
            return ClangASTType(m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetTypeClass();
            
        case clang::Type::Attributed:               break;
        case clang::Type::TemplateTypeParm:         break;
        case clang::Type::SubstTemplateTypeParm:    break;
        case clang::Type::SubstTemplateTypeParmPack:break;
        case clang::Type::Auto:                     break;
        case clang::Type::InjectedClassName:        break;
        case clang::Type::DependentName:            break;
        case clang::Type::DependentTemplateSpecialization: break;
        case clang::Type::PackExpansion:            break;
            
        case clang::Type::TypeOfExpr:               break;
        case clang::Type::TypeOf:                   break;
        case clang::Type::Decltype:                 break;
        case clang::Type::TemplateSpecialization:   break;
        case clang::Type::Atomic:                   break;

        // pointer type decayed from an array or function type.
        case clang::Type::Decayed:                  break;
        case clang::Type::Adjusted:                 break;
    }
    // We don't know hot to display this type...
    return lldb::eTypeClassOther;
    
}

void
ClangASTType::SetClangType (clang::ASTContext *ast, clang::QualType qual_type)
{
    m_ast = ast;
    m_type = qual_type.getAsOpaquePtr();
}

unsigned
ClangASTType::GetTypeQualifiers() const
{
    if (IsValid())
        return GetQualType().getQualifiers().getCVRQualifiers();
    return 0;
}

//----------------------------------------------------------------------
// Creating related types
//----------------------------------------------------------------------

ClangASTType
ClangASTType::AddConstModifier () const
{
    if (m_type)
    {
        clang::QualType result(GetQualType());
        result.addConst();
        return ClangASTType (m_ast, result);
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::AddRestrictModifier () const
{
    if (m_type)
    {
        clang::QualType result(GetQualType());
        result.getQualifiers().setRestrict (true);
        return ClangASTType (m_ast, result);
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::AddVolatileModifier () const
{
    if (m_type)
    {
        clang::QualType result(GetQualType());
        result.getQualifiers().setVolatile (true);
        return ClangASTType (m_ast, result);
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetArrayElementType (uint64_t *stride) const
{
    if (IsValid())
    {
        clang::QualType qual_type(GetCanonicalQualType());
        
        const clang::Type *array_elem_type = qual_type.getTypePtr()->getArrayElementTypeNoTypeQual();
        
        if (!array_elem_type)
            return ClangASTType();
        
        ClangASTType element_type (m_ast, array_elem_type->getCanonicalTypeUnqualified());
        
        // TODO: the real stride will be >= this value.. find the real one!
        if (stride)
            *stride = element_type.GetByteSize(nullptr);
        
        return element_type;
        
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetCanonicalType () const
{
    if (IsValid())
        return ClangASTType (m_ast, GetCanonicalQualType());
    return ClangASTType();
}

static clang::QualType
GetFullyUnqualifiedType_Impl (clang::ASTContext *ast, clang::QualType qual_type)
{
    if (qual_type->isPointerType())
        qual_type = ast->getPointerType(GetFullyUnqualifiedType_Impl(ast, qual_type->getPointeeType()));
    else
        qual_type = qual_type.getUnqualifiedType();
    qual_type.removeLocalConst();
    qual_type.removeLocalRestrict();
    qual_type.removeLocalVolatile();
    return qual_type;
}

ClangASTType
ClangASTType::GetFullyUnqualifiedType () const
{
    if (IsValid())
        return ClangASTType(m_ast, GetFullyUnqualifiedType_Impl(m_ast, GetQualType()));
    return ClangASTType();
}


int
ClangASTType::GetFunctionArgumentCount () const
{
    if (IsValid())
    {
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(GetCanonicalQualType());
        if (func)
            return func->getNumParams();
    }
    return -1;
}

ClangASTType
ClangASTType::GetFunctionArgumentTypeAtIndex (size_t idx) const
{
    if (IsValid())
    {
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(GetCanonicalQualType());
        if (func)
        {
            const uint32_t num_args = func->getNumParams();
            if (idx < num_args)
                return ClangASTType(m_ast, func->getParamType(idx));
        }
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetFunctionReturnType () const
{
    if (IsValid())
    {
        clang::QualType qual_type(GetCanonicalQualType());
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
            return ClangASTType(m_ast, func->getReturnType());
    }
    return ClangASTType();
}

size_t
ClangASTType::GetNumMemberFunctions () const
{
    size_t num_functions = 0;
    if (IsValid())
    {
        clang::QualType qual_type(GetCanonicalQualType());
        switch (qual_type->getTypeClass()) {
            case clang::Type::Record:
                if (GetCompleteQualType (m_ast, qual_type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    assert(record_decl);
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    if (cxx_record_decl)
                        num_functions = std::distance(cxx_record_decl->method_begin(), cxx_record_decl->method_end());
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (GetCompleteType())
                {
                    const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                        if (class_interface_decl)
                            num_functions = std::distance(class_interface_decl->meth_begin(), class_interface_decl->meth_end());
                    }
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType())
                {
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        if (class_interface_decl)
                            num_functions = std::distance(class_interface_decl->meth_begin(), class_interface_decl->meth_end());
                    }
                }
                break;
                
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumMemberFunctions();
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumMemberFunctions();
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumMemberFunctions();
                
            default:
                break;
        }
    }
    return num_functions;
}

TypeMemberFunctionImpl
ClangASTType::GetMemberFunctionAtIndex (size_t idx)
{
    std::string name("");
    MemberFunctionKind kind(MemberFunctionKind::eMemberFunctionKindUnknown);
    ClangASTType type{};
    clang::ObjCMethodDecl *method_decl(nullptr);
    if (IsValid())
    {
        clang::QualType qual_type(GetCanonicalQualType());
        switch (qual_type->getTypeClass()) {
            case clang::Type::Record:
                if (GetCompleteQualType (m_ast, qual_type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    assert(record_decl);
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    if (cxx_record_decl)
                    {
                        auto method_iter = cxx_record_decl->method_begin();
                        auto method_end = cxx_record_decl->method_end();
                        if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                        {
                            std::advance(method_iter, idx);
                            auto method_decl = method_iter->getCanonicalDecl();
                            if (method_decl)
                            {
                                if (!method_decl->getName().empty())
                                    name.assign(method_decl->getName().data());
                                else
                                    name.clear();
                                if (method_decl->isStatic())
                                    kind = lldb::eMemberFunctionKindStaticMethod;
                                else if (llvm::isa<clang::CXXConstructorDecl>(method_decl))
                                    kind = lldb::eMemberFunctionKindConstructor;
                                else if (llvm::isa<clang::CXXDestructorDecl>(method_decl))
                                    kind = lldb::eMemberFunctionKindDestructor;
                                else
                                    kind = lldb::eMemberFunctionKindInstanceMethod;
                                type = ClangASTType(m_ast,method_decl->getType().getAsOpaquePtr());
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (GetCompleteType())
                {
                    const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                        if (class_interface_decl)
                        {
                            auto method_iter = class_interface_decl->meth_begin();
                            auto method_end = class_interface_decl->meth_end();
                            if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                            {
                                std::advance(method_iter, idx);
                                method_decl = method_iter->getCanonicalDecl();
                                if (method_decl)
                                {
                                    name = method_decl->getSelector().getAsString();
                                    if (method_decl->isClassMethod())
                                        kind = lldb::eMemberFunctionKindStaticMethod;
                                    else
                                        kind = lldb::eMemberFunctionKindInstanceMethod;
                                }
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType())
                {
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        if (class_interface_decl)
                        {
                            auto method_iter = class_interface_decl->meth_begin();
                            auto method_end = class_interface_decl->meth_end();
                            if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                            {
                                std::advance(method_iter, idx);
                                method_decl = method_iter->getCanonicalDecl();
                                if (method_decl)
                                {
                                    name = method_decl->getSelector().getAsString();
                                    if (method_decl->isClassMethod())
                                        kind = lldb::eMemberFunctionKindStaticMethod;
                                    else
                                        kind = lldb::eMemberFunctionKindInstanceMethod;
                                }
                            }
                        }
                    }
                }
                break;

            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetMemberFunctionAtIndex(idx);
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetMemberFunctionAtIndex(idx);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetMemberFunctionAtIndex(idx);
                
            default:
                break;
        }
    }
    
    if (kind == eMemberFunctionKindUnknown)
        return TypeMemberFunctionImpl();
    if (method_decl)
        return TypeMemberFunctionImpl(method_decl, name, kind);
    if (type)
        return TypeMemberFunctionImpl(type, name, kind);
    
    return TypeMemberFunctionImpl();
}

ClangASTType
ClangASTType::GetLValueReferenceType () const
{
    if (IsValid())
    {
        return ClangASTType(m_ast, m_ast->getLValueReferenceType(GetQualType()));
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetRValueReferenceType () const
{
    if (IsValid())
    {
        return ClangASTType(m_ast, m_ast->getRValueReferenceType(GetQualType()));
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetNonReferenceType () const
{
    if (IsValid())
        return ClangASTType(m_ast, GetQualType().getNonReferenceType());
    return ClangASTType();
}

ClangASTType
ClangASTType::CreateTypedefType (const char *typedef_name,
                                 clang::DeclContext *decl_ctx) const
{
    if (IsValid() && typedef_name && typedef_name[0])
    {
        clang::QualType qual_type (GetQualType());
        if (decl_ctx == nullptr)
            decl_ctx = m_ast->getTranslationUnitDecl();
        clang::TypedefDecl *decl = clang::TypedefDecl::Create (*m_ast,
                                                               decl_ctx,
                                                               clang::SourceLocation(),
                                                               clang::SourceLocation(),
                                                               &m_ast->Idents.get(typedef_name),
                                                               m_ast->getTrivialTypeSourceInfo(qual_type));
        
        decl->setAccess(clang::AS_public); // TODO respect proper access specifier
        
        // Get a uniqued clang::QualType for the typedef decl type
        return ClangASTType (m_ast, m_ast->getTypedefType (decl));
    }
    return ClangASTType();

}

ClangASTType
ClangASTType::GetPointeeType () const
{
    if (m_type)
    {
        clang::QualType qual_type(GetQualType());
        return ClangASTType (m_ast, qual_type.getTypePtr()->getPointeeType());
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetPointerType () const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                return ClangASTType(m_ast, m_ast->getObjCObjectPointerType(qual_type).getAsOpaquePtr());
                
            default:
                return ClangASTType(m_ast, m_ast->getPointerType(qual_type).getAsOpaquePtr());
        }
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetTypedefedType () const
{
    if (IsValid())
    {
        const clang::TypedefType *typedef_type = llvm::dyn_cast<clang::TypedefType>(GetQualType());
        if (typedef_type)
            return ClangASTType (m_ast, typedef_type->getDecl()->getUnderlyingType());
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::RemoveFastQualifiers () const
{
    if (m_type)
    {
        clang::QualType qual_type(GetQualType());
        qual_type.getQualifiers().removeFastQualifiers();
        return ClangASTType (m_ast, qual_type);
    }
    return ClangASTType();
}


//----------------------------------------------------------------------
// Create related types using the current type's AST
//----------------------------------------------------------------------

ClangASTType
ClangASTType::GetBasicTypeFromAST (lldb::BasicType basic_type) const
{
    if (IsValid())
        return ClangASTContext::GetBasicType(m_ast, basic_type);
    return ClangASTType();
}
//----------------------------------------------------------------------
// Exploring the type
//----------------------------------------------------------------------

uint64_t
ClangASTType::GetBitSize (ExecutionContextScope *exe_scope) const
{
    if (GetCompleteType ())
    {
        clang::QualType qual_type(GetCanonicalQualType());
        switch (qual_type->getTypeClass())
        {
            case clang::Type::ObjCInterface:
            case clang::Type::ObjCObject:
            {
                ExecutionContext exe_ctx (exe_scope);
                Process *process = exe_ctx.GetProcessPtr();
                if (process)
                {
                    ObjCLanguageRuntime *objc_runtime = process->GetObjCLanguageRuntime();
                    if (objc_runtime)
                    {
                        uint64_t bit_size = 0;
                        if (objc_runtime->GetTypeBitSize(*this, bit_size))
                            return bit_size;
                    }
                }
                else
                {
                    static bool g_printed = false;
                    if (!g_printed)
                    {
                        StreamString s;
                        DumpTypeDescription(&s);

                        llvm::outs() << "warning: trying to determine the size of type ";
                        llvm::outs() << s.GetString() << "\n";
                        llvm::outs() << "without a valid ExecutionContext. this is not reliable. please file a bug against LLDB.\n";
                        llvm::outs() << "backtrace:\n";
                        llvm::sys::PrintStackTrace(llvm::outs());
                        llvm::outs() << "\n";
                        g_printed = true;
                    }
                }
            }
                // fallthrough
            default:
                const uint32_t bit_size = m_ast->getTypeSize (qual_type);
                if (bit_size == 0)
                {
                    if (qual_type->isIncompleteArrayType())
                        return m_ast->getTypeSize (qual_type->getArrayElementTypeNoTypeQual()->getCanonicalTypeUnqualified());
                }
                if (qual_type->isObjCObjectOrInterfaceType())
                    return bit_size + m_ast->getTypeSize(m_ast->ObjCBuiltinClassTy);
                return bit_size;
        }
    }
    return 0;
}

uint64_t
ClangASTType::GetByteSize (ExecutionContextScope *exe_scope) const
{
    return (GetBitSize (exe_scope) + 7) / 8;
}


size_t
ClangASTType::GetTypeBitAlign () const
{
    if (GetCompleteType ())
        return m_ast->getTypeAlign(GetQualType());
    return 0;
}


lldb::Encoding
ClangASTType::GetEncoding (uint64_t &count) const
{
    if (!IsValid())
        return lldb::eEncodingInvalid;
    
    count = 1;
    clang::QualType qual_type(GetCanonicalQualType());
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:
            break;
            
        case clang::Type::FunctionNoProto:
        case clang::Type::FunctionProto:
            break;
            
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            break;
            
        case clang::Type::ConstantArray:
            break;
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
            // TODO: Set this to more than one???
            break;
            
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            default: assert(0 && "Unknown builtin type!");
            case clang::BuiltinType::Void:
                break;
                
            case clang::BuiltinType::Bool:
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Char16:
            case clang::BuiltinType::Char32:
            case clang::BuiltinType::Short:
            case clang::BuiltinType::Int:
            case clang::BuiltinType::Long:
            case clang::BuiltinType::LongLong:
            case clang::BuiltinType::Int128:        return lldb::eEncodingSint;
                
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:
            case clang::BuiltinType::UShort:
            case clang::BuiltinType::UInt:
            case clang::BuiltinType::ULong:
            case clang::BuiltinType::ULongLong:
            case clang::BuiltinType::UInt128:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::Float:
            case clang::BuiltinType::Double:
            case clang::BuiltinType::LongDouble:    return lldb::eEncodingIEEE754;
                
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCSel:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::NullPtr:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::Kind::ARCUnbridgedCast:
            case clang::BuiltinType::Kind::BoundMember:
            case clang::BuiltinType::Kind::BuiltinFn:
            case clang::BuiltinType::Kind::Dependent:
            case clang::BuiltinType::Kind::Half:
            case clang::BuiltinType::Kind::OCLEvent:
            case clang::BuiltinType::Kind::OCLImage1d:
            case clang::BuiltinType::Kind::OCLImage1dArray:
            case clang::BuiltinType::Kind::OCLImage1dBuffer:
            case clang::BuiltinType::Kind::OCLImage2d:
            case clang::BuiltinType::Kind::OCLImage2dArray:
            case clang::BuiltinType::Kind::OCLImage3d:
            case clang::BuiltinType::Kind::OCLSampler:
            case clang::BuiltinType::Kind::Overload:
            case clang::BuiltinType::Kind::PseudoObject:
            case clang::BuiltinType::Kind::UnknownAny:
                break;
        }
            break;
            // All pointer types are represented as unsigned integer encodings.
            // We may nee to add a eEncodingPointer if we ever need to know the
            // difference
        case clang::Type::ObjCObjectPointer:
        case clang::Type::BlockPointer:
        case clang::Type::Pointer:
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
        case clang::Type::MemberPointer:            return lldb::eEncodingUint;
        case clang::Type::Complex:
        {
            lldb::Encoding encoding = lldb::eEncodingIEEE754;
            if (qual_type->isComplexType())
                encoding = lldb::eEncodingIEEE754;
            else
            {
                const clang::ComplexType *complex_type = qual_type->getAsComplexIntegerType();
                if (complex_type)
                    encoding = ClangASTType(m_ast, complex_type->getElementType()).GetEncoding(count);
                else
                    encoding = lldb::eEncodingSint;
            }
            count = 2;
            return encoding;
        }
            
        case clang::Type::ObjCInterface:            break;
        case clang::Type::Record:                   break;
        case clang::Type::Enum:                     return lldb::eEncodingSint;
        case clang::Type::Typedef:
            return ClangASTType(m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetEncoding(count);
            
        case clang::Type::Elaborated:
            return ClangASTType(m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetEncoding(count);
            
        case clang::Type::Paren:
            return ClangASTType(m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetEncoding(count);
            
        case clang::Type::DependentSizedArray:
        case clang::Type::DependentSizedExtVector:
        case clang::Type::UnresolvedUsing:
        case clang::Type::Attributed:
        case clang::Type::TemplateTypeParm:
        case clang::Type::SubstTemplateTypeParm:
        case clang::Type::SubstTemplateTypeParmPack:
        case clang::Type::Auto:
        case clang::Type::InjectedClassName:
        case clang::Type::DependentName:
        case clang::Type::DependentTemplateSpecialization:
        case clang::Type::PackExpansion:
        case clang::Type::ObjCObject:
            
        case clang::Type::TypeOfExpr:
        case clang::Type::TypeOf:
        case clang::Type::Decltype:
        case clang::Type::TemplateSpecialization:
        case clang::Type::Atomic:
        case clang::Type::Adjusted:
            break;

        // pointer type decayed from an array or function type.
        case clang::Type::Decayed:
            break;
    }
    count = 0;
    return lldb::eEncodingInvalid;
}

lldb::Format
ClangASTType::GetFormat () const
{
    if (!IsValid())
        return lldb::eFormatDefault;
    
    clang::QualType qual_type(GetCanonicalQualType());
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:
            break;
            
        case clang::Type::FunctionNoProto:
        case clang::Type::FunctionProto:
            break;
            
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            break;
            
        case clang::Type::ConstantArray:
            return lldb::eFormatVoid; // no value
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
            break;
            
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
                //default: assert(0 && "Unknown builtin type!");
            case clang::BuiltinType::UnknownAny:
            case clang::BuiltinType::Void:
            case clang::BuiltinType::BoundMember:
                break;
                
            case clang::BuiltinType::Bool:          return lldb::eFormatBoolean;
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:       return lldb::eFormatChar;
            case clang::BuiltinType::Char16:        return lldb::eFormatUnicode16;
            case clang::BuiltinType::Char32:        return lldb::eFormatUnicode32;
            case clang::BuiltinType::UShort:        return lldb::eFormatUnsigned;
            case clang::BuiltinType::Short:         return lldb::eFormatDecimal;
            case clang::BuiltinType::UInt:          return lldb::eFormatUnsigned;
            case clang::BuiltinType::Int:           return lldb::eFormatDecimal;
            case clang::BuiltinType::ULong:         return lldb::eFormatUnsigned;
            case clang::BuiltinType::Long:          return lldb::eFormatDecimal;
            case clang::BuiltinType::ULongLong:     return lldb::eFormatUnsigned;
            case clang::BuiltinType::LongLong:      return lldb::eFormatDecimal;
            case clang::BuiltinType::UInt128:       return lldb::eFormatUnsigned;
            case clang::BuiltinType::Int128:        return lldb::eFormatDecimal;
            case clang::BuiltinType::Float:         return lldb::eFormatFloat;
            case clang::BuiltinType::Double:        return lldb::eFormatFloat;
            case clang::BuiltinType::LongDouble:    return lldb::eFormatFloat;
            case clang::BuiltinType::NullPtr:
            case clang::BuiltinType::Overload:
            case clang::BuiltinType::Dependent:
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCSel:
            case clang::BuiltinType::Half:
            case clang::BuiltinType::ARCUnbridgedCast:
            case clang::BuiltinType::PseudoObject:
            case clang::BuiltinType::BuiltinFn:
            case clang::BuiltinType::OCLEvent:
            case clang::BuiltinType::OCLImage1d:
            case clang::BuiltinType::OCLImage1dArray:
            case clang::BuiltinType::OCLImage1dBuffer:
            case clang::BuiltinType::OCLImage2d:
            case clang::BuiltinType::OCLImage2dArray:
            case clang::BuiltinType::OCLImage3d:
            case clang::BuiltinType::OCLSampler:
                return lldb::eFormatHex;
        }
            break;
        case clang::Type::ObjCObjectPointer:        return lldb::eFormatHex;
        case clang::Type::BlockPointer:             return lldb::eFormatHex;
        case clang::Type::Pointer:                  return lldb::eFormatHex;
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:          return lldb::eFormatHex;
        case clang::Type::MemberPointer:            break;
        case clang::Type::Complex:
        {
            if (qual_type->isComplexType())
                return lldb::eFormatComplex;
            else
                return lldb::eFormatComplexInteger;
        }
        case clang::Type::ObjCInterface:            break;
        case clang::Type::Record:                   break;
        case clang::Type::Enum:                     return lldb::eFormatEnum;
        case clang::Type::Typedef:
            return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetFormat();
        case clang::Type::Auto:
            return ClangASTType (m_ast, llvm::cast<clang::AutoType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Paren:
            return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetFormat();
        case clang::Type::DependentSizedArray:
        case clang::Type::DependentSizedExtVector:
        case clang::Type::UnresolvedUsing:
        case clang::Type::Attributed:
        case clang::Type::TemplateTypeParm:
        case clang::Type::SubstTemplateTypeParm:
        case clang::Type::SubstTemplateTypeParmPack:
        case clang::Type::InjectedClassName:
        case clang::Type::DependentName:
        case clang::Type::DependentTemplateSpecialization:
        case clang::Type::PackExpansion:
        case clang::Type::ObjCObject:
            
        case clang::Type::TypeOfExpr:
        case clang::Type::TypeOf:
        case clang::Type::Decltype:
        case clang::Type::TemplateSpecialization:
        case clang::Type::Atomic:
        case clang::Type::Adjusted:
            break;

        // pointer type decayed from an array or function type.
        case clang::Type::Decayed:
            break;
    }
    // We don't know hot to display this type...
    return lldb::eFormatBytes;
}

static bool
ObjCDeclHasIVars (clang::ObjCInterfaceDecl *class_interface_decl, bool check_superclass)
{
    while (class_interface_decl)
    {
        if (class_interface_decl->ivar_size() > 0)
            return true;
        
        if (check_superclass)
            class_interface_decl = class_interface_decl->getSuperClass();
        else
            break;
    }
    return false;
}

uint32_t
ClangASTType::GetNumChildren (bool omit_empty_base_classes) const
{
    if (!IsValid())
        return 0;
    
    uint32_t num_children = 0;
    clang::QualType qual_type(GetQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            case clang::BuiltinType::ObjCId:    // child is Class
            case clang::BuiltinType::ObjCClass: // child is Class
                num_children = 1;
                break;
                
            default:
                break;
        }
            break;
            
        case clang::Type::Complex: return 0;
            
        case clang::Type::Record:
            if (GetCompleteQualType (m_ast, qual_type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    if (omit_empty_base_classes)
                    {
                        // Check each base classes to see if it or any of its
                        // base classes contain any fields. This can help
                        // limit the noise in variable views by not having to
                        // show base classes that contain no members.
                        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            
                            // Skip empty base classes
                            if (ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                            
                            num_children++;
                        }
                    }
                    else
                    {
                        // Include all base classes
                        num_children += cxx_record_decl->getNumBases();
                    }
                    
                }
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                    ++num_children;
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteQualType (m_ast, qual_type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                if (ObjCDeclHasIVars (superclass_interface_decl, true))
                                    ++num_children;
                            }
                            else
                                ++num_children;
                        }
                        
                        num_children += class_interface_decl->ivar_size();
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
        {
            const clang::ObjCObjectPointerType *pointer_type = llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr());
            clang::QualType pointee_type = pointer_type->getPointeeType();
            uint32_t num_pointee_children = ClangASTType (m_ast,pointee_type).GetNumChildren (omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
            break;
            
        case clang::Type::Vector:
        case clang::Type::ExtVector:
            num_children = llvm::cast<clang::VectorType>(qual_type.getTypePtr())->getNumElements();
            break;
            
        case clang::Type::ConstantArray:
            num_children = llvm::cast<clang::ConstantArrayType>(qual_type.getTypePtr())->getSize().getLimitedValue();
            break;
            
        case clang::Type::Pointer:
        {
            const clang::PointerType *pointer_type = llvm::cast<clang::PointerType>(qual_type.getTypePtr());
            clang::QualType pointee_type (pointer_type->getPointeeType());
            uint32_t num_pointee_children = ClangASTType (m_ast,pointee_type).GetNumChildren (omit_empty_base_classes);
            if (num_pointee_children == 0)
            {
                // We have a pointer to a pointee type that claims it has no children.
                // We will want to look at
                num_children = ClangASTType (m_ast, pointee_type).GetNumPointeeChildren();
            }
            else
                num_children = num_pointee_children;
        }
            break;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
        {
            const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
            clang::QualType pointee_type = reference_type->getPointeeType();
            uint32_t num_pointee_children = ClangASTType (m_ast, pointee_type).GetNumChildren (omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
            break;
            
            
        case clang::Type::Typedef:
            num_children = ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Elaborated:
            num_children = ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Paren:
            num_children = ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumChildren (omit_empty_base_classes);
            break;
        default:
            break;
    }
    return num_children;
}

lldb::BasicType
ClangASTType::GetBasicTypeEnumeration () const
{
    if (IsValid())
    {
        clang::QualType qual_type(GetQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        if (type_class == clang::Type::Builtin)
        {
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                case clang::BuiltinType::Void:      return eBasicTypeVoid;
                case clang::BuiltinType::Bool:      return eBasicTypeBool;
                case clang::BuiltinType::Char_S:    return eBasicTypeSignedChar;
                case clang::BuiltinType::Char_U:    return eBasicTypeUnsignedChar;
                case clang::BuiltinType::Char16:    return eBasicTypeChar16;
                case clang::BuiltinType::Char32:    return eBasicTypeChar32;
                case clang::BuiltinType::UChar:     return eBasicTypeUnsignedChar;
                case clang::BuiltinType::SChar:     return eBasicTypeSignedChar;
                case clang::BuiltinType::WChar_S:   return eBasicTypeSignedWChar;
                case clang::BuiltinType::WChar_U:   return eBasicTypeUnsignedWChar;
                case clang::BuiltinType::Short:     return eBasicTypeShort;
                case clang::BuiltinType::UShort:    return eBasicTypeUnsignedShort;
                case clang::BuiltinType::Int:       return eBasicTypeInt;
                case clang::BuiltinType::UInt:      return eBasicTypeUnsignedInt;
                case clang::BuiltinType::Long:      return eBasicTypeLong;
                case clang::BuiltinType::ULong:     return eBasicTypeUnsignedLong;
                case clang::BuiltinType::LongLong:  return eBasicTypeLongLong;
                case clang::BuiltinType::ULongLong: return eBasicTypeUnsignedLongLong;
                case clang::BuiltinType::Int128:    return eBasicTypeInt128;
                case clang::BuiltinType::UInt128:   return eBasicTypeUnsignedInt128;
                    
                case clang::BuiltinType::Half:      return eBasicTypeHalf;
                case clang::BuiltinType::Float:     return eBasicTypeFloat;
                case clang::BuiltinType::Double:    return eBasicTypeDouble;
                case clang::BuiltinType::LongDouble:return eBasicTypeLongDouble;
                    
                case clang::BuiltinType::NullPtr:   return eBasicTypeNullPtr;
                case clang::BuiltinType::ObjCId:    return eBasicTypeObjCID;
                case clang::BuiltinType::ObjCClass: return eBasicTypeObjCClass;
                case clang::BuiltinType::ObjCSel:   return eBasicTypeObjCSel;
                case clang::BuiltinType::Dependent:
                case clang::BuiltinType::Overload:
                case clang::BuiltinType::BoundMember:
                case clang::BuiltinType::PseudoObject:
                case clang::BuiltinType::UnknownAny:
                case clang::BuiltinType::BuiltinFn:
                case clang::BuiltinType::ARCUnbridgedCast:
                case clang::BuiltinType::OCLEvent:
                case clang::BuiltinType::OCLImage1d:
                case clang::BuiltinType::OCLImage1dArray:
                case clang::BuiltinType::OCLImage1dBuffer:
                case clang::BuiltinType::OCLImage2d:
                case clang::BuiltinType::OCLImage2dArray:
                case clang::BuiltinType::OCLImage3d:
                case clang::BuiltinType::OCLSampler:
                    return eBasicTypeOther;
            }
        }
    }
    return eBasicTypeInvalid;
}


#pragma mark Aggregate Types

uint32_t
ClangASTType::GetNumDirectBaseClasses () const
{
    if (!IsValid())
        return 0;
    
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumBases();
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            count = GetPointeeType().GetNumDirectBaseClasses();
            break;
            
        case clang::Type::ObjCObject:
            if (GetCompleteType())
            {
                const clang::ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;
        case clang::Type::ObjCInterface:
            if (GetCompleteType())
            {
                const clang::ObjCInterfaceType *objc_interface_type = qual_type->getAs<clang::ObjCInterfaceType>();
                if (objc_interface_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_interface_type->getInterface();
                    
                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            count = ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumDirectBaseClasses ();
            break;
            
        case clang::Type::Elaborated:
            count = ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumDirectBaseClasses ();
            break;
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumDirectBaseClasses ();
            
        default:
            break;
    }
    return count;
}

uint32_t
ClangASTType::GetNumVirtualBaseClasses () const
{
    if (!IsValid())
        return 0;
    
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumVBases();
            }
            break;
            
        case clang::Type::Typedef:
            count = ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumVirtualBaseClasses();
            break;
            
        case clang::Type::Elaborated:
            count = ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumVirtualBaseClasses();
            break;
            
        case clang::Type::Paren:
            count = ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumVirtualBaseClasses();
            break;
            
        default:
            break;
    }
    return count;
}

uint32_t
ClangASTType::GetNumFields () const
{
    if (!IsValid())
        return 0;
    
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const clang::RecordType *record_type = llvm::dyn_cast<clang::RecordType>(qual_type.getTypePtr());
                if (record_type)
                {
                    clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        uint32_t field_idx = 0;
                        clang::RecordDecl::field_iterator field, field_end;
                        for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                            ++field_idx;
                        count = field_idx;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            count = ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumFields();
            break;
            
        case clang::Type::Elaborated:
            count = ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumFields();
            break;
            
        case clang::Type::Paren:
            count = ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumFields();
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType())
            {
                const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    
                    if (class_interface_decl)
                        count = class_interface_decl->ivar_size();
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType())
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                        count = class_interface_decl->ivar_size();
                }
            }
            break;
            
        default:
            break;
    }
    return count;
}

ClangASTType
ClangASTType::GetDirectBaseClassAtIndex (size_t idx, uint32_t *bit_offset_ptr) const
{
    if (!IsValid())
        return ClangASTType();
    
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const clang::ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(cxx_record_decl);
                                const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                                if (base_class->isVirtual())
                                    *bit_offset_ptr = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                                else
                                    *bit_offset_ptr = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                            }
                            return ClangASTType (m_ast, base_class->getType());
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            return GetPointeeType().GetDirectBaseClassAtIndex(idx,bit_offset_ptr);
            
        case clang::Type::ObjCObject:
            if (idx == 0 && GetCompleteType())
            {
                const clang::ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (bit_offset_ptr)
                                *bit_offset_ptr = 0;
                            return ClangASTType (m_ast, m_ast->getObjCInterfaceType(superclass_interface_decl));
                        }
                    }
                }
            }
            break;
        case clang::Type::ObjCInterface:
            if (idx == 0 && GetCompleteType())
            {
                const clang::ObjCObjectType *objc_interface_type = qual_type->getAs<clang::ObjCInterfaceType>();
                if (objc_interface_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_interface_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (bit_offset_ptr)
                                *bit_offset_ptr = 0;
                            return ClangASTType (m_ast, m_ast->getObjCInterfaceType(superclass_interface_decl));
                        }
                    }
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetDirectBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetDirectBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetDirectBaseClassAtIndex (idx, bit_offset_ptr);
            
        default:
            break;
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetVirtualBaseClassAtIndex (size_t idx, uint32_t *bit_offset_ptr) const
{
    if (!IsValid())
        return ClangASTType();
    
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->vbases_begin(), base_class_end = cxx_record_decl->vbases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const clang::ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(cxx_record_decl);
                                const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                                *bit_offset_ptr = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                                
                            }
                            return ClangASTType (m_ast, base_class->getType());
                        }
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetVirtualBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetVirtualBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Paren:
            return  ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetVirtualBaseClassAtIndex (idx, bit_offset_ptr);
            
        default:
            break;
    }
    return ClangASTType();
}

static clang_type_t
GetObjCFieldAtIndex (clang::ASTContext *ast,
                     clang::ObjCInterfaceDecl *class_interface_decl,
                     size_t idx,
                     std::string& name,
                     uint64_t *bit_offset_ptr,
                     uint32_t *bitfield_bit_size_ptr,
                     bool *is_bitfield_ptr)
{
    if (class_interface_decl)
    {
        if (idx < (class_interface_decl->ivar_size()))
        {
            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
            uint32_t ivar_idx = 0;
            
            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++ivar_idx)
            {
                if (ivar_idx == idx)
                {
                    const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                    
                    clang::QualType ivar_qual_type(ivar_decl->getType());
                    
                    name.assign(ivar_decl->getNameAsString());
                    
                    if (bit_offset_ptr)
                    {
                        const clang::ASTRecordLayout &interface_layout = ast->getASTObjCInterfaceLayout(class_interface_decl);
                        *bit_offset_ptr = interface_layout.getFieldOffset (ivar_idx);
                    }
                    
                    const bool is_bitfield = ivar_pos->isBitField();
                    
                    if (bitfield_bit_size_ptr)
                    {
                        *bitfield_bit_size_ptr = 0;
                        
                        if (is_bitfield && ast)
                        {
                            clang::Expr *bitfield_bit_size_expr = ivar_pos->getBitWidth();
                            llvm::APSInt bitfield_apsint;
                            if (bitfield_bit_size_expr && bitfield_bit_size_expr->EvaluateAsInt(bitfield_apsint, *ast))
                            {
                                *bitfield_bit_size_ptr = bitfield_apsint.getLimitedValue();
                            }
                        }
                    }
                    if (is_bitfield_ptr)
                        *is_bitfield_ptr = is_bitfield;
                    
                    return ivar_qual_type.getAsOpaquePtr();
                }
            }
        }
    }
    return nullptr;
}

ClangASTType
ClangASTType::GetFieldAtIndex (size_t idx,
                               std::string& name,
                               uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) const
{
    if (!IsValid())
        return ClangASTType();
    
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx)
                {
                    if (idx == field_idx)
                    {
                        // Print the member type if requested
                        // Print the member name and equal sign
                        name.assign(field->getNameAsString());
                        
                        // Figure out the type byte size (field_type_info.first) and
                        // alignment (field_type_info.second) from the AST context.
                        if (bit_offset_ptr)
                        {
                            const clang::ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(record_decl);
                            *bit_offset_ptr = record_layout.getFieldOffset (field_idx);
                        }
                        
                        const bool is_bitfield = field->isBitField();
                        
                        if (bitfield_bit_size_ptr)
                        {
                            *bitfield_bit_size_ptr = 0;
                            
                            if (is_bitfield)
                            {
                                clang::Expr *bitfield_bit_size_expr = field->getBitWidth();
                                llvm::APSInt bitfield_apsint;
                                if (bitfield_bit_size_expr && bitfield_bit_size_expr->EvaluateAsInt(bitfield_apsint, *m_ast))
                                {
                                    *bitfield_bit_size_ptr = bitfield_apsint.getLimitedValue();
                                }
                            }
                        }
                        if (is_bitfield_ptr)
                            *is_bitfield_ptr = is_bitfield;
                        
                        return ClangASTType (m_ast, field->getType());
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType())
            {
                const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    return ClangASTType (m_ast, GetObjCFieldAtIndex(m_ast, class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType())
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    return ClangASTType (m_ast, GetObjCFieldAtIndex(m_ast, class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).
                        GetFieldAtIndex (idx,
                                         name,
                                         bit_offset_ptr,
                                         bitfield_bit_size_ptr,
                                         is_bitfield_ptr);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).
                        GetFieldAtIndex (idx,
                                         name,
                                         bit_offset_ptr,
                                         bitfield_bit_size_ptr,
                                         is_bitfield_ptr);
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).
                        GetFieldAtIndex (idx,
                                         name,
                                         bit_offset_ptr,
                                         bitfield_bit_size_ptr,
                                         is_bitfield_ptr);
            
        default:
            break;
    }
    return ClangASTType();
}

uint32_t
ClangASTType::GetIndexOfFieldWithName (const char* name,
                                       ClangASTType* field_clang_type_ptr,
                                       uint64_t *bit_offset_ptr,
                                       uint32_t *bitfield_bit_size_ptr,
                                       bool *is_bitfield_ptr) const
{
    unsigned count = GetNumFields();
    std::string field_name;
    for (unsigned index = 0; index < count; index++)
    {
        ClangASTType field_clang_type (GetFieldAtIndex(index, field_name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
        if (strcmp(field_name.c_str(), name) == 0)
        {
            if (field_clang_type_ptr)
                *field_clang_type_ptr = field_clang_type;
            return index;
        }
    }
    return UINT32_MAX;
}

// If a pointer to a pointee type (the clang_type arg) says that it has no
// children, then we either need to trust it, or override it and return a
// different result. For example, an "int *" has one child that is an integer,
// but a function pointer doesn't have any children. Likewise if a Record type
// claims it has no children, then there really is nothing to show.
uint32_t
ClangASTType::GetNumPointeeChildren () const
{
    if (!IsValid())
        return 0;
    
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            case clang::BuiltinType::UnknownAny:
            case clang::BuiltinType::Void:
            case clang::BuiltinType::NullPtr:
            case clang::BuiltinType::OCLEvent:
            case clang::BuiltinType::OCLImage1d:
            case clang::BuiltinType::OCLImage1dArray:
            case clang::BuiltinType::OCLImage1dBuffer:
            case clang::BuiltinType::OCLImage2d:
            case clang::BuiltinType::OCLImage2dArray:
            case clang::BuiltinType::OCLImage3d:
            case clang::BuiltinType::OCLSampler:
                return 0;
            case clang::BuiltinType::Bool:
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:
            case clang::BuiltinType::Char16:
            case clang::BuiltinType::Char32:
            case clang::BuiltinType::UShort:
            case clang::BuiltinType::UInt:
            case clang::BuiltinType::ULong:
            case clang::BuiltinType::ULongLong:
            case clang::BuiltinType::UInt128:
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Short:
            case clang::BuiltinType::Int:
            case clang::BuiltinType::Long:
            case clang::BuiltinType::LongLong:
            case clang::BuiltinType::Int128:
            case clang::BuiltinType::Float:
            case clang::BuiltinType::Double:
            case clang::BuiltinType::LongDouble:
            case clang::BuiltinType::Dependent:
            case clang::BuiltinType::Overload:
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCSel:
            case clang::BuiltinType::BoundMember:
            case clang::BuiltinType::Half:
            case clang::BuiltinType::ARCUnbridgedCast:
            case clang::BuiltinType::PseudoObject:
            case clang::BuiltinType::BuiltinFn:
                return 1;
        }
            break;
            
        case clang::Type::Complex:                  return 1;
        case clang::Type::Pointer:                  return 1;
        case clang::Type::BlockPointer:             return 0;   // If block pointers don't have debug info, then no children for them
        case clang::Type::LValueReference:          return 1;
        case clang::Type::RValueReference:          return 1;
        case clang::Type::MemberPointer:            return 0;
        case clang::Type::ConstantArray:            return 0;
        case clang::Type::IncompleteArray:          return 0;
        case clang::Type::VariableArray:            return 0;
        case clang::Type::DependentSizedArray:      return 0;
        case clang::Type::DependentSizedExtVector:  return 0;
        case clang::Type::Vector:                   return 0;
        case clang::Type::ExtVector:                return 0;
        case clang::Type::FunctionProto:            return 0;   // When we function pointers, they have no children...
        case clang::Type::FunctionNoProto:          return 0;   // When we function pointers, they have no children...
        case clang::Type::UnresolvedUsing:          return 0;
        case clang::Type::Paren:                    return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumPointeeChildren ();
        case clang::Type::Typedef:                  return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumPointeeChildren ();
        case clang::Type::Elaborated:               return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumPointeeChildren ();
        case clang::Type::TypeOfExpr:               return 0;
        case clang::Type::TypeOf:                   return 0;
        case clang::Type::Decltype:                 return 0;
        case clang::Type::Record:                   return 0;
        case clang::Type::Enum:                     return 1;
        case clang::Type::TemplateTypeParm:         return 1;
        case clang::Type::SubstTemplateTypeParm:    return 1;
        case clang::Type::TemplateSpecialization:   return 1;
        case clang::Type::InjectedClassName:        return 0;
        case clang::Type::DependentName:            return 1;
        case clang::Type::DependentTemplateSpecialization:  return 1;
        case clang::Type::ObjCObject:               return 0;
        case clang::Type::ObjCInterface:            return 0;
        case clang::Type::ObjCObjectPointer:        return 1;
        default: 
            break;
    }
    return 0;
}


ClangASTType
ClangASTType::GetChildClangTypeAtIndex (ExecutionContext *exe_ctx,
                                        size_t idx,
                                        bool transparent_pointers,
                                        bool omit_empty_base_classes,
                                        bool ignore_array_bounds,
                                        std::string& child_name,
                                        uint32_t &child_byte_size,
                                        int32_t &child_byte_offset,
                                        uint32_t &child_bitfield_bit_size,
                                        uint32_t &child_bitfield_bit_offset,
                                        bool &child_is_base_class,
                                        bool &child_is_deref_of_parent,
                                        ValueObject *valobj) const
{
    if (!IsValid())
        return ClangASTType();
    
    clang::QualType parent_qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass parent_type_class = parent_qual_type->getTypeClass();
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    
    const bool idx_is_valid = idx < GetNumChildren (omit_empty_base_classes);
    uint32_t bit_offset;
    switch (parent_type_class)
    {
        case clang::Type::Builtin:
            if (idx_is_valid)
            {
                switch (llvm::cast<clang::BuiltinType>(parent_qual_type)->getKind())
                {
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                        child_name = "isa";
                        child_byte_size = m_ast->getTypeSize(m_ast->ObjCBuiltinClassTy) / CHAR_BIT;
                        return ClangASTType (m_ast, m_ast->ObjCBuiltinClassTy);
                        
                    default:
                        break;
                }
            }
            break;
            
        case clang::Type::Record:
            if (idx_is_valid && GetCompleteType())
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(parent_qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const clang::ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(record_decl);
                uint32_t child_idx = 0;
                
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    // We might have base classes to print out first
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const clang::CXXRecordDecl *base_class_decl = nullptr;
                        
                        // Skip empty base classes
                        if (omit_empty_base_classes)
                        {
                            base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            if (ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                        }
                        
                        if (idx == child_idx)
                        {
                            if (base_class_decl == nullptr)
                                base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            
                            
                            if (base_class->isVirtual())
                            {
                                bool handled = false;
                                if (valobj)
                                {
                                    Error err;
                                    AddressType addr_type = eAddressTypeInvalid;
                                    lldb::addr_t vtable_ptr_addr = valobj->GetCPPVTableAddress(addr_type);
                                    
                                    if (vtable_ptr_addr != LLDB_INVALID_ADDRESS && addr_type == eAddressTypeLoad)
                                    {
                                        
                                        ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
                                        Process *process = exe_ctx.GetProcessPtr();
                                        if (process)
                                        {
                                            clang::VTableContextBase *vtable_ctx = m_ast->getVTableContext();
                                            if (vtable_ctx)
                                            {
                                                if (vtable_ctx->isMicrosoft())
                                                {
                                                    clang::MicrosoftVTableContext *msoft_vtable_ctx = static_cast<clang::MicrosoftVTableContext *>(vtable_ctx);
                                                    
                                                    if (vtable_ptr_addr)
                                                    {
                                                        const lldb::addr_t vbtable_ptr_addr = vtable_ptr_addr + record_layout.getVBPtrOffset().getQuantity();
                                                        
                                                        const lldb::addr_t vbtable_ptr = process->ReadPointerFromMemory(vbtable_ptr_addr, err);
                                                        if (vbtable_ptr != LLDB_INVALID_ADDRESS)
                                                        {
                                                            // Get the index into the virtual base table. The index is the index in uint32_t from vbtable_ptr
                                                            const unsigned vbtable_index = msoft_vtable_ctx->getVBTableIndex(cxx_record_decl, base_class_decl);
                                                            const lldb::addr_t base_offset_addr = vbtable_ptr + vbtable_index * 4;
                                                            const uint32_t base_offset = process->ReadUnsignedIntegerFromMemory(base_offset_addr, 4, UINT32_MAX, err);
                                                            if (base_offset != UINT32_MAX)
                                                            {
                                                                handled = true;
                                                                bit_offset = base_offset * 8;
                                                            }
                                                        }
                                                    }
                                                }
                                                else
                                                {
                                                    clang::ItaniumVTableContext *itanium_vtable_ctx = static_cast<clang::ItaniumVTableContext *>(vtable_ctx);
                                                    if (vtable_ptr_addr)
                                                    {
                                                        const lldb::addr_t vtable_ptr = process->ReadPointerFromMemory(vtable_ptr_addr, err);
                                                        if (vtable_ptr != LLDB_INVALID_ADDRESS)
                                                        {
                                                            clang::CharUnits base_offset_offset = itanium_vtable_ctx->getVirtualBaseOffsetOffset(cxx_record_decl, base_class_decl);
                                                            const lldb::addr_t base_offset_addr = vtable_ptr + base_offset_offset.getQuantity();
                                                            const uint32_t base_offset = process->ReadUnsignedIntegerFromMemory(base_offset_addr, 4, UINT32_MAX, err);
                                                            if (base_offset != UINT32_MAX)
                                                            {
                                                                handled = true;
                                                                bit_offset = base_offset * 8;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                }
                                if (!handled)
                                    bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                            }
                            else
                                bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                            
                            // Base classes should be a multiple of 8 bits in size
                            child_byte_offset = bit_offset/8;
                            ClangASTType base_class_clang_type(m_ast, base_class->getType());
                            child_name = base_class_clang_type.GetTypeName().AsCString("");
                            uint64_t base_class_clang_type_bit_size = base_class_clang_type.GetBitSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                            
                            // Base classes bit sizes should be a multiple of 8 bits in size
                            assert (base_class_clang_type_bit_size % 8 == 0);
                            child_byte_size = base_class_clang_type_bit_size / 8;
                            child_is_base_class = true;
                            return base_class_clang_type;
                        }
                        // We don't increment the child index in the for loop since we might
                        // be skipping empty base classes
                        ++child_idx;
                    }
                }
                // Make sure index is in range...
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
                {
                    if (idx == child_idx)
                    {
                        // Print the member type if requested
                        // Print the member name and equal sign
                        child_name.assign(field->getNameAsString().c_str());
                        
                        // Figure out the type byte size (field_type_info.first) and
                        // alignment (field_type_info.second) from the AST context.
                        ClangASTType field_clang_type (m_ast, field->getType());
                        assert(field_idx < record_layout.getFieldCount());
                        child_byte_size = field_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        
                        // Figure out the field offset within the current struct/union/class type
                        bit_offset = record_layout.getFieldOffset (field_idx);
                        child_byte_offset = bit_offset / 8;
                        if (ClangASTContext::FieldIsBitfield (m_ast, *field, child_bitfield_bit_size))
                            child_bitfield_bit_offset = bit_offset % 8;
                        
                        return field_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (idx_is_valid && GetCompleteType())
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(parent_qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    uint32_t child_idx = 0;
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        const clang::ASTRecordLayout &interface_layout = m_ast->getASTObjCInterfaceLayout(class_interface_decl);
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                ClangASTType base_class_clang_type (m_ast, m_ast->getObjCInterfaceType(superclass_interface_decl));
                                if (base_class_clang_type.GetNumChildren(omit_empty_base_classes) > 0)
                                {
                                    if (idx == 0)
                                    {
                                        clang::QualType ivar_qual_type(m_ast->getObjCInterfaceType(superclass_interface_decl));
                                        
                                        
                                        child_name.assign(superclass_interface_decl->getNameAsString().c_str());
                                        
                                        clang::TypeInfo ivar_type_info = m_ast->getTypeInfo(ivar_qual_type.getTypePtr());
                                        
                                        child_byte_size = ivar_type_info.Width / 8;
                                        child_byte_offset = 0;
                                        child_is_base_class = true;
                                        
                                        return ClangASTType (m_ast, ivar_qual_type);
                                    }
                                    
                                    ++child_idx;
                                }
                            }
                            else
                                ++child_idx;
                        }
                        
                        const uint32_t superclass_idx = child_idx;
                        
                        if (idx < (child_idx + class_interface_decl->ivar_size()))
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos)
                            {
                                if (child_idx == idx)
                                {
                                    clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                    
                                    clang::QualType ivar_qual_type(ivar_decl->getType());
                                    
                                    child_name.assign(ivar_decl->getNameAsString().c_str());
                                    
                                    clang::TypeInfo  ivar_type_info = m_ast->getTypeInfo(ivar_qual_type.getTypePtr());
                                    
                                    child_byte_size = ivar_type_info.Width / 8;
                                    
                                    // Figure out the field offset within the current struct/union/class type
                                    // For ObjC objects, we can't trust the bit offset we get from the Clang AST, since
                                    // that doesn't account for the space taken up by unbacked properties, or from
                                    // the changing size of base classes that are newer than this class.
                                    // So if we have a process around that we can ask about this object, do so.
                                    child_byte_offset = LLDB_INVALID_IVAR_OFFSET;
                                    Process *process = nullptr;
                                    if (exe_ctx)
                                        process = exe_ctx->GetProcessPtr();
                                    if (process)
                                    {
                                        ObjCLanguageRuntime *objc_runtime = process->GetObjCLanguageRuntime();
                                        if (objc_runtime != nullptr)
                                        {
                                            ClangASTType parent_ast_type (m_ast, parent_qual_type);
                                            child_byte_offset = objc_runtime->GetByteOffsetForIvar (parent_ast_type, ivar_decl->getNameAsString().c_str());
                                        }
                                    }
                                    
                                    // Setting this to UINT32_MAX to make sure we don't compute it twice...
                                    bit_offset = UINT32_MAX;
                                    
                                    if (child_byte_offset == static_cast<int32_t>(LLDB_INVALID_IVAR_OFFSET))
                                    {
                                        bit_offset = interface_layout.getFieldOffset (child_idx - superclass_idx);
                                        child_byte_offset = bit_offset / 8;
                                    }
                                    
                                    // Note, the ObjC Ivar Byte offset is just that, it doesn't account for the bit offset
                                    // of a bitfield within its containing object.  So regardless of where we get the byte
                                    // offset from, we still need to get the bit offset for bitfields from the layout.
                                    
                                    if (ClangASTContext::FieldIsBitfield (m_ast, ivar_decl, child_bitfield_bit_size))
                                    {
                                        if (bit_offset == UINT32_MAX)
                                            bit_offset = interface_layout.getFieldOffset (child_idx - superclass_idx);
                                        
                                        child_bitfield_bit_offset = bit_offset % 8;
                                    }
                                    return ClangASTType (m_ast, ivar_qual_type);
                                }
                                ++child_idx;
                            }
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (idx_is_valid)
            {
                ClangASTType pointee_clang_type (GetPointeeType());
                
                if (transparent_pointers && pointee_clang_type.IsAggregateType())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                        idx,
                                                                        transparent_pointers,
                                                                        omit_empty_base_classes,
                                                                        ignore_array_bounds,
                                                                        child_name,
                                                                        child_byte_size,
                                                                        child_byte_offset,
                                                                        child_bitfield_bit_size,
                                                                        child_bitfield_bit_offset,
                                                                        child_is_base_class,
                                                                        tmp_child_is_deref_of_parent,
                                                                        valobj);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0 && pointee_clang_type.GetCompleteType())
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::Vector:
        case clang::Type::ExtVector:
            if (idx_is_valid)
            {
                const clang::VectorType *array = llvm::cast<clang::VectorType>(parent_qual_type.getTypePtr());
                if (array)
                {
                    ClangASTType element_type (m_ast, array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf(element_name, sizeof(element_name), "[%" PRIu64 "]", static_cast<uint64_t>(idx));
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = (int32_t)idx * (int32_t)child_byte_size;
                        return element_type;
                    }
                }
            }
            break;
            
        case clang::Type::ConstantArray:
        case clang::Type::IncompleteArray:
            if (ignore_array_bounds || idx_is_valid)
            {
                const clang::ArrayType *array = GetQualType()->getAsArrayTypeUnsafe();
                if (array)
                {
                    ClangASTType element_type (m_ast, array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf(element_name, sizeof(element_name), "[%" PRIu64 "]", static_cast<uint64_t>(idx));
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = (int32_t)idx * (int32_t)child_byte_size;
                        return element_type;
                    }
                }
            }
            break;
            
            
        case clang::Type::Pointer:
            if (idx_is_valid)
            {
                ClangASTType pointee_clang_type (GetPointeeType());
                
                // Don't dereference "void *" pointers
                if (pointee_clang_type.IsVoidType())
                    return ClangASTType();
                
                if (transparent_pointers && pointee_clang_type.IsAggregateType ())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                        idx,
                                                                        transparent_pointers,
                                                                        omit_empty_base_classes,
                                                                        ignore_array_bounds,
                                                                        child_name,
                                                                        child_byte_size,
                                                                        child_byte_offset,
                                                                        child_bitfield_bit_size,
                                                                        child_bitfield_bit_offset,
                                                                        child_is_base_class,
                                                                        tmp_child_is_deref_of_parent,
                                                                        valobj);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            if (idx_is_valid)
            {
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(parent_qual_type.getTypePtr());
                ClangASTType pointee_clang_type (m_ast, reference_type->getPointeeType());
                if (transparent_pointers && pointee_clang_type.IsAggregateType ())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                        idx,
                                                                        transparent_pointers,
                                                                        omit_empty_base_classes,
                                                                        ignore_array_bounds,
                                                                        child_name,
                                                                        child_byte_size,
                                                                        child_byte_offset,
                                                                        child_bitfield_bit_size,
                                                                        child_bitfield_bit_offset,
                                                                        child_is_base_class,
                                                                        tmp_child_is_deref_of_parent,
                                                                        valobj);
                }
                else
                {
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '&');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            {
                ClangASTType typedefed_clang_type (m_ast, llvm::cast<clang::TypedefType>(parent_qual_type)->getDecl()->getUnderlyingType());
                return typedefed_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                      idx,
                                                                      transparent_pointers,
                                                                      omit_empty_base_classes,
                                                                      ignore_array_bounds,
                                                                      child_name,
                                                                      child_byte_size,
                                                                      child_byte_offset,
                                                                      child_bitfield_bit_size,
                                                                      child_bitfield_bit_offset,
                                                                      child_is_base_class,
                                                                      child_is_deref_of_parent,
                                                                      valobj);
            }
            break;
            
        case clang::Type::Elaborated:
            {
                ClangASTType elaborated_clang_type (m_ast, llvm::cast<clang::ElaboratedType>(parent_qual_type)->getNamedType());
                return elaborated_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                       idx,
                                                                       transparent_pointers,
                                                                       omit_empty_base_classes,
                                                                       ignore_array_bounds,
                                                                       child_name,
                                                                       child_byte_size,
                                                                       child_byte_offset,
                                                                       child_bitfield_bit_size,
                                                                       child_bitfield_bit_offset,
                                                                       child_is_base_class,
                                                                       child_is_deref_of_parent,
                                                                       valobj);
            }
            
        case clang::Type::Paren:
            {
                ClangASTType paren_clang_type (m_ast, llvm::cast<clang::ParenType>(parent_qual_type)->desugar());
                return paren_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                  idx,
                                                                  transparent_pointers,
                                                                  omit_empty_base_classes,
                                                                  ignore_array_bounds,
                                                                  child_name,
                                                                  child_byte_size,
                                                                  child_byte_offset,
                                                                  child_bitfield_bit_size,
                                                                  child_bitfield_bit_offset,
                                                                  child_is_base_class,
                                                                  child_is_deref_of_parent,
                                                                  valobj);
            }
            
            
        default:
            break;
    }
    return ClangASTType();
}

static inline bool
BaseSpecifierIsEmpty (const clang::CXXBaseSpecifier *b)
{
    return ClangASTContext::RecordHasFields(b->getType()->getAsCXXRecordDecl()) == false;
}

static uint32_t
GetIndexForRecordBase
(
 const clang::RecordDecl *record_decl,
 const clang::CXXBaseSpecifier *base_spec,
 bool omit_empty_base_classes
 )
{
    uint32_t child_idx = 0;
    
    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
    
    //    const char *super_name = record_decl->getNameAsCString();
    //    const char *base_name = base_spec->getType()->getAs<clang::RecordType>()->getDecl()->getNameAsCString();
    //    printf ("GetIndexForRecordChild (%s, %s)\n", super_name, base_name);
    //
    if (cxx_record_decl)
    {
        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
             base_class != base_class_end;
             ++base_class)
        {
            if (omit_empty_base_classes)
            {
                if (BaseSpecifierIsEmpty (base_class))
                    continue;
            }
            
            //            printf ("GetIndexForRecordChild (%s, %s) base[%u] = %s\n", super_name, base_name,
            //                    child_idx,
            //                    base_class->getType()->getAs<clang::RecordType>()->getDecl()->getNameAsCString());
            //
            //
            if (base_class == base_spec)
                return child_idx;
            ++child_idx;
        }
    }
    
    return UINT32_MAX;
}


static uint32_t
GetIndexForRecordChild (const clang::RecordDecl *record_decl,
                        clang::NamedDecl *canonical_decl,
                        bool omit_empty_base_classes)
{
    uint32_t child_idx = ClangASTContext::GetNumBaseClasses (llvm::dyn_cast<clang::CXXRecordDecl>(record_decl),
                                                             omit_empty_base_classes);
    
    clang::RecordDecl::field_iterator field, field_end;
    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
         field != field_end;
         ++field, ++child_idx)
    {
        if (field->getCanonicalDecl() == canonical_decl)
            return child_idx;
    }
    
    return UINT32_MAX;
}

// Look for a child member (doesn't include base classes, but it does include
// their members) in the type hierarchy. Returns an index path into "clang_type"
// on how to reach the appropriate member.
//
//    class A
//    {
//    public:
//        int m_a;
//        int m_b;
//    };
//
//    class B
//    {
//    };
//
//    class C :
//        public B,
//        public A
//    {
//    };
//
// If we have a clang type that describes "class C", and we wanted to looked
// "m_b" in it:
//
// With omit_empty_base_classes == false we would get an integer array back with:
// { 1,  1 }
// The first index 1 is the child index for "class A" within class C
// The second index 1 is the child index for "m_b" within class A
//
// With omit_empty_base_classes == true we would get an integer array back with:
// { 0,  1 }
// The first index 0 is the child index for "class A" within class C (since class B doesn't have any members it doesn't count)
// The second index 1 is the child index for "m_b" within class A

size_t
ClangASTType::GetIndexOfChildMemberWithName (const char *name,
                                             bool omit_empty_base_classes,
                                             std::vector<uint32_t>& child_indexes) const
{
    if (IsValid() && name && name[0])
    {
        clang::QualType qual_type(GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    
                    // Try and find a field that matches NAME
                    clang::RecordDecl::field_iterator field, field_end;
                    llvm::StringRef name_sref(name);
                    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                         field != field_end;
                         ++field, ++child_idx)
                    {
                        llvm::StringRef field_name = field->getName();
                        if (field_name.empty())
                        {
                            ClangASTType field_type(m_ast,field->getType());
                            child_indexes.push_back(child_idx);
                            if (field_type.GetIndexOfChildMemberWithName(name,  omit_empty_base_classes, child_indexes))
                                return child_indexes.size();
                            child_indexes.pop_back();
                                
                        }
                        else if (field_name.equals (name_sref))
                        {
                            // We have to add on the number of base classes to this index!
                            child_indexes.push_back (child_idx + ClangASTContext::GetNumBaseClasses (cxx_record_decl, omit_empty_base_classes));
                            return child_indexes.size();
                        }
                    }
                    
                    if (cxx_record_decl)
                    {
                        const clang::RecordDecl *parent_record_decl = cxx_record_decl;
                        
                        //printf ("parent = %s\n", parent_record_decl->getNameAsCString());
                        
                        //const Decl *root_cdecl = cxx_record_decl->getCanonicalDecl();
                        // Didn't find things easily, lets let clang do its thang...
                        clang::IdentifierInfo & ident_ref = m_ast->Idents.get(name_sref);
                        clang::DeclarationName decl_name(&ident_ref);
                        
                        clang::CXXBasePaths paths;
                        if (cxx_record_decl->lookupInBases(clang::CXXRecordDecl::FindOrdinaryMember,
                                                           decl_name.getAsOpaquePtr(),
                                                           paths))
                        {
                            clang::CXXBasePaths::const_paths_iterator path, path_end = paths.end();
                            for (path = paths.begin(); path != path_end; ++path)
                            {
                                const size_t num_path_elements = path->size();
                                for (size_t e=0; e<num_path_elements; ++e)
                                {
                                    clang::CXXBasePathElement elem = (*path)[e];
                                    
                                    child_idx = GetIndexForRecordBase (parent_record_decl, elem.Base, omit_empty_base_classes);
                                    if (child_idx == UINT32_MAX)
                                    {
                                        child_indexes.clear();
                                        return 0;
                                    }
                                    else
                                    {
                                        child_indexes.push_back (child_idx);
                                        parent_record_decl = llvm::cast<clang::RecordDecl>(elem.Base->getType()->getAs<clang::RecordType>()->getDecl());
                                    }
                                }
                                for (clang::NamedDecl *path_decl : path->Decls)
                                {
                                    child_idx = GetIndexForRecordChild (parent_record_decl, path_decl, omit_empty_base_classes);
                                    if (child_idx == UINT32_MAX)
                                    {
                                        child_indexes.clear();
                                        return 0;
                                    }
                                    else
                                    {
                                        child_indexes.push_back (child_idx);
                                    }
                                }
                            }
                            return child_indexes.size();
                        }
                    }
                    
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType ())
                {
                    llvm::StringRef name_sref(name);
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
                                if (ivar_decl->getName().equals (name_sref))
                                {
                                    if ((!omit_empty_base_classes && superclass_interface_decl) ||
                                        ( omit_empty_base_classes && ObjCDeclHasIVars (superclass_interface_decl, true)))
                                        ++child_idx;
                                    
                                    child_indexes.push_back (child_idx);
                                    return child_indexes.size();
                                }
                            }
                            
                            if (superclass_interface_decl)
                            {
                                // The super class index is always zero for ObjC classes,
                                // so we push it onto the child indexes in case we find
                                // an ivar in our superclass...
                                child_indexes.push_back (0);
                                
                                ClangASTType superclass_clang_type (m_ast, m_ast->getObjCInterfaceType(superclass_interface_decl));
                                if (superclass_clang_type.GetIndexOfChildMemberWithName (name,
                                                                                         omit_empty_base_classes,
                                                                                         child_indexes))
                                {
                                    // We did find an ivar in a superclass so just
                                    // return the results!
                                    return child_indexes.size();
                                }
                                
                                // We didn't find an ivar matching "name" in our
                                // superclass, pop the superclass zero index that
                                // we pushed on above.
                                child_indexes.pop_back();
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                {
                    ClangASTType objc_object_clang_type (m_ast, llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                    return objc_object_clang_type.GetIndexOfChildMemberWithName (name,
                                                                                 omit_empty_base_classes,
                                                                                 child_indexes);
                }
                break;
                
                
            case clang::Type::ConstantArray:
            {
                //                const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(parent_qual_type.getTypePtr());
                //                const uint64_t element_count = array->getSize().getLimitedValue();
                //
                //                if (idx < element_count)
                //                {
                //                    std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(array->getElementType());
                //
                //                    char element_name[32];
                //                    ::snprintf (element_name, sizeof (element_name), "%s[%u]", parent_name ? parent_name : "", idx);
                //
                //                    child_name.assign(element_name);
                //                    assert(field_type_info.first % 8 == 0);
                //                    child_byte_size = field_type_info.first / 8;
                //                    child_byte_offset = idx * child_byte_size;
                //                    return array->getElementType().getAsOpaquePtr();
                //                }
            }
                break;
                
                //        case clang::Type::MemberPointerType:
                //            {
                //                MemberPointerType *mem_ptr_type = llvm::cast<MemberPointerType>(qual_type.getTypePtr());
                //                clang::QualType pointee_type = mem_ptr_type->getPointeeType();
                //
                //                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                //                {
                //                    return GetIndexOfChildWithName (ast,
                //                                                    mem_ptr_type->getPointeeType().getAsOpaquePtr(),
                //                                                    name);
                //                }
                //            }
                //            break;
                //
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                    clang::QualType pointee_type(reference_type->getPointeeType());
                    ClangASTType pointee_clang_type (m_ast, pointee_type);
                    
                    if (pointee_clang_type.IsAggregateType ())
                    {
                        return pointee_clang_type.GetIndexOfChildMemberWithName (name,
                                                                                 omit_empty_base_classes,
                                                                                 child_indexes);
                    }
                }
                break;
                
            case clang::Type::Pointer:
            {
                ClangASTType pointee_clang_type (GetPointeeType());
                
                if (pointee_clang_type.IsAggregateType ())
                {
                    return pointee_clang_type.GetIndexOfChildMemberWithName (name,
                                                                             omit_empty_base_classes,
                                                                             child_indexes);
                }
            }
                break;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildMemberWithName (name,
                                                                                                                                         omit_empty_base_classes,
                                                                                                                                         child_indexes);
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildMemberWithName (name,
                                                                                                                            omit_empty_base_classes,
                                                                                                                            child_indexes);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildMemberWithName (name,
                                                                                                                         omit_empty_base_classes,
                                                                                                                         child_indexes);
                
            default:
                break;
        }
    }
    return 0;
}


// Get the index of the child of "clang_type" whose name matches. This function
// doesn't descend into the children, but only looks one level deep and name
// matches can include base class names.

uint32_t
ClangASTType::GetIndexOfChildWithName (const char *name, bool omit_empty_base_classes) const
{
    if (IsValid() && name && name[0])
    {
        clang::QualType qual_type(GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    
                    if (cxx_record_decl)
                    {
                        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            // Skip empty base classes
                            clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            if (omit_empty_base_classes && ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                            
                            ClangASTType base_class_clang_type (m_ast, base_class->getType());
                            std::string base_class_type_name (base_class_clang_type.GetTypeName().AsCString(""));
                            if (base_class_type_name.compare (name) == 0)
                                return child_idx;
                            ++child_idx;
                        }
                    }
                    
                    // Try and find a field that matches NAME
                    clang::RecordDecl::field_iterator field, field_end;
                    llvm::StringRef name_sref(name);
                    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                         field != field_end;
                         ++field, ++child_idx)
                    {
                        if (field->getName().equals (name_sref))
                            return child_idx;
                    }
                    
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType())
                {
                    llvm::StringRef name_sref(name);
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
                                if (ivar_decl->getName().equals (name_sref))
                                {
                                    if ((!omit_empty_base_classes && superclass_interface_decl) ||
                                        ( omit_empty_base_classes && ObjCDeclHasIVars (superclass_interface_decl, true)))
                                        ++child_idx;
                                    
                                    return child_idx;
                                }
                            }
                            
                            if (superclass_interface_decl)
                            {
                                if (superclass_interface_decl->getName().equals (name_sref))
                                    return 0;
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                {
                    ClangASTType pointee_clang_type (m_ast, llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                    return pointee_clang_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                }
                break;
                
            case clang::Type::ConstantArray:
            {
                //                const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(parent_qual_type.getTypePtr());
                //                const uint64_t element_count = array->getSize().getLimitedValue();
                //
                //                if (idx < element_count)
                //                {
                //                    std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(array->getElementType());
                //
                //                    char element_name[32];
                //                    ::snprintf (element_name, sizeof (element_name), "%s[%u]", parent_name ? parent_name : "", idx);
                //
                //                    child_name.assign(element_name);
                //                    assert(field_type_info.first % 8 == 0);
                //                    child_byte_size = field_type_info.first / 8;
                //                    child_byte_offset = idx * child_byte_size;
                //                    return array->getElementType().getAsOpaquePtr();
                //                }
            }
                break;
                
                //        case clang::Type::MemberPointerType:
                //            {
                //                MemberPointerType *mem_ptr_type = llvm::cast<MemberPointerType>(qual_type.getTypePtr());
                //                clang::QualType pointee_type = mem_ptr_type->getPointeeType();
                //
                //                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                //                {
                //                    return GetIndexOfChildWithName (ast,
                //                                                    mem_ptr_type->getPointeeType().getAsOpaquePtr(),
                //                                                    name);
                //                }
                //            }
                //            break;
                //
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                    ClangASTType pointee_type (m_ast, reference_type->getPointeeType());
                    
                    if (pointee_type.IsAggregateType ())
                    {
                        return pointee_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                    }
                }
                break;
                
            case clang::Type::Pointer:
                {
                    const clang::PointerType *pointer_type = llvm::cast<clang::PointerType>(qual_type.getTypePtr());
                    ClangASTType pointee_type (m_ast, pointer_type->getPointeeType());
                    
                    if (pointee_type.IsAggregateType ())
                    {
                        return pointee_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                    }
                    else
                    {
                        //                    if (parent_name)
                        //                    {
                        //                        child_name.assign(1, '*');
                        //                        child_name += parent_name;
                        //                    }
                        //
                        //                    // We have a pointer to an simple type
                        //                    if (idx == 0)
                        //                    {
                        //                        std::pair<uint64_t, unsigned> clang_type_info = ast->getTypeInfo(pointee_type);
                        //                        assert(clang_type_info.first % 8 == 0);
                        //                        child_byte_size = clang_type_info.first / 8;
                        //                        child_byte_offset = 0;
                        //                        return pointee_type.getAsOpaquePtr();
                        //                    }
                    }
                }
                break;
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            default:
                break;
        }
    }
    return UINT32_MAX;
}


size_t
ClangASTType::GetNumTemplateArguments () const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                    if (cxx_record_decl)
                    {
                        const clang::ClassTemplateSpecializationDecl *template_decl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(cxx_record_decl);
                        if (template_decl)
                            return template_decl->getTemplateArgs().size();
                    }
                }
                break;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumTemplateArguments();
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumTemplateArguments();
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumTemplateArguments();
                
            default:
                break;
        }
    }
    return 0;
}

ClangASTType
ClangASTType::GetTemplateArgument (size_t arg_idx, lldb::TemplateArgumentKind &kind) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                    if (cxx_record_decl)
                    {
                        const clang::ClassTemplateSpecializationDecl *template_decl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(cxx_record_decl);
                        if (template_decl && arg_idx < template_decl->getTemplateArgs().size())
                        {
                            const clang::TemplateArgument &template_arg = template_decl->getTemplateArgs()[arg_idx];
                            switch (template_arg.getKind())
                            {
                                case clang::TemplateArgument::Null:
                                    kind = eTemplateArgumentKindNull;
                                    return ClangASTType();
                                    
                                case clang::TemplateArgument::Type:
                                    kind = eTemplateArgumentKindType;
                                    return ClangASTType(m_ast, template_arg.getAsType());
                                    
                                case clang::TemplateArgument::Declaration:
                                    kind = eTemplateArgumentKindDeclaration;
                                    return ClangASTType();
                                    
                                case clang::TemplateArgument::Integral:
                                    kind = eTemplateArgumentKindIntegral;
                                    return ClangASTType(m_ast, template_arg.getIntegralType());
                                    
                                case clang::TemplateArgument::Template:
                                    kind = eTemplateArgumentKindTemplate;
                                    return ClangASTType();
                                    
                                case clang::TemplateArgument::TemplateExpansion:
                                    kind = eTemplateArgumentKindTemplateExpansion;
                                    return ClangASTType();
                                    
                                case clang::TemplateArgument::Expression:
                                    kind = eTemplateArgumentKindExpression;
                                    return ClangASTType();
                                    
                                case clang::TemplateArgument::Pack:
                                    kind = eTemplateArgumentKindPack;
                                    return ClangASTType();
                                    
                                default:
                                    assert (!"Unhandled clang::TemplateArgument::ArgKind");
                                    break;
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetTemplateArgument (arg_idx, kind);
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetTemplateArgument (arg_idx, kind);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetTemplateArgument (arg_idx, kind);
                
            default:
                break;
        }
    }
    kind = eTemplateArgumentKindNull;
    return ClangASTType ();
}

static bool
IsOperator (const char *name, clang::OverloadedOperatorKind &op_kind)
{
    if (name == nullptr || name[0] == '\0')
        return false;
    
#define OPERATOR_PREFIX "operator"
#define OPERATOR_PREFIX_LENGTH (sizeof (OPERATOR_PREFIX) - 1)
    
    const char *post_op_name = nullptr;
    
    bool no_space = true;
    
    if (::strncmp(name, OPERATOR_PREFIX, OPERATOR_PREFIX_LENGTH))
        return false;
    
    post_op_name = name + OPERATOR_PREFIX_LENGTH;
    
    if (post_op_name[0] == ' ')
    {
        post_op_name++;
        no_space = false;
    }
    
#undef OPERATOR_PREFIX
#undef OPERATOR_PREFIX_LENGTH
    
    // This is an operator, set the overloaded operator kind to invalid
    // in case this is a conversion operator...
    op_kind = clang::NUM_OVERLOADED_OPERATORS;
    
    switch (post_op_name[0])
    {
        default:
            if (no_space)
                return false;
            break;
        case 'n':
            if (no_space)
                return false;
            if  (strcmp (post_op_name, "new") == 0)
                op_kind = clang::OO_New;
            else if (strcmp (post_op_name, "new[]") == 0)
                op_kind = clang::OO_Array_New;
            break;
            
        case 'd':
            if (no_space)
                return false;
            if (strcmp (post_op_name, "delete") == 0)
                op_kind = clang::OO_Delete;
            else if (strcmp (post_op_name, "delete[]") == 0)
                op_kind = clang::OO_Array_Delete;
            break;
            
        case '+':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Plus;
            else if (post_op_name[2] == '\0')
            {
                if (post_op_name[1] == '=')
                    op_kind = clang::OO_PlusEqual;
                else if (post_op_name[1] == '+')
                    op_kind = clang::OO_PlusPlus;
            }
            break;
            
        case '-':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Minus;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_MinusEqual; break;
                    case '-': op_kind = clang::OO_MinusMinus; break;
                    case '>': op_kind = clang::OO_Arrow; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '*')
                    op_kind = clang::OO_ArrowStar; break;
            }
            break;
            
        case '*':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Star;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_StarEqual;
            break;
            
        case '/':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Slash;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_SlashEqual;
            break;
            
        case '%':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Percent;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_PercentEqual;
            break;
            
            
        case '^':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Caret;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_CaretEqual;
            break;
            
        case '&':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Amp;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_AmpEqual; break;
                    case '&': op_kind = clang::OO_AmpAmp; break;
                }
            }
            break;
            
        case '|':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Pipe;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_PipeEqual; break;
                    case '|': op_kind = clang::OO_PipePipe; break;
                }
            }
            break;
            
        case '~':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Tilde;
            break;
            
        case '!':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Exclaim;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_ExclaimEqual;
            break;
            
        case '=':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Equal;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_EqualEqual;
            break;
            
        case '<':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Less;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '<': op_kind = clang::OO_LessLess; break;
                    case '=': op_kind = clang::OO_LessEqual; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '=')
                    op_kind = clang::OO_LessLessEqual;
            }
            break;
            
        case '>':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Greater;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '>': op_kind = clang::OO_GreaterGreater; break;
                    case '=': op_kind = clang::OO_GreaterEqual; break;
                }
            }
            else if (post_op_name[1] == '>' &&
                     post_op_name[2] == '=' &&
                     post_op_name[3] == '\0')
            {
                op_kind = clang::OO_GreaterGreaterEqual;
            }
            break;
            
        case ',':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Comma;
            break;
            
        case '(':
            if (post_op_name[1] == ')' && post_op_name[2] == '\0')
                op_kind = clang::OO_Call;
            break;
            
        case '[':
            if (post_op_name[1] == ']' && post_op_name[2] == '\0')
                op_kind = clang::OO_Subscript;
            break;
    }
    
    return true;
}

clang::EnumDecl *
ClangASTType::GetAsEnumDecl () const
{
    const clang::EnumType *enum_type = llvm::dyn_cast<clang::EnumType>(GetCanonicalQualType());
    if (enum_type)
        return enum_type->getDecl();
    return NULL;
}

clang::RecordDecl *
ClangASTType::GetAsRecordDecl () const
{
    const clang::RecordType *record_type = llvm::dyn_cast<clang::RecordType>(GetCanonicalQualType());
    if (record_type)
        return record_type->getDecl();
    return nullptr;
}

clang::CXXRecordDecl *
ClangASTType::GetAsCXXRecordDecl () const
{
    return GetCanonicalQualType()->getAsCXXRecordDecl();
}

clang::ObjCInterfaceDecl *
ClangASTType::GetAsObjCInterfaceDecl () const
{
    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(GetCanonicalQualType());
    if (objc_class_type)
        return objc_class_type->getInterface();
    return nullptr;
}

clang::FieldDecl *
ClangASTType::AddFieldToRecordType (const char *name,
                                    const ClangASTType &field_clang_type,
                                    AccessType access,
                                    uint32_t bitfield_bit_size)
{
    if (!IsValid() || !field_clang_type.IsValid())
        return nullptr;
    
    clang::FieldDecl *field = nullptr;

    clang::Expr *bit_width = nullptr;
    if (bitfield_bit_size != 0)
    {
        llvm::APInt bitfield_bit_size_apint(m_ast->getTypeSize(m_ast->IntTy), bitfield_bit_size);
        bit_width = new (*m_ast)clang::IntegerLiteral (*m_ast, bitfield_bit_size_apint, m_ast->IntTy, clang::SourceLocation());
    }

    clang::RecordDecl *record_decl = GetAsRecordDecl ();
    if (record_decl)
    {
        field = clang::FieldDecl::Create (*m_ast,
                                          record_decl,
                                          clang::SourceLocation(),
                                          clang::SourceLocation(),
                                          name ? &m_ast->Idents.get(name) : nullptr,  // Identifier
                                          field_clang_type.GetQualType(),             // Field type
                                          nullptr,                                    // TInfo *
                                          bit_width,                                  // BitWidth
                                          false,                                      // Mutable
                                          clang::ICIS_NoInit);                        // HasInit
        
        if (!name)
        {
            // Determine whether this field corresponds to an anonymous
            // struct or union.
            if (const clang::TagType *TagT = field->getType()->getAs<clang::TagType>()) {
                if (clang::RecordDecl *Rec = llvm::dyn_cast<clang::RecordDecl>(TagT->getDecl()))
                    if (!Rec->getDeclName()) {
                        Rec->setAnonymousStructOrUnion(true);
                        field->setImplicit();
                        
                    }
            }
        }
        
        if (field)
        {
            field->setAccess (ClangASTContext::ConvertAccessTypeToAccessSpecifier (access));
            
            record_decl->addDecl(field);
            
#ifdef LLDB_CONFIGURATION_DEBUG
            VerifyDecl(field);
#endif
        }
    }
    else
    {
        clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
        
        if (class_interface_decl)
        {
            const bool is_synthesized = false;
            
            field_clang_type.GetCompleteType();

            field = clang::ObjCIvarDecl::Create (*m_ast,
                                          class_interface_decl,
                                          clang::SourceLocation(),
                                          clang::SourceLocation(),
                                          name ? &m_ast->Idents.get(name) : nullptr,   // Identifier
                                          field_clang_type.GetQualType(),           // Field type
                                          nullptr,                                     // TypeSourceInfo *
                                          ConvertAccessTypeToObjCIvarAccessControl (access),
                                          bit_width,
                                          is_synthesized);
            
            if (field)
            {
                class_interface_decl->addDecl(field);
                
#ifdef LLDB_CONFIGURATION_DEBUG
                VerifyDecl(field);
#endif
            }
        }
    }
    return field;
}

void
ClangASTType::BuildIndirectFields ()
{
    clang::RecordDecl *record_decl = GetAsRecordDecl();
    
    if (!record_decl)
        return;
    
    typedef llvm::SmallVector <clang::IndirectFieldDecl *, 1> IndirectFieldVector;
    
    IndirectFieldVector indirect_fields;
    clang::RecordDecl::field_iterator field_pos;
    clang::RecordDecl::field_iterator field_end_pos = record_decl->field_end();
    clang::RecordDecl::field_iterator last_field_pos = field_end_pos;
    for (field_pos = record_decl->field_begin(); field_pos != field_end_pos; last_field_pos = field_pos++)
    {
        if (field_pos->isAnonymousStructOrUnion())
        {
            clang::QualType field_qual_type = field_pos->getType();
            
            const clang::RecordType *field_record_type = field_qual_type->getAs<clang::RecordType>();
            
            if (!field_record_type)
                continue;
            
            clang::RecordDecl *field_record_decl = field_record_type->getDecl();
            
            if (!field_record_decl)
                continue;
            
            for (clang::RecordDecl::decl_iterator di = field_record_decl->decls_begin(), de = field_record_decl->decls_end();
                 di != de;
                 ++di)
            {
                if (clang::FieldDecl *nested_field_decl = llvm::dyn_cast<clang::FieldDecl>(*di))
                {
                    clang::NamedDecl **chain = new (*m_ast) clang::NamedDecl*[2];
                    chain[0] = *field_pos;
                    chain[1] = nested_field_decl;
                    clang::IndirectFieldDecl *indirect_field = clang::IndirectFieldDecl::Create(*m_ast,
                                                                                                record_decl,
                                                                                                clang::SourceLocation(),
                                                                                                nested_field_decl->getIdentifier(),
                                                                                                nested_field_decl->getType(),
                                                                                                chain,
                                                                                                2);
                    
                    indirect_field->setImplicit();
                    
                    indirect_field->setAccess(ClangASTContext::UnifyAccessSpecifiers(field_pos->getAccess(),
                                                                                     nested_field_decl->getAccess()));
                    
                    indirect_fields.push_back(indirect_field);
                }
                else if (clang::IndirectFieldDecl *nested_indirect_field_decl = llvm::dyn_cast<clang::IndirectFieldDecl>(*di))
                {
                    int nested_chain_size = nested_indirect_field_decl->getChainingSize();
                    clang::NamedDecl **chain = new (*m_ast) clang::NamedDecl*[nested_chain_size + 1];
                    chain[0] = *field_pos;
                    
                    int chain_index = 1;
                    for (clang::IndirectFieldDecl::chain_iterator nci = nested_indirect_field_decl->chain_begin(),
                         nce = nested_indirect_field_decl->chain_end();
                         nci < nce;
                         ++nci)
                    {
                        chain[chain_index] = *nci;
                        chain_index++;
                    }
                    
                    clang::IndirectFieldDecl *indirect_field = clang::IndirectFieldDecl::Create(*m_ast,
                                                                                                record_decl,
                                                                                                clang::SourceLocation(),
                                                                                                nested_indirect_field_decl->getIdentifier(),
                                                                                                nested_indirect_field_decl->getType(),
                                                                                                chain,
                                                                                                nested_chain_size + 1);
                    
                    indirect_field->setImplicit();
                    
                    indirect_field->setAccess(ClangASTContext::UnifyAccessSpecifiers(field_pos->getAccess(),
                                                                                     nested_indirect_field_decl->getAccess()));
                    
                    indirect_fields.push_back(indirect_field);
                }
            }
        }
    }
    
    // Check the last field to see if it has an incomplete array type as its
    // last member and if it does, the tell the record decl about it
    if (last_field_pos != field_end_pos)
    {
        if (last_field_pos->getType()->isIncompleteArrayType())
            record_decl->hasFlexibleArrayMember();
    }
    
    for (IndirectFieldVector::iterator ifi = indirect_fields.begin(), ife = indirect_fields.end();
         ifi < ife;
         ++ifi)
    {
        record_decl->addDecl(*ifi);
    }
}

void
ClangASTType::SetIsPacked ()
{
    clang::RecordDecl *record_decl = GetAsRecordDecl();
    
    if (!record_decl)
        return;
    
    record_decl->addAttr(clang::PackedAttr::CreateImplicit(*m_ast));
}

clang::VarDecl *
ClangASTType::AddVariableToRecordType (const char *name,
                                       const ClangASTType &var_type,
                                       AccessType access)
{
    clang::VarDecl *var_decl = nullptr;
    
    if (!IsValid() || !var_type.IsValid())
        return nullptr;
    
    clang::RecordDecl *record_decl = GetAsRecordDecl ();
    if (record_decl)
    {        
        var_decl = clang::VarDecl::Create (*m_ast,                                     // ASTContext &
                                           record_decl,                                // DeclContext *
                                           clang::SourceLocation(),                    // clang::SourceLocation StartLoc
                                           clang::SourceLocation(),                    // clang::SourceLocation IdLoc
                                           name ? &m_ast->Idents.get(name) : nullptr,  // clang::IdentifierInfo *
                                           var_type.GetQualType(),                     // Variable clang::QualType
                                           nullptr,                                    // TypeSourceInfo *
                                           clang::SC_Static);                          // StorageClass
        if (var_decl)
        {
            var_decl->setAccess(ClangASTContext::ConvertAccessTypeToAccessSpecifier (access));
            record_decl->addDecl(var_decl);
            
#ifdef LLDB_CONFIGURATION_DEBUG
            VerifyDecl(var_decl);
#endif
        }
    }
    return var_decl;
}


clang::CXXMethodDecl *
ClangASTType::AddMethodToCXXRecordType (const char *name,
                                        const ClangASTType &method_clang_type,
                                        lldb::AccessType access,
                                        bool is_virtual,
                                        bool is_static,
                                        bool is_inline,
                                        bool is_explicit,
                                        bool is_attr_used,
                                        bool is_artificial)
{
    if (!IsValid() || !method_clang_type.IsValid() || name == nullptr || name[0] == '\0')
        return nullptr;
    
    clang::QualType record_qual_type(GetCanonicalQualType());
    
    clang::CXXRecordDecl *cxx_record_decl = record_qual_type->getAsCXXRecordDecl();
    
    if (cxx_record_decl == nullptr)
        return nullptr;
    
    clang::QualType method_qual_type (method_clang_type.GetQualType());
    
    clang::CXXMethodDecl *cxx_method_decl = nullptr;
    
    clang::DeclarationName decl_name (&m_ast->Idents.get(name));
    
    const clang::FunctionType *function_type = llvm::dyn_cast<clang::FunctionType>(method_qual_type.getTypePtr());
    
    if (function_type == nullptr)
        return nullptr;
    
    const clang::FunctionProtoType *method_function_prototype (llvm::dyn_cast<clang::FunctionProtoType>(function_type));
    
    if (!method_function_prototype)
        return nullptr;
    
    unsigned int num_params = method_function_prototype->getNumParams();
    
    clang::CXXDestructorDecl *cxx_dtor_decl(nullptr);
    clang::CXXConstructorDecl *cxx_ctor_decl(nullptr);
    
    if (is_artificial)
        return nullptr; // skip everything artificial
    
    if (name[0] == '~')
    {
        cxx_dtor_decl = clang::CXXDestructorDecl::Create (*m_ast,
                                                          cxx_record_decl,
                                                          clang::SourceLocation(),
                                                          clang::DeclarationNameInfo (m_ast->DeclarationNames.getCXXDestructorName (m_ast->getCanonicalType (record_qual_type)), clang::SourceLocation()),
                                                          method_qual_type,
                                                          nullptr,
                                                          is_inline,
                                                          is_artificial);
        cxx_method_decl = cxx_dtor_decl;
    }
    else if (decl_name == cxx_record_decl->getDeclName())
    {
        cxx_ctor_decl = clang::CXXConstructorDecl::Create (*m_ast,
                                                           cxx_record_decl,
                                                           clang::SourceLocation(),
                                                           clang::DeclarationNameInfo (m_ast->DeclarationNames.getCXXConstructorName (m_ast->getCanonicalType (record_qual_type)), clang::SourceLocation()),
                                                           method_qual_type,
                                                           nullptr, // TypeSourceInfo *
                                                           is_explicit,
                                                           is_inline,
                                                           is_artificial,
                                                           false /*is_constexpr*/);
        cxx_method_decl = cxx_ctor_decl;
    }
    else
    {
        clang::StorageClass SC = is_static ? clang::SC_Static : clang::SC_None;
        clang::OverloadedOperatorKind op_kind = clang::NUM_OVERLOADED_OPERATORS;
        
        if (IsOperator (name, op_kind))
        {
            if (op_kind != clang::NUM_OVERLOADED_OPERATORS)
            {
                // Check the number of operator parameters. Sometimes we have
                // seen bad DWARF that doesn't correctly describe operators and
                // if we try to create a method and add it to the class, clang
                // will assert and crash, so we need to make sure things are
                // acceptable.
                if (!ClangASTContext::CheckOverloadedOperatorKindParameterCount (op_kind, num_params))
                    return nullptr;
                cxx_method_decl = clang::CXXMethodDecl::Create (*m_ast,
                                                                cxx_record_decl,
                                                                clang::SourceLocation(),
                                                                clang::DeclarationNameInfo (m_ast->DeclarationNames.getCXXOperatorName (op_kind), clang::SourceLocation()),
                                                                method_qual_type,
                                                                nullptr, // TypeSourceInfo *
                                                                SC,
                                                                is_inline,
                                                                false /*is_constexpr*/,
                                                                clang::SourceLocation());
            }
            else if (num_params == 0)
            {
                // Conversion operators don't take params...
                cxx_method_decl = clang::CXXConversionDecl::Create (*m_ast,
                                                                    cxx_record_decl,
                                                                    clang::SourceLocation(),
                                                                    clang::DeclarationNameInfo (m_ast->DeclarationNames.getCXXConversionFunctionName (m_ast->getCanonicalType (function_type->getReturnType())), clang::SourceLocation()),
                                                                    method_qual_type,
                                                                    nullptr, // TypeSourceInfo *
                                                                    is_inline,
                                                                    is_explicit,
                                                                    false /*is_constexpr*/,
                                                                    clang::SourceLocation());
            }
        }
        
        if (cxx_method_decl == nullptr)
        {
            cxx_method_decl = clang::CXXMethodDecl::Create (*m_ast,
                                                            cxx_record_decl,
                                                            clang::SourceLocation(),
                                                            clang::DeclarationNameInfo (decl_name, clang::SourceLocation()),
                                                            method_qual_type,
                                                            nullptr, // TypeSourceInfo *
                                                            SC,
                                                            is_inline,
                                                            false /*is_constexpr*/,
                                                            clang::SourceLocation());
        }
    }
    
    clang::AccessSpecifier access_specifier = ClangASTContext::ConvertAccessTypeToAccessSpecifier (access);
    
    cxx_method_decl->setAccess (access_specifier);
    cxx_method_decl->setVirtualAsWritten (is_virtual);
    
    if (is_attr_used)
        cxx_method_decl->addAttr(clang::UsedAttr::CreateImplicit(*m_ast));
    
    // Populate the method decl with parameter decls
    
    llvm::SmallVector<clang::ParmVarDecl *, 12> params;
    
    for (unsigned param_index = 0;
         param_index < num_params;
         ++param_index)
    {
        params.push_back (clang::ParmVarDecl::Create (*m_ast,
                                                      cxx_method_decl,
                                                      clang::SourceLocation(),
                                                      clang::SourceLocation(),
                                                      nullptr, // anonymous
                                                      method_function_prototype->getParamType(param_index),
                                                      nullptr,
                                                      clang::SC_None,
                                                      nullptr));
    }
    
    cxx_method_decl->setParams (llvm::ArrayRef<clang::ParmVarDecl*>(params));
    
    cxx_record_decl->addDecl (cxx_method_decl);
    
    // Sometimes the debug info will mention a constructor (default/copy/move),
    // destructor, or assignment operator (copy/move) but there won't be any
    // version of this in the code. So we check if the function was artificially
    // generated and if it is trivial and this lets the compiler/backend know
    // that it can inline the IR for these when it needs to and we can avoid a
    // "missing function" error when running expressions.
    
    if (is_artificial)
    {
        if (cxx_ctor_decl &&
            ((cxx_ctor_decl->isDefaultConstructor() && cxx_record_decl->hasTrivialDefaultConstructor ()) ||
             (cxx_ctor_decl->isCopyConstructor()    && cxx_record_decl->hasTrivialCopyConstructor    ()) ||
             (cxx_ctor_decl->isMoveConstructor()    && cxx_record_decl->hasTrivialMoveConstructor    ()) ))
        {
            cxx_ctor_decl->setDefaulted();
            cxx_ctor_decl->setTrivial(true);
        }
        else if (cxx_dtor_decl)
        {
            if (cxx_record_decl->hasTrivialDestructor())
            {
                cxx_dtor_decl->setDefaulted();
                cxx_dtor_decl->setTrivial(true);
            }
        }
        else if ((cxx_method_decl->isCopyAssignmentOperator() && cxx_record_decl->hasTrivialCopyAssignment()) ||
                 (cxx_method_decl->isMoveAssignmentOperator() && cxx_record_decl->hasTrivialMoveAssignment()))
        {
            cxx_method_decl->setDefaulted();
            cxx_method_decl->setTrivial(true);
        }
    }
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(cxx_method_decl);
#endif
    
    //    printf ("decl->isPolymorphic()             = %i\n", cxx_record_decl->isPolymorphic());
    //    printf ("decl->isAggregate()               = %i\n", cxx_record_decl->isAggregate());
    //    printf ("decl->isPOD()                     = %i\n", cxx_record_decl->isPOD());
    //    printf ("decl->isEmpty()                   = %i\n", cxx_record_decl->isEmpty());
    //    printf ("decl->isAbstract()                = %i\n", cxx_record_decl->isAbstract());
    //    printf ("decl->hasTrivialConstructor()     = %i\n", cxx_record_decl->hasTrivialConstructor());
    //    printf ("decl->hasTrivialCopyConstructor() = %i\n", cxx_record_decl->hasTrivialCopyConstructor());
    //    printf ("decl->hasTrivialCopyAssignment()  = %i\n", cxx_record_decl->hasTrivialCopyAssignment());
    //    printf ("decl->hasTrivialDestructor()      = %i\n", cxx_record_decl->hasTrivialDestructor());
    return cxx_method_decl;
}


#pragma mark C++ Base Classes

clang::CXXBaseSpecifier *
ClangASTType::CreateBaseClassSpecifier (AccessType access, bool is_virtual, bool base_of_class)
{
    if (IsValid())
        return new clang::CXXBaseSpecifier (clang::SourceRange(),
                                            is_virtual,
                                            base_of_class,
                                            ClangASTContext::ConvertAccessTypeToAccessSpecifier (access),
                                            m_ast->getTrivialTypeSourceInfo (GetQualType()),
                                            clang::SourceLocation());
    return nullptr;
}

void
ClangASTType::DeleteBaseClassSpecifiers (clang::CXXBaseSpecifier **base_classes, unsigned num_base_classes)
{
    for (unsigned i=0; i<num_base_classes; ++i)
    {
        delete base_classes[i];
        base_classes[i] = nullptr;
    }
}

bool
ClangASTType::SetBaseClassesForClassType (clang::CXXBaseSpecifier const * const *base_classes,
                                          unsigned num_base_classes)
{
    if (IsValid())
    {
        clang::CXXRecordDecl *cxx_record_decl = GetAsCXXRecordDecl();
        if (cxx_record_decl)
        {
            cxx_record_decl->setBases(base_classes, num_base_classes);
            return true;
        }
    }
    return false;
}

bool
ClangASTType::SetObjCSuperClass (const ClangASTType &superclass_clang_type)
{
    if (IsValid() && superclass_clang_type.IsValid())
    {
        clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
        clang::ObjCInterfaceDecl *super_interface_decl = superclass_clang_type.GetAsObjCInterfaceDecl ();
        if (class_interface_decl && super_interface_decl)
        {

            class_interface_decl->setSuperClass(
                    m_ast->getTrivialTypeSourceInfo(m_ast->getObjCInterfaceType(super_interface_decl)));
            return true;
        }
    }
    return false;
}

bool
ClangASTType::AddObjCClassProperty (const char *property_name,
                                    const ClangASTType &property_clang_type,
                                    clang::ObjCIvarDecl *ivar_decl,
                                    const char *property_setter_name,
                                    const char *property_getter_name,
                                    uint32_t property_attributes,
                                    ClangASTMetadata *metadata)
{
    if (!IsValid() || !property_clang_type.IsValid() || property_name == nullptr || property_name[0] == '\0')
        return false;
        
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
    
    if (class_interface_decl)
    {
        ClangASTType property_clang_type_to_access;
        
        if (property_clang_type.IsValid())
            property_clang_type_to_access = property_clang_type;
        else if (ivar_decl)
            property_clang_type_to_access = ClangASTType (m_ast, ivar_decl->getType());
        
        if (class_interface_decl && property_clang_type_to_access.IsValid())
        {
            clang::TypeSourceInfo *prop_type_source;
            if (ivar_decl)
                prop_type_source = m_ast->getTrivialTypeSourceInfo (ivar_decl->getType());
            else
                prop_type_source = m_ast->getTrivialTypeSourceInfo (property_clang_type.GetQualType());
            
            clang::ObjCPropertyDecl *property_decl = clang::ObjCPropertyDecl::Create (*m_ast,
                                                                                      class_interface_decl,
                                                                                      clang::SourceLocation(), // Source Location
                                                                                      &m_ast->Idents.get(property_name),
                                                                                      clang::SourceLocation(), //Source Location for AT
                                                                                      clang::SourceLocation(), //Source location for (
                                                                                      ivar_decl ? ivar_decl->getType() : property_clang_type.GetQualType(),
                                                                                      prop_type_source);
            
            if (property_decl)
            {
                if (metadata)
                    ClangASTContext::SetMetadata(m_ast, property_decl, *metadata);
                
                class_interface_decl->addDecl (property_decl);
                
                clang::Selector setter_sel, getter_sel;
                
                if (property_setter_name != nullptr)
                {
                    std::string property_setter_no_colon(property_setter_name, strlen(property_setter_name) - 1);
                    clang::IdentifierInfo *setter_ident = &m_ast->Idents.get(property_setter_no_colon.c_str());
                    setter_sel = m_ast->Selectors.getSelector(1, &setter_ident);
                }
                else if (!(property_attributes & DW_APPLE_PROPERTY_readonly))
                {
                    std::string setter_sel_string("set");
                    setter_sel_string.push_back(::toupper(property_name[0]));
                    setter_sel_string.append(&property_name[1]);
                    clang::IdentifierInfo *setter_ident = &m_ast->Idents.get(setter_sel_string.c_str());
                    setter_sel = m_ast->Selectors.getSelector(1, &setter_ident);
                }
                property_decl->setSetterName(setter_sel);
                property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_setter);
                
                if (property_getter_name != nullptr)
                {
                    clang::IdentifierInfo *getter_ident = &m_ast->Idents.get(property_getter_name);
                    getter_sel = m_ast->Selectors.getSelector(0, &getter_ident);
                }
                else
                {
                    clang::IdentifierInfo *getter_ident = &m_ast->Idents.get(property_name);
                    getter_sel = m_ast->Selectors.getSelector(0, &getter_ident);
                }
                property_decl->setGetterName(getter_sel);
                property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_getter);
                
                if (ivar_decl)
                    property_decl->setPropertyIvarDecl (ivar_decl);
                
                if (property_attributes & DW_APPLE_PROPERTY_readonly)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_readonly);
                if (property_attributes & DW_APPLE_PROPERTY_readwrite)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_readwrite);
                if (property_attributes & DW_APPLE_PROPERTY_assign)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_assign);
                if (property_attributes & DW_APPLE_PROPERTY_retain)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_retain);
                if (property_attributes & DW_APPLE_PROPERTY_copy)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_copy);
                if (property_attributes & DW_APPLE_PROPERTY_nonatomic)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_nonatomic);
                
                if (!getter_sel.isNull() && !class_interface_decl->lookupInstanceMethod(getter_sel))
                {
                    const bool isInstance = true;
                    const bool isVariadic = false;
                    const bool isSynthesized = false;
                    const bool isImplicitlyDeclared = true;
                    const bool isDefined = false;
                    const clang::ObjCMethodDecl::ImplementationControl impControl = clang::ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    clang::ObjCMethodDecl *getter = clang::ObjCMethodDecl::Create (*m_ast,
                                                                                   clang::SourceLocation(),
                                                                                   clang::SourceLocation(),
                                                                                   getter_sel,
                                                                                   property_clang_type_to_access.GetQualType(),
                                                                                   nullptr,
                                                                                   class_interface_decl,
                                                                                   isInstance,
                                                                                   isVariadic,
                                                                                   isSynthesized,
                                                                                   isImplicitlyDeclared,
                                                                                   isDefined,
                                                                                   impControl,
                                                                                   HasRelatedResultType);
                    
                    if (getter && metadata)
                        ClangASTContext::SetMetadata(m_ast, getter, *metadata);
                    
                    if (getter)
                    {
                        getter->setMethodParams(*m_ast, llvm::ArrayRef<clang::ParmVarDecl*>(), llvm::ArrayRef<clang::SourceLocation>());
                    
                        class_interface_decl->addDecl(getter);
                    }
                }
                
                if (!setter_sel.isNull() && !class_interface_decl->lookupInstanceMethod(setter_sel))
                {
                    clang::QualType result_type = m_ast->VoidTy;
                    
                    const bool isInstance = true;
                    const bool isVariadic = false;
                    const bool isSynthesized = false;
                    const bool isImplicitlyDeclared = true;
                    const bool isDefined = false;
                    const clang::ObjCMethodDecl::ImplementationControl impControl = clang::ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    clang::ObjCMethodDecl *setter = clang::ObjCMethodDecl::Create (*m_ast,
                                                                                   clang::SourceLocation(),
                                                                                   clang::SourceLocation(),
                                                                                   setter_sel,
                                                                                   result_type,
                                                                                   nullptr,
                                                                                   class_interface_decl,
                                                                                   isInstance,
                                                                                   isVariadic,
                                                                                   isSynthesized,
                                                                                   isImplicitlyDeclared,
                                                                                   isDefined,
                                                                                   impControl,
                                                                                   HasRelatedResultType);
                    
                    if (setter && metadata)
                        ClangASTContext::SetMetadata(m_ast, setter, *metadata);
                    
                    llvm::SmallVector<clang::ParmVarDecl *, 1> params;
                    
                    params.push_back (clang::ParmVarDecl::Create (*m_ast,
                                                                  setter,
                                                                  clang::SourceLocation(),
                                                                  clang::SourceLocation(),
                                                                  nullptr, // anonymous
                                                                  property_clang_type_to_access.GetQualType(),
                                                                  nullptr,
                                                                  clang::SC_Auto,
                                                                  nullptr));
                    
                    if (setter)
                    {
                        setter->setMethodParams(*m_ast, llvm::ArrayRef<clang::ParmVarDecl*>(params), llvm::ArrayRef<clang::SourceLocation>());
                    
                        class_interface_decl->addDecl(setter);
                    }
                }
                
                return true;
            }
        }
    }
    return false;
}

bool
ClangASTType::IsObjCClassTypeAndHasIVars (bool check_superclass) const
{
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
    if (class_interface_decl)
        return ObjCDeclHasIVars (class_interface_decl, check_superclass);
    return false;
}


clang::ObjCMethodDecl *
ClangASTType::AddMethodToObjCObjectType (const char *name,  // the full symbol name as seen in the symbol table ("-[NString stringWithCString:]")
                                         const ClangASTType &method_clang_type,
                                         lldb::AccessType access,
                                         bool is_artificial)
{
    if (!IsValid() || !method_clang_type.IsValid())
        return nullptr;

    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl();
    
    if (class_interface_decl == nullptr)
        return nullptr;
    
    const char *selector_start = ::strchr (name, ' ');
    if (selector_start == nullptr)
        return nullptr;
    
    selector_start++;
    llvm::SmallVector<clang::IdentifierInfo *, 12> selector_idents;
    
    size_t len = 0;
    const char *start;
    //printf ("name = '%s'\n", name);
    
    unsigned num_selectors_with_args = 0;
    for (start = selector_start;
         start && *start != '\0' && *start != ']';
         start += len)
    {
        len = ::strcspn(start, ":]");
        bool has_arg = (start[len] == ':');
        if (has_arg)
            ++num_selectors_with_args;
        selector_idents.push_back (&m_ast->Idents.get (llvm::StringRef (start, len)));
        if (has_arg)
            len += 1;
    }
    
    
    if (selector_idents.size() == 0)
        return nullptr;
    
    clang::Selector method_selector = m_ast->Selectors.getSelector (num_selectors_with_args ? selector_idents.size() : 0,
                                                                    selector_idents.data());
    
    clang::QualType method_qual_type (method_clang_type.GetQualType());
    
    // Populate the method decl with parameter decls
    const clang::Type *method_type(method_qual_type.getTypePtr());
    
    if (method_type == nullptr)
        return nullptr;
    
    const clang::FunctionProtoType *method_function_prototype (llvm::dyn_cast<clang::FunctionProtoType>(method_type));
    
    if (!method_function_prototype)
        return nullptr;
    
    
    bool is_variadic = false;
    bool is_synthesized = false;
    bool is_defined = false;
    clang::ObjCMethodDecl::ImplementationControl imp_control = clang::ObjCMethodDecl::None;
    
    const unsigned num_args = method_function_prototype->getNumParams();
    
    if (num_args != num_selectors_with_args)
        return nullptr; // some debug information is corrupt.  We are not going to deal with it.
    
    clang::ObjCMethodDecl *objc_method_decl = clang::ObjCMethodDecl::Create (*m_ast,
                                                                             clang::SourceLocation(), // beginLoc,
                                                                             clang::SourceLocation(), // endLoc,
                                                                             method_selector,
                                                                             method_function_prototype->getReturnType(),
                                                                             nullptr, // TypeSourceInfo *ResultTInfo,
                                                                             GetDeclContextForType (),
                                                                             name[0] == '-',
                                                                             is_variadic,
                                                                             is_synthesized,
                                                                             true, // is_implicitly_declared; we force this to true because we don't have source locations
                                                                             is_defined,
                                                                             imp_control,
                                                                             false /*has_related_result_type*/);
    
    
    if (objc_method_decl == nullptr)
        return nullptr;
    
    if (num_args > 0)
    {
        llvm::SmallVector<clang::ParmVarDecl *, 12> params;
        
        for (unsigned param_index = 0; param_index < num_args; ++param_index)
        {
            params.push_back (clang::ParmVarDecl::Create (*m_ast,
                                                          objc_method_decl,
                                                          clang::SourceLocation(),
                                                          clang::SourceLocation(),
                                                          nullptr, // anonymous
                                                          method_function_prototype->getParamType(param_index),
                                                          nullptr,
                                                          clang::SC_Auto,
                                                          nullptr));
        }
        
        objc_method_decl->setMethodParams(*m_ast, llvm::ArrayRef<clang::ParmVarDecl*>(params), llvm::ArrayRef<clang::SourceLocation>());
    }
    
    class_interface_decl->addDecl (objc_method_decl);
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(objc_method_decl);
#endif
    
    return objc_method_decl;
}


clang::DeclContext *
ClangASTType::GetDeclContextForType () const
{
    if (!IsValid())
        return nullptr;
    
    clang::QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::UnaryTransform:           break;
        case clang::Type::FunctionNoProto:          break;
        case clang::Type::FunctionProto:            break;
        case clang::Type::IncompleteArray:          break;
        case clang::Type::VariableArray:            break;
        case clang::Type::ConstantArray:            break;
        case clang::Type::DependentSizedArray:      break;
        case clang::Type::ExtVector:                break;
        case clang::Type::DependentSizedExtVector:  break;
        case clang::Type::Vector:                   break;
        case clang::Type::Builtin:                  break;
        case clang::Type::BlockPointer:             break;
        case clang::Type::Pointer:                  break;
        case clang::Type::LValueReference:          break;
        case clang::Type::RValueReference:          break;
        case clang::Type::MemberPointer:            break;
        case clang::Type::Complex:                  break;
        case clang::Type::ObjCObject:               break;
        case clang::Type::ObjCInterface:            return llvm::cast<clang::ObjCObjectType>(qual_type.getTypePtr())->getInterface();
        case clang::Type::ObjCObjectPointer:        return ClangASTType (m_ast, llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType()).GetDeclContextForType();
        case clang::Type::Record:                   return llvm::cast<clang::RecordType>(qual_type)->getDecl();
        case clang::Type::Enum:                     return llvm::cast<clang::EnumType>(qual_type)->getDecl();
        case clang::Type::Typedef:                  return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetDeclContextForType();
        case clang::Type::Elaborated:               return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetDeclContextForType();
        case clang::Type::Paren:                    return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).GetDeclContextForType();
        case clang::Type::TypeOfExpr:               break;
        case clang::Type::TypeOf:                   break;
        case clang::Type::Decltype:                 break;
            //case clang::Type::QualifiedName:          break;
        case clang::Type::TemplateSpecialization:   break;
        case clang::Type::DependentTemplateSpecialization:  break;
        case clang::Type::TemplateTypeParm:         break;
        case clang::Type::SubstTemplateTypeParm:    break;
        case clang::Type::SubstTemplateTypeParmPack:break;
        case clang::Type::PackExpansion:            break;
        case clang::Type::UnresolvedUsing:          break;
        case clang::Type::Attributed:               break;
        case clang::Type::Auto:                     break;
        case clang::Type::InjectedClassName:        break;
        case clang::Type::DependentName:            break;
        case clang::Type::Atomic:                   break;
        case clang::Type::Adjusted:                 break;

        // pointer type decayed from an array or function type.
        case clang::Type::Decayed:                  break;
    }
    // No DeclContext in this type...
    return nullptr;
}

bool
ClangASTType::SetDefaultAccessForRecordFields (int default_accessibility,
                                               int *assigned_accessibilities,
                                               size_t num_assigned_accessibilities)
{
    if (IsValid())
    {
        clang::RecordDecl *record_decl = GetAsRecordDecl();
        if (record_decl)
        {
            uint32_t field_idx;
            clang::RecordDecl::field_iterator field, field_end;
            for (field = record_decl->field_begin(), field_end = record_decl->field_end(), field_idx = 0;
                 field != field_end;
                 ++field, ++field_idx)
            {
                // If no accessibility was assigned, assign the correct one
                if (field_idx < num_assigned_accessibilities && assigned_accessibilities[field_idx] == clang::AS_none)
                    field->setAccess ((clang::AccessSpecifier)default_accessibility);
            }
            return true;
        }
    }
    return false;
}


bool
ClangASTType::SetHasExternalStorage (bool has_extern)
{
    if (!IsValid())
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType());
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
        {
            clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
            if (cxx_record_decl)
            {
                cxx_record_decl->setHasExternalLexicalStorage (has_extern);
                cxx_record_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
            break;
            
        case clang::Type::Enum:
        {
            clang::EnumDecl *enum_decl = llvm::cast<clang::EnumType>(qual_type)->getDecl();
            if (enum_decl)
            {
                enum_decl->setHasExternalLexicalStorage (has_extern);
                enum_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
        {
            const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
            assert (objc_class_type);
            if (objc_class_type)
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                
                if (class_interface_decl)
                {
                    class_interface_decl->setHasExternalLexicalStorage (has_extern);
                    class_interface_decl->setHasExternalVisibleStorage (has_extern);
                    return true;
                }
            }
        }
            break;
            
        case clang::Type::Typedef:
            return ClangASTType (m_ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).SetHasExternalStorage (has_extern);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).SetHasExternalStorage (has_extern);
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).SetHasExternalStorage (has_extern);
            
        default:
            break;
    }
    return false;
}

bool
ClangASTType::SetTagTypeKind (int kind) const
{
    if (IsValid())
    {
        clang::QualType tag_qual_type(GetQualType());
        const clang::Type *clang_type = tag_qual_type.getTypePtr();
        if (clang_type)
        {
            const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(clang_type);
            if (tag_type)
            {
                clang::TagDecl *tag_decl = llvm::dyn_cast<clang::TagDecl>(tag_type->getDecl());
                if (tag_decl)
                {
                    tag_decl->setTagKind ((clang::TagDecl::TagKind)kind);
                    return true;
                }
            }
        }
    }
    return false;
}


#pragma mark TagDecl

bool
ClangASTType::StartTagDeclarationDefinition ()
{
    if (IsValid())
    {
        clang::QualType qual_type (GetQualType());
        const clang::Type *t = qual_type.getTypePtr();
        if (t)
        {
            const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(t);
            if (tag_type)
            {
                clang::TagDecl *tag_decl = tag_type->getDecl();
                if (tag_decl)
                {
                    tag_decl->startDefinition();
                    return true;
                }
            }
            
            const clang::ObjCObjectType *object_type = llvm::dyn_cast<clang::ObjCObjectType>(t);
            if (object_type)
            {
                clang::ObjCInterfaceDecl *interface_decl = object_type->getInterface();
                if (interface_decl)
                {
                    interface_decl->startDefinition();
                    return true;
                }
            }
        }
    }
    return false;
}

bool
ClangASTType::CompleteTagDeclarationDefinition ()
{
    if (IsValid())
    {
        clang::QualType qual_type (GetQualType());
        
        clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
        
        if (cxx_record_decl)
        {
            cxx_record_decl->completeDefinition();
            
            return true;
        }
        
        const clang::EnumType *enum_type = llvm::dyn_cast<clang::EnumType>(qual_type.getTypePtr());
        
        if (enum_type)
        {
            clang::EnumDecl *enum_decl = enum_type->getDecl();
            
            if (enum_decl)
            {
                /// TODO This really needs to be fixed.
                
                unsigned NumPositiveBits = 1;
                unsigned NumNegativeBits = 0;
                                
                clang::QualType promotion_qual_type;
                // If the enum integer type is less than an integer in bit width,
                // then we must promote it to an integer size.
                if (m_ast->getTypeSize(enum_decl->getIntegerType()) < m_ast->getTypeSize(m_ast->IntTy))
                {
                    if (enum_decl->getIntegerType()->isSignedIntegerType())
                        promotion_qual_type = m_ast->IntTy;
                    else
                        promotion_qual_type = m_ast->UnsignedIntTy;
                }
                else
                    promotion_qual_type = enum_decl->getIntegerType();
                
                enum_decl->completeDefinition(enum_decl->getIntegerType(), promotion_qual_type, NumPositiveBits, NumNegativeBits);
                return true;
            }
        }
    }
    return false;
}







bool
ClangASTType::AddEnumerationValueToEnumerationType (const ClangASTType &enumerator_clang_type,
                                                    const Declaration &decl,
                                                    const char *name,
                                                    int64_t enum_value,
                                                    uint32_t enum_value_bit_size)
{
    if (IsValid() && enumerator_clang_type.IsValid() && name && name[0])
    {
        clang::QualType enum_qual_type (GetCanonicalQualType());
        
        bool is_signed = false;
        enumerator_clang_type.IsIntegerType (is_signed);
        const clang::Type *clang_type = enum_qual_type.getTypePtr();
        if (clang_type)
        {
            const clang::EnumType *enum_type = llvm::dyn_cast<clang::EnumType>(clang_type);
            
            if (enum_type)
            {
                llvm::APSInt enum_llvm_apsint(enum_value_bit_size, is_signed);
                enum_llvm_apsint = enum_value;
                clang::EnumConstantDecl *enumerator_decl =
                clang::EnumConstantDecl::Create (*m_ast,
                                                 enum_type->getDecl(),
                                                 clang::SourceLocation(),
                                                 name ? &m_ast->Idents.get(name) : nullptr,    // Identifier
                                                 enumerator_clang_type.GetQualType(),
                                                 nullptr,
                                                 enum_llvm_apsint);
                
                if (enumerator_decl)
                {
                    enum_type->getDecl()->addDecl(enumerator_decl);
                    
#ifdef LLDB_CONFIGURATION_DEBUG
                    VerifyDecl(enumerator_decl);
#endif
                    
                    return true;
                }
            }
        }
    }
    return false;
}


ClangASTType
ClangASTType::GetEnumerationIntegerType () const
{
    clang::QualType enum_qual_type (GetCanonicalQualType());
    const clang::Type *clang_type = enum_qual_type.getTypePtr();
    if (clang_type)
    {
        const clang::EnumType *enum_type = llvm::dyn_cast<clang::EnumType>(clang_type);
        if (enum_type)
        {
            clang::EnumDecl *enum_decl = enum_type->getDecl();
            if (enum_decl)
                return ClangASTType (m_ast, enum_decl->getIntegerType());
        }
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::CreateMemberPointerType (const ClangASTType &pointee_type) const
{
    if (IsValid() && pointee_type.IsValid())
    {
        return ClangASTType (m_ast, m_ast->getMemberPointerType (pointee_type.GetQualType(),
                                                                 GetQualType().getTypePtr()));
    }
    return ClangASTType();
}


size_t
ClangASTType::ConvertStringToFloatValue (const char *s, uint8_t *dst, size_t dst_size) const
{
    if (IsValid())
    {
        clang::QualType qual_type (GetCanonicalQualType());
        uint32_t count = 0;
        bool is_complex = false;
        if (IsFloatingPointType (count, is_complex))
        {
            // TODO: handle complex and vector types
            if (count != 1)
                return false;
            
            llvm::StringRef s_sref(s);
            llvm::APFloat ap_float(m_ast->getFloatTypeSemantics(qual_type), s_sref);
            
            const uint64_t bit_size = m_ast->getTypeSize (qual_type);
            const uint64_t byte_size = bit_size / 8;
            if (dst_size >= byte_size)
            {
                if (bit_size == sizeof(float)*8)
                {
                    float float32 = ap_float.convertToFloat();
                    ::memcpy (dst, &float32, byte_size);
                    return byte_size;
                }
                else if (bit_size >= 64)
                {
                    llvm::APInt ap_int(ap_float.bitcastToAPInt());
                    ::memcpy (dst, ap_int.getRawData(), byte_size);
                    return byte_size;
                }
            }
        }
    }
    return 0;
}



//----------------------------------------------------------------------
// Dumping types
//----------------------------------------------------------------------
#define DEPTH_INCREMENT 2

void
ClangASTType::DumpValue (ExecutionContext *exe_ctx,
                         Stream *s,
                         lldb::Format format,
                         const lldb_private::DataExtractor &data,
                         lldb::offset_t data_byte_offset,
                         size_t data_byte_size,
                         uint32_t bitfield_bit_size,
                         uint32_t bitfield_bit_offset,
                         bool show_types,
                         bool show_summary,
                         bool verbose,
                         uint32_t depth)
{
    if (!IsValid())
        return;

    clang::QualType qual_type(GetQualType());
    switch (qual_type->getTypeClass())
    {
    case clang::Type::Record:
        if (GetCompleteType ())
        {
            const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
            const clang::RecordDecl *record_decl = record_type->getDecl();
            assert(record_decl);
            uint32_t field_bit_offset = 0;
            uint32_t field_byte_offset = 0;
            const clang::ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(record_decl);
            uint32_t child_idx = 0;

            const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
            if (cxx_record_decl)
            {
                // We might have base classes to print out first
                clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                     base_class != base_class_end;
                     ++base_class)
                {
                    const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());

                    // Skip empty base classes
                    if (verbose == false && ClangASTContext::RecordHasFields(base_class_decl) == false)
                        continue;

                    if (base_class->isVirtual())
                        field_bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                    else
                        field_bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                    field_byte_offset = field_bit_offset / 8;
                    assert (field_bit_offset % 8 == 0);
                    if (child_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');

                    clang::QualType base_class_qual_type = base_class->getType();
                    std::string base_class_type_name(base_class_qual_type.getAsString());

                    // Indent and print the base class type name
                    s->Printf("\n%*s%s ", depth + DEPTH_INCREMENT, "", base_class_type_name.c_str());

                    clang::TypeInfo base_class_type_info = m_ast->getTypeInfo(base_class_qual_type);

                    // Dump the value of the member
                    ClangASTType base_clang_type(m_ast, base_class_qual_type);
                    base_clang_type.DumpValue (exe_ctx,
                                               s,                                   // Stream to dump to
                                               base_clang_type.GetFormat(),         // The format with which to display the member
                                               data,                                // Data buffer containing all bytes for this type
                                               data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                                               base_class_type_info.Width / 8,      // Size of this type in bytes
                                               0,                                   // Bitfield bit size
                                               0,                                   // Bitfield bit offset
                                               show_types,                          // Boolean indicating if we should show the variable types
                                               show_summary,                        // Boolean indicating if we should show a summary for the current type
                                               verbose,                             // Verbose output?
                                               depth + DEPTH_INCREMENT);            // Scope depth for any types that have children
                    
                    ++child_idx;
                }
            }
            uint32_t field_idx = 0;
            clang::RecordDecl::field_iterator field, field_end;
            for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
            {
                // Print the starting squiggly bracket (if this is the
                // first member) or comma (for member 2 and beyond) for
                // the struct/union/class member.
                if (child_idx == 0)
                    s->PutChar('{');
                else
                    s->PutChar(',');

                // Indent
                s->Printf("\n%*s", depth + DEPTH_INCREMENT, "");

                clang::QualType field_type = field->getType();
                // Print the member type if requested
                // Figure out the type byte size (field_type_info.first) and
                // alignment (field_type_info.second) from the AST context.
                clang::TypeInfo field_type_info = m_ast->getTypeInfo(field_type);
                assert(field_idx < record_layout.getFieldCount());
                // Figure out the field offset within the current struct/union/class type
                field_bit_offset = record_layout.getFieldOffset (field_idx);
                field_byte_offset = field_bit_offset / 8;
                uint32_t field_bitfield_bit_size = 0;
                uint32_t field_bitfield_bit_offset = 0;
                if (ClangASTContext::FieldIsBitfield (m_ast, *field, field_bitfield_bit_size))
                    field_bitfield_bit_offset = field_bit_offset % 8;

                if (show_types)
                {
                    std::string field_type_name(field_type.getAsString());
                    if (field_bitfield_bit_size > 0)
                        s->Printf("(%s:%u) ", field_type_name.c_str(), field_bitfield_bit_size);
                    else
                        s->Printf("(%s) ", field_type_name.c_str());
                }
                // Print the member name and equal sign
                s->Printf("%s = ", field->getNameAsString().c_str());


                // Dump the value of the member
                ClangASTType field_clang_type (m_ast, field_type);
                field_clang_type.DumpValue (exe_ctx,
                                            s,                              // Stream to dump to
                                            field_clang_type.GetFormat(),   // The format with which to display the member
                                            data,                           // Data buffer containing all bytes for this type
                                            data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                                            field_type_info.Width / 8,      // Size of this type in bytes
                                            field_bitfield_bit_size,        // Bitfield bit size
                                            field_bitfield_bit_offset,      // Bitfield bit offset
                                            show_types,                     // Boolean indicating if we should show the variable types
                                            show_summary,                   // Boolean indicating if we should show a summary for the current type
                                            verbose,                        // Verbose output?
                                            depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
            }

            // Indent the trailing squiggly bracket
            if (child_idx > 0)
                s->Printf("\n%*s}", depth, "");
        }
        return;

    case clang::Type::Enum:
        if (GetCompleteType ())
        {
            const clang::EnumType *enum_type = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
            const clang::EnumDecl *enum_decl = enum_type->getDecl();
            assert(enum_decl);
            clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
            lldb::offset_t offset = data_byte_offset;
            const int64_t enum_value = data.GetMaxU64Bitfield(&offset, data_byte_size, bitfield_bit_size, bitfield_bit_offset);
            for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
            {
                if (enum_pos->getInitVal() == enum_value)
                {
                    s->Printf("%s", enum_pos->getNameAsString().c_str());
                    return;
                }
            }
            // If we have gotten here we didn't get find the enumerator in the
            // enum decl, so just print the integer.
            s->Printf("%" PRIi64, enum_value);
        }
        return;

    case clang::Type::ConstantArray:
        {
            const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(qual_type.getTypePtr());
            bool is_array_of_characters = false;
            clang::QualType element_qual_type = array->getElementType();

            const clang::Type *canonical_type = element_qual_type->getCanonicalTypeInternal().getTypePtr();
            if (canonical_type)
                is_array_of_characters = canonical_type->isCharType();

            const uint64_t element_count = array->getSize().getLimitedValue();

            clang::TypeInfo field_type_info = m_ast->getTypeInfo(element_qual_type);

            uint32_t element_idx = 0;
            uint32_t element_offset = 0;
            uint64_t element_byte_size = field_type_info.Width / 8;
            uint32_t element_stride = element_byte_size;

            if (is_array_of_characters)
            {
                s->PutChar('"');
                data.Dump(s, data_byte_offset, lldb::eFormatChar, element_byte_size, element_count, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                s->PutChar('"');
                return;
            }
            else
            {
                ClangASTType element_clang_type(m_ast, element_qual_type);
                lldb::Format element_format = element_clang_type.GetFormat();

                for (element_idx = 0; element_idx < element_count; ++element_idx)
                {
                    // Print the starting squiggly bracket (if this is the
                    // first member) or comma (for member 2 and beyond) for
                    // the struct/union/class member.
                    if (element_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');

                    // Indent and print the index
                    s->Printf("\n%*s[%u] ", depth + DEPTH_INCREMENT, "", element_idx);

                    // Figure out the field offset within the current struct/union/class type
                    element_offset = element_idx * element_stride;

                    // Dump the value of the member
                    element_clang_type.DumpValue (exe_ctx,
                                                  s,                              // Stream to dump to
                                                  element_format,                 // The format with which to display the element
                                                  data,                           // Data buffer containing all bytes for this type
                                                  data_byte_offset + element_offset,// Offset into "data" where to grab value from
                                                  element_byte_size,              // Size of this type in bytes
                                                  0,                              // Bitfield bit size
                                                  0,                              // Bitfield bit offset
                                                  show_types,                     // Boolean indicating if we should show the variable types
                                                  show_summary,                   // Boolean indicating if we should show a summary for the current type
                                                  verbose,                        // Verbose output?
                                                  depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
                }

                // Indent the trailing squiggly bracket
                if (element_idx > 0)
                    s->Printf("\n%*s}", depth, "");
            }
        }
        return;

    case clang::Type::Typedef:
        {
            clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
            
            ClangASTType typedef_clang_type (m_ast, typedef_qual_type);
            lldb::Format typedef_format = typedef_clang_type.GetFormat();
            clang::TypeInfo typedef_type_info = m_ast->getTypeInfo(typedef_qual_type);
            uint64_t typedef_byte_size = typedef_type_info.Width / 8;

            return typedef_clang_type.DumpValue (exe_ctx,
                                                 s,                  // Stream to dump to
                                                 typedef_format,     // The format with which to display the element
                                                 data,               // Data buffer containing all bytes for this type
                                                 data_byte_offset,   // Offset into "data" where to grab value from
                                                 typedef_byte_size,  // Size of this type in bytes
                                                 bitfield_bit_size,  // Bitfield bit size
                                                 bitfield_bit_offset,// Bitfield bit offset
                                                 show_types,         // Boolean indicating if we should show the variable types
                                                 show_summary,       // Boolean indicating if we should show a summary for the current type
                                                 verbose,            // Verbose output?
                                                 depth);             // Scope depth for any types that have children
        }
        break;

    case clang::Type::Elaborated:
        {
            clang::QualType elaborated_qual_type = llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType();
            ClangASTType elaborated_clang_type (m_ast, elaborated_qual_type);
            lldb::Format elaborated_format = elaborated_clang_type.GetFormat();
            clang::TypeInfo elaborated_type_info = m_ast->getTypeInfo(elaborated_qual_type);
            uint64_t elaborated_byte_size = elaborated_type_info.Width / 8;

            return elaborated_clang_type.DumpValue (exe_ctx,
                                                    s,                  // Stream to dump to
                                                    elaborated_format,  // The format with which to display the element
                                                    data,               // Data buffer containing all bytes for this type
                                                    data_byte_offset,   // Offset into "data" where to grab value from
                                                    elaborated_byte_size,  // Size of this type in bytes
                                                    bitfield_bit_size,  // Bitfield bit size
                                                    bitfield_bit_offset,// Bitfield bit offset
                                                    show_types,         // Boolean indicating if we should show the variable types
                                                    show_summary,       // Boolean indicating if we should show a summary for the current type
                                                    verbose,            // Verbose output?
                                                    depth);             // Scope depth for any types that have children
        }
        break;
            
    case clang::Type::Paren:
        {
            clang::QualType desugar_qual_type = llvm::cast<clang::ParenType>(qual_type)->desugar();
            ClangASTType desugar_clang_type (m_ast, desugar_qual_type);

            lldb::Format desugar_format = desugar_clang_type.GetFormat();
            clang::TypeInfo desugar_type_info = m_ast->getTypeInfo(desugar_qual_type);
            uint64_t desugar_byte_size = desugar_type_info.Width / 8;
            
            return desugar_clang_type.DumpValue (exe_ctx,
                                                 s,                  // Stream to dump to
                                                 desugar_format,  // The format with which to display the element
                                                 data,               // Data buffer containing all bytes for this type
                                                 data_byte_offset,   // Offset into "data" where to grab value from
                                                 desugar_byte_size,  // Size of this type in bytes
                                                 bitfield_bit_size,  // Bitfield bit size
                                                 bitfield_bit_offset,// Bitfield bit offset
                                                 show_types,         // Boolean indicating if we should show the variable types
                                                 show_summary,       // Boolean indicating if we should show a summary for the current type
                                                 verbose,            // Verbose output?
                                                 depth);             // Scope depth for any types that have children
        }
        break;

    default:
        // We are down to a scalar type that we just need to display.
        data.Dump(s,
                  data_byte_offset,
                  format,
                  data_byte_size,
                  1,
                  UINT32_MAX,
                  LLDB_INVALID_ADDRESS,
                  bitfield_bit_size,
                  bitfield_bit_offset);

        if (show_summary)
            DumpSummary (exe_ctx, s, data, data_byte_offset, data_byte_size);
        break;
    }
}




bool
ClangASTType::DumpTypeValue (Stream *s,
                             lldb::Format format,
                             const lldb_private::DataExtractor &data,
                             lldb::offset_t byte_offset,
                             size_t byte_size,
                             uint32_t bitfield_bit_size,
                             uint32_t bitfield_bit_offset,
                             ExecutionContextScope *exe_scope)
{
    if (!IsValid())
        return false;
    if (IsAggregateType())
    {
        return false;
    }
    else
    {
        clang::QualType qual_type(GetQualType());

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::Typedef:
            {
                clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
                ClangASTType typedef_clang_type (m_ast, typedef_qual_type);
                if (format == eFormatDefault)
                    format = typedef_clang_type.GetFormat();
                clang::TypeInfo typedef_type_info = m_ast->getTypeInfo(typedef_qual_type);
                uint64_t typedef_byte_size = typedef_type_info.Width / 8;

                return typedef_clang_type.DumpTypeValue (s,
                                                         format,                 // The format with which to display the element
                                                         data,                   // Data buffer containing all bytes for this type
                                                         byte_offset,            // Offset into "data" where to grab value from
                                                         typedef_byte_size,      // Size of this type in bytes
                                                         bitfield_bit_size,      // Size in bits of a bitfield value, if zero don't treat as a bitfield
                                                         bitfield_bit_offset,    // Offset in bits of a bitfield value if bitfield_bit_size != 0
                                                         exe_scope);
            }
            break;

        case clang::Type::Enum:
            // If our format is enum or default, show the enumeration value as
            // its enumeration string value, else just display it as requested.
            if ((format == eFormatEnum || format == eFormatDefault) && GetCompleteType ())
            {
                const clang::EnumType *enum_type = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
                const clang::EnumDecl *enum_decl = enum_type->getDecl();
                assert(enum_decl);
                clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
                const bool is_signed = qual_type->isSignedIntegerOrEnumerationType();
                lldb::offset_t offset = byte_offset;
                if (is_signed)
                {
                    const int64_t enum_svalue = data.GetMaxS64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                    for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                    {
                        if (enum_pos->getInitVal().getSExtValue() == enum_svalue)
                        {
                            s->PutCString (enum_pos->getNameAsString().c_str());
                            return true;
                        }
                    }
                    // If we have gotten here we didn't get find the enumerator in the
                    // enum decl, so just print the integer.                    
                    s->Printf("%" PRIi64, enum_svalue);
                }
                else
                {
                    const uint64_t enum_uvalue = data.GetMaxU64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                    for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                    {
                        if (enum_pos->getInitVal().getZExtValue() == enum_uvalue)
                        {
                            s->PutCString (enum_pos->getNameAsString().c_str());
                            return true;
                        }
                    }
                    // If we have gotten here we didn't get find the enumerator in the
                    // enum decl, so just print the integer.
                    s->Printf("%" PRIu64, enum_uvalue);
                }
                return true;
            }
            // format was not enum, just fall through and dump the value as requested....
                
        default:
            // We are down to a scalar type that we just need to display.
            {
                uint32_t item_count = 1;
                // A few formats, we might need to modify our size and count for depending
                // on how we are trying to display the value...
                switch (format)
                {
                    default:
                    case eFormatBoolean:
                    case eFormatBinary:
                    case eFormatComplex:
                    case eFormatCString:         // NULL terminated C strings
                    case eFormatDecimal:
                    case eFormatEnum:
                    case eFormatHex:
                    case eFormatHexUppercase:
                    case eFormatFloat:
                    case eFormatOctal:
                    case eFormatOSType:
                    case eFormatUnsigned:
                    case eFormatPointer:
                    case eFormatVectorOfChar:
                    case eFormatVectorOfSInt8:
                    case eFormatVectorOfUInt8:
                    case eFormatVectorOfSInt16:
                    case eFormatVectorOfUInt16:
                    case eFormatVectorOfSInt32:
                    case eFormatVectorOfUInt32:
                    case eFormatVectorOfSInt64:
                    case eFormatVectorOfUInt64:
                    case eFormatVectorOfFloat32:
                    case eFormatVectorOfFloat64:
                    case eFormatVectorOfUInt128:
                        break;

                    case eFormatChar: 
                    case eFormatCharPrintable:  
                    case eFormatCharArray:
                    case eFormatBytes:
                    case eFormatBytesWithASCII:
                        item_count = byte_size;
                        byte_size = 1; 
                        break;

                    case eFormatUnicode16:
                        item_count = byte_size / 2; 
                        byte_size = 2; 
                        break;

                    case eFormatUnicode32:
                        item_count = byte_size / 4; 
                        byte_size = 4; 
                        break;
                }
                return data.Dump (s,
                                  byte_offset,
                                  format,
                                  byte_size,
                                  item_count,
                                  UINT32_MAX,
                                  LLDB_INVALID_ADDRESS,
                                  bitfield_bit_size,
                                  bitfield_bit_offset,
                                  exe_scope);
            }
            break;
        }
    }
    return 0;
}



void
ClangASTType::DumpSummary (ExecutionContext *exe_ctx,
                           Stream *s,
                           const lldb_private::DataExtractor &data,
                           lldb::offset_t data_byte_offset,
                           size_t data_byte_size)
{
    uint32_t length = 0;
    if (IsCStringType (length))
    {
        if (exe_ctx)
        {
            Process *process = exe_ctx->GetProcessPtr();
            if (process)
            {
                lldb::offset_t offset = data_byte_offset;
                lldb::addr_t pointer_address = data.GetMaxU64(&offset, data_byte_size);
                std::vector<uint8_t> buf;
                if (length > 0)
                    buf.resize (length);
                else
                    buf.resize (256);

                lldb_private::DataExtractor cstr_data(&buf.front(), buf.size(), process->GetByteOrder(), 4);
                buf.back() = '\0';
                size_t bytes_read;
                size_t total_cstr_len = 0;
                Error error;
                while ((bytes_read = process->ReadMemory (pointer_address, &buf.front(), buf.size(), error)) > 0)
                {
                    const size_t len = strlen((const char *)&buf.front());
                    if (len == 0)
                        break;
                    if (total_cstr_len == 0)
                        s->PutCString (" \"");
                    cstr_data.Dump(s, 0, lldb::eFormatChar, 1, len, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                    total_cstr_len += len;
                    if (len < buf.size())
                        break;
                    pointer_address += total_cstr_len;
                }
                if (total_cstr_len > 0)
                    s->PutChar ('"');
            }
        }
    }
}

void
ClangASTType::DumpTypeDescription () const
{
    StreamFile s (stdout, false);
    DumpTypeDescription (&s);
    ClangASTMetadata *metadata = ClangASTContext::GetMetadata (m_ast, m_type);
    if (metadata)
    {
        metadata->Dump (&s);
    }
}

void
ClangASTType::DumpTypeDescription (Stream *s) const
{
    if (IsValid())
    {
        clang::QualType qual_type(GetQualType());

        llvm::SmallVector<char, 1024> buf;
        llvm::raw_svector_ostream llvm_ostrm (buf);

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            {
                GetCompleteType ();
                
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    if (class_interface_decl)
                    {
                        clang::PrintingPolicy policy = m_ast->getPrintingPolicy();
                        class_interface_decl->print(llvm_ostrm, policy, s->GetIndentLevel());
                    }
                }
            }
            break;
        
        case clang::Type::Typedef:
            {
                const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
                if (typedef_type)
                {
                    const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
                    std::string clang_typedef_name (typedef_decl->getQualifiedNameAsString());
                    if (!clang_typedef_name.empty())
                    {
                        s->PutCString ("typedef ");
                        s->PutCString (clang_typedef_name.c_str());
                    }
                }
            }
            break;

        case clang::Type::Elaborated:
            ClangASTType (m_ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).DumpTypeDescription(s);
            return;

        case clang::Type::Paren:
            ClangASTType (m_ast, llvm::cast<clang::ParenType>(qual_type)->desugar()).DumpTypeDescription(s);
            return;

        case clang::Type::Record:
            {
                GetCompleteType ();
                
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);

                if (cxx_record_decl)
                    cxx_record_decl->print(llvm_ostrm, m_ast->getPrintingPolicy(), s->GetIndentLevel());
                else
                    record_decl->print(llvm_ostrm, m_ast->getPrintingPolicy(), s->GetIndentLevel());
            }
            break;

        default:
            {
                const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
                if (tag_type)
                {
                    clang::TagDecl *tag_decl = tag_type->getDecl();
                    if (tag_decl)
                        tag_decl->print(llvm_ostrm, 0);
                }
                else
                {
                    std::string clang_type_name(qual_type.getAsString());
                    if (!clang_type_name.empty())
                        s->PutCString (clang_type_name.c_str());
                }
            }
        }
        
        llvm_ostrm.flush();
        if (buf.size() > 0)
        {
            s->Write (buf.data(), buf.size());
        }
    }
}

bool
ClangASTType::GetValueAsScalar (const lldb_private::DataExtractor &data,
                                lldb::offset_t data_byte_offset,
                                size_t data_byte_size,
                                Scalar &value) const
{
    if (!IsValid())
        return false;

    if (IsAggregateType ())
    {
        return false;   // Aggregate types don't have scalar values
    }
    else
    {
        uint64_t count = 0;
        lldb::Encoding encoding = GetEncoding (count);

        if (encoding == lldb::eEncodingInvalid || count != 1)
            return false;

        const uint64_t byte_size = GetByteSize(nullptr);
        lldb::offset_t offset = data_byte_offset;
        switch (encoding)
        {
        case lldb::eEncodingInvalid:
            break;
        case lldb::eEncodingVector:
            break;
        case lldb::eEncodingUint:
            if (byte_size <= sizeof(unsigned long long))
            {
                uint64_t uval64 = data.GetMaxU64 (&offset, byte_size);
                if (byte_size <= sizeof(unsigned int))
                {
                    value = (unsigned int)uval64;
                    return true;
                }
                else if (byte_size <= sizeof(unsigned long))
                {
                    value = (unsigned long)uval64;
                    return true;
                }
                else if (byte_size <= sizeof(unsigned long long))
                {
                    value = (unsigned long long )uval64;
                    return true;
                }
                else
                    value.Clear();
            }
            break;

        case lldb::eEncodingSint:
            if (byte_size <= sizeof(long long))
            {
                int64_t sval64 = data.GetMaxS64 (&offset, byte_size);
                if (byte_size <= sizeof(int))
                {
                    value = (int)sval64;
                    return true;
                }
                else if (byte_size <= sizeof(long))
                {
                    value = (long)sval64;
                    return true;
                }
                else if (byte_size <= sizeof(long long))
                {
                    value = (long long )sval64;
                    return true;
                }
                else
                    value.Clear();
            }
            break;

        case lldb::eEncodingIEEE754:
            if (byte_size <= sizeof(long double))
            {
                uint32_t u32;
                uint64_t u64;
                if (byte_size == sizeof(float))
                {
                    if (sizeof(float) == sizeof(uint32_t))
                    {
                        u32 = data.GetU32(&offset);
                        value = *((float *)&u32);
                        return true;
                    }
                    else if (sizeof(float) == sizeof(uint64_t))
                    {
                        u64 = data.GetU64(&offset);
                        value = *((float *)&u64);
                        return true;
                    }
                }
                else
                if (byte_size == sizeof(double))
                {
                    if (sizeof(double) == sizeof(uint32_t))
                    {
                        u32 = data.GetU32(&offset);
                        value = *((double *)&u32);
                        return true;
                    }
                    else if (sizeof(double) == sizeof(uint64_t))
                    {
                        u64 = data.GetU64(&offset);
                        value = *((double *)&u64);
                        return true;
                    }
                }
                else
                if (byte_size == sizeof(long double))
                {
                    if (sizeof(long double) == sizeof(uint32_t))
                    {
                        u32 = data.GetU32(&offset);
                        value = *((long double *)&u32);
                        return true;
                    }
                    else if (sizeof(long double) == sizeof(uint64_t))
                    {
                        u64 = data.GetU64(&offset);
                        value = *((long double *)&u64);
                        return true;
                    }
                }
            }
            break;
        }
    }
    return false;
}

bool
ClangASTType::SetValueFromScalar (const Scalar &value, Stream &strm)
{
    // Aggregate types don't have scalar values
    if (!IsAggregateType ())
    {
        strm.GetFlags().Set(Stream::eBinary);
        uint64_t count = 0;
        lldb::Encoding encoding = GetEncoding (count);

        if (encoding == lldb::eEncodingInvalid || count != 1)
            return false;

        const uint64_t bit_width = GetBitSize(nullptr);
        // This function doesn't currently handle non-byte aligned assignments
        if ((bit_width % 8) != 0)
            return false;

        const uint64_t byte_size = (bit_width + 7 ) / 8;
        switch (encoding)
        {
        case lldb::eEncodingInvalid:
            break;
        case lldb::eEncodingVector:
            break;
        case lldb::eEncodingUint:
            switch (byte_size)
            {
            case 1: strm.PutHex8(value.UInt()); return true;
            case 2: strm.PutHex16(value.UInt()); return true;
            case 4: strm.PutHex32(value.UInt()); return true;
            case 8: strm.PutHex64(value.ULongLong()); return true;
            default:
                break;
            }
            break;

        case lldb::eEncodingSint:
            switch (byte_size)
            {
            case 1: strm.PutHex8(value.SInt()); return true;
            case 2: strm.PutHex16(value.SInt()); return true;
            case 4: strm.PutHex32(value.SInt()); return true;
            case 8: strm.PutHex64(value.SLongLong()); return true;
            default:
                break;
            }
            break;

        case lldb::eEncodingIEEE754:
            if (byte_size <= sizeof(long double))
            {
                if (byte_size == sizeof(float))
                {
                    strm.PutFloat(value.Float());
                    return true;
                }
                else
                if (byte_size == sizeof(double))
                {
                    strm.PutDouble(value.Double());
                    return true;
                }
                else
                if (byte_size == sizeof(long double))
                {
                    strm.PutDouble(value.LongDouble());
                    return true;
                }
            }
            break;
        }
    }
    return false;
}

bool
ClangASTType::ReadFromMemory (lldb_private::ExecutionContext *exe_ctx,
                              lldb::addr_t addr,
                              AddressType address_type,
                              lldb_private::DataExtractor &data)
{
    if (!IsValid())
        return false;

    // Can't convert a file address to anything valid without more
    // context (which Module it came from)
    if (address_type == eAddressTypeFile)
        return false;
    
    if (!GetCompleteType())
        return false;
    
    const uint64_t byte_size = GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
    if (data.GetByteSize() < byte_size)
    {
        lldb::DataBufferSP data_sp(new DataBufferHeap (byte_size, '\0'));
        data.SetData(data_sp);
    }
    
    uint8_t* dst = (uint8_t*)data.PeekData(0, byte_size);
    if (dst != nullptr)
    {
        if (address_type == eAddressTypeHost)
        {
            if (addr == 0)
                return false;
            // The address is an address in this process, so just copy it
            memcpy (dst, (uint8_t*)nullptr + addr, byte_size);
            return true;
        }
        else
        {
            Process *process = nullptr;
            if (exe_ctx)
                process = exe_ctx->GetProcessPtr();
            if (process)
            {
                Error error;
                return process->ReadMemory(addr, dst, byte_size, error) == byte_size;
            }
        }
    }
    return false;
}

bool
ClangASTType::WriteToMemory (lldb_private::ExecutionContext *exe_ctx,
                             lldb::addr_t addr,
                             AddressType address_type,
                             StreamString &new_value)
{
    if (!IsValid())
        return false;

    // Can't convert a file address to anything valid without more
    // context (which Module it came from)
    if (address_type == eAddressTypeFile)
        return false;
        
    if (!GetCompleteType())
        return false;

    const uint64_t byte_size = GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);

    if (byte_size > 0)
    {
        if (address_type == eAddressTypeHost)
        {
            // The address is an address in this process, so just copy it
            memcpy ((void *)addr, new_value.GetData(), byte_size);
            return true;
        }
        else
        {
            Process *process = nullptr;
            if (exe_ctx)
                process = exe_ctx->GetProcessPtr();
            if (process)
            {
                Error error;
                return process->WriteMemory(addr, new_value.GetData(), byte_size, error) == byte_size;
            }
        }
    }
    return false;
}


//clang::CXXRecordDecl *
//ClangASTType::GetAsCXXRecordDecl (lldb::clang_type_t opaque_clang_qual_type)
//{
//    if (opaque_clang_qual_type)
//        return clang::QualType::getFromOpaquePtr(opaque_clang_qual_type)->getAsCXXRecordDecl();
//    return NULL;
//}

bool
lldb_private::operator == (const lldb_private::ClangASTType &lhs, const lldb_private::ClangASTType &rhs)
{
    return lhs.GetASTContext() == rhs.GetASTContext() && lhs.GetOpaqueQualType() == rhs.GetOpaqueQualType();
}


bool
lldb_private::operator != (const lldb_private::ClangASTType &lhs, const lldb_private::ClangASTType &rhs)
{
    return lhs.GetASTContext() != rhs.GetASTContext() || lhs.GetOpaqueQualType() != rhs.GetOpaqueQualType();
}



