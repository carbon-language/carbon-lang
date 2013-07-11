//===-- ClangASTType.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

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

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/VerifyDecl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"

#include <mutex>

using namespace lldb;
using namespace lldb_private;
using namespace clang;
using namespace llvm;

static bool
GetCompleteQualType (ASTContext *ast, QualType qual_type, bool allow_completion = true)
{
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::ConstantArray:
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
        {
            const ArrayType *array_type = dyn_cast<ArrayType>(qual_type.getTypePtr());
            
            if (array_type)
                return GetCompleteQualType (ast, array_type->getElementType(), allow_completion);
        }
            break;
            
        case clang::Type::Record:
        case clang::Type::Enum:
        {
            const TagType *tag_type = dyn_cast<TagType>(qual_type.getTypePtr());
            if (tag_type)
            {
                TagDecl *tag_decl = tag_type->getDecl();
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
                            ExternalASTSource *external_ast_source = ast->getExternalSource();
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
            const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type);
            if (objc_class_type)
            {
                ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
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
                            ExternalASTSource *external_ast_source = ast->getExternalSource();
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
            return GetCompleteQualType (ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType(), allow_completion);
            
        case clang::Type::Elaborated:
            return GetCompleteQualType (ast, cast<ElaboratedType>(qual_type)->getNamedType(), allow_completion);
            
        case clang::Type::Paren:
            return GetCompleteQualType (ast, cast<ParenType>(qual_type)->desugar(), allow_completion);
            
        default:
            break;
    }
    
    return true;
}

static ObjCIvarDecl::AccessControl
ConvertAccessTypeToObjCIvarAccessControl (AccessType access)
{
    switch (access)
    {
        case eAccessNone:      return ObjCIvarDecl::None;
        case eAccessPublic:    return ObjCIvarDecl::Public;
        case eAccessPrivate:   return ObjCIvarDecl::Private;
        case eAccessProtected: return ObjCIvarDecl::Protected;
        case eAccessPackage:   return ObjCIvarDecl::Package;
    }
    return ObjCIvarDecl::None;
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
    
    QualType qual_type (GetCanonicalQualType());
    
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
            return ClangASTType(m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).IsAggregateType();
        case clang::Type::Typedef:
            return ClangASTType(m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsAggregateType();
        case clang::Type::Paren:
            return ClangASTType(m_ast, cast<ParenType>(qual_type)->desugar()).IsAggregateType();
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
        QualType qual_type (GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            default:
                break;
                
            case clang::Type::ConstantArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, cast<ConstantArrayType>(qual_type)->getElementType());
                if (size)
                    *size = cast<ConstantArrayType>(qual_type)->getSize().getLimitedValue(ULLONG_MAX);
                return true;
                
            case clang::Type::IncompleteArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, cast<IncompleteArrayType>(qual_type)->getElementType());
                if (size)
                    *size = 0;
                if (is_incomplete)
                    *is_incomplete = true;
                return true;
                
            case clang::Type::VariableArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, cast<VariableArrayType>(qual_type)->getElementType());
                if (size)
                    *size = 0;
                return true;
                
            case clang::Type::DependentSizedArray:
                if (element_type_ptr)
                    element_type_ptr->SetClangType (m_ast, cast<DependentSizedArrayType>(qual_type)->getElementType());
                if (size)
                    *size = 0;
                return true;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsArrayType (element_type_ptr,
                                                                                                                       size,
                                                                                                                       is_incomplete);
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).IsArrayType (element_type_ptr,
                                                                                                          size,
                                                                                                          is_incomplete);
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).IsArrayType (element_type_ptr,
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
                length = cast<ConstantArrayType>(GetCanonicalQualType().getTypePtr())->getSize().getLimitedValue();
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
        QualType qual_type (GetCanonicalQualType());
        
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
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsFunctionType();
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).IsFunctionType();
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).IsFunctionType();
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
                    if (reference_type)
                        return ClangASTType (m_ast, reference_type->getPointeeType()).IsFunctionType();
                }
                break;
        }
    }
    return false;
}


bool
ClangASTType::IsFunctionPointerType () const
{
    if (IsValid())
    {
        QualType qual_type (GetCanonicalQualType());
        
        if (qual_type->isFunctionPointerType())
            return true;
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        default:
            break;
        case clang::Type::Typedef:
            return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsFunctionPointerType();
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).IsFunctionPointerType();
        case clang::Type::Paren:
            return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).IsFunctionPointerType();
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            {
                const ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
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
    
    QualType qual_type (GetCanonicalQualType());
    const BuiltinType *builtin_type = dyn_cast<BuiltinType>(qual_type->getCanonicalTypeInternal());
    
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
        QualType qual_type (GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (cast<clang::BuiltinType>(qual_type)->getKind())
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
                    pointee_type->SetClangType (m_ast, cast<ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetClangType (m_ast, cast<BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetClangType (m_ast, cast<PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetClangType (m_ast, cast<MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Typedef:
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsPointerType(pointee_type);
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).IsPointerType(pointee_type);
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).IsPointerType(pointee_type);
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
        QualType qual_type (GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (cast<clang::BuiltinType>(qual_type)->getKind())
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
                    pointee_type->SetClangType(m_ast, cast<ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, cast<BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, cast<PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, cast<MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, cast<LValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, cast<LValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::Typedef:
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsPointerOrReferenceType(pointee_type);
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).IsPointerOrReferenceType(pointee_type);
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).IsPointerOrReferenceType(pointee_type);
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}


bool
ClangASTType::IsReferenceType (ClangASTType *pointee_type) const
{
    if (IsValid())
    {
        QualType qual_type (GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, cast<LValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(m_ast, cast<LValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::Typedef:
                return ClangASTType(m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsReferenceType(pointee_type);
            case clang::Type::Elaborated:
                return ClangASTType(m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).IsReferenceType(pointee_type);
            case clang::Type::Paren:
                return ClangASTType(m_ast, cast<clang::ParenType>(qual_type)->desugar()).IsReferenceType(pointee_type);
                
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
        QualType qual_type (GetCanonicalQualType());
        
        if (const BuiltinType *BT = dyn_cast<BuiltinType>(qual_type->getCanonicalTypeInternal()))
        {
            clang::BuiltinType::Kind kind = BT->getKind();
            if (kind >= BuiltinType::Float && kind <= BuiltinType::LongDouble)
            {
                count = 1;
                is_complex = false;
                return true;
            }
        }
        else if (const ComplexType *CT = dyn_cast<ComplexType>(qual_type->getCanonicalTypeInternal()))
        {
            if (ClangASTType (m_ast, CT->getElementType()).IsFloatingPointType (count, is_complex))
            {
                count = 2;
                is_complex = true;
                return true;
            }
        }
        else if (const VectorType *VT = dyn_cast<VectorType>(qual_type->getCanonicalTypeInternal()))
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

    QualType qual_type(GetQualType());
    const TagType *tag_type = dyn_cast<TagType>(qual_type.getTypePtr());
    if (tag_type)
    {
        TagDecl *tag_decl = tag_type->getDecl();
        if (tag_decl)
            return tag_decl->isCompleteDefinition();
        return false;
    }
    else
    {
        const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type);
        if (objc_class_type)
        {
            ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            if (class_interface_decl)
                return class_interface_decl->getDefinition() != NULL;
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
        QualType qual_type (GetCanonicalQualType());
    
        const ObjCObjectPointerType *obj_pointer_type = dyn_cast<ObjCObjectPointerType>(qual_type);

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
        QualType qual_type(GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType())
                {
                    const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                    const RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
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
    QualType pointee_qual_type;
    if (m_type)
    {
        QualType qual_type (GetCanonicalQualType());
        bool success = false;
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                if (check_objc && cast<BuiltinType>(qual_type)->getKind() == BuiltinType::ObjCId)
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
                        dynamic_pointee_type->SetClangType(m_ast, cast<ObjCObjectPointerType>(qual_type)->getPointeeType());
                    return true;
                }
                break;
                
            case clang::Type::Pointer:
                pointee_qual_type = cast<PointerType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                pointee_qual_type = cast<ReferenceType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast,
                                     cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).IsPossibleDynamicType (dynamic_pointee_type,
                                                                                                                          check_cplusplus,
                                                                                                                          check_objc);
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast,
                                     cast<ElaboratedType>(qual_type)->getNamedType()).IsPossibleDynamicType (dynamic_pointee_type,
                                                                                                             check_cplusplus,
                                                                                                             check_objc);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast,
                                     cast<ParenType>(qual_type)->desugar()).IsPossibleDynamicType (dynamic_pointee_type,
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
                    switch (cast<BuiltinType>(pointee_qual_type)->getKind())
                {
                    case BuiltinType::UnknownAny:
                    case BuiltinType::Void:
                        if (dynamic_pointee_type)
                            dynamic_pointee_type->SetClangType(m_ast, pointee_qual_type);
                        return true;
                        
                    case BuiltinType::NullPtr:
                    case BuiltinType::Bool:
                    case BuiltinType::Char_U:
                    case BuiltinType::UChar:
                    case BuiltinType::WChar_U:
                    case BuiltinType::Char16:
                    case BuiltinType::Char32:
                    case BuiltinType::UShort:
                    case BuiltinType::UInt:
                    case BuiltinType::ULong:
                    case BuiltinType::ULongLong:
                    case BuiltinType::UInt128:
                    case BuiltinType::Char_S:
                    case BuiltinType::SChar:
                    case BuiltinType::WChar_S:
                    case BuiltinType::Short:
                    case BuiltinType::Int:
                    case BuiltinType::Long:
                    case BuiltinType::LongLong:
                    case BuiltinType::Int128:
                    case BuiltinType::Float:
                    case BuiltinType::Double:
                    case BuiltinType::LongDouble:
                    case BuiltinType::Dependent:
                    case BuiltinType::Overload:
                    case BuiltinType::ObjCId:
                    case BuiltinType::ObjCClass:
                    case BuiltinType::ObjCSel:
                    case BuiltinType::BoundMember:
                    case BuiltinType::Half:
                    case BuiltinType::ARCUnbridgedCast:
                    case BuiltinType::PseudoObject:
                    case BuiltinType::BuiltinFn:
                    case BuiltinType::OCLEvent:
                    case BuiltinType::OCLImage1d:
                    case BuiltinType::OCLImage1dArray:
                    case BuiltinType::OCLImage1dBuffer:
                    case BuiltinType::OCLImage2d:
                    case BuiltinType::OCLImage2dArray:
                    case BuiltinType::OCLImage3d:
                    case BuiltinType::OCLSampler:
                        break;
                }
                    break;
                    
                case clang::Type::Record:
                    if (check_cplusplus)
                    {
                        CXXRecordDecl *cxx_record_decl = pointee_qual_type->getAsCXXRecordDecl();
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

    return (GetTypeInfo (NULL) & eTypeIsScalar) != 0;
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
    if (IsArrayType(&element_type, NULL, NULL))
        return element_type.IsScalarType();
    return false;
}


bool
ClangASTType::GetCXXClassName (std::string &class_name) const
{
    if (IsValid())
    {
        QualType qual_type (GetCanonicalQualType());
        
        CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
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
    
    QualType qual_type (GetCanonicalQualType());
    if (qual_type->getAsCXXRecordDecl() != NULL)
        return true;
    return false;
}

bool
ClangASTType::IsBeingDefined () const
{
    if (!IsValid())
        return false;
    QualType qual_type (GetCanonicalQualType());
    const clang::TagType *tag_type = dyn_cast<clang::TagType>(qual_type);
    if (tag_type)
        return tag_type->isBeingDefined();
    return false;
}

bool
ClangASTType::IsObjCObjectPointerType (ClangASTType *class_type_ptr)
{
    if (!IsValid())
        return false;
    
    QualType qual_type (GetCanonicalQualType());

    if (qual_type->isObjCObjectPointerType())
    {
        if (class_type_ptr)
        {
            if (!qual_type->isObjCClassType() &&
                !qual_type->isObjCIdType())
            {
                const ObjCObjectPointerType *obj_pointer_type = dyn_cast<ObjCObjectPointerType>(qual_type);
                if (obj_pointer_type == NULL)
                    class_type_ptr->Clear();
                else
                    class_type_ptr->SetClangType (m_ast, QualType(obj_pointer_type->getInterfaceType(), 0));
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
    
    QualType qual_type (GetCanonicalQualType());

    const ObjCObjectType *object_type = dyn_cast<ObjCObjectType>(qual_type);
    if (object_type)
    {
        const ObjCInterfaceDecl *interface = object_type->getInterface();
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
        std::string type_name (GetTypeName());
        if (!type_name.empty())
            return ConstString (type_name.c_str());
    }
    return ConstString("<invalid>");
}

std::string
ClangASTType::GetTypeName () const
{
    std::string type_name;
    if (IsValid())
    {
        PrintingPolicy printing_policy (m_ast->getPrintingPolicy());
        QualType qual_type(GetQualType());
        printing_policy.SuppressTagKeyword = true;
        printing_policy.LangOpts.WChar = true;
        const TypedefType *typedef_type = qual_type->getAs<TypedefType>();
        if (typedef_type)
        {
            const TypedefNameDecl *typedef_decl = typedef_type->getDecl();
            type_name = typedef_decl->getQualifiedNameAsString(printing_policy);
        }
        else
        {
            type_name = qual_type.getAsString(printing_policy);
        }
    }
    return type_name;
}


uint32_t
ClangASTType::GetTypeInfo (ClangASTType *pointee_or_element_clang_type) const
{
    if (!IsValid())
        return 0;
    
    if (pointee_or_element_clang_type)
        pointee_or_element_clang_type->Clear();
    
    QualType qual_type (GetQualType());
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
        {
            const BuiltinType *builtin_type = dyn_cast<BuiltinType>(qual_type->getCanonicalTypeInternal());
            
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
            const ComplexType *complex_type = dyn_cast<ComplexType>(qual_type->getCanonicalTypeInternal());
            if (complex_type)
            {
                QualType complex_element_type (complex_type->getElementType());
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
                pointee_or_element_clang_type->SetClangType(m_ast, cast<ArrayType>(qual_type.getTypePtr())->getElementType());
            return eTypeHasChildren | eTypeIsArray;
            
        case clang::Type::DependentName:                    return 0;
        case clang::Type::DependentSizedExtVector:          return eTypeHasChildren | eTypeIsVector;
        case clang::Type::DependentTemplateSpecialization:  return eTypeIsTemplate;
        case clang::Type::Decltype:                         return 0;
            
        case clang::Type::Enum:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, cast<EnumType>(qual_type)->getDecl()->getIntegerType());
            return eTypeIsEnumeration | eTypeHasValue;
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::Paren:
            return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetTypeInfo (pointee_or_element_clang_type);
            
        case clang::Type::FunctionProto:                    return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::FunctionNoProto:                  return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::InjectedClassName:                return 0;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(m_ast, cast<ReferenceType>(qual_type.getTypePtr())->getPointeeType());
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
            return eTypeIsTypedef | ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::TypeOfExpr:                       return 0;
        case clang::Type::TypeOf:                           return 0;
        case clang::Type::UnresolvedUsing:                  return 0;
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
        {
            uint32_t vector_type_flags = eTypeHasChildren | eTypeIsVector;
            const VectorType *vector_type = dyn_cast<VectorType>(qual_type->getCanonicalTypeInternal());
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
    QualType qual_type (GetCanonicalQualType().getNonReferenceType());
    if (qual_type->isAnyPointerType())
    {
        if (qual_type->isObjCObjectPointerType())
            return lldb::eLanguageTypeObjC;
        
        QualType pointee_type (qual_type->getPointeeType());
        if (pointee_type->getPointeeCXXRecordDecl() != NULL)
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
                switch (cast<BuiltinType>(qual_type)->getKind())
            {
                default:
                case BuiltinType::Void:
                case BuiltinType::Bool:
                case BuiltinType::Char_U:
                case BuiltinType::UChar:
                case BuiltinType::WChar_U:
                case BuiltinType::Char16:
                case BuiltinType::Char32:
                case BuiltinType::UShort:
                case BuiltinType::UInt:
                case BuiltinType::ULong:
                case BuiltinType::ULongLong:
                case BuiltinType::UInt128:
                case BuiltinType::Char_S:
                case BuiltinType::SChar:
                case BuiltinType::WChar_S:
                case BuiltinType::Short:
                case BuiltinType::Int:
                case BuiltinType::Long:
                case BuiltinType::LongLong:
                case BuiltinType::Int128:
                case BuiltinType::Float:
                case BuiltinType::Double:
                case BuiltinType::LongDouble:
                    break;
                    
                case BuiltinType::NullPtr:
                    return eLanguageTypeC_plus_plus;
                    
                case BuiltinType::ObjCId:
                case BuiltinType::ObjCClass:
                case BuiltinType::ObjCSel:
                    return eLanguageTypeObjC;
                    
                case BuiltinType::Dependent:
                case BuiltinType::Overload:
                case BuiltinType::BoundMember:
                case BuiltinType::UnknownAny:
                    break;
            }
                break;
            case clang::Type::Typedef:
                return ClangASTType(m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetMinimumLanguage();
        }
    }
    return lldb::eLanguageTypeC;
}

lldb::TypeClass
ClangASTType::GetTypeClass () const
{
    if (!IsValid())
        return lldb::eTypeClassInvalid;
    
    QualType qual_type(GetQualType());
    
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
                const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();
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
            return ClangASTType(m_ast, cast<ParenType>(qual_type)->desugar()).GetTypeClass();
        case clang::Type::Elaborated:
            return ClangASTType(m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetTypeClass();
            
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
        QualType result(GetQualType());
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
        QualType result(GetQualType());
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
        QualType result(GetQualType());
        result.getQualifiers().setVolatile (true);
        return ClangASTType (m_ast, result);
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetArrayElementType (uint64_t& stride) const
{
    if (IsValid())
    {
        QualType qual_type(GetCanonicalQualType());
        
        ClangASTType element_type (m_ast, qual_type.getTypePtr()->getArrayElementTypeNoTypeQual()->getCanonicalTypeUnqualified());
        
        // TODO: the real stride will be >= this value.. find the real one!
        stride = element_type.GetByteSize();
        
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

static QualType
GetFullyUnqualifiedType_Impl (ASTContext *ast, QualType qual_type)
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
        const FunctionProtoType* func = dyn_cast<FunctionProtoType>(GetCanonicalQualType());
        if (func)
            return func->getNumArgs();
    }
    return -1;
}

ClangASTType
ClangASTType::GetFunctionArgumentTypeAtIndex (size_t idx)
{
    if (IsValid())
    {
        const FunctionProtoType* func = dyn_cast<FunctionProtoType>(GetCanonicalQualType());
        if (func)
        {
            const uint32_t num_args = func->getNumArgs();
            if (idx < num_args)
                return ClangASTType(m_ast, func->getArgType(idx));
        }
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetFunctionReturnType () const
{
    if (IsValid())
    {
        QualType qual_type(GetCanonicalQualType());
        const FunctionProtoType* func = dyn_cast<FunctionProtoType>(qual_type.getTypePtr());
        if (func)
            return ClangASTType(m_ast, func->getResultType());
    }
    return ClangASTType();
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
        QualType qual_type (GetQualType());
        if (decl_ctx == NULL)
            decl_ctx = m_ast->getTranslationUnitDecl();
        TypedefDecl *decl = TypedefDecl::Create (*m_ast,
                                                 decl_ctx,
                                                 SourceLocation(),
                                                 SourceLocation(),
                                                 &m_ast->Idents.get(typedef_name),
                                                 m_ast->getTrivialTypeSourceInfo(qual_type));
        
        decl->setAccess(AS_public); // TODO respect proper access specifier
        
        // Get a uniqued QualType for the typedef decl type
        return ClangASTType (m_ast, m_ast->getTypedefType (decl));
    }
    return ClangASTType();

}

ClangASTType
ClangASTType::GetPointeeType () const
{
    if (m_type)
    {
        QualType qual_type(GetQualType());
        return ClangASTType (m_ast, qual_type.getTypePtr()->getPointeeType());
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetPointerType () const
{
    if (IsValid())
    {
        QualType qual_type (GetQualType());
        
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
        const TypedefType *typedef_type = dyn_cast<TypedefType>(GetQualType());
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
        QualType qual_type(GetQualType());
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
ClangASTType::GetBitSize () const
{
    if (GetCompleteType ())
    {
        QualType qual_type(GetCanonicalQualType());
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
    return 0;
}

uint64_t
ClangASTType::GetByteSize () const
{
    return (GetBitSize () + 7) / 8;
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
    QualType qual_type(GetCanonicalQualType());
    
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
            switch (cast<BuiltinType>(qual_type)->getKind())
        {
            default: assert(0 && "Unknown builtin type!");
            case BuiltinType::Void:
                break;
                
            case BuiltinType::Bool:
            case BuiltinType::Char_S:
            case BuiltinType::SChar:
            case BuiltinType::WChar_S:
            case BuiltinType::Char16:
            case BuiltinType::Char32:
            case BuiltinType::Short:
            case BuiltinType::Int:
            case BuiltinType::Long:
            case BuiltinType::LongLong:
            case BuiltinType::Int128:        return lldb::eEncodingSint;
                
            case BuiltinType::Char_U:
            case BuiltinType::UChar:
            case BuiltinType::WChar_U:
            case BuiltinType::UShort:
            case BuiltinType::UInt:
            case BuiltinType::ULong:
            case BuiltinType::ULongLong:
            case BuiltinType::UInt128:       return lldb::eEncodingUint;
                
            case BuiltinType::Float:
            case BuiltinType::Double:
            case BuiltinType::LongDouble:    return lldb::eEncodingIEEE754;
                
            case BuiltinType::ObjCClass:
            case BuiltinType::ObjCId:
            case BuiltinType::ObjCSel:       return lldb::eEncodingUint;
                
            case BuiltinType::NullPtr:       return lldb::eEncodingUint;
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
                const ComplexType *complex_type = qual_type->getAsComplexIntegerType();
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
            return ClangASTType(m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetEncoding(count);
            
        case clang::Type::Elaborated:
            return ClangASTType(m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetEncoding(count);
            
        case clang::Type::Paren:
            return ClangASTType(m_ast, cast<ParenType>(qual_type)->desugar()).GetEncoding(count);
            
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
    
    QualType qual_type(GetCanonicalQualType());
    
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
            switch (cast<BuiltinType>(qual_type)->getKind())
        {
                //default: assert(0 && "Unknown builtin type!");
            case BuiltinType::UnknownAny:
            case BuiltinType::Void:
            case BuiltinType::BoundMember:
                break;
                
            case BuiltinType::Bool:          return lldb::eFormatBoolean;
            case BuiltinType::Char_S:
            case BuiltinType::SChar:
            case BuiltinType::WChar_S:
            case BuiltinType::Char_U:
            case BuiltinType::UChar:
            case BuiltinType::WChar_U:       return lldb::eFormatChar;
            case BuiltinType::Char16:        return lldb::eFormatUnicode16;
            case BuiltinType::Char32:        return lldb::eFormatUnicode32;
            case BuiltinType::UShort:        return lldb::eFormatUnsigned;
            case BuiltinType::Short:         return lldb::eFormatDecimal;
            case BuiltinType::UInt:          return lldb::eFormatUnsigned;
            case BuiltinType::Int:           return lldb::eFormatDecimal;
            case BuiltinType::ULong:         return lldb::eFormatUnsigned;
            case BuiltinType::Long:          return lldb::eFormatDecimal;
            case BuiltinType::ULongLong:     return lldb::eFormatUnsigned;
            case BuiltinType::LongLong:      return lldb::eFormatDecimal;
            case BuiltinType::UInt128:       return lldb::eFormatUnsigned;
            case BuiltinType::Int128:        return lldb::eFormatDecimal;
            case BuiltinType::Float:         return lldb::eFormatFloat;
            case BuiltinType::Double:        return lldb::eFormatFloat;
            case BuiltinType::LongDouble:    return lldb::eFormatFloat;
            case BuiltinType::NullPtr:
            case BuiltinType::Overload:
            case BuiltinType::Dependent:
            case BuiltinType::ObjCId:
            case BuiltinType::ObjCClass:
            case BuiltinType::ObjCSel:
            case BuiltinType::Half:
            case BuiltinType::ARCUnbridgedCast:
            case BuiltinType::PseudoObject:
            case BuiltinType::BuiltinFn:
            case BuiltinType::OCLEvent:
            case BuiltinType::OCLImage1d:
            case BuiltinType::OCLImage1dArray:
            case BuiltinType::OCLImage1dBuffer:
            case BuiltinType::OCLImage2d:
            case BuiltinType::OCLImage2dArray:
            case BuiltinType::OCLImage3d:
            case BuiltinType::OCLSampler:
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
            return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetFormat();
        case clang::Type::Auto:
            return ClangASTType (m_ast, cast<AutoType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Paren:
            return ClangASTType (m_ast, cast<ParenType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetFormat();
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
            break;
    }
    // We don't know hot to display this type...
    return lldb::eFormatBytes;
}

static bool
ObjCDeclHasIVars (ObjCInterfaceDecl *class_interface_decl, bool check_superclass)
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
    QualType qual_type(GetQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (cast<BuiltinType>(qual_type)->getKind())
        {
            case BuiltinType::ObjCId:    // child is Class
            case BuiltinType::ObjCClass: // child is Class
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
                const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    if (omit_empty_base_classes)
                    {
                        // Check each base classes to see if it or any of its
                        // base classes contain any fields. This can help
                        // limit the noise in variable views by not having to
                        // show base classes that contain no members.
                        CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            const CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
                            
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
                RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                    ++num_children;
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteQualType (m_ast, qual_type))
            {
                const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
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
            const ObjCObjectPointerType *pointer_type = cast<ObjCObjectPointerType>(qual_type.getTypePtr());
            QualType pointee_type = pointer_type->getPointeeType();
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
            num_children = cast<VectorType>(qual_type.getTypePtr())->getNumElements();
            break;
            
        case clang::Type::ConstantArray:
            num_children = cast<ConstantArrayType>(qual_type.getTypePtr())->getSize().getLimitedValue();
            break;
            
        case clang::Type::Pointer:
        {
            const PointerType *pointer_type = cast<PointerType>(qual_type.getTypePtr());
            QualType pointee_type (pointer_type->getPointeeType());
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
            const ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
            QualType pointee_type = reference_type->getPointeeType();
            uint32_t num_pointee_children = ClangASTType (m_ast, pointee_type).GetNumChildren (omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
            break;
            
            
        case clang::Type::Typedef:
            num_children = ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Elaborated:
            num_children = ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Paren:
            num_children = ClangASTType (m_ast, cast<ParenType>(qual_type)->desugar()).GetNumChildren (omit_empty_base_classes);
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
        QualType qual_type(GetQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        if (type_class == clang::Type::Builtin)
        {
            switch (cast<clang::BuiltinType>(qual_type)->getKind())
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
    QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumBases();
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType())
            {
                const ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType())
            {
                const ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            count = ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumDirectBaseClasses ();
            break;
            
        case clang::Type::Elaborated:
            count = ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetNumDirectBaseClasses ();
            break;
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetNumDirectBaseClasses ();
            
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
    QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumVBases();
            }
            break;
            
        case clang::Type::Typedef:
            count = ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumVirtualBaseClasses();
            break;
            
        case clang::Type::Elaborated:
            count = ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetNumVirtualBaseClasses();
            break;
            
        case clang::Type::Paren:
            count = ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetNumVirtualBaseClasses();
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
    QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const RecordType *record_type = dyn_cast<RecordType>(qual_type.getTypePtr());
                if (record_type)
                {
                    RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        uint32_t field_idx = 0;
                        RecordDecl::field_iterator field, field_end;
                        for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                            ++field_idx;
                        count = field_idx;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            count = ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumFields();
            break;
            
        case clang::Type::Elaborated:
            count = ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetNumFields();
            break;
            
        case clang::Type::Paren:
            count = ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetNumFields();
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType())
            {
                const ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    
                    if (class_interface_decl)
                        count = class_interface_decl->ivar_size();
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType())
            {
                const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
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
    
    QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(cxx_record_decl);
                                const CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
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
            if (idx == 0 && GetCompleteType())
            {
                const ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    if (class_interface_decl)
                    {
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
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
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (idx == 0 && GetCompleteType())
            {
                const ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
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
            return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetDirectBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetDirectBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetDirectBaseClassAtIndex (idx, bit_offset_ptr);
            
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
    
    QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->vbases_begin(), base_class_end = cxx_record_decl->vbases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(cxx_record_decl);
                                const CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
                                *bit_offset_ptr = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                                
                            }
                            return ClangASTType (m_ast, base_class->getType());
                        }
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetVirtualBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetVirtualBaseClassAtIndex (idx, bit_offset_ptr);
            
        case clang::Type::Paren:
            return  ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetVirtualBaseClassAtIndex (idx, bit_offset_ptr);
            
        default:
            break;
    }
    return ClangASTType();
}

static clang_type_t
GetObjCFieldAtIndex (clang::ASTContext *ast,
                     ObjCInterfaceDecl *class_interface_decl,
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
            ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
            uint32_t ivar_idx = 0;
            
            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++ivar_idx)
            {
                if (ivar_idx == idx)
                {
                    const ObjCIvarDecl* ivar_decl = *ivar_pos;
                    
                    QualType ivar_qual_type(ivar_decl->getType());
                    
                    name.assign(ivar_decl->getNameAsString());
                    
                    if (bit_offset_ptr)
                    {
                        const ASTRecordLayout &interface_layout = ast->getASTObjCInterfaceLayout(class_interface_decl);
                        *bit_offset_ptr = interface_layout.getFieldOffset (ivar_idx);
                    }
                    
                    const bool is_bitfield = ivar_pos->isBitField();
                    
                    if (bitfield_bit_size_ptr)
                    {
                        *bitfield_bit_size_ptr = 0;
                        
                        if (is_bitfield && ast)
                        {
                            Expr *bitfield_bit_size_expr = ivar_pos->getBitWidth();
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
    return NULL;
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
    
    QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType())
            {
                const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();
                uint32_t field_idx = 0;
                RecordDecl::field_iterator field, field_end;
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
                            const ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(record_decl);
                            *bit_offset_ptr = record_layout.getFieldOffset (field_idx);
                        }
                        
                        const bool is_bitfield = field->isBitField();
                        
                        if (bitfield_bit_size_ptr)
                        {
                            *bitfield_bit_size_ptr = 0;
                            
                            if (is_bitfield)
                            {
                                Expr *bitfield_bit_size_expr = field->getBitWidth();
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
                const ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    return ClangASTType (m_ast, GetObjCFieldAtIndex(m_ast, class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType())
            {
                const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    return ClangASTType (m_ast, GetObjCFieldAtIndex(m_ast, class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).
                        GetFieldAtIndex (idx,
                                         name,
                                         bit_offset_ptr,
                                         bitfield_bit_size_ptr,
                                         is_bitfield_ptr);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).
                        GetFieldAtIndex (idx,
                                         name,
                                         bit_offset_ptr,
                                         bitfield_bit_size_ptr,
                                         is_bitfield_ptr);
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).
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
    
    QualType qual_type(GetCanonicalQualType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (cast<clang::BuiltinType>(qual_type)->getKind())
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
        case clang::Type::Paren:                    return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetNumPointeeChildren ();
        case clang::Type::Typedef:                  return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumPointeeChildren ();
        case clang::Type::Elaborated:               return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetNumPointeeChildren ();
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
                                        const char *parent_name,
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
                                        bool &child_is_deref_of_parent) const
{
    if (!IsValid())
        return ClangASTType();
    
    QualType parent_qual_type(GetCanonicalQualType());
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
                switch (cast<clang::BuiltinType>(parent_qual_type)->getKind())
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
                const RecordType *record_type = cast<RecordType>(parent_qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(record_decl);
                uint32_t child_idx = 0;
                
                const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    // We might have base classes to print out first
                    CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const CXXRecordDecl *base_class_decl = NULL;
                        
                        // Skip empty base classes
                        if (omit_empty_base_classes)
                        {
                            base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
                            if (ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                        }
                        
                        if (idx == child_idx)
                        {
                            if (base_class_decl == NULL)
                                base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
                            
                            
                            if (base_class->isVirtual())
                                bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                            else
                                bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                            
                            // Base classes should be a multiple of 8 bits in size
                            child_byte_offset = bit_offset/8;
                            ClangASTType base_class_clang_type(m_ast, base_class->getType());
                            child_name = base_class_clang_type.GetTypeName();
                            uint64_t base_class_clang_type_bit_size = base_class_clang_type.GetBitSize();
                            
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
                RecordDecl::field_iterator field, field_end;
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
                        child_byte_size = field_clang_type.GetByteSize();
                        
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
                const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(parent_qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    uint32_t child_idx = 0;
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        const ASTRecordLayout &interface_layout = m_ast->getASTObjCInterfaceLayout(class_interface_decl);
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                ClangASTType base_class_clang_type (m_ast, m_ast->getObjCInterfaceType(superclass_interface_decl));
                                if (base_class_clang_type.GetNumChildren(omit_empty_base_classes) > 0)
                                {
                                    if (idx == 0)
                                    {
                                        QualType ivar_qual_type(m_ast->getObjCInterfaceType(superclass_interface_decl));
                                        
                                        
                                        child_name.assign(superclass_interface_decl->getNameAsString().c_str());
                                        
                                        std::pair<uint64_t, unsigned> ivar_type_info = m_ast->getTypeInfo(ivar_qual_type.getTypePtr());
                                        
                                        child_byte_size = ivar_type_info.first / 8;
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
                            ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos)
                            {
                                if (child_idx == idx)
                                {
                                    ObjCIvarDecl* ivar_decl = *ivar_pos;
                                    
                                    QualType ivar_qual_type(ivar_decl->getType());
                                    
                                    child_name.assign(ivar_decl->getNameAsString().c_str());
                                    
                                    std::pair<uint64_t, unsigned> ivar_type_info = m_ast->getTypeInfo(ivar_qual_type.getTypePtr());
                                    
                                    child_byte_size = ivar_type_info.first / 8;
                                    
                                    // Figure out the field offset within the current struct/union/class type
                                    // For ObjC objects, we can't trust the bit offset we get from the Clang AST, since
                                    // that doesn't account for the space taken up by unbacked properties, or from
                                    // the changing size of base classes that are newer than this class.
                                    // So if we have a process around that we can ask about this object, do so.
                                    child_byte_offset = LLDB_INVALID_IVAR_OFFSET;
                                    Process *process = NULL;
                                    if (exe_ctx)
                                        process = exe_ctx->GetProcessPtr();
                                    if (process)
                                    {
                                        ObjCLanguageRuntime *objc_runtime = process->GetObjCLanguageRuntime();
                                        if (objc_runtime != NULL)
                                        {
                                            ClangASTType parent_ast_type (m_ast, parent_qual_type);
                                            child_byte_offset = objc_runtime->GetByteOffsetForIvar (parent_ast_type, ivar_decl->getNameAsString().c_str());
                                        }
                                    }
                                    
                                    // Setting this to UINT32_MAX to make sure we don't compute it twice...
                                    bit_offset = UINT32_MAX;
                                    
                                    if (child_byte_offset == LLDB_INVALID_IVAR_OFFSET)
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
                                                                        parent_name,
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
                                                                        tmp_child_is_deref_of_parent);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0 && pointee_clang_type.GetCompleteType())
                    {
                        child_byte_size = pointee_clang_type.GetByteSize();
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
                const VectorType *array = cast<VectorType>(parent_qual_type.getTypePtr());
                if (array)
                {
                    ClangASTType element_type (m_ast, array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf (element_name, sizeof (element_name), "[%zu]", idx);
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize();
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
                const ArrayType *array = cast<ArrayType>(parent_qual_type.getTypePtr());
                if (array)
                {
                    ClangASTType element_type (m_ast, array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf (element_name, sizeof (element_name), "[%zu]", idx);
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize();
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
                                                                        parent_name,
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
                                                                        tmp_child_is_deref_of_parent);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize();
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
                const ReferenceType *reference_type = cast<ReferenceType>(parent_qual_type.getTypePtr());
                ClangASTType pointee_clang_type (m_ast, reference_type->getPointeeType());
                if (transparent_pointers && pointee_clang_type.IsAggregateType ())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                        parent_name,
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
                                                                        tmp_child_is_deref_of_parent);
                }
                else
                {
                    if (parent_name)
                    {
                        child_name.assign(1, '&');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize();
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            {
                ClangASTType typedefed_clang_type (m_ast, cast<TypedefType>(parent_qual_type)->getDecl()->getUnderlyingType());
                return typedefed_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                      parent_name,
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
                                                                      child_is_deref_of_parent);
            }
            break;
            
        case clang::Type::Elaborated:
            {
                ClangASTType elaborated_clang_type (m_ast, cast<ElaboratedType>(parent_qual_type)->getNamedType());
                return elaborated_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                       parent_name,
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
                                                                       child_is_deref_of_parent);
            }
            
        case clang::Type::Paren:
            {
                ClangASTType paren_clang_type (m_ast, llvm::cast<clang::ParenType>(parent_qual_type)->desugar());
                return paren_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                  parent_name,
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
                                                                  child_is_deref_of_parent);
            }
            
            
        default:
            break;
    }
    return ClangASTType();
}

static inline bool
BaseSpecifierIsEmpty (const CXXBaseSpecifier *b)
{
    return ClangASTContext::RecordHasFields(b->getType()->getAsCXXRecordDecl()) == false;
}

static uint32_t
GetIndexForRecordBase
(
 const RecordDecl *record_decl,
 const CXXBaseSpecifier *base_spec,
 bool omit_empty_base_classes
 )
{
    uint32_t child_idx = 0;
    
    const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
    
    //    const char *super_name = record_decl->getNameAsCString();
    //    const char *base_name = base_spec->getType()->getAs<RecordType>()->getDecl()->getNameAsCString();
    //    printf ("GetIndexForRecordChild (%s, %s)\n", super_name, base_name);
    //
    if (cxx_record_decl)
    {
        CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
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
            //                    base_class->getType()->getAs<RecordType>()->getDecl()->getNameAsCString());
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
GetIndexForRecordChild (const RecordDecl *record_decl,
                        NamedDecl *canonical_decl,
                        bool omit_empty_base_classes)
{
    uint32_t child_idx = ClangASTContext::GetNumBaseClasses (dyn_cast<CXXRecordDecl>(record_decl),
                                                             omit_empty_base_classes);
    
    RecordDecl::field_iterator field, field_end;
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
        QualType qual_type(GetCanonicalQualType());
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                    const RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
                    
                    // Try and find a field that matches NAME
                    RecordDecl::field_iterator field, field_end;
                    StringRef name_sref(name);
                    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                         field != field_end;
                         ++field, ++child_idx)
                    {
                        if (field->getName().equals (name_sref))
                        {
                            // We have to add on the number of base classes to this index!
                            child_indexes.push_back (child_idx + ClangASTContext::GetNumBaseClasses (cxx_record_decl, omit_empty_base_classes));
                            return child_indexes.size();
                        }
                    }
                    
                    if (cxx_record_decl)
                    {
                        const RecordDecl *parent_record_decl = cxx_record_decl;
                        
                        //printf ("parent = %s\n", parent_record_decl->getNameAsCString());
                        
                        //const Decl *root_cdecl = cxx_record_decl->getCanonicalDecl();
                        // Didn't find things easily, lets let clang do its thang...
                        IdentifierInfo & ident_ref = m_ast->Idents.get(name_sref);
                        DeclarationName decl_name(&ident_ref);
                        
                        CXXBasePaths paths;
                        if (cxx_record_decl->lookupInBases(CXXRecordDecl::FindOrdinaryMember,
                                                           decl_name.getAsOpaquePtr(),
                                                           paths))
                        {
                            CXXBasePaths::const_paths_iterator path, path_end = paths.end();
                            for (path = paths.begin(); path != path_end; ++path)
                            {
                                const size_t num_path_elements = path->size();
                                for (size_t e=0; e<num_path_elements; ++e)
                                {
                                    CXXBasePathElement elem = (*path)[e];
                                    
                                    child_idx = GetIndexForRecordBase (parent_record_decl, elem.Base, omit_empty_base_classes);
                                    if (child_idx == UINT32_MAX)
                                    {
                                        child_indexes.clear();
                                        return 0;
                                    }
                                    else
                                    {
                                        child_indexes.push_back (child_idx);
                                        parent_record_decl = cast<RecordDecl>(elem.Base->getType()->getAs<RecordType>()->getDecl());
                                    }
                                }
                                for (NamedDecl *path_decl : path->Decls)
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
                    StringRef name_sref(name);
                    const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
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
                    ClangASTType objc_object_clang_type (m_ast, cast<ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                    return objc_object_clang_type.GetIndexOfChildMemberWithName (name,
                                                                                 omit_empty_base_classes,
                                                                                 child_indexes);
                }
                break;
                
                
            case clang::Type::ConstantArray:
            {
                //                const ConstantArrayType *array = cast<ConstantArrayType>(parent_qual_type.getTypePtr());
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
                //                MemberPointerType *mem_ptr_type = cast<MemberPointerType>(qual_type.getTypePtr());
                //                QualType pointee_type = mem_ptr_type->getPointeeType();
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
                    const ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
                    QualType pointee_type(reference_type->getPointeeType());
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
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildMemberWithName (name,
                                                                                                                                         omit_empty_base_classes,
                                                                                                                                         child_indexes);
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildMemberWithName (name,
                                                                                                                            omit_empty_base_classes,
                                                                                                                            child_indexes);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildMemberWithName (name,
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
        QualType qual_type(GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                    const RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
                    
                    if (cxx_record_decl)
                    {
                        CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            // Skip empty base classes
                            CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
                            if (omit_empty_base_classes && ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                            
                            ClangASTType base_class_clang_type (m_ast, base_class->getType());
                            std::string base_class_type_name (base_class_clang_type.GetTypeName());
                            if (base_class_type_name.compare (name) == 0)
                                return child_idx;
                            ++child_idx;
                        }
                    }
                    
                    // Try and find a field that matches NAME
                    RecordDecl::field_iterator field, field_end;
                    StringRef name_sref(name);
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
                    StringRef name_sref(name);
                    const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
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
                    ClangASTType pointee_clang_type (m_ast, cast<ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                    return pointee_clang_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                }
                break;
                
            case clang::Type::ConstantArray:
            {
                //                const ConstantArrayType *array = cast<ConstantArrayType>(parent_qual_type.getTypePtr());
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
                //                MemberPointerType *mem_ptr_type = cast<MemberPointerType>(qual_type.getTypePtr());
                //                QualType pointee_type = mem_ptr_type->getPointeeType();
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
                    const ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
                    ClangASTType pointee_type (m_ast, reference_type->getPointeeType());
                    
                    if (pointee_type.IsAggregateType ())
                    {
                        return pointee_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                    }
                }
                break;
                
            case clang::Type::Pointer:
                {
                    const PointerType *pointer_type = cast<PointerType>(qual_type.getTypePtr());
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
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
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
        QualType qual_type (GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                    if (cxx_record_decl)
                    {
                        const ClassTemplateSpecializationDecl *template_decl = dyn_cast<ClassTemplateSpecializationDecl>(cxx_record_decl);
                        if (template_decl)
                            return template_decl->getTemplateArgs().size();
                    }
                }
                break;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumTemplateArguments();
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetNumTemplateArguments();
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<ParenType>(qual_type)->desugar()).GetNumTemplateArguments();
                
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
        QualType qual_type (GetCanonicalQualType());
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType ())
                {
                    const CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                    if (cxx_record_decl)
                    {
                        const ClassTemplateSpecializationDecl *template_decl = dyn_cast<ClassTemplateSpecializationDecl>(cxx_record_decl);
                        if (template_decl && arg_idx < template_decl->getTemplateArgs().size())
                        {
                            const TemplateArgument &template_arg = template_decl->getTemplateArgs()[arg_idx];
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
                                    assert (!"Unhandled TemplateArgument::ArgKind");
                                    break;
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::Typedef:
                return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetTemplateArgument (arg_idx, kind);
                
            case clang::Type::Elaborated:
                return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetTemplateArgument (arg_idx, kind);
                
            case clang::Type::Paren:
                return ClangASTType (m_ast, cast<ParenType>(qual_type)->desugar()).GetTemplateArgument (arg_idx, kind);
                
            default:
                break;
        }
    }
    kind = eTemplateArgumentKindNull;
    return ClangASTType ();
}

static bool
IsOperator (const char *name, OverloadedOperatorKind &op_kind)
{
    if (name == NULL || name[0] == '\0')
        return false;
    
#define OPERATOR_PREFIX "operator"
#define OPERATOR_PREFIX_LENGTH (sizeof (OPERATOR_PREFIX) - 1)
    
    const char *post_op_name = NULL;
    
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
    op_kind = NUM_OVERLOADED_OPERATORS;
    
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
                op_kind = OO_New;
            else if (strcmp (post_op_name, "new[]") == 0)
                op_kind = OO_Array_New;
            break;
            
        case 'd':
            if (no_space)
                return false;
            if (strcmp (post_op_name, "delete") == 0)
                op_kind = OO_Delete;
            else if (strcmp (post_op_name, "delete[]") == 0)
                op_kind = OO_Array_Delete;
            break;
            
        case '+':
            if (post_op_name[1] == '\0')
                op_kind = OO_Plus;
            else if (post_op_name[2] == '\0')
            {
                if (post_op_name[1] == '=')
                    op_kind = OO_PlusEqual;
                else if (post_op_name[1] == '+')
                    op_kind = OO_PlusPlus;
            }
            break;
            
        case '-':
            if (post_op_name[1] == '\0')
                op_kind = OO_Minus;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = OO_MinusEqual; break;
                    case '-': op_kind = OO_MinusMinus; break;
                    case '>': op_kind = OO_Arrow; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '*')
                    op_kind = OO_ArrowStar; break;
            }
            break;
            
        case '*':
            if (post_op_name[1] == '\0')
                op_kind = OO_Star;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = OO_StarEqual;
            break;
            
        case '/':
            if (post_op_name[1] == '\0')
                op_kind = OO_Slash;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = OO_SlashEqual;
            break;
            
        case '%':
            if (post_op_name[1] == '\0')
                op_kind = OO_Percent;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = OO_PercentEqual;
            break;
            
            
        case '^':
            if (post_op_name[1] == '\0')
                op_kind = OO_Caret;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = OO_CaretEqual;
            break;
            
        case '&':
            if (post_op_name[1] == '\0')
                op_kind = OO_Amp;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = OO_AmpEqual; break;
                    case '&': op_kind = OO_AmpAmp; break;
                }
            }
            break;
            
        case '|':
            if (post_op_name[1] == '\0')
                op_kind = OO_Pipe;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = OO_PipeEqual; break;
                    case '|': op_kind = OO_PipePipe; break;
                }
            }
            break;
            
        case '~':
            if (post_op_name[1] == '\0')
                op_kind = OO_Tilde;
            break;
            
        case '!':
            if (post_op_name[1] == '\0')
                op_kind = OO_Exclaim;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = OO_ExclaimEqual;
            break;
            
        case '=':
            if (post_op_name[1] == '\0')
                op_kind = OO_Equal;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = OO_EqualEqual;
            break;
            
        case '<':
            if (post_op_name[1] == '\0')
                op_kind = OO_Less;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '<': op_kind = OO_LessLess; break;
                    case '=': op_kind = OO_LessEqual; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '=')
                    op_kind = OO_LessLessEqual;
            }
            break;
            
        case '>':
            if (post_op_name[1] == '\0')
                op_kind = OO_Greater;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '>': op_kind = OO_GreaterGreater; break;
                    case '=': op_kind = OO_GreaterEqual; break;
                }
            }
            else if (post_op_name[1] == '>' &&
                     post_op_name[2] == '=' &&
                     post_op_name[3] == '\0')
            {
                op_kind = OO_GreaterGreaterEqual;
            }
            break;
            
        case ',':
            if (post_op_name[1] == '\0')
                op_kind = OO_Comma;
            break;
            
        case '(':
            if (post_op_name[1] == ')' && post_op_name[2] == '\0')
                op_kind = OO_Call;
            break;
            
        case '[':
            if (post_op_name[1] == ']' && post_op_name[2] == '\0')
                op_kind = OO_Subscript;
            break;
    }
    
    return true;
}

static inline bool
check_op_param (uint32_t op_kind, bool unary, bool binary, uint32_t num_params)
{
    // Special-case call since it can take any number of operands
    if(op_kind == OO_Call)
        return true;
    
    // The parameter count doens't include "this"
    if (num_params == 0)
        return unary;
    if (num_params == 1)
        return binary;
    else
        return false;
}

clang::RecordDecl *
ClangASTType::GetAsRecordDecl () const
{
    const RecordType *record_type = dyn_cast<RecordType>(GetCanonicalQualType());
    if (record_type)
        return record_type->getDecl();
    return NULL;
}

clang::CXXRecordDecl *
ClangASTType::GetAsCXXRecordDecl () const
{
    return GetCanonicalQualType()->getAsCXXRecordDecl();
}

ObjCInterfaceDecl *
ClangASTType::GetAsObjCInterfaceDecl () const
{
    const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(GetCanonicalQualType());
    if (objc_class_type)
        return objc_class_type->getInterface();
    return NULL;
}

clang::FieldDecl *
ClangASTType::AddFieldToRecordType (const char *name,
                                    const ClangASTType &field_clang_type,
                                    AccessType access,
                                    uint32_t bitfield_bit_size)
{
    if (!IsValid() || !field_clang_type.IsValid())
        return NULL;
    
    FieldDecl *field = NULL;

    clang::Expr *bit_width = NULL;
    if (bitfield_bit_size != 0)
    {
        APInt bitfield_bit_size_apint(m_ast->getTypeSize(m_ast->IntTy), bitfield_bit_size);
        bit_width = new (*m_ast)IntegerLiteral (*m_ast, bitfield_bit_size_apint, m_ast->IntTy, SourceLocation());
    }

    RecordDecl *record_decl = GetAsRecordDecl ();
    if (record_decl)
    {
        field = FieldDecl::Create (*m_ast,
                                   record_decl,
                                   SourceLocation(),
                                   SourceLocation(),
                                   name ? &m_ast->Idents.get(name) : NULL,  // Identifier
                                   field_clang_type.GetQualType(),          // Field type
                                   NULL,            // TInfo *
                                   bit_width,       // BitWidth
                                   false,           // Mutable
                                   ICIS_NoInit);    // HasInit
        
        if (!name)
        {
            // Determine whether this field corresponds to an anonymous
            // struct or union.
            if (const TagType *TagT = field->getType()->getAs<TagType>()) {
                if (RecordDecl *Rec = dyn_cast<RecordDecl>(TagT->getDecl()))
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
        ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
        
        if (class_interface_decl)
        {
            const bool is_synthesized = false;
            
            field_clang_type.GetCompleteType();

            field = ObjCIvarDecl::Create (*m_ast,
                                          class_interface_decl,
                                          SourceLocation(),
                                          SourceLocation(),
                                          name ? &m_ast->Idents.get(name) : NULL,   // Identifier
                                          field_clang_type.GetQualType(),           // Field type
                                          NULL,                                     // TypeSourceInfo *
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
    RecordDecl *record_decl = GetAsRecordDecl();
    
    if (!record_decl)
        return;
    
    typedef llvm::SmallVector <IndirectFieldDecl *, 1> IndirectFieldVector;
    
    IndirectFieldVector indirect_fields;
    RecordDecl::field_iterator field_pos;
    RecordDecl::field_iterator field_end_pos = record_decl->field_end();
    RecordDecl::field_iterator last_field_pos = field_end_pos;
    for (field_pos = record_decl->field_begin(); field_pos != field_end_pos; last_field_pos = field_pos++)
    {
        if (field_pos->isAnonymousStructOrUnion())
        {
            QualType field_qual_type = field_pos->getType();
            
            const RecordType *field_record_type = field_qual_type->getAs<RecordType>();
            
            if (!field_record_type)
                continue;
            
            RecordDecl *field_record_decl = field_record_type->getDecl();
            
            if (!field_record_decl)
                continue;
            
            for (RecordDecl::decl_iterator di = field_record_decl->decls_begin(), de = field_record_decl->decls_end();
                 di != de;
                 ++di)
            {
                if (FieldDecl *nested_field_decl = dyn_cast<FieldDecl>(*di))
                {
                    NamedDecl **chain = new (*m_ast) NamedDecl*[2];
                    chain[0] = *field_pos;
                    chain[1] = nested_field_decl;
                    IndirectFieldDecl *indirect_field = IndirectFieldDecl::Create(*m_ast,
                                                                                  record_decl,
                                                                                  SourceLocation(),
                                                                                  nested_field_decl->getIdentifier(),
                                                                                  nested_field_decl->getType(),
                                                                                  chain,
                                                                                  2);
                    
                    indirect_field->setAccess(ClangASTContext::UnifyAccessSpecifiers(field_pos->getAccess(),
                                                                                     nested_field_decl->getAccess()));
                    
                    indirect_fields.push_back(indirect_field);
                }
                else if (IndirectFieldDecl *nested_indirect_field_decl = dyn_cast<IndirectFieldDecl>(*di))
                {
                    int nested_chain_size = nested_indirect_field_decl->getChainingSize();
                    NamedDecl **chain = new (*m_ast) NamedDecl*[nested_chain_size + 1];
                    chain[0] = *field_pos;
                    
                    int chain_index = 1;
                    for (IndirectFieldDecl::chain_iterator nci = nested_indirect_field_decl->chain_begin(),
                         nce = nested_indirect_field_decl->chain_end();
                         nci < nce;
                         ++nci)
                    {
                        chain[chain_index] = *nci;
                        chain_index++;
                    }
                    
                    IndirectFieldDecl *indirect_field = IndirectFieldDecl::Create(*m_ast,
                                                                                  record_decl,
                                                                                  SourceLocation(),
                                                                                  nested_indirect_field_decl->getIdentifier(),
                                                                                  nested_indirect_field_decl->getType(),
                                                                                  chain,
                                                                                  nested_chain_size + 1);
                    
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

clang::VarDecl *
ClangASTType::AddVariableToRecordType (const char *name,
                                       const ClangASTType &var_type,
                                       AccessType access)
{
    clang::VarDecl *var_decl = NULL;
    
    if (!IsValid() || !var_type.IsValid())
        return NULL;
    
    RecordDecl *record_decl = GetAsRecordDecl ();
    if (record_decl)
    {        
        var_decl = VarDecl::Create (*m_ast,                                     // ASTContext &
                                    record_decl,                                // DeclContext *
                                    SourceLocation(),                           // SourceLocation StartLoc
                                    SourceLocation(),                           // SourceLocation IdLoc
                                    name ? &m_ast->Idents.get(name) : NULL,     // IdentifierInfo *
                                    var_type.GetQualType(),                     // Variable QualType
                                    NULL,                                       // TypeSourceInfo *
                                    SC_Static);                                 // StorageClass
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


CXXMethodDecl *
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
    if (!IsValid() || !method_clang_type.IsValid() || name == NULL || name[0] == '\0')
        return NULL;
    
    QualType record_qual_type(GetCanonicalQualType());
    
    CXXRecordDecl *cxx_record_decl = record_qual_type->getAsCXXRecordDecl();
    
    if (cxx_record_decl == NULL)
        return NULL;
    
    QualType method_qual_type (method_clang_type.GetQualType());
    
    CXXMethodDecl *cxx_method_decl = NULL;
    
    DeclarationName decl_name (&m_ast->Idents.get(name));
    
    const clang::FunctionType *function_type = dyn_cast<FunctionType>(method_qual_type.getTypePtr());
    
    if (function_type == NULL)
        return NULL;
    
    const FunctionProtoType *method_function_prototype (dyn_cast<FunctionProtoType>(function_type));
    
    if (!method_function_prototype)
        return NULL;
    
    unsigned int num_params = method_function_prototype->getNumArgs();
    
    CXXDestructorDecl *cxx_dtor_decl(NULL);
    CXXConstructorDecl *cxx_ctor_decl(NULL);
    
    if (is_artificial)
        return NULL; // skip everything artificial
    
    if (name[0] == '~')
    {
        cxx_dtor_decl = CXXDestructorDecl::Create (*m_ast,
                                                   cxx_record_decl,
                                                   SourceLocation(),
                                                   DeclarationNameInfo (m_ast->DeclarationNames.getCXXDestructorName (m_ast->getCanonicalType (record_qual_type)), SourceLocation()),
                                                   method_qual_type,
                                                   NULL,
                                                   is_inline,
                                                   is_artificial);
        cxx_method_decl = cxx_dtor_decl;
    }
    else if (decl_name == cxx_record_decl->getDeclName())
    {
        cxx_ctor_decl = CXXConstructorDecl::Create (*m_ast,
                                                    cxx_record_decl,
                                                    SourceLocation(),
                                                    DeclarationNameInfo (m_ast->DeclarationNames.getCXXConstructorName (m_ast->getCanonicalType (record_qual_type)), SourceLocation()),
                                                    method_qual_type,
                                                    NULL, // TypeSourceInfo *
                                                    is_explicit,
                                                    is_inline,
                                                    is_artificial,
                                                    false /*is_constexpr*/);
        cxx_method_decl = cxx_ctor_decl;
    }
    else
    {
        clang::StorageClass SC = is_static ? SC_Static : SC_None;
        OverloadedOperatorKind op_kind = NUM_OVERLOADED_OPERATORS;
        
        if (IsOperator (name, op_kind))
        {
            if (op_kind != NUM_OVERLOADED_OPERATORS)
            {
                // Check the number of operator parameters. Sometimes we have
                // seen bad DWARF that doesn't correctly describe operators and
                // if we try to create a methed and add it to the class, clang
                // will assert and crash, so we need to make sure things are
                // acceptable.
                if (!ClangASTContext::CheckOverloadedOperatorKindParameterCount (op_kind, num_params))
                    return NULL;
                cxx_method_decl = CXXMethodDecl::Create (*m_ast,
                                                         cxx_record_decl,
                                                         SourceLocation(),
                                                         DeclarationNameInfo (m_ast->DeclarationNames.getCXXOperatorName (op_kind), SourceLocation()),
                                                         method_qual_type,
                                                         NULL, // TypeSourceInfo *
                                                         SC,
                                                         is_inline,
                                                         false /*is_constexpr*/,
                                                         SourceLocation());
            }
            else if (num_params == 0)
            {
                // Conversion operators don't take params...
                cxx_method_decl = CXXConversionDecl::Create (*m_ast,
                                                             cxx_record_decl,
                                                             SourceLocation(),
                                                             DeclarationNameInfo (m_ast->DeclarationNames.getCXXConversionFunctionName (m_ast->getCanonicalType (function_type->getResultType())), SourceLocation()),
                                                             method_qual_type,
                                                             NULL, // TypeSourceInfo *
                                                             is_inline,
                                                             is_explicit,
                                                             false /*is_constexpr*/,
                                                             SourceLocation());
            }
        }
        
        if (cxx_method_decl == NULL)
        {
            cxx_method_decl = CXXMethodDecl::Create (*m_ast,
                                                     cxx_record_decl,
                                                     SourceLocation(),
                                                     DeclarationNameInfo (decl_name, SourceLocation()),
                                                     method_qual_type,
                                                     NULL, // TypeSourceInfo *
                                                     SC,
                                                     is_inline,
                                                     false /*is_constexpr*/,
                                                     SourceLocation());
        }
    }
    
    AccessSpecifier access_specifier = ClangASTContext::ConvertAccessTypeToAccessSpecifier (access);
    
    cxx_method_decl->setAccess (access_specifier);
    cxx_method_decl->setVirtualAsWritten (is_virtual);
    
    if (is_attr_used)
        cxx_method_decl->addAttr(::new (*m_ast) UsedAttr(SourceRange(), *m_ast));
    
    // Populate the method decl with parameter decls
    
    llvm::SmallVector<ParmVarDecl *, 12> params;
    
    for (unsigned param_index = 0;
         param_index < num_params;
         ++param_index)
    {
        params.push_back (ParmVarDecl::Create (*m_ast,
                                               cxx_method_decl,
                                               SourceLocation(),
                                               SourceLocation(),
                                               NULL, // anonymous
                                               method_function_prototype->getArgType(param_index),
                                               NULL,
                                               SC_None,
                                               NULL));
    }
    
    cxx_method_decl->setParams (ArrayRef<ParmVarDecl*>(params));
    
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

CXXBaseSpecifier *
ClangASTType::CreateBaseClassSpecifier (AccessType access, bool is_virtual, bool base_of_class)
{
    if (IsValid())
        return new CXXBaseSpecifier (SourceRange(),
                                     is_virtual,
                                     base_of_class,
                                     ClangASTContext::ConvertAccessTypeToAccessSpecifier (access),
                                     m_ast->getTrivialTypeSourceInfo (GetQualType()),
                                     SourceLocation());
    return NULL;
}

void
ClangASTType::DeleteBaseClassSpecifiers (CXXBaseSpecifier **base_classes, unsigned num_base_classes)
{
    for (unsigned i=0; i<num_base_classes; ++i)
    {
        delete base_classes[i];
        base_classes[i] = NULL;
    }
}

bool
ClangASTType::SetBaseClassesForClassType (CXXBaseSpecifier const * const *base_classes,
                                          unsigned num_base_classes)
{
    if (IsValid())
    {
        CXXRecordDecl *cxx_record_decl = GetAsCXXRecordDecl();
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
        ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
        ObjCInterfaceDecl *super_interface_decl = superclass_clang_type.GetAsObjCInterfaceDecl ();
        if (class_interface_decl && super_interface_decl)
        {
            class_interface_decl->setSuperClass(super_interface_decl);
            return true;
        }
    }
    return false;
}

bool
ClangASTType::AddObjCClassProperty (const char *property_name,
                                    const ClangASTType &property_clang_type,
                                    ObjCIvarDecl *ivar_decl,
                                    const char *property_setter_name,
                                    const char *property_getter_name,
                                    uint32_t property_attributes,
                                    ClangASTMetadata *metadata)
{
    if (!IsValid() || !property_clang_type.IsValid() || property_name == NULL || property_name[0] == '\0')
        return false;
        
    ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
    
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
            
            ObjCPropertyDecl *property_decl = ObjCPropertyDecl::Create (*m_ast,
                                                                        class_interface_decl,
                                                                        SourceLocation(), // Source Location
                                                                        &m_ast->Idents.get(property_name),
                                                                        SourceLocation(), //Source Location for AT
                                                                        SourceLocation(), //Source location for (
                                                                        prop_type_source);
            
            if (property_decl)
            {
                if (metadata)
                    ClangASTContext::SetMetadata(m_ast, property_decl, *metadata);
                
                class_interface_decl->addDecl (property_decl);
                
                Selector setter_sel, getter_sel;
                
                if (property_setter_name != NULL)
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
                
                if (property_getter_name != NULL)
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
                    const ObjCMethodDecl::ImplementationControl impControl = ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    ObjCMethodDecl *getter = ObjCMethodDecl::Create (*m_ast,
                                                                     SourceLocation(),
                                                                     SourceLocation(),
                                                                     getter_sel,
                                                                     property_clang_type_to_access.GetQualType(),
                                                                     NULL,
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
                    
                    getter->setMethodParams(*m_ast, ArrayRef<ParmVarDecl*>(), ArrayRef<SourceLocation>());
                    
                    class_interface_decl->addDecl(getter);
                }
                
                if (!setter_sel.isNull() && !class_interface_decl->lookupInstanceMethod(setter_sel))
                {
                    QualType result_type = m_ast->VoidTy;
                    
                    const bool isInstance = true;
                    const bool isVariadic = false;
                    const bool isSynthesized = false;
                    const bool isImplicitlyDeclared = true;
                    const bool isDefined = false;
                    const ObjCMethodDecl::ImplementationControl impControl = ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    ObjCMethodDecl *setter = ObjCMethodDecl::Create (*m_ast,
                                                                     SourceLocation(),
                                                                     SourceLocation(),
                                                                     setter_sel,
                                                                     result_type,
                                                                     NULL,
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
                    
                    llvm::SmallVector<ParmVarDecl *, 1> params;
                    
                    params.push_back (ParmVarDecl::Create (*m_ast,
                                                           setter,
                                                           SourceLocation(),
                                                           SourceLocation(),
                                                           NULL, // anonymous
                                                           property_clang_type_to_access.GetQualType(),
                                                           NULL,
                                                           SC_Auto,
                                                           NULL));
                    
                    setter->setMethodParams(*m_ast, ArrayRef<ParmVarDecl*>(params), ArrayRef<SourceLocation>());
                    
                    class_interface_decl->addDecl(setter);
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
    ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl ();
    if (class_interface_decl)
        return ObjCDeclHasIVars (class_interface_decl, check_superclass);
    return false;
}


ObjCMethodDecl *
ClangASTType::AddMethodToObjCObjectType (const char *name,  // the full symbol name as seen in the symbol table ("-[NString stringWithCString:]")
                                         const ClangASTType &method_clang_type,
                                         lldb::AccessType access,
                                         bool is_artificial)
{
    if (!IsValid() || !method_clang_type.IsValid())
        return NULL;

    ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl();
    
    if (class_interface_decl == NULL)
        return NULL;
    
    const char *selector_start = ::strchr (name, ' ');
    if (selector_start == NULL)
        return NULL;
    
    selector_start++;
    llvm::SmallVector<IdentifierInfo *, 12> selector_idents;
    
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
        selector_idents.push_back (&m_ast->Idents.get (StringRef (start, len)));
        if (has_arg)
            len += 1;
    }
    
    
    if (selector_idents.size() == 0)
        return 0;
    
    clang::Selector method_selector = m_ast->Selectors.getSelector (num_selectors_with_args ? selector_idents.size() : 0,
                                                                    selector_idents.data());
    
    QualType method_qual_type (method_clang_type.GetQualType());
    
    // Populate the method decl with parameter decls
    const clang::Type *method_type(method_qual_type.getTypePtr());
    
    if (method_type == NULL)
        return NULL;
    
    const FunctionProtoType *method_function_prototype (dyn_cast<FunctionProtoType>(method_type));
    
    if (!method_function_prototype)
        return NULL;
    
    
    bool is_variadic = false;
    bool is_synthesized = false;
    bool is_defined = false;
    ObjCMethodDecl::ImplementationControl imp_control = ObjCMethodDecl::None;
    
    const unsigned num_args = method_function_prototype->getNumArgs();
    
    if (num_args != num_selectors_with_args)
        return NULL; // some debug information is corrupt.  We are not going to deal with it.
    
    ObjCMethodDecl *objc_method_decl = ObjCMethodDecl::Create (*m_ast,
                                                               SourceLocation(), // beginLoc,
                                                               SourceLocation(), // endLoc,
                                                               method_selector,
                                                               method_function_prototype->getResultType(),
                                                               NULL, // TypeSourceInfo *ResultTInfo,
                                                               GetDeclContextForType (),
                                                               name[0] == '-',
                                                               is_variadic,
                                                               is_synthesized,
                                                               true, // is_implicitly_declared; we force this to true because we don't have source locations
                                                               is_defined,
                                                               imp_control,
                                                               false /*has_related_result_type*/);
    
    
    if (objc_method_decl == NULL)
        return NULL;
    
    if (num_args > 0)
    {
        llvm::SmallVector<ParmVarDecl *, 12> params;
        
        for (unsigned param_index = 0; param_index < num_args; ++param_index)
        {
            params.push_back (ParmVarDecl::Create (*m_ast,
                                                   objc_method_decl,
                                                   SourceLocation(),
                                                   SourceLocation(),
                                                   NULL, // anonymous
                                                   method_function_prototype->getArgType(param_index),
                                                   NULL,
                                                   SC_Auto,
                                                   NULL));
        }
        
        objc_method_decl->setMethodParams(*m_ast, ArrayRef<ParmVarDecl*>(params), ArrayRef<SourceLocation>());
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
        return NULL;
    
    QualType qual_type(GetCanonicalQualType());
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
        case clang::Type::ObjCInterface:            return cast<ObjCObjectType>(qual_type.getTypePtr())->getInterface();
        case clang::Type::ObjCObjectPointer:        return ClangASTType (m_ast, cast<ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType()).GetDeclContextForType();
        case clang::Type::Record:                   return cast<RecordType>(qual_type)->getDecl();
        case clang::Type::Enum:                     return cast<EnumType>(qual_type)->getDecl();
        case clang::Type::Typedef:                  return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetDeclContextForType();
        case clang::Type::Elaborated:               return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).GetDeclContextForType();
        case clang::Type::Paren:                    return ClangASTType (m_ast, cast<ParenType>(qual_type)->desugar()).GetDeclContextForType();
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
    }
    // No DeclContext in this type...
    return NULL;
}

bool
ClangASTType::SetDefaultAccessForRecordFields (int default_accessibility,
                                               int *assigned_accessibilities,
                                               size_t num_assigned_accessibilities)
{
    if (IsValid())
    {
        RecordDecl *record_decl = GetAsRecordDecl();
        if (record_decl)
        {
            uint32_t field_idx;
            RecordDecl::field_iterator field, field_end;
            for (field = record_decl->field_begin(), field_end = record_decl->field_end(), field_idx = 0;
                 field != field_end;
                 ++field, ++field_idx)
            {
                // If no accessibility was assigned, assign the correct one
                if (field_idx < num_assigned_accessibilities && assigned_accessibilities[field_idx] == clang::AS_none)
                    field->setAccess ((AccessSpecifier)default_accessibility);
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
    
    QualType qual_type (GetCanonicalQualType());
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
        {
            CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
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
            EnumDecl *enum_decl = cast<EnumType>(qual_type)->getDecl();
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
            const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
            assert (objc_class_type);
            if (objc_class_type)
            {
                ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                
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
            return ClangASTType (m_ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType()).SetHasExternalStorage (has_extern);
            
        case clang::Type::Elaborated:
            return ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).SetHasExternalStorage (has_extern);
            
        case clang::Type::Paren:
            return ClangASTType (m_ast, cast<ParenType>(qual_type)->desugar()).SetHasExternalStorage (has_extern);
            
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
        QualType tag_qual_type(GetQualType());
        const clang::Type *clang_type = tag_qual_type.getTypePtr();
        if (clang_type)
        {
            const TagType *tag_type = dyn_cast<TagType>(clang_type);
            if (tag_type)
            {
                TagDecl *tag_decl = dyn_cast<TagDecl>(tag_type->getDecl());
                if (tag_decl)
                {
                    tag_decl->setTagKind ((TagDecl::TagKind)kind);
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
        QualType qual_type (GetQualType());
        const clang::Type *t = qual_type.getTypePtr();
        if (t)
        {
            const TagType *tag_type = dyn_cast<TagType>(t);
            if (tag_type)
            {
                TagDecl *tag_decl = tag_type->getDecl();
                if (tag_decl)
                {
                    tag_decl->startDefinition();
                    return true;
                }
            }
            
            const ObjCObjectType *object_type = dyn_cast<ObjCObjectType>(t);
            if (object_type)
            {
                ObjCInterfaceDecl *interface_decl = object_type->getInterface();
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
        QualType qual_type (GetQualType());
        
        CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
        
        if (cxx_record_decl)
        {
            cxx_record_decl->completeDefinition();
            
            return true;
        }
        
        const EnumType *enum_type = dyn_cast<EnumType>(qual_type.getTypePtr());
        
        if (enum_type)
        {
            EnumDecl *enum_decl = enum_type->getDecl();
            
            if (enum_decl)
            {
                /// TODO This really needs to be fixed.
                
                unsigned NumPositiveBits = 1;
                unsigned NumNegativeBits = 0;
                                
                QualType promotion_qual_type;
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
        QualType enum_qual_type (GetCanonicalQualType());
        
        bool is_signed = false;
        enumerator_clang_type.IsIntegerType (is_signed);
        const clang::Type *clang_type = enum_qual_type.getTypePtr();
        if (clang_type)
        {
            const EnumType *enum_type = dyn_cast<EnumType>(clang_type);
            
            if (enum_type)
            {
                llvm::APSInt enum_llvm_apsint(enum_value_bit_size, is_signed);
                enum_llvm_apsint = enum_value;
                EnumConstantDecl *enumerator_decl =
                EnumConstantDecl::Create (*m_ast,
                                          enum_type->getDecl(),
                                          SourceLocation(),
                                          name ? &m_ast->Idents.get(name) : NULL,    // Identifier
                                          enumerator_clang_type.GetQualType(),
                                          NULL,
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
    QualType enum_qual_type (GetCanonicalQualType());
    const clang::Type *clang_type = enum_qual_type.getTypePtr();
    if (clang_type)
    {
        const EnumType *enum_type = dyn_cast<EnumType>(clang_type);
        if (enum_type)
        {
            EnumDecl *enum_decl = enum_type->getDecl();
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
        QualType qual_type (GetCanonicalQualType());
        uint32_t count = 0;
        bool is_complex = false;
        if (IsFloatingPointType (count, is_complex))
        {
            // TODO: handle complex and vector types
            if (count != 1)
                return false;
            
            StringRef s_sref(s);
            APFloat ap_float(m_ast->getFloatTypeSemantics(qual_type), s_sref);
            
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

    QualType qual_type(GetQualType());
    switch (qual_type->getTypeClass())
    {
    case clang::Type::Record:
        if (GetCompleteType ())
        {
            const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
            const RecordDecl *record_decl = record_type->getDecl();
            assert(record_decl);
            uint32_t field_bit_offset = 0;
            uint32_t field_byte_offset = 0;
            const ASTRecordLayout &record_layout = m_ast->getASTRecordLayout(record_decl);
            uint32_t child_idx = 0;

            const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
            if (cxx_record_decl)
            {
                // We might have base classes to print out first
                CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                     base_class != base_class_end;
                     ++base_class)
                {
                    const CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());

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

                    QualType base_class_qual_type = base_class->getType();
                    std::string base_class_type_name(base_class_qual_type.getAsString());

                    // Indent and print the base class type name
                    s->Printf("\n%*s%s ", depth + DEPTH_INCREMENT, "", base_class_type_name.c_str());

                    std::pair<uint64_t, unsigned> base_class_type_info = m_ast->getTypeInfo(base_class_qual_type);

                    // Dump the value of the member
                    ClangASTType base_clang_type(m_ast, base_class_qual_type);
                    base_clang_type.DumpValue (exe_ctx,
                                               s,                                   // Stream to dump to
                                               base_clang_type.GetFormat(),         // The format with which to display the member
                                               data,                                // Data buffer containing all bytes for this type
                                               data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                                               base_class_type_info.first / 8,      // Size of this type in bytes
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
            RecordDecl::field_iterator field, field_end;
            for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
            {
                // Print the starting squiggly bracket (if this is the
                // first member) or comman (for member 2 and beyong) for
                // the struct/union/class member.
                if (child_idx == 0)
                    s->PutChar('{');
                else
                    s->PutChar(',');

                // Indent
                s->Printf("\n%*s", depth + DEPTH_INCREMENT, "");

                QualType field_type = field->getType();
                // Print the member type if requested
                // Figure out the type byte size (field_type_info.first) and
                // alignment (field_type_info.second) from the AST context.
                std::pair<uint64_t, unsigned> field_type_info = m_ast->getTypeInfo(field_type);
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
                                            field_type_info.first / 8,      // Size of this type in bytes
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
            const EnumType *enum_type = cast<EnumType>(qual_type.getTypePtr());
            const EnumDecl *enum_decl = enum_type->getDecl();
            assert(enum_decl);
            EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
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
            const ConstantArrayType *array = cast<ConstantArrayType>(qual_type.getTypePtr());
            bool is_array_of_characters = false;
            QualType element_qual_type = array->getElementType();

            const clang::Type *canonical_type = element_qual_type->getCanonicalTypeInternal().getTypePtr();
            if (canonical_type)
                is_array_of_characters = canonical_type->isCharType();

            const uint64_t element_count = array->getSize().getLimitedValue();

            std::pair<uint64_t, unsigned> field_type_info = m_ast->getTypeInfo(element_qual_type);

            uint32_t element_idx = 0;
            uint32_t element_offset = 0;
            uint64_t element_byte_size = field_type_info.first / 8;
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
                    // first member) or comman (for member 2 and beyong) for
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
            QualType typedef_qual_type = cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType();
            
            ClangASTType typedef_clang_type (m_ast, typedef_qual_type);
            lldb::Format typedef_format = typedef_clang_type.GetFormat();
            std::pair<uint64_t, unsigned> typedef_type_info = m_ast->getTypeInfo(typedef_qual_type);
            uint64_t typedef_byte_size = typedef_type_info.first / 8;

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
            QualType elaborated_qual_type = cast<ElaboratedType>(qual_type)->getNamedType();
            ClangASTType elaborated_clang_type (m_ast, elaborated_qual_type);
            lldb::Format elaborated_format = elaborated_clang_type.GetFormat();
            std::pair<uint64_t, unsigned> elaborated_type_info = m_ast->getTypeInfo(elaborated_qual_type);
            uint64_t elaborated_byte_size = elaborated_type_info.first / 8;

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
            QualType desugar_qual_type = cast<ParenType>(qual_type)->desugar();
            ClangASTType desugar_clang_type (m_ast, desugar_qual_type);

            lldb::Format desugar_format = desugar_clang_type.GetFormat();
            std::pair<uint64_t, unsigned> desugar_type_info = m_ast->getTypeInfo(desugar_qual_type);
            uint64_t desugar_byte_size = desugar_type_info.first / 8;
            
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
        // We are down the a scalar type that we just need to display.
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
        QualType qual_type(GetQualType());

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::Typedef:
            {
                QualType typedef_qual_type = cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType();
                ClangASTType typedef_clang_type (m_ast, typedef_qual_type);
                if (format == eFormatDefault)
                    format = typedef_clang_type.GetFormat();
                std::pair<uint64_t, unsigned> typedef_type_info = m_ast->getTypeInfo(typedef_qual_type);
                uint64_t typedef_byte_size = typedef_type_info.first / 8;

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
                const EnumType *enum_type = cast<EnumType>(qual_type.getTypePtr());
                const EnumDecl *enum_decl = enum_type->getDecl();
                assert(enum_decl);
                EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
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
            // We are down the a scalar type that we just need to display.
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
                lldb::addr_t pointer_addresss = data.GetMaxU64(&offset, data_byte_size);
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
                while ((bytes_read = process->ReadMemory (pointer_addresss, &buf.front(), buf.size(), error)) > 0)
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
                    pointer_addresss += total_cstr_len;
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
        QualType qual_type(GetQualType());

        SmallVector<char, 1024> buf;
        raw_svector_ostream llvm_ostrm (buf);

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            {
                GetCompleteType ();
                
                const ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    if (class_interface_decl)
                    {
                        PrintingPolicy policy = m_ast->getPrintingPolicy();
                        class_interface_decl->print(llvm_ostrm, policy, s->GetIndentLevel());
                    }
                }
            }
            break;
        
        case clang::Type::Typedef:
            {
                const TypedefType *typedef_type = qual_type->getAs<TypedefType>();
                if (typedef_type)
                {
                    const TypedefNameDecl *typedef_decl = typedef_type->getDecl();
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
            ClangASTType (m_ast, cast<ElaboratedType>(qual_type)->getNamedType()).DumpTypeDescription(s);
            return;

        case clang::Type::Paren:
            ClangASTType (m_ast, cast<ParenType>(qual_type)->desugar()).DumpTypeDescription(s);
            return;

        case clang::Type::Record:
            {
                GetCompleteType ();
                
                const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();
                const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);

                if (cxx_record_decl)
                    cxx_record_decl->print(llvm_ostrm, m_ast->getPrintingPolicy(), s->GetIndentLevel());
                else
                    record_decl->print(llvm_ostrm, m_ast->getPrintingPolicy(), s->GetIndentLevel());
            }
            break;

        default:
            {
                const TagType *tag_type = dyn_cast<TagType>(qual_type.getTypePtr());
                if (tag_type)
                {
                    TagDecl *tag_decl = tag_type->getDecl();
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

        const uint64_t byte_size = GetByteSize();
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

        const uint64_t bit_width = GetBitSize();
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
    
    const uint64_t byte_size = GetByteSize();
    if (data.GetByteSize() < byte_size)
    {
        lldb::DataBufferSP data_sp(new DataBufferHeap (byte_size, '\0'));
        data.SetData(data_sp);
    }
    
    uint8_t* dst = (uint8_t*)data.PeekData(0, byte_size);
    if (dst != NULL)
    {
        if (address_type == eAddressTypeHost)
        {
            if (addr == 0)
                return false;
            // The address is an address in this process, so just copy it
            memcpy (dst, (uint8_t*)NULL + addr, byte_size);
            return true;
        }
        else
        {
            Process *process = NULL;
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

    const uint64_t byte_size = GetByteSize();

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
            Process *process = NULL;
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


//CXXRecordDecl *
//ClangASTType::GetAsCXXRecordDecl (lldb::clang_type_t opaque_clang_qual_type)
//{
//    if (opaque_clang_qual_type)
//        return QualType::getFromOpaquePtr(opaque_clang_qual_type)->getAsCXXRecordDecl();
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


