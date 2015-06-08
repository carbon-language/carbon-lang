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

ClangASTType::ClangASTType (TypeSystem *type_system,
                            void* type) :
m_type (type),
m_type_system (type_system)
{
}

ClangASTType::ClangASTType (clang::ASTContext *ast,
                            clang::QualType qual_type) :
    m_type (qual_type.getAsOpaquePtr()),
m_type_system (ClangASTContext::GetASTContext(ast))
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
    if (IsValid())
        return m_type_system->IsAggregateType(m_type);
    return false;
}

bool
ClangASTType::IsArrayType (ClangASTType *element_type_ptr,
                           uint64_t *size,
                           bool *is_incomplete) const
{
    if (IsValid())
        return m_type_system->IsArrayType(m_type, element_type_ptr, size, is_incomplete);

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
        return m_type_system->IsVectorType(m_type, element_type, size);
    return false;
}

bool
ClangASTType::IsRuntimeGeneratedType () const
{
    if (IsValid())
        return m_type_system->IsRuntimeGeneratedType(m_type);
    return false;
}

bool
ClangASTType::IsCharType () const
{
    if (IsValid())
        return m_type_system->IsCharType(m_type);
    return false;
}


bool
ClangASTType::IsCompleteType () const
{
    if (IsValid())
        return m_type_system->IsCompleteType(m_type);
    return false;
}

bool
ClangASTType::IsConst() const
{
    if (IsValid())
        return m_type_system->IsConst(m_type);
    return false;
}

bool
ClangASTType::IsCStringType (uint32_t &length) const
{
    if (IsValid())
        return m_type_system->IsCStringType(m_type, length);
    return false;
}

bool
ClangASTType::IsFunctionType (bool *is_variadic_ptr) const
{
    if (IsValid())
        return m_type_system->IsFunctionType(m_type, is_variadic_ptr);
    return false;
}

// Used to detect "Homogeneous Floating-point Aggregates"
uint32_t
ClangASTType::IsHomogeneousAggregate (ClangASTType* base_type_ptr) const
{
    if (IsValid())
        return m_type_system->IsHomogeneousAggregate(m_type, base_type_ptr);
    return 0;
}

size_t
ClangASTType::GetNumberOfFunctionArguments () const
{
    if (IsValid())
        return m_type_system->GetNumberOfFunctionArguments(m_type);
    return 0;
}

ClangASTType
ClangASTType::GetFunctionArgumentAtIndex (const size_t index) const
{
    if (IsValid())
        return m_type_system->GetFunctionArgumentAtIndex(m_type, index);
    return ClangASTType();
}

bool
ClangASTType::IsFunctionPointerType () const
{
    if (IsValid())
        return m_type_system->IsFunctionPointerType(m_type);
    return false;

}

bool
ClangASTType::IsIntegerType (bool &is_signed) const
{
    if (IsValid())
        return m_type_system->IsIntegerType(m_type, is_signed);
    return false;
}

bool
ClangASTType::IsPointerType (ClangASTType *pointee_type) const
{
    if (IsValid())
    {
        return m_type_system->IsPointerType(m_type, pointee_type);
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
        return m_type_system->IsPointerOrReferenceType(m_type, pointee_type);
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
        return m_type_system->IsReferenceType(m_type, pointee_type, is_rvalue);
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
        return m_type_system->IsFloatingPointType(m_type, count, is_complex);
    }
    count = 0;
    is_complex = false;
    return false;
}


bool
ClangASTType::IsDefined() const
{
    if (IsValid())
        return m_type_system->IsDefined(m_type);
    return true;
}

bool
ClangASTType::IsPolymorphicClass () const
{
    if (IsValid())
    {
        return m_type_system->IsPolymorphicClass(m_type);
    }
    return false;
}

bool
ClangASTType::IsPossibleDynamicType (ClangASTType *dynamic_pointee_type,
                                     bool check_cplusplus,
                                     bool check_objc) const
{
    if (IsValid())
        return m_type_system->IsPossibleDynamicType(m_type, dynamic_pointee_type, check_cplusplus, check_objc);
    return false;
}


bool
ClangASTType::IsScalarType () const
{
    if (!IsValid())
        return false;

    return m_type_system->IsScalarType(m_type);
}

bool
ClangASTType::IsTypedefType () const
{
    if (!IsValid())
        return false;
    return m_type_system->IsTypedefType(m_type);
}

bool
ClangASTType::IsVoidType () const
{
    if (!IsValid())
        return false;
    return m_type_system->IsVoidType(m_type);
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
ClangASTType::IsBeingDefined () const
{
    if (!IsValid())
        return false;
    return m_type_system->IsBeingDefined(m_type);
}

//----------------------------------------------------------------------
// Type Completion
//----------------------------------------------------------------------

bool
ClangASTType::GetCompleteType () const
{
    if (!IsValid())
        return false;
    return m_type_system->GetCompleteType(m_type);
}

//----------------------------------------------------------------------
// AST related queries
//----------------------------------------------------------------------
size_t
ClangASTType::GetPointerByteSize () const
{
    if (m_type_system)
        return m_type_system->GetPointerByteSize();
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
        m_type_system->GetTypeName(m_type);
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
    
    return m_type_system->GetTypeInfo(m_type);
}



lldb::LanguageType
ClangASTType::GetMinimumLanguage ()
{
    if (!IsValid())
        return lldb::eLanguageTypeC;
    
    return m_type_system->GetMinimumLanguage(m_type);
}

lldb::TypeClass
ClangASTType::GetTypeClass () const
{
    if (!IsValid())
        return lldb::eTypeClassInvalid;
    
    return m_type_system->GetTypeClass(m_type);
    
}

void
ClangASTType::SetClangType (TypeSystem* type_system, void*  type)
{
    m_type_system = type_system;
    m_type = type;
}

void
ClangASTType::SetClangType (clang::ASTContext *ast, clang::QualType qual_type)
{
    m_type_system = ClangASTContext::GetASTContext(ast);
    m_type = qual_type.getAsOpaquePtr();
}

unsigned
ClangASTType::GetTypeQualifiers() const
{
    if (IsValid())
        return m_type_system->GetTypeQualifiers(m_type);
    return 0;
}

//----------------------------------------------------------------------
// Creating related types
//----------------------------------------------------------------------

ClangASTType
ClangASTType::GetArrayElementType (uint64_t *stride) const
{
    if (IsValid())
    {
        return m_type_system->GetArrayElementType(m_type, stride);
        
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetCanonicalType () const
{
    if (IsValid())
        return m_type_system->GetCanonicalType(m_type);
    return ClangASTType();
}

ClangASTType
ClangASTType::GetFullyUnqualifiedType () const
{
    if (IsValid())
        return m_type_system->GetFullyUnqualifiedType(m_type);
    return ClangASTType();
}


int
ClangASTType::GetFunctionArgumentCount () const
{
    if (IsValid())
    {
        return m_type_system->GetFunctionArgumentCount(m_type);
    }
    return -1;
}

ClangASTType
ClangASTType::GetFunctionArgumentTypeAtIndex (size_t idx) const
{
    if (IsValid())
    {
        return m_type_system->GetFunctionArgumentTypeAtIndex(m_type, idx);
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetFunctionReturnType () const
{
    if (IsValid())
    {
        return m_type_system->GetFunctionReturnType(m_type);
    }
    return ClangASTType();
}

size_t
ClangASTType::GetNumMemberFunctions () const
{
    if (IsValid())
    {
        return m_type_system->GetNumMemberFunctions(m_type);
    }
    return 0;
}

TypeMemberFunctionImpl
ClangASTType::GetMemberFunctionAtIndex (size_t idx)
{
    if (IsValid())
    {
        return m_type_system->GetMemberFunctionAtIndex(m_type, idx);
    }
    return TypeMemberFunctionImpl();
}

ClangASTType
ClangASTType::GetNonReferenceType () const
{
    if (IsValid())
        return m_type_system->GetNonReferenceType(m_type);
    return ClangASTType();
}

ClangASTType
ClangASTType::GetPointeeType () const
{
    if (IsValid())
    {
        return m_type_system->GetPointeeType(m_type);
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetPointerType () const
{
    if (IsValid())
    {
        return m_type_system->GetPointerType(m_type);
    }
    return ClangASTType();
}

ClangASTType
ClangASTType::GetTypedefedType () const
{
    if (IsValid())
    {
        return m_type_system->GetTypedefedType(m_type);
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
        return m_type_system->GetBasicTypeFromAST(m_type, basic_type);
    return ClangASTType();
}
//----------------------------------------------------------------------
// Exploring the type
//----------------------------------------------------------------------

uint64_t
ClangASTType::GetBitSize (ExecutionContextScope *exe_scope) const
{
    if (IsValid())
    {
        return m_type_system->GetBitSize(m_type, exe_scope);
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
    if (IsValid())
        return m_type_system->GetTypeBitAlign(m_type);
    return 0;
}


lldb::Encoding
ClangASTType::GetEncoding (uint64_t &count) const
{
    if (!IsValid())
        return lldb::eEncodingInvalid;
    
    return m_type_system->GetEncoding(m_type, count);
}

lldb::Format
ClangASTType::GetFormat () const
{
    if (!IsValid())
        return lldb::eFormatDefault;
    
    return m_type_system->GetFormat(m_type);
}

uint32_t
ClangASTType::GetNumChildren (bool omit_empty_base_classes) const
{
    if (!IsValid())
        return 0;
    return m_type_system->GetNumChildren(m_type, omit_empty_base_classes);
}

lldb::BasicType
ClangASTType::GetBasicTypeEnumeration () const
{
    if (IsValid())
    {
        return m_type_system->GetBasicTypeEnumeration(m_type);
    }
    return eBasicTypeInvalid;
}

#pragma mark Aggregate Types

uint32_t
ClangASTType::GetNumFields () const
{
    if (!IsValid())
        return 0;
    return m_type_system->GetNumFields(m_type);
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
    return m_type_system->GetFieldAtIndex(m_type, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr);
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
    return m_type_system->GetChildClangTypeAtIndex(m_type, exe_ctx, idx, transparent_pointers, omit_empty_base_classes, ignore_array_bounds, child_name, child_byte_size, child_byte_offset, child_bitfield_bit_size, child_bitfield_bit_offset, child_is_base_class, child_is_deref_of_parent, valobj);
}



size_t
ClangASTType::GetIndexOfChildMemberWithName (const char *name,
                                             bool omit_empty_base_classes,
                                             std::vector<uint32_t>& child_indexes) const
{
    if (IsValid() && name && name[0])
    {
        return m_type_system->GetIndexOfChildMemberWithName(m_type, name, omit_empty_base_classes, child_indexes);
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
        return m_type_system->GetIndexOfChildWithName(m_type, name, omit_empty_base_classes);
    return UINT32_MAX;
}

#pragma mark TagDecl


size_t
ClangASTType::ConvertStringToFloatValue (const char *s, uint8_t *dst, size_t dst_size) const
{
    if (IsValid())
        return m_type_system->ConvertStringToFloatValue(m_type, s, dst, dst_size);
    return 0;
}



//----------------------------------------------------------------------
// Dumping types
//----------------------------------------------------------------------

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
    m_type_system->DumpValue(m_type, exe_ctx, s, format, data, data_byte_offset, data_byte_size, bitfield_bit_size, bitfield_bit_offset, show_types, show_summary, verbose, depth);
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
    return m_type_system->DumpTypeValue(m_type, s, format, data, byte_offset, byte_size, bitfield_bit_size, bitfield_bit_offset, exe_scope);
}



void
ClangASTType::DumpSummary (ExecutionContext *exe_ctx,
                           Stream *s,
                           const lldb_private::DataExtractor &data,
                           lldb::offset_t data_byte_offset,
                           size_t data_byte_size)
{
    if (IsValid())
        m_type_system->DumpSummary(m_type, exe_ctx, s, data, data_byte_offset, data_byte_size);
}

void
ClangASTType::DumpTypeDescription () const
{
    if (IsValid())
        m_type_system->DumpTypeDescription(m_type);
}

void
ClangASTType::DumpTypeDescription (Stream *s) const
{
    if (IsValid())
    {
        m_type_system->DumpTypeDescription(m_type, s);
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
    if (!IsValid())
        return false;

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
    return lhs.GetTypeSystem() == rhs.GetTypeSystem() && lhs.GetOpaqueQualType() == rhs.GetOpaqueQualType();
}


bool
lldb_private::operator != (const lldb_private::ClangASTType &lhs, const lldb_private::ClangASTType &rhs)
{
    return lhs.GetTypeSystem() != rhs.GetTypeSystem() || lhs.GetOpaqueQualType() != rhs.GetOpaqueQualType();
}



