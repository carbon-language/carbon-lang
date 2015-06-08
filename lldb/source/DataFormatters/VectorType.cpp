//===-- VectorType.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTType.h"

#include "lldb/Utility/LLDBAssert.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

static ClangASTType
GetClangTypeForFormat (lldb::Format format,
                       ClangASTType element_type,
                       ClangASTContext *ast_ctx)
{
    lldbassert(ast_ctx && "ast_ctx needs to be not NULL");
    
    switch (format)
    {
        case lldb::eFormatAddressInfo:
        case lldb::eFormatPointer:
            return ast_ctx->GetPointerSizedIntType(false);
            
        case lldb::eFormatBoolean:
            return ast_ctx->GetBasicType(lldb::eBasicTypeBool);
            
        case lldb::eFormatBytes:
        case lldb::eFormatBytesWithASCII:
        case lldb::eFormatChar:
        case lldb::eFormatCharArray:
        case lldb::eFormatCharPrintable:
            return ast_ctx->GetBasicType(lldb::eBasicTypeChar);

        case lldb::eFormatComplex /* lldb::eFormatComplexFloat */:
            return ast_ctx->GetBasicType(lldb::eBasicTypeFloatComplex);

        case lldb::eFormatCString:
            return ast_ctx->GetBasicType(lldb::eBasicTypeChar).GetPointerType();

        case lldb::eFormatFloat:
            return ast_ctx->GetBasicType(lldb::eBasicTypeFloat);
            
        case lldb::eFormatHex:
        case lldb::eFormatHexUppercase:
        case lldb::eFormatOctal:
            return ast_ctx->GetBasicType(lldb::eBasicTypeInt);

        case lldb::eFormatHexFloat:
            return ast_ctx->GetBasicType(lldb::eBasicTypeFloat);

        case lldb::eFormatUnicode16:
        case lldb::eFormatUnicode32:

        case lldb::eFormatUnsigned:
            return ast_ctx->GetBasicType(lldb::eBasicTypeUnsignedInt);

        case lldb::eFormatVectorOfChar:
            return ast_ctx->GetBasicType(lldb::eBasicTypeChar);
            
        case lldb::eFormatVectorOfFloat32:
            return ast_ctx->GetFloatTypeFromBitSize(32);
            
        case lldb::eFormatVectorOfFloat64:
            return ast_ctx->GetFloatTypeFromBitSize(64);
            
        case lldb::eFormatVectorOfSInt16:
            return ast_ctx->GetIntTypeFromBitSize(16, true);
            
        case lldb::eFormatVectorOfSInt32:
            return ast_ctx->GetIntTypeFromBitSize(32, true);

        case lldb::eFormatVectorOfSInt64:
            return ast_ctx->GetIntTypeFromBitSize(64, true);
            
        case lldb::eFormatVectorOfSInt8:
            return ast_ctx->GetIntTypeFromBitSize(8, true);

        case lldb::eFormatVectorOfUInt128:
            return ast_ctx->GetIntTypeFromBitSize(128, false);

        case lldb::eFormatVectorOfUInt16:
            return ast_ctx->GetIntTypeFromBitSize(16, false);

        case lldb::eFormatVectorOfUInt32:
            return ast_ctx->GetIntTypeFromBitSize(32, false);

        case lldb::eFormatVectorOfUInt64:
            return ast_ctx->GetIntTypeFromBitSize(64, false);

        case lldb::eFormatVectorOfUInt8:
            return ast_ctx->GetIntTypeFromBitSize(8, false);
            
        case lldb::eFormatDefault:
            return element_type;
        
        case lldb::eFormatBinary:
        case lldb::eFormatComplexInteger:
        case lldb::eFormatDecimal:
        case lldb::eFormatEnum:
        case lldb::eFormatInstruction:
        case lldb::eFormatOSType:
        case lldb::eFormatVoid:
        default:
            return ast_ctx->GetIntTypeFromBitSize(8, false);
    }
}

static lldb::Format
GetItemFormatForFormat (lldb::Format format,
                        ClangASTType element_type)
{
    switch (format)
    {
        case lldb::eFormatVectorOfChar:
            return lldb::eFormatChar;
            
        case lldb::eFormatVectorOfFloat32:
        case lldb::eFormatVectorOfFloat64:
            return lldb::eFormatFloat;
            
        case lldb::eFormatVectorOfSInt16:
        case lldb::eFormatVectorOfSInt32:
        case lldb::eFormatVectorOfSInt64:
        case lldb::eFormatVectorOfSInt8:
            return lldb::eFormatDecimal;
            
        case lldb::eFormatVectorOfUInt128:
        case lldb::eFormatVectorOfUInt16:
        case lldb::eFormatVectorOfUInt32:
        case lldb::eFormatVectorOfUInt64:
        case lldb::eFormatVectorOfUInt8:
            return lldb::eFormatUnsigned;
            
        case lldb::eFormatBinary:
        case lldb::eFormatComplexInteger:
        case lldb::eFormatDecimal:
        case lldb::eFormatEnum:
        case lldb::eFormatInstruction:
        case lldb::eFormatOSType:
        case lldb::eFormatVoid:
            return eFormatHex;

        case lldb::eFormatDefault:
        {
            // special case the (default, char) combination to actually display as an integer value
            // most often, you won't want to see the ASCII characters... (and if you do, eFormatChar is a keystroke away)
            bool is_char = element_type.IsCharType();
            bool is_signed = false;
            element_type.IsIntegerType(is_signed);
            return is_char ? (is_signed ? lldb::eFormatDecimal : eFormatHex) : format;
        }
            break;
            
        default:
            return format;
    }
}

static size_t
CalculateNumChildren (ClangASTType container_type,
                      ClangASTType element_type,
                      lldb_private::ExecutionContextScope *exe_scope = nullptr // does not matter here because all we trade in are basic types
                      )
{
    auto container_size = container_type.GetByteSize(exe_scope);
    auto element_size = element_type.GetByteSize(exe_scope);
    
    if (element_size)
    {
        if (container_size % element_size)
            return 0;
        return container_size / element_size;
    }
    return 0;
}

namespace lldb_private {
    namespace formatters {

        class VectorTypeSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            VectorTypeSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
            SyntheticChildrenFrontEnd(*valobj_sp),
            m_parent_format (eFormatInvalid),
            m_item_format(eFormatInvalid),
            m_child_type(),
            m_num_children(0)
            {}
            
            virtual size_t
            CalculateNumChildren ()
            {
                return m_num_children;
            }
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx)
            {
                if (idx >= CalculateNumChildren())
                    return lldb::ValueObjectSP();
                auto offset = idx * m_child_type.GetByteSize(nullptr);
                ValueObjectSP child_sp(m_backend.GetSyntheticChildAtOffset(offset, m_child_type, true));
                if (!child_sp)
                    return child_sp;
                
                StreamString idx_name;
                idx_name.Printf("[%" PRIu64 "]", (uint64_t)idx);
                child_sp->SetName( ConstString( idx_name.GetData() ) );
                
                child_sp->SetFormat(m_item_format);
                
                return child_sp;
            }

            virtual bool
            Update()
            {
                m_parent_format = m_backend.GetFormat();
                ClangASTType parent_type(m_backend.GetClangType());
                ClangASTType element_type;
                parent_type.IsVectorType(&element_type, nullptr);
                m_child_type = ::GetClangTypeForFormat(m_parent_format, element_type, ClangASTContext::GetASTContext(parent_type.GetASTContext()));
                m_num_children = ::CalculateNumChildren(parent_type,
                                                        m_child_type);
                m_item_format = GetItemFormatForFormat(m_parent_format,
                                                       m_child_type);
                return false;
            }
            
            virtual bool
            MightHaveChildren ()
            {
                return true;
            }
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name)
            {
                const char* item_name = name.GetCString();
                uint32_t idx = ExtractIndexFromString(item_name);
                if (idx < UINT32_MAX && idx >= CalculateNumChildren())
                    return UINT32_MAX;
                return idx;
            }
            
            virtual
            ~VectorTypeSyntheticFrontEnd () {}
            
        private:
            lldb::Format m_parent_format;
            lldb::Format m_item_format;
            ClangASTType m_child_type;
            size_t m_num_children;
        };
    }
}

lldb_private::SyntheticChildrenFrontEnd*
lldb_private::formatters::VectorTypeSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new VectorTypeSyntheticFrontEnd(valobj_sp));
}
