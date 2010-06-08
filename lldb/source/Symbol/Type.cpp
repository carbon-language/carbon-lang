//===-- Type.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Other libraries and framework includes
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecordLayout.h"

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"

#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContextScope.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"

lldb_private::Type::Type
(
    lldb::user_id_t uid,
    SymbolFile* symbol_file,
    const ConstString &name,
    uint64_t byte_size,
    SymbolContextScope *context,
    lldb::user_id_t encoding_uid,
    EncodingUIDType encoding_uid_type,
    const Declaration& decl,
    void *clang_type
) :
    UserID (uid),
    m_name (name),
    m_byte_size (byte_size),
    m_symbol_file (symbol_file),
    m_context (context),
    m_encoding_uid (encoding_uid),
    m_encoding_uid_type (encoding_uid_type),
    m_decl (decl),
    m_clang_qual_type (clang_type)
{
}

lldb_private::Type::Type () :
    UserID (0),
    m_name ("<INVALID TYPE>"),
    m_byte_size (0),
    m_symbol_file (NULL),
    m_context (),
    m_encoding_uid (0),
    m_encoding_uid_type (eTypeInvalid),
    m_decl (),
    m_clang_qual_type (NULL)
{
}


const lldb_private::Type&
lldb_private::Type::operator= (const Type& rhs)
{
    if (this != &rhs)
    {
        UserID::operator= (rhs);
        m_name = rhs.m_name;
        m_byte_size = rhs.m_byte_size;
        m_symbol_file = rhs.m_symbol_file;
        m_context = rhs.m_context;
        m_encoding_uid = rhs.m_encoding_uid;
        m_decl = rhs.m_decl;
        m_clang_qual_type = rhs.m_clang_qual_type;
    }
    return *this;
}


void
lldb_private::Type::Dump (Stream *s, bool show_context)
{
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    *s << "Type" << (const UserID&)*this << ' ';
    if (m_name)
        *s << ", name = \"" << m_name << "\"";

    if (m_byte_size != 0)
        s->Printf(", size = %zu", m_byte_size);

    if (show_context && m_context != NULL)
    {
        s->PutCString(", context = ( ");
        m_context->DumpSymbolContext(s);
        s->PutCString(" )");
    }

    m_decl.Dump(s);

    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(m_clang_qual_type));

    if (qual_type.getTypePtr())
    {
        *s << ", clang_type = ";

        clang::TagType *tag_type = dyn_cast<clang::TagType>(qual_type.getTypePtr());
        clang::TagDecl *tag_decl = NULL;
        if (tag_type)
            tag_decl = tag_type->getDecl();

        if (tag_decl)
        {
            s->EOL();
            s->EOL();
            tag_decl->print(llvm::fouts(), 0);
            s->EOL();
        }
        else
        {
            const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
            if (typedef_type)
            {
                const clang::TypedefDecl *typedef_decl = typedef_type->getDecl();
                std::string clang_typedef_name (typedef_decl->getQualifiedNameAsString());
                if (!clang_typedef_name.empty())
                    *s << " (" << clang_typedef_name.c_str() << ')';
            }
            else
            {
                // We have a clang type, lets show it
                TypeList *type_list = GetTypeList();
                if (type_list)
                {
                    clang::ASTContext *ast_context = GetClangAST();
                    if (ast_context)
                    {
                        std::string clang_type_name(qual_type.getAsString());
                        if (!clang_type_name.empty())
                            *s << " (" << clang_type_name.c_str() << ')';
                    }
                }
            }
        }
    }
    else if (m_encoding_uid != LLDB_INVALID_UID)
    {
        *s << ", type_uid = " << m_encoding_uid;
        switch (m_encoding_uid_type)
        {
        case eIsTypeWithUID: s->PutCString(" (unresolved type)"); break;
        case eIsConstTypeWithUID: s->PutCString(" (unresolved const type)"); break;
        case eIsRestrictTypeWithUID: s->PutCString(" (unresolved restrict type)"); break;
        case eIsVolatileTypeWithUID: s->PutCString(" (unresolved volatile type)"); break;
        case eTypedefToTypeWithUID: s->PutCString(" (unresolved typedef)"); break;
        case ePointerToTypeWithUID: s->PutCString(" (unresolved pointer)"); break;
        case eLValueReferenceToTypeWithUID: s->PutCString(" (unresolved L value reference)"); break;
        case eRValueReferenceToTypeWithUID: s->PutCString(" (unresolved R value reference)"); break;
        }
    }

//
//  if (m_access)
//      s->Printf(", access = %u", m_access);
    s->EOL();
}

const lldb_private::ConstString &
lldb_private::Type::GetName()
{
    if (!(m_name))
    {
        if (ResolveClangType())
        {
            std::string type_name = ClangASTContext::GetTypeName (m_clang_qual_type);
            if (!type_name.empty())
                m_name.SetCString (type_name.c_str());
        }
    }
    return m_name;
}

int
lldb_private::Type::DumpClangTypeName(Stream *s, void *clang_type)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    std::string type_name;
    const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
    if (typedef_type)
    {
        const clang::TypedefDecl *typedef_decl = typedef_type->getDecl();
        type_name = typedef_decl->getQualifiedNameAsString();
    }
    else
    {
        type_name = qual_type.getAsString();
    }
    if (!type_name.empty())
        return s->Printf("(%s) ", type_name.c_str());
    return 0;
}

lldb_private::ConstString
lldb_private::Type::GetClangTypeName (void *clang_type)
{
    ConstString clang_type_name;
    if (clang_type)
    {
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

        const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
        if (typedef_type)
        {
            const clang::TypedefDecl *typedef_decl = typedef_type->getDecl();
            std::string clang_typedef_name (typedef_decl->getQualifiedNameAsString());
            if (!clang_typedef_name.empty())
                clang_type_name.SetCString (clang_typedef_name.c_str());
        }
        else
        {
            std::string type_name(qual_type.getAsString());
            if (!type_name.empty())
                clang_type_name.SetCString (type_name.c_str());
        }
    }
    else
    {
        clang_type_name.SetCString ("<invalid>");
    }

    return clang_type_name;
}



void
lldb_private::Type::DumpTypeName(Stream *s)
{
    GetName().Dump(s, "<invalid-type-name>");
}


void
lldb_private::Type::DumpValue
(
    lldb_private::ExecutionContext *exe_ctx,
    lldb_private::Stream *s,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    bool show_types,
    bool show_summary,
    bool verbose,
    lldb::Format format
)
{
    if (ResolveClangType())
    {
        if (show_types)
        {
            s->PutChar('(');
            if (verbose)
                s->Printf("Type{0x%8.8x} ", GetID());
            DumpTypeName (s);
            s->PutCString(") ");
        }

        lldb_private::Type::DumpValue (exe_ctx,
                                       GetClangAST (),
                                       m_clang_qual_type,
                                       s,
                                       format == lldb::eFormatDefault ? GetFormat() : format,
                                       data,
                                       data_byte_offset,
                                       GetByteSize(),
                                       0, // Bitfield bit size
                                       0, // Bitfield bit offset
                                       show_types,
                                       show_summary,
                                       verbose,
                                       0);
    }
}


void
lldb_private::Type::DumpSummary
(
    ExecutionContext *exe_ctx,
    clang::ASTContext *ast_context,
    void *clang_type,
    Stream *s,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size
)
{
    uint32_t length = 0;
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    if (ClangASTContext::IsCStringType (clang_type, length))
    {

        if (exe_ctx && exe_ctx->process)
        {
            uint32_t offset = data_byte_offset;
            lldb::addr_t pointer_addresss = data.GetMaxU64(&offset, data_byte_size);
            const size_t k_max_buf_size = length ? length : 256;
            uint8_t buf[k_max_buf_size + 1];
            lldb_private::DataExtractor data(buf, k_max_buf_size, exe_ctx->process->GetByteOrder(), 4);
            buf[k_max_buf_size] = '\0';
            size_t bytes_read;
            size_t total_cstr_len = 0;
            Error error;
            while ((bytes_read = exe_ctx->process->ReadMemory (pointer_addresss, buf, k_max_buf_size, error)) > 0)
            {
                const size_t len = strlen((const char *)buf);
                if (len == 0)
                    break;
                if (total_cstr_len == 0)
                    s->PutCString (" \"");
                data.Dump(s, 0, lldb::eFormatChar, 1, len, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                total_cstr_len += len;
                if (len < k_max_buf_size)
                    break;
                pointer_addresss += total_cstr_len;
            }
            if (total_cstr_len > 0)
                s->PutChar ('"');
        }
    }
}

#define DEPTH_INCREMENT 2
void
lldb_private::Type::DumpValue
(
    ExecutionContext *exe_ctx,
    clang::ASTContext *ast_context,
    void *clang_type,
    Stream *s,
    lldb::Format format,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size,
    uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset,
    bool show_types,
    bool show_summary,
    bool verbose,
    uint32_t depth
)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    switch (qual_type->getTypeClass())
    {
    case clang::Type::Record:
        {
            const clang::RecordType *record_type = cast<clang::RecordType>(qual_type.getTypePtr());
            const clang::RecordDecl *record_decl = record_type->getDecl();
            assert(record_decl);
            uint32_t field_bit_offset = 0;
            uint32_t field_byte_offset = 0;
            const clang::ASTRecordLayout &record_layout = ast_context->getASTRecordLayout(record_decl);
            uint32_t child_idx = 0;


            const clang::CXXRecordDecl *cxx_record_decl = dyn_cast<clang::CXXRecordDecl>(record_decl);
            if (cxx_record_decl)
            {
                // We might have base classes to print out first
                clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                     base_class != base_class_end;
                     ++base_class)
                {
                    const clang::CXXRecordDecl *base_class_decl = cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());

                    // Skip empty base classes
                    if (verbose == false && ClangASTContext::RecordHasFields(base_class_decl) == false)
                        continue;

                    if (base_class->isVirtual())
                        field_bit_offset = record_layout.getVBaseClassOffset(base_class_decl);
                    else
                        field_bit_offset = record_layout.getBaseClassOffset(base_class_decl);
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

                    std::pair<uint64_t, unsigned> base_class_type_info = ast_context->getTypeInfo(base_class_qual_type);

                    // Dump the value of the member
                    Type::DumpValue (
                        exe_ctx,
                        ast_context,                        // The clang AST context for this type
                        base_class_qual_type.getAsOpaquePtr(),// The clang type we want to dump
                        s,                                  // Stream to dump to
                        Type::GetFormat(base_class_qual_type.getAsOpaquePtr()), // The format with which to display the member
                        data,                               // Data buffer containing all bytes for this type
                        data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                        base_class_type_info.first / 8,     // Size of this type in bytes
                        0,                                  // Bitfield bit size
                        0,                                  // Bitfield bit offset
                        show_types,                         // Boolean indicating if we should show the variable types
                        show_summary,                       // Boolean indicating if we should show a summary for the current type
                        verbose,                            // Verbose output?
                        depth + DEPTH_INCREMENT);           // Scope depth for any types that have children

                    ++child_idx;
                }
            }
            const unsigned num_fields = record_layout.getFieldCount();

            uint32_t field_idx = 0;
            clang::RecordDecl::field_iterator field, field_end;
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

                clang::QualType field_type = field->getType();
                // Print the member type if requested
                // Figure out the type byte size (field_type_info.first) and
                // alignment (field_type_info.second) from the AST context.
                std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(field_type);
                assert(field_idx < num_fields);
                // Figure out the field offset within the current struct/union/class type
                field_bit_offset = record_layout.getFieldOffset (field_idx);
                field_byte_offset = field_bit_offset / 8;
                uint32_t field_bitfield_bit_size = 0;
                uint32_t field_bitfield_bit_offset = 0;
                if (ClangASTContext::FieldIsBitfield (ast_context, *field, field_bitfield_bit_size))
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
                Type::DumpValue (
                    exe_ctx,
                    ast_context,                    // The clang AST context for this type
                    field_type.getAsOpaquePtr(),    // The clang type we want to dump
                    s,                              // Stream to dump to
                    Type::GetFormat(field_type.getAsOpaquePtr()),   // The format with which to display the member
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
        {
            const clang::EnumType *enum_type = cast<clang::EnumType>(qual_type.getTypePtr());
            const clang::EnumDecl *enum_decl = enum_type->getDecl();
            assert(enum_decl);
            clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
            uint32_t offset = data_byte_offset;
            const int64_t enum_value = data.GetMaxU64Bitfield(&offset, data_byte_size, bitfield_bit_size, bitfield_bit_offset);
            for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
            {
                if (enum_pos->getInitVal() == enum_value)
                {
                    s->Printf("%s", enum_pos->getNameAsCString());
                    return;
                }
            }
            // If we have gotten here we didn't get find the enumerator in the
            // enum decl, so just print the integer.
            s->Printf("%lli", enum_value);
        }
        return;

    case clang::Type::ConstantArray:
        {
            const clang::ConstantArrayType *array = cast<clang::ConstantArrayType>(qual_type.getTypePtr());
            bool is_array_of_characters = false;
            clang::QualType element_qual_type = array->getElementType();

            clang::Type *canonical_type = element_qual_type->getCanonicalTypeInternal().getTypePtr();
            if (canonical_type)
                is_array_of_characters = canonical_type->isCharType();

            const uint64_t element_count = array->getSize().getLimitedValue();

            std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(element_qual_type);

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
                lldb::Format element_format = Type::GetFormat(element_qual_type.getAsOpaquePtr());

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
                    Type::DumpValue (
                        exe_ctx,
                        ast_context,                    // The clang AST context for this type
                        element_qual_type.getAsOpaquePtr(), // The clang type we want to dump
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
            clang::QualType typedef_qual_type = cast<clang::TypedefType>(qual_type)->LookThroughTypedefs();
            lldb::Format typedef_format = lldb_private::Type::GetFormat(typedef_qual_type.getAsOpaquePtr());
            std::pair<uint64_t, unsigned> typedef_type_info = ast_context->getTypeInfo(typedef_qual_type);
            uint64_t typedef_byte_size = typedef_type_info.first / 8;

            return Type::DumpValue(
                        exe_ctx,
                        ast_context,        // The clang AST context for this type
                        typedef_qual_type.getAsOpaquePtr(), // The clang type we want to dump
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

    default:
        // We are down the a scalar type that we just need to display.
        data.Dump(s, data_byte_offset, format, data_byte_size, 1, UINT32_MAX, LLDB_INVALID_ADDRESS, bitfield_bit_size, bitfield_bit_offset);

        if (show_summary)
            Type::DumpSummary (exe_ctx, ast_context, clang_type, s, data, data_byte_offset, data_byte_size);
        break;
    }
}

bool
lldb_private::Type::DumpTypeValue
(
    Stream *s,
    clang::ASTContext *ast_context,
    void *clang_type,
    lldb::Format format,
    const lldb_private::DataExtractor &data,
    uint32_t byte_offset,
    size_t byte_size,
    uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset
)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    if (ClangASTContext::IsAggregateType (clang_type))
    {
        return 0;
    }
    else
    {
        switch (qual_type->getTypeClass())
        {
        case clang::Type::Enum:
            {
                const clang::EnumType *enum_type = cast<clang::EnumType>(qual_type.getTypePtr());
                const clang::EnumDecl *enum_decl = enum_type->getDecl();
                assert(enum_decl);
                clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
                uint32_t offset = byte_offset;
                const int64_t enum_value = data.GetMaxU64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                {
                    if (enum_pos->getInitVal() == enum_value)
                    {
                        s->PutCString (enum_pos->getNameAsCString());
                        return true;
                    }
                }
                // If we have gotten here we didn't get find the enumerator in the
                // enum decl, so just print the integer.

                s->Printf("%lli", enum_value);
                return true;
            }
            break;

        case clang::Type::Typedef:
            {
                clang::QualType typedef_qual_type = cast<clang::TypedefType>(qual_type)->LookThroughTypedefs();
                lldb::Format typedef_format = Type::GetFormat(typedef_qual_type.getAsOpaquePtr());
                std::pair<uint64_t, unsigned> typedef_type_info = ast_context->getTypeInfo(typedef_qual_type);
                uint64_t typedef_byte_size = typedef_type_info.first / 8;

                return Type::DumpTypeValue(
                            s,
                            ast_context,            // The clang AST context for this type
                            typedef_qual_type.getAsOpaquePtr(),     // The clang type we want to dump
                            typedef_format,         // The format with which to display the element
                            data,                   // Data buffer containing all bytes for this type
                            byte_offset,            // Offset into "data" where to grab value from
                            typedef_byte_size,      // Size of this type in bytes
                            bitfield_bit_size,      // Size in bits of a bitfield value, if zero don't treat as a bitfield
                            bitfield_bit_offset);   // Offset in bits of a bitfield value if bitfield_bit_size != 0
            }
            break;

        default:
            // We are down the a scalar type that we just need to display.
            return data.Dump(s,
                             byte_offset,
                             format,
                             byte_size,
                             1,
                             UINT32_MAX,
                             LLDB_INVALID_ADDRESS,
                             bitfield_bit_size,
                             bitfield_bit_offset);
            break;
        }
    }
    return 0;
}

bool
lldb_private::Type::GetValueAsScalar
(
    clang::ASTContext *ast_context,
    void *clang_type,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size,
    Scalar &value
)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    if (ClangASTContext::IsAggregateType (clang_type))
    {
        return false;   // Aggregate types don't have scalar values
    }
    else
    {
        uint32_t count = 0;
        lldb::Encoding encoding = Type::GetEncoding (clang_type, count);

        if (encoding == lldb::eEncodingInvalid || count != 1)
            return false;

        uint64_t bit_width = ast_context->getTypeSize(qual_type);
        uint32_t byte_size = (bit_width + 7 ) / 8;
        uint32_t offset = data_byte_offset;
        switch (encoding)
        {
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
                int64_t sval64 = (int64_t)data.GetMaxU64 (&offset, byte_size);
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
lldb_private::Type::SetValueFromScalar
(
    clang::ASTContext *ast_context,
    void *clang_type,
    const Scalar &value,
    Stream &strm
)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    // Aggregate types don't have scalar values
    if (!ClangASTContext::IsAggregateType (clang_type))
    {
        strm.GetFlags().Set(Stream::eBinary);
        uint32_t count = 0;
        lldb::Encoding encoding = Type::GetEncoding (clang_type, count);

        if (encoding == lldb::eEncodingInvalid || count != 1)
            return false;

        uint64_t bit_width = ast_context->getTypeSize(qual_type);
        // This function doesn't currently handle non-byte aligned assignments
        if ((bit_width % 8) != 0)
            return false;

        uint32_t byte_size = (bit_width + 7 ) / 8;
        switch (encoding)
        {
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


uint64_t
lldb_private::Type::GetByteSize()
{
    if (m_byte_size == 0)
    {
        switch (m_encoding_uid_type)
        {
        case eIsTypeWithUID:
        case eIsConstTypeWithUID:
        case eIsRestrictTypeWithUID:
        case eIsVolatileTypeWithUID:
        case eTypedefToTypeWithUID:
            if (m_encoding_uid != LLDB_INVALID_UID)
            {
                Type *encoding_type = m_symbol_file->ResolveTypeUID (m_encoding_uid);
                if (encoding_type)
                    m_byte_size = encoding_type->GetByteSize();
            }
            if (m_byte_size == 0)
            {
                uint64_t bit_width = GetClangAST()->getTypeSize(clang::QualType::getFromOpaquePtr(GetOpaqueClangQualType()));
                m_byte_size = (bit_width + 7 ) / 8;
            }
            break;

        // If we are a pointer or reference, then this is just a pointer size;
        case ePointerToTypeWithUID:
        case eLValueReferenceToTypeWithUID:
        case eRValueReferenceToTypeWithUID:
            m_byte_size = GetTypeList()->GetClangASTContext().GetPointerBitSize() / 8;
            break;
        }
    }
    return m_byte_size;
}


uint32_t
lldb_private::Type::GetNumChildren (bool omit_empty_base_classes)
{
    if (!ResolveClangType())
        return 0;
    return ClangASTContext::GetNumChildren (m_clang_qual_type, omit_empty_base_classes);

}

bool
lldb_private::Type::IsAggregateType ()
{
    if (ResolveClangType())
        return ClangASTContext::IsAggregateType (m_clang_qual_type);
    return false;
}

lldb::Format
lldb_private::Type::GetFormat ()
{
    // Make sure we resolve our type if it already hasn't been.
    if (!ResolveClangType())
        return lldb::eFormatInvalid;
    return lldb_private::Type::GetFormat (m_clang_qual_type);
}


lldb::Format
lldb_private::Type::GetFormat (void *clang_type)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    switch (qual_type->getTypeClass())
    {
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
        break;

    case clang::Type::Builtin:
        switch (cast<clang::BuiltinType>(qual_type)->getKind())
        {
        default: assert(0 && "Unknown builtin type!");
        case clang::BuiltinType::Void:
            break;

        case clang::BuiltinType::Bool:          return lldb::eFormatBoolean;
        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::SChar:
        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::UChar:
        case clang::BuiltinType::WChar:         return lldb::eFormatChar;
        case clang::BuiltinType::Char16:        return lldb::eFormatUnicode16;
        case clang::BuiltinType::Char32:        return lldb::eFormatUnicode32;
        case clang::BuiltinType::UShort:        return lldb::eFormatHex;
        case clang::BuiltinType::Short:         return lldb::eFormatDecimal;
        case clang::BuiltinType::UInt:          return lldb::eFormatHex;
        case clang::BuiltinType::Int:           return lldb::eFormatDecimal;
        case clang::BuiltinType::ULong:         return lldb::eFormatHex;
        case clang::BuiltinType::Long:          return lldb::eFormatDecimal;
        case clang::BuiltinType::ULongLong:     return lldb::eFormatHex;
        case clang::BuiltinType::LongLong:      return lldb::eFormatDecimal;
        case clang::BuiltinType::UInt128:       return lldb::eFormatHex;
        case clang::BuiltinType::Int128:        return lldb::eFormatDecimal;
        case clang::BuiltinType::Float:         return lldb::eFormatFloat;
        case clang::BuiltinType::Double:        return lldb::eFormatFloat;
        case clang::BuiltinType::LongDouble:    return lldb::eFormatFloat;
        case clang::BuiltinType::NullPtr:       return lldb::eFormatHex;
        }
        break;
    case clang::Type::ObjCObjectPointer:        return lldb::eFormatHex;
    case clang::Type::BlockPointer:             return lldb::eFormatHex;
    case clang::Type::Pointer:                  return lldb::eFormatHex;
    case clang::Type::LValueReference:
    case clang::Type::RValueReference:          return lldb::eFormatHex;
    case clang::Type::MemberPointer:            break;
    case clang::Type::Complex:                  return lldb::eFormatComplex;
    case clang::Type::ObjCInterface:            break;
    case clang::Type::Record:                   break;
    case clang::Type::Enum:                     return lldb::eFormatEnum;
    case clang::Type::Typedef:
        return lldb_private::Type::GetFormat(cast<clang::TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr());

    case clang::Type::TypeOfExpr:
    case clang::Type::TypeOf:
    case clang::Type::Decltype:
//    case clang::Type::QualifiedName:
    case clang::Type::TemplateSpecialization:   break;
    }
    // We don't know hot to display this type...
    return lldb::eFormatBytes;
}


lldb::Encoding
lldb_private::Type::GetEncoding (uint32_t &count)
{
    // Make sure we resolve our type if it already hasn't been.
    if (!ResolveClangType())
        return lldb::eEncodingInvalid;

    return Type::GetEncoding (m_clang_qual_type, count);
}


lldb::Encoding
lldb_private::Type::GetEncoding (void *clang_type, uint32_t &count)
{
    count = 1;
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    switch (qual_type->getTypeClass())
    {
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
        switch (cast<clang::BuiltinType>(qual_type)->getKind())
        {
        default: assert(0 && "Unknown builtin type!");
        case clang::BuiltinType::Void:
            break;

        case clang::BuiltinType::Bool:
        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::SChar:
        case clang::BuiltinType::WChar:
        case clang::BuiltinType::Char16:
        case clang::BuiltinType::Char32:
        case clang::BuiltinType::Short:
        case clang::BuiltinType::Int:
        case clang::BuiltinType::Long:
        case clang::BuiltinType::LongLong:
        case clang::BuiltinType::Int128:        return lldb::eEncodingSint;

        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::UChar:
        case clang::BuiltinType::UShort:
        case clang::BuiltinType::UInt:
        case clang::BuiltinType::ULong:
        case clang::BuiltinType::ULongLong:
        case clang::BuiltinType::UInt128:       return lldb::eEncodingUint;

        case clang::BuiltinType::Float:
        case clang::BuiltinType::Double:
        case clang::BuiltinType::LongDouble:    return lldb::eEncodingIEEE754;

        case clang::BuiltinType::NullPtr:       return lldb::eEncodingUint;
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
    // Complex numbers are made up of floats
    case clang::Type::Complex:
        count = 2;
        return lldb::eEncodingIEEE754;

    case clang::Type::ObjCInterface:            break;
    case clang::Type::Record:                   break;
    case clang::Type::Enum:                     return lldb::eEncodingSint;
    case clang::Type::Typedef:
        return Type::GetEncoding(cast<clang::TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr(), count);
        break;

    case clang::Type::TypeOfExpr:
    case clang::Type::TypeOf:
    case clang::Type::Decltype:
//    case clang::Type::QualifiedName:
    case clang::Type::TemplateSpecialization:   break;
    }
    count = 0;
    return lldb::eEncodingInvalid;
}


bool
lldb_private::Type::DumpValueInMemory
(
    lldb_private::ExecutionContext *exe_ctx,
    lldb_private::Stream *s,
    lldb::addr_t address,
    lldb::AddressType address_type,
    bool show_types,
    bool show_summary,
    bool verbose
)
{
    if (address != LLDB_INVALID_ADDRESS)
    {
        lldb_private::DataExtractor data;
        data.SetByteOrder (exe_ctx->process->GetByteOrder());
        if (ReadFromMemory (exe_ctx, address, address_type, data))
        {
            DumpValue(exe_ctx, s, data, 0, show_types, show_summary, verbose);
            return true;
        }
    }
    return false;
}

bool
lldb_private::Type::ReadFromMemory
(
    lldb_private::ExecutionContext *exe_ctx,
    clang::ASTContext *ast_context,
    void *clang_type,
    lldb::addr_t addr,
    lldb::AddressType address_type,
    lldb_private::DataExtractor &data
)
{
    if (address_type == lldb::eAddressTypeFile)
    {
        // Can't convert a file address to anything valid without more
        // context (which Module it came from)
        return false;
    }
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    const uint32_t byte_size = (ast_context->getTypeSize (qual_type) + 7) / 8;
    if (data.GetByteSize() < byte_size)
    {
        lldb::DataBufferSP data_sp(new DataBufferHeap (byte_size, '\0'));
        data.SetData(data_sp);
    }

    uint8_t* dst = (uint8_t*)data.PeekData(0, byte_size);
    if (dst != NULL)
    {
        if (address_type == lldb::eAddressTypeHost)
        {
            // The address is an address in this process, so just copy it
            memcpy (dst, (uint8_t*)NULL + addr, byte_size);
            return true;
        }
        else
        {
            if (exe_ctx && exe_ctx->process)
            {
                Error error;
                return exe_ctx->process->ReadMemory(addr, dst, byte_size, error) == byte_size;
            }
        }
    }
    return false;
}

bool
lldb_private::Type::WriteToMemory
(
    lldb_private::ExecutionContext *exe_ctx,
    clang::ASTContext *ast_context,
    void *clang_type,
    lldb::addr_t addr,
    lldb::AddressType address_type,
    StreamString &new_value
)
{
    if (address_type == lldb::eAddressTypeFile)
    {
        // Can't convert a file address to anything valid without more
        // context (which Module it came from)
        return false;
    }
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    const uint32_t byte_size = (ast_context->getTypeSize (qual_type) + 7) / 8;

    if (byte_size > 0)
    {
        if (address_type == lldb::eAddressTypeHost)
        {
            // The address is an address in this process, so just copy it
            memcpy ((void *)addr, new_value.GetData(), byte_size);
            return true;
        }
        else
        {
            if (exe_ctx && exe_ctx->process)
            {
                Error error;
                return exe_ctx->process->WriteMemory(addr, new_value.GetData(), byte_size, error) == byte_size;
            }
        }
    }
    return false;
}


bool
lldb_private::Type::ReadFromMemory (lldb_private::ExecutionContext *exe_ctx, lldb::addr_t addr, lldb::AddressType address_type, lldb_private::DataExtractor &data)
{
    if (address_type == lldb::eAddressTypeFile)
    {
        // Can't convert a file address to anything valid without more
        // context (which Module it came from)
        return false;
    }

    const uint32_t byte_size = GetByteSize();
    if (data.GetByteSize() < byte_size)
    {
        lldb::DataBufferSP data_sp(new DataBufferHeap (byte_size, '\0'));
        data.SetData(data_sp);
    }

    uint8_t* dst = (uint8_t*)data.PeekData(0, byte_size);
    if (dst != NULL)
    {
        if (address_type == lldb::eAddressTypeHost)
        {
            // The address is an address in this process, so just copy it
            memcpy (dst, (uint8_t*)NULL + addr, byte_size);
            return true;
        }
        else
        {
            if (exe_ctx && exe_ctx->process)
            {
                Error error;
                return exe_ctx->process->ReadMemory(addr, dst, byte_size, error) == byte_size;
            }
        }
    }
    return false;
}


bool
lldb_private::Type::WriteToMemory (lldb_private::ExecutionContext *exe_ctx, lldb::addr_t addr, lldb::AddressType address_type, lldb_private::DataExtractor &data)
{
    return false;
}


lldb_private::TypeList*
lldb_private::Type::GetTypeList()
{
    return GetSymbolFile()->GetObjectFile()->GetModule()->GetTypeList();
}


bool
lldb_private::Type::ResolveClangType()
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(m_clang_qual_type));
    if (qual_type.getTypePtr() == NULL)
    {
        clang::QualType resolved_qual_type;
        TypeList *type_list = GetTypeList();
        if (m_encoding_uid != LLDB_INVALID_UID)
        {
            Type *encoding_type = m_symbol_file->ResolveTypeUID(m_encoding_uid);
            if (encoding_type)
            {

                switch (m_encoding_uid_type)
                {
                case eIsTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(encoding_type->GetOpaqueClangQualType());
                    break;

                case eIsConstTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(ClangASTContext::AddConstModifier (encoding_type->GetOpaqueClangQualType()));
                    break;

                case eIsRestrictTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(ClangASTContext::AddRestrictModifier (encoding_type->GetOpaqueClangQualType()));
                    break;

                case eIsVolatileTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(ClangASTContext::AddVolatileModifier (encoding_type->GetOpaqueClangQualType()));
                    break;

                case eTypedefToTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->CreateClangTypedefType (this, encoding_type));
                    // Clear the name so it can get fully qualified in case the
                    // typedef is in a namespace.
                    m_name.Clear();
                    break;

                case ePointerToTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->CreateClangPointerType (encoding_type));
                    break;

                case eLValueReferenceToTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->CreateClangLValueReferenceType (encoding_type));
                    break;

                case eRValueReferenceToTypeWithUID:
                    resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->CreateClangRValueReferenceType (encoding_type));
                    break;

                default:
                    assert(!"Unhandled encoding_uid_type.");
                    break;
                }
            }
        }
        else
        {
            // We have no encoding type, return void?
            void *void_clang_type = type_list->GetClangASTContext().GetVoidBuiltInType();
            switch (m_encoding_uid_type)
            {
            case eIsTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr(void_clang_type);
                break;

            case eIsConstTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr (ClangASTContext::AddConstModifier (void_clang_type));
                break;

            case eIsRestrictTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr (ClangASTContext::AddRestrictModifier (void_clang_type));
                break;

            case eIsVolatileTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr (ClangASTContext::AddVolatileModifier (void_clang_type));
                break;

            case eTypedefToTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->GetClangASTContext().CreateTypedefType (m_name.AsCString(), void_clang_type, NULL));
                break;

            case ePointerToTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->GetClangASTContext().CreatePointerType (void_clang_type));
                break;

            case eLValueReferenceToTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->GetClangASTContext().CreateLValueReferenceType (void_clang_type));
                break;

            case eRValueReferenceToTypeWithUID:
                resolved_qual_type = clang::QualType::getFromOpaquePtr(type_list->GetClangASTContext().CreateRValueReferenceType (void_clang_type));
                break;

            default:
                assert(!"Unhandled encoding_uid_type.");
                break;
            }
        }
        if (resolved_qual_type.getTypePtr())
        {
            m_clang_qual_type = resolved_qual_type.getAsOpaquePtr();
        }

    }
    return m_clang_qual_type != NULL;
}

void *
lldb_private::Type::GetChildClangTypeAtIndex
(
    const char *parent_name,
    uint32_t idx,
    bool transparent_pointers,
    bool omit_empty_base_classes,
    ConstString& name,
    uint32_t &child_byte_size,
    int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset
)
{
    if (!ResolveClangType())
        return false;

    std::string name_str;
    void *child_qual_type = GetClangASTContext().GetChildClangTypeAtIndex (
            parent_name,
            m_clang_qual_type,
            idx,
            transparent_pointers,
            omit_empty_base_classes,
            name_str,
            child_byte_size,
            child_byte_offset,
            child_bitfield_bit_size,
            child_bitfield_bit_offset);

    if (child_qual_type)
    {
        if (!name_str.empty())
            name.SetCString(name_str.c_str());
        else
            name.Clear();
    }
    return child_qual_type;
}



void *
lldb_private::Type::GetOpaqueClangQualType ()
{
    ResolveClangType();
    return m_clang_qual_type;
}

clang::ASTContext *
lldb_private::Type::GetClangAST ()
{
    TypeList *type_list = GetTypeList();
    if (type_list)
        return type_list->GetClangASTContext().getASTContext();
    return NULL;
}

lldb_private::ClangASTContext &
lldb_private::Type::GetClangASTContext ()
{
    return GetTypeList()->GetClangASTContext();
}

int
lldb_private::Type::Compare(const Type &a, const Type &b)
{
    // Just compare the UID values for now...
    lldb::user_id_t a_uid = a.GetID();
    lldb::user_id_t b_uid = b.GetID();
    if (a_uid < b_uid)
        return -1;
    if (a_uid > b_uid)
        return 1;
    return 0;
//  if (a.getQualType() == b.getQualType())
//      return 0;
}

