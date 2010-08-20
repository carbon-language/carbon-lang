//===-- Type.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Other libraries and framework includes

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"

#include "lldb/Symbol/ClangASTType.h"
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
lldb_private::Type::GetDescription (Stream *s, lldb::DescriptionLevel level, bool show_name)
{
    if (show_name)
    {
        if (m_name)
            *s << '\"' << m_name << "\", ";
    }

    *s << "id = " << (const UserID&)*this;

    if (m_byte_size != 0)
        s->Printf(", byte-size = %zu", m_byte_size);

    m_decl.Dump(s);

    if (m_clang_qual_type)
    {
        *s << ", clang_type = " << m_clang_qual_type << ' ';

        ClangASTType::DumpTypeDescription (GetClangAST(), m_clang_qual_type, s);
    }
    else if (m_encoding_uid != LLDB_INVALID_UID)
    {
        s->Printf(", type_uid = 0x%8.8x", m_encoding_uid);
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

    if (m_clang_qual_type)
    {
        *s << ", clang_type = " << m_clang_qual_type << ' ';

        ClangASTType::DumpTypeDescription (GetClangAST(), m_clang_qual_type, s);
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

        lldb_private::ClangASTType::DumpValue (GetClangAST (),
                                               m_clang_qual_type,
                                               exe_ctx,
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
                uint64_t bit_width = ClangASTType::GetClangTypeBitWidth (GetClangAST(), GetOpaqueClangQualType());
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
    return lldb_private::ClangASTType::GetFormat (m_clang_qual_type);
}



lldb::Encoding
lldb_private::Type::GetEncoding (uint32_t &count)
{
    // Make sure we resolve our type if it already hasn't been.
    if (!ResolveClangType())
        return lldb::eEncodingInvalid;

    return lldb_private::ClangASTType::GetEncoding (m_clang_qual_type, count);
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

const lldb_private::Declaration &
lldb_private::Type::GetDeclaration () const
{
    return m_decl;
}

bool
lldb_private::Type::ResolveClangType()
{
    if (m_clang_qual_type == NULL)
    {
        TypeList *type_list = GetTypeList();
        if (m_encoding_uid != LLDB_INVALID_UID)
        {
            Type *encoding_type = m_symbol_file->ResolveTypeUID(m_encoding_uid);
            if (encoding_type)
            {

                switch (m_encoding_uid_type)
                {
                case eIsTypeWithUID:
                    m_clang_qual_type = encoding_type->GetOpaqueClangQualType();
                    break;

                case eIsConstTypeWithUID:
                    m_clang_qual_type = ClangASTContext::AddConstModifier (encoding_type->GetOpaqueClangQualType());
                    break;

                case eIsRestrictTypeWithUID:
                    m_clang_qual_type = ClangASTContext::AddRestrictModifier (encoding_type->GetOpaqueClangQualType());
                    break;

                case eIsVolatileTypeWithUID:
                    m_clang_qual_type = ClangASTContext::AddVolatileModifier (encoding_type->GetOpaqueClangQualType());
                    break;

                case eTypedefToTypeWithUID:
                    m_clang_qual_type = type_list->CreateClangTypedefType (this, encoding_type);
                    // Clear the name so it can get fully qualified in case the
                    // typedef is in a namespace.
                    m_name.Clear();
                    break;

                case ePointerToTypeWithUID:
                    m_clang_qual_type = type_list->CreateClangPointerType (encoding_type);
                    break;

                case eLValueReferenceToTypeWithUID:
                    m_clang_qual_type = type_list->CreateClangLValueReferenceType (encoding_type);
                    break;

                case eRValueReferenceToTypeWithUID:
                    m_clang_qual_type = type_list->CreateClangRValueReferenceType (encoding_type);
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
            void *void_clang_type = type_list->GetClangASTContext().GetBuiltInType_void();
            switch (m_encoding_uid_type)
            {
            case eIsTypeWithUID:
                m_clang_qual_type = void_clang_type;
                break;

            case eIsConstTypeWithUID:
                m_clang_qual_type = ClangASTContext::AddConstModifier (void_clang_type);
                break;

            case eIsRestrictTypeWithUID:
                m_clang_qual_type = ClangASTContext::AddRestrictModifier (void_clang_type);
                break;

            case eIsVolatileTypeWithUID:
                m_clang_qual_type = ClangASTContext::AddVolatileModifier (void_clang_type);
                break;

            case eTypedefToTypeWithUID:
                m_clang_qual_type = type_list->GetClangASTContext().CreateTypedefType (m_name.AsCString(), void_clang_type, NULL);
                break;

            case ePointerToTypeWithUID:
                m_clang_qual_type = type_list->GetClangASTContext().CreatePointerType (void_clang_type);
                break;

            case eLValueReferenceToTypeWithUID:
                m_clang_qual_type = type_list->GetClangASTContext().CreateLValueReferenceType (void_clang_type);
                break;

            case eRValueReferenceToTypeWithUID:
                m_clang_qual_type = type_list->GetClangASTContext().CreateRValueReferenceType (void_clang_type);
                break;

            default:
                assert(!"Unhandled encoding_uid_type.");
                break;
            }
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
        return NULL;

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

