//===-- SBType.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include "lldb/API/SBType.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTType.h"

using namespace lldb;
using namespace lldb_private;


bool
SBType::IsPointerType (void *opaque_type)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    //if (log)
    //    log->Printf ("SBType::IsPointerType (%p)", opaque_type);
    
    bool ret_value = ClangASTContext::IsPointerType (opaque_type);

    if (log)
        log->Printf ("SBType::IsPointerType (opaque_type=%p) ==> '%s'", opaque_type, (ret_value ? "true" : "false"));

    return ret_value;
}


SBType::SBType (void *ast, void *clang_type) :
    m_ast (ast),
    m_type (clang_type)
{
}

SBType::SBType (const SBType &rhs) :
    m_ast (rhs.m_ast),
    m_type (rhs.m_type)
{
}

const SBType &
SBType::operator =(const SBType &rhs)
{
    m_ast = rhs.m_ast;
    m_type = rhs.m_type;
    return *this;
}

SBType::~SBType ()
{
}

bool
SBType::IsValid ()
{
    return m_ast != NULL && m_type != NULL;
}

const char *
SBType::GetName ()
{
    if (IsValid ())
        return ClangASTType::GetClangTypeName (m_type).AsCString(NULL);
    return NULL;
}

uint64_t
SBType::GetByteSize()
{
    if (IsValid ())
        return ClangASTType::GetClangTypeBitWidth (static_cast<clang::ASTContext *>(m_ast), m_type);
    return NULL;
}

Encoding
SBType::GetEncoding (uint32_t &count)
{
    if (IsValid ())
        return ClangASTType::GetEncoding (m_type, count);
    count = 0;
    return eEncodingInvalid;
}

uint64_t
SBType::GetNumberChildren (bool omit_empty_base_classes)
{
    if (IsValid ())
        return ClangASTContext::GetNumChildren (static_cast<clang::ASTContext *>(m_ast),
                                                m_type, 
                                                omit_empty_base_classes);
    return 0;
}


bool
SBType::GetChildAtIndex (bool omit_empty_base_classes, uint32_t idx, SBTypeMember &member)
{
    void *child_clang_type = NULL;
    std::string child_name;
    uint32_t child_byte_size = 0;
    int32_t child_byte_offset = 0;
    uint32_t child_bitfield_bit_size = 0;
    uint32_t child_bitfield_bit_offset = 0;
    bool child_is_base_class = false;

    if (IsValid ())
    {

        child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (static_cast<clang::ASTContext *>(m_ast),
                                                                      NULL,
                                                                      m_type,
                                                                      idx,
                                                                      false, // transparent pointers
                                                                      omit_empty_base_classes,
                                                                      child_name,
                                                                      child_byte_size,
                                                                      child_byte_offset,
                                                                      child_bitfield_bit_size,
                                                                      child_bitfield_bit_offset,
                                                                      child_is_base_class);
        
    }
    
    if (child_clang_type)
    {
        member.m_ast = m_ast;
        member.m_parent_type = m_type;
        member.m_member_type = child_clang_type,
        member.SetName (child_name.c_str());
        member.m_offset = child_byte_offset;
        member.m_bit_size = child_bitfield_bit_size;
        member.m_bit_offset = child_bitfield_bit_offset;
        member.m_is_base_class = child_is_base_class;
    }
    else
    {
        member.Clear();
    }

    return child_clang_type != NULL;
}

uint32_t
SBType::GetChildIndexForName (bool omit_empty_base_classes, const char *name)
{
    return ClangASTContext::GetIndexOfChildWithName (static_cast<clang::ASTContext *>(m_ast),
                                                     m_type,
                                                     name,
                                                     omit_empty_base_classes);
}

bool
SBType::IsPointerType ()
{
    return ClangASTContext::IsPointerType (m_type);
}

SBType
SBType::GetPointeeType ()
{
    void *pointee_type = NULL;
    if (IsPointerType ())
    {
        pointee_type = ClangASTType::GetPointeeType (m_type);
    }
    return SBType (pointee_type ? m_ast : NULL, pointee_type);
}

bool
SBType::GetDescription (SBStream &description)
{
    const char *name = GetName();
    uint64_t byte_size = GetByteSize();
    uint64_t num_children = GetNumberChildren (true);
    bool is_ptr = IsPointerType ();

    description.Printf ("type_name: %s, size: %d bytes", (name != NULL ? name : "<unknown type name>"), byte_size);
    if (is_ptr)
    {
        SBType pointee_type = GetPointeeType();
        const char *pointee_name = pointee_type.GetName();
        description.Printf (", (* %s)", (pointee_name != NULL ? pointee_name : "<unknown type name>"));
    }
    else if (num_children > 0)
    {
        description.Printf (", %d members:\n", num_children);
        for (uint32_t i = 0; i < num_children; ++i)
        {
            SBTypeMember field;
            GetChildAtIndex (true, i, field);
            const char *field_name = field.GetName();
            SBType field_type = field.GetType();
            const char *field_type_name = field_type.GetName();
            
            description.Printf ("     %s (type: %s", (field_name != NULL ? field_name : "<unknown member name>"), 
                                (field_type_name != NULL ? field_type_name : "<unknown type name>"));

            if (field.IsBitfield())
            {
                size_t width = field.GetBitfieldWidth ();
                description.Printf (" , %d bits", (int) width);
            }
            description.Printf (")\n");
        }
    }
    return true;
}

SBTypeMember::SBTypeMember () :
    m_ast (NULL),
    m_parent_type (NULL),
    m_member_type (NULL),
    m_member_name (NULL),
    m_offset (0),
    m_bit_size (0),
    m_bit_offset (0),
    m_is_base_class (false)
    
{
}

SBTypeMember::SBTypeMember (const SBTypeMember &rhs) :
    m_ast (rhs.m_ast),
    m_parent_type (rhs.m_parent_type),
    m_member_type (rhs.m_member_type),
    m_member_name (rhs.m_member_name),
    m_offset (rhs.m_offset),
    m_bit_size (rhs.m_bit_size),
    m_bit_offset (rhs.m_bit_offset),
    m_is_base_class (rhs.m_is_base_class)
{
}

const SBTypeMember&
SBTypeMember::operator =(const SBTypeMember &rhs)
{
    if (this != &rhs)
    {
        m_ast = rhs.m_ast;
        m_parent_type = rhs.m_parent_type;
        m_member_type = rhs.m_member_type;
        m_member_name = rhs.m_member_name;
        m_offset = rhs.m_offset;
        m_bit_size = rhs.m_bit_size;
        m_bit_offset = rhs.m_bit_offset;
        m_is_base_class = rhs.m_is_base_class;
    }
    return *this;
}

SBTypeMember::~SBTypeMember ()
{
    SetName (NULL);
}

void
SBTypeMember::SetName (const char *name)
{
    if (m_member_name)  
        free (m_member_name);
    if (name && name[0])
        m_member_name = ::strdup (name);
    else
        m_member_name = NULL;
}

void
SBTypeMember::Clear()
{
    m_ast = NULL;
    m_parent_type = NULL;
    m_member_type = NULL;
    SetName (NULL);
    m_offset = 0;
    m_bit_size  = 0;
    m_bit_offset = 0;
    m_is_base_class = false;
}

bool
SBTypeMember::IsValid ()
{
    return m_member_type != NULL;
}

bool
SBTypeMember::IsBitfield ()
{
    return m_bit_size != 0;
}

size_t
SBTypeMember::GetBitfieldWidth ()
{
    return m_bit_size;
}

size_t
SBTypeMember::GetBitfieldOffset ()
{
    return m_bit_offset;
}

bool
SBTypeMember::IsBaseClass ()
{
    return m_is_base_class;
}

size_t
SBTypeMember::GetOffset ()
{
    return m_offset;
}

SBType
SBTypeMember::GetType()
{
    return SBType (m_ast, m_member_type);
}

SBType
SBTypeMember::GetParentType()
{
    return SBType (m_ast, m_parent_type);
}


const char *
SBTypeMember::GetName ()
{
    return m_member_name;
}

