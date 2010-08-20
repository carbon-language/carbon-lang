//===-- Type.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Type_h_
#define liblldb_Type_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/UserID.h"
#include "lldb/Symbol/Declaration.h"
#include <set>

namespace lldb_private {

class Type : public UserID
{
public:
    typedef enum
    {
        eTypeInvalid,
        eIsTypeWithUID,                 ///< This type is the type whose UID is m_encoding_uid
        eIsConstTypeWithUID,            ///< This type is the type whose UID is m_encoding_uid with the const qualifier added
        eIsRestrictTypeWithUID,         ///< This type is the type whose UID is m_encoding_uid with the restrict qualifier added
        eIsVolatileTypeWithUID,         ///< This type is the type whose UID is m_encoding_uid with the volatile qualifier added
        eTypedefToTypeWithUID,          ///< This type is pointer to a type whose UID is m_encoding_uid
        ePointerToTypeWithUID,          ///< This type is pointer to a type whose UID is m_encoding_uid
        eLValueReferenceToTypeWithUID,  ///< This type is L value reference to a type whose UID is m_encoding_uid
        eRValueReferenceToTypeWithUID,   ///< This type is R value reference to a type whose UID is m_encoding_uid
        eTypeUIDSynthetic
    } EncodingUIDType;

    Type(lldb::user_id_t uid,
           SymbolFile* symbol_file,
           const ConstString &name,
           uint64_t byte_size,
           SymbolContextScope *context,
           lldb::user_id_t encoding_uid,
           EncodingUIDType encoding_type,
           const Declaration& decl,
           void *clang_qual_type);

    // This makes an invalid type.  Used for functions that return a Type when they
    // get an error.
    Type();

    const Type&
    operator= (const Type& rhs);

    void
    Dump(Stream *s, bool show_context);

    void
    DumpTypeName(Stream *s);


    void
    GetDescription (Stream *s, lldb::DescriptionLevel level, bool show_name);

    SymbolFile *
    GetSymbolFile()
    {
        return m_symbol_file;
    }
    const SymbolFile *
    GetSymbolFile() const
    {
        return m_symbol_file;
    }

    TypeList*
    GetTypeList();

    const ConstString&
    GetName();

    uint64_t
    GetByteSize();

    uint32_t
    GetNumChildren (bool omit_empty_base_classes);

    bool
    IsAggregateType ();

    bool
    IsValidType ()
    {
        return m_encoding_uid_type != eTypeInvalid;
    }

    void
    SetByteSize(uint32_t byte_size);

    const ConstString &
    GetName () const
    {
        return m_name;
    }

    void
    DumpValue(ExecutionContext *exe_ctx,
              Stream *s,
              const DataExtractor &data,
              uint32_t data_offset,
              bool show_type,
              bool show_summary,
              bool verbose,
              lldb::Format format = lldb::eFormatDefault);

    bool
    DumpValueInMemory(ExecutionContext *exe_ctx,
                      Stream *s,
                      lldb::addr_t address,
                      lldb::AddressType address_type,
                      bool show_types,
                      bool show_summary,
                      bool verbose);

    bool
    ReadFromMemory (ExecutionContext *exe_ctx,
                    lldb::addr_t address,
                    lldb::AddressType address_type,
                    DataExtractor &data);

    bool
    WriteToMemory (ExecutionContext *exe_ctx,
                   lldb::addr_t address,
                   lldb::AddressType address_type,
                   DataExtractor &data);

    bool
    GetIsDeclaration() const;

    void
    SetIsDeclaration(bool b);

    bool
    GetIsExternal() const;

    void
    SetIsExternal(bool b);

    lldb::Format
    GetFormat ();

    lldb::Encoding
    GetEncoding (uint32_t &count);

    SymbolContextScope *
    GetSymbolContextScope()
    {
        return m_context;
    }
    const SymbolContextScope *
    GetSymbolContextScope() const
    {
        return m_context;
    }
    void
    SetSymbolContextScope(SymbolContextScope *context)
    {
        m_context = context;
    }

    const lldb_private::Declaration &
    GetDeclaration () const;

    void *
    GetOpaqueClangQualType ();

    clang::ASTContext *
    GetClangAST ();

    ClangASTContext &
    GetClangASTContext ();

    void *
    GetChildClangTypeAtIndex (const char *parent_name,
                              uint32_t idx,
                              bool transparent_pointers,
                              bool omit_empty_base_classes,
                              ConstString& name,
                              uint32_t &child_byte_size,
                              int32_t &child_byte_offset,
                              uint32_t &child_bitfield_bit_size,
                              uint32_t &child_bitfield_bit_offset);

    static int
    Compare(const Type &a, const Type &b);

protected:
    ConstString m_name;
    uint64_t m_byte_size;
    SymbolFile *m_symbol_file;
    SymbolContextScope *m_context; // The symbol context in which this type is defined
    lldb::user_id_t m_encoding_uid;
    EncodingUIDType m_encoding_uid_type;
    Declaration m_decl;
    void *m_clang_qual_type;

    bool ResolveClangType();
};

} // namespace lldb_private

#endif  // liblldb_Type_h_

