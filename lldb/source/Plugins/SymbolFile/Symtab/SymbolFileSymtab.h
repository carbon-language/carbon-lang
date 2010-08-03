//===-- SymbolFileSymtab.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolFileSymtab_h_
#define liblldb_SymbolFileSymtab_h_

#include "lldb/Symbol/SymbolFile.h"
#include <vector>

class SymbolFileSymtab : public lldb_private::SymbolFile
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::SymbolFile*
    CreateInstance (lldb_private::ObjectFile* obj_file);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SymbolFileSymtab(lldb_private::ObjectFile* obj_file);

    virtual
    ~SymbolFileSymtab();

    virtual uint32_t        GetAbilities ();

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    virtual uint32_t
    GetNumCompileUnits();

    virtual lldb::CompUnitSP
    ParseCompileUnitAtIndex(uint32_t index);

    virtual size_t
    ParseCompileUnitFunctions (const lldb_private::SymbolContext& sc);

    virtual bool
    ParseCompileUnitLineTable (const lldb_private::SymbolContext& sc);

    virtual bool
    ParseCompileUnitSupportFiles (const lldb_private::SymbolContext& sc, lldb_private::FileSpecList &support_files);

    virtual size_t
    ParseFunctionBlocks (const lldb_private::SymbolContext& sc);

    virtual size_t
    ParseTypes (const lldb_private::SymbolContext& sc);

    virtual size_t
    ParseVariablesForContext (const lldb_private::SymbolContext& sc);

    virtual lldb_private::Type*
    ResolveTypeUID(lldb::user_id_t type_uid);

    virtual uint32_t
    ResolveSymbolContext (const lldb_private::Address& so_addr, uint32_t resolve_scope, lldb_private::SymbolContext& sc);

    virtual uint32_t
    ResolveSymbolContext (const lldb_private::FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, lldb_private::SymbolContextList& sc_list);

    virtual uint32_t
    FindGlobalVariables(const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::VariableList& variables);

    virtual uint32_t
    FindGlobalVariables(const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::VariableList& variables);

    virtual uint32_t
    FindFunctions(const lldb_private::ConstString &name, uint32_t name_type_mask, bool append, lldb_private::SymbolContextList& sc_list);

    virtual uint32_t
    FindFunctions(const lldb_private::RegularExpression& regex, bool append, lldb_private::SymbolContextList& sc_list);

    virtual uint32_t
    FindTypes (const lldb_private::SymbolContext& sc, const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::TypeList& types);

//  virtual uint32_t
//  FindTypes(const lldb_private::SymbolContext& sc, const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::TypeList& types);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual void
    GetPluginCommandHelp (const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand (lldb_private::Args &command, lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging (lldb_private::Stream *strm, lldb_private::Args &command);



protected:
    std::vector<uint32_t>   m_source_indexes;
    std::vector<uint32_t>   m_func_indexes;
    std::vector<uint32_t>   m_code_indexes;
    std::vector<uint32_t>   m_data_indexes;
    std::vector<uint32_t>   m_addr_indexes; // Anything that needs to go into an search by address

private:
    DISALLOW_COPY_AND_ASSIGN (SymbolFileSymtab);
};


#endif  // liblldb_SymbolFileSymtab_h_
