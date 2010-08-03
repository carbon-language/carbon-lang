//===-- SymbolFileDWARFDebugMap.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolFileDWARFDebugMap_h_
#define liblldb_SymbolFileDWARFDebugMap_h_


#include <vector>
#include <bitset>
#include "lldb/Symbol/SymbolFile.h"

class SymbolFileDWARF;

class SymbolFileDWARFDebugMap : public lldb_private::SymbolFile
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

    static lldb_private::SymbolFile *
    CreateInstance (lldb_private::ObjectFile* obj_file);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
                            SymbolFileDWARFDebugMap (lldb_private::ObjectFile* ofile);
    virtual               ~ SymbolFileDWARFDebugMap ();

    virtual uint32_t        GetAbilities ();

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    virtual uint32_t        GetNumCompileUnits ();
    virtual lldb::CompUnitSP ParseCompileUnitAtIndex (uint32_t index);

    virtual size_t          ParseCompileUnitFunctions (const lldb_private::SymbolContext& sc);
    virtual bool            ParseCompileUnitLineTable (const lldb_private::SymbolContext& sc);
    virtual bool            ParseCompileUnitSupportFiles (const lldb_private::SymbolContext& sc, lldb_private::FileSpecList &support_files);
    virtual size_t          ParseFunctionBlocks (const lldb_private::SymbolContext& sc);
    virtual size_t          ParseTypes (const lldb_private::SymbolContext& sc);
    virtual size_t          ParseVariablesForContext (const lldb_private::SymbolContext& sc);

    virtual lldb_private::Type*     ResolveTypeUID (lldb::user_id_t type_uid);
    virtual uint32_t        ResolveSymbolContext (const lldb_private::Address& so_addr, uint32_t resolve_scope, lldb_private::SymbolContext& sc);
    virtual uint32_t        ResolveSymbolContext (const lldb_private::FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindGlobalVariables (const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindGlobalVariables (const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindFunctions (const lldb_private::ConstString &name, uint32_t name_type_mask, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindFunctions (const lldb_private::RegularExpression& regex, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindTypes (const lldb_private::SymbolContext& sc, const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::TypeList& types);
//  virtual uint32_t        FindTypes (const lldb_private::SymbolContext& sc, const RegularExpression& regex, bool append, uint32_t max_matches, Type::Encoding encoding, lldb::user_id_t udt_uid, TypeList& types);

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
    enum
    {
        kHaveInitializedOSOs = (1 << 0),
        kNumFlags
    };

    //------------------------------------------------------------------
    // Class specific types
    //------------------------------------------------------------------
    struct CompileUnitInfo
    {
        lldb_private::FileSpec so_file;
        lldb_private::Symbol *so_symbol;
        lldb_private::Symbol *oso_symbol;
        lldb::ModuleSP oso_module_sp;
        lldb::CompUnitSP oso_compile_unit_sp;
        lldb_private::SymbolVendor *oso_symbol_vendor;
//      lldb_private::shared_ptr<SymbolFileDWARF> oso_dwarf_sp;
//      lldb_private::shared_ptr<SymbolVendor> oso_dwarf_sp;
        std::vector<uint32_t> function_indexes;
        std::vector<uint32_t> static_indexes;
        lldb::SharedPtr<lldb_private::SectionList>::Type debug_map_sections_sp;

        CompileUnitInfo() :
            so_file(),
            so_symbol(NULL),
            oso_symbol(NULL),
            oso_module_sp(),
            oso_compile_unit_sp(),
            oso_symbol_vendor(NULL),
//          oso_dwarf_sp(),
            function_indexes(),
            static_indexes(),
            debug_map_sections_sp()
        {
        }
    };

    //------------------------------------------------------------------
    // Protected Member Functions
    //------------------------------------------------------------------
    void
    InitOSO ();

    bool
    GetFileSpecForSO (uint32_t oso_idx, lldb_private::FileSpec &file_spec);

    CompileUnitInfo *
    GetCompUnitInfo (const lldb_private::SymbolContext& sc);

    lldb_private::Module *
    GetModuleByCompUnitInfo (CompileUnitInfo *comp_unit_info);

    lldb_private::Module *
    GetModuleByOSOIndex (uint32_t oso_idx);

    lldb_private::ObjectFile *
    GetObjectFileByCompUnitInfo (CompileUnitInfo *comp_unit_info);

    lldb_private::ObjectFile *
    GetObjectFileByOSOIndex (uint32_t oso_idx);

    SymbolFileDWARF *
    GetSymbolFile (const lldb_private::SymbolContext& sc);

    SymbolFileDWARF *
    GetSymbolFileByCompUnitInfo (CompileUnitInfo *comp_unit_info);

    SymbolFileDWARF *
    GetSymbolFileByOSOIndex (uint32_t oso_idx);

    CompileUnitInfo*
    GetCompileUnitInfoForSymbolWithIndex (uint32_t symbol_idx, uint32_t *oso_idx_ptr);

    static int
    SymbolContainsSymbolIndex (uint32_t *symbol_idx_ptr, const CompileUnitInfo *comp_unit_info);

    uint32_t
    PrivateFindGlobalVariables (const lldb_private::ConstString &name,
                                const std::vector<uint32_t> &name_symbol_indexes,
                                uint32_t max_matches,
                                lldb_private::VariableList& variables);

    //------------------------------------------------------------------
    // Member Variables
    //------------------------------------------------------------------
    std::bitset<kNumFlags> m_flags;
    std::vector<CompileUnitInfo> m_compile_unit_infos;
    std::vector<uint32_t> m_func_indexes;   // Sorted by address
    std::vector<uint32_t> m_glob_indexes;
};

#endif // #ifndef liblldb_SymbolFileDWARFDebugMap_h_
