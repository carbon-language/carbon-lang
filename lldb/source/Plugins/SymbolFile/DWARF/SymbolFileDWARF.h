//===-- SymbolFileDWARF.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_SymbolFileDWARF_h_
#define SymbolFileDWARF_SymbolFileDWARF_h_

// C Includes
// C++ Includes
#include <list>
#include <map>
#include <set>
#include <vector>

// Other libraries and framework includes
#include "llvm/ADT/DenseMap.h"

#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolContext.h"

// Project includes
#include "DWARFDefines.h"
#include "DWARFDataExtractor.h"
#include "HashedNameToDIE.h"
#include "NameToDIE.h"
#include "UniqueDWARFASTType.h"

//----------------------------------------------------------------------
// Forward Declarations for this DWARF plugin
//----------------------------------------------------------------------
class DebugMapModule;
class DWARFAbbreviationDeclaration;
class DWARFAbbreviationDeclarationSet;
class DWARFileUnit;
class DWARFDebugAbbrev;
class DWARFDebugAranges;
class DWARFDebugInfo;
class DWARFDebugInfoEntry;
class DWARFDebugLine;
class DWARFDebugPubnames;
class DWARFDebugRanges;
class DWARFDeclContext;
class DWARFDIECollection;
class DWARFFormValue;
class SymbolFileDWARFDebugMap;

#define DIE_IS_BEING_PARSED ((lldb_private::Type*)1)

class SymbolFileDWARF : public lldb_private::SymbolFile, public lldb_private::UserID
{
public:
    friend class SymbolFileDWARFDebugMap;
    friend class SymbolFileDWARFDwo;
    friend class DebugMapModule;
    friend class DWARFCompileUnit;
    friend class DWARFASTParserClang;
    friend class DWARFASTParserGo;

    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static void
    DebuggerInitialize(lldb_private::Debugger &debugger);

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::SymbolFile*
    CreateInstance (lldb_private::ObjectFile* obj_file);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------

    SymbolFileDWARF(lldb_private::ObjectFile* ofile);

    ~SymbolFileDWARF() override;

    uint32_t
    CalculateAbilities () override;

    void
    InitializeObject() override;

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------

    uint32_t
    GetNumCompileUnits() override;

    lldb::CompUnitSP
    ParseCompileUnitAtIndex(uint32_t index) override;

    lldb::LanguageType
    ParseCompileUnitLanguage (const lldb_private::SymbolContext& sc) override;

    size_t
    ParseCompileUnitFunctions (const lldb_private::SymbolContext& sc) override;

    bool
    ParseCompileUnitLineTable (const lldb_private::SymbolContext& sc) override;

    bool
    ParseCompileUnitSupportFiles (const lldb_private::SymbolContext& sc,
                                  lldb_private::FileSpecList& support_files) override;

    bool
    ParseImportedModules (const lldb_private::SymbolContext &sc,
                          std::vector<lldb_private::ConstString> &imported_modules) override;

    size_t
    ParseFunctionBlocks (const lldb_private::SymbolContext& sc) override;

    size_t
    ParseTypes (const lldb_private::SymbolContext& sc) override;

    size_t
    ParseVariablesForContext (const lldb_private::SymbolContext& sc) override;

    lldb_private::Type *
    ResolveTypeUID(lldb::user_id_t type_uid) override;

    bool
    CompleteType (lldb_private::CompilerType& clang_type) override;

    lldb_private::Type *
    ResolveType (const DWARFDIE &die,
                 bool assert_not_being_parsed = true);

    lldb_private::CompilerDecl
    GetDeclForUID (lldb::user_id_t uid) override;

    lldb_private::CompilerDeclContext
    GetDeclContextForUID (lldb::user_id_t uid) override;

    lldb_private::CompilerDeclContext
    GetDeclContextContainingUID (lldb::user_id_t uid) override;

    void
    ParseDeclsForContext (lldb_private::CompilerDeclContext decl_ctx) override;
    

    uint32_t
    ResolveSymbolContext (const lldb_private::Address& so_addr,
                          uint32_t resolve_scope,
                          lldb_private::SymbolContext& sc) override;

    uint32_t
    ResolveSymbolContext (const lldb_private::FileSpec& file_spec,
                          uint32_t line,
                          bool check_inlines,
                          uint32_t resolve_scope,
                          lldb_private::SymbolContextList& sc_list) override;

    uint32_t
    FindGlobalVariables (const lldb_private::ConstString &name,
                         const lldb_private::CompilerDeclContext *parent_decl_ctx,
                         bool append,
                         uint32_t max_matches,
                         lldb_private::VariableList& variables) override;

    uint32_t
    FindGlobalVariables (const lldb_private::RegularExpression& regex,
                         bool append,
                         uint32_t max_matches,
                         lldb_private::VariableList& variables) override;

    uint32_t
    FindFunctions (const lldb_private::ConstString &name,
                   const lldb_private::CompilerDeclContext *parent_decl_ctx,
                   uint32_t name_type_mask,
                   bool include_inlines,
                   bool append,
                   lldb_private::SymbolContextList& sc_list) override;

    uint32_t
    FindFunctions (const lldb_private::RegularExpression& regex,
                   bool include_inlines,
                   bool append,
                   lldb_private::SymbolContextList& sc_list) override;

    uint32_t
    FindTypes (const lldb_private::SymbolContext& sc,
               const lldb_private::ConstString &name,
               const lldb_private::CompilerDeclContext *parent_decl_ctx,
               bool append,
               uint32_t max_matches,
               lldb_private::TypeList& types) override;

    lldb_private::TypeList *
    GetTypeList () override;

    size_t
    GetTypes (lldb_private::SymbolContextScope *sc_scope,
              uint32_t type_mask,
              lldb_private::TypeList &type_list) override;

    lldb_private::TypeSystem *
    GetTypeSystemForLanguage (lldb::LanguageType language) override;

    lldb_private::CompilerDeclContext
    FindNamespace (const lldb_private::SymbolContext& sc,
                   const lldb_private::ConstString &name,
                   const lldb_private::CompilerDeclContext *parent_decl_ctx) override;


    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    lldb_private::ConstString
    GetPluginName() override;

    uint32_t
    GetPluginVersion() override;

    const lldb_private::DWARFDataExtractor&     get_debug_abbrev_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_addr_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_aranges_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_frame_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_info_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_line_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_loc_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_ranges_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_str_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_str_offsets_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_names_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_types_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_namespaces_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_objc_data ();


    DWARFDebugAbbrev*
    DebugAbbrev();

    const DWARFDebugAbbrev*
    DebugAbbrev() const;

    DWARFDebugInfo*
    DebugInfo();

    const DWARFDebugInfo*
    DebugInfo() const;

    DWARFDebugRanges*
    DebugRanges();
    const DWARFDebugRanges*
    DebugRanges() const;

    virtual const lldb_private::DWARFDataExtractor&
    GetCachedSectionData (uint32_t got_flag, 
                          lldb::SectionType sect_type, 
                          lldb_private::DWARFDataExtractor &data);

    static bool
    SupportedVersion(uint16_t version);

    DWARFDIE
    GetDeclContextDIEContainingDIE (const DWARFDIE &die);

    lldb_private::Flags&
    GetFlags ()
    {
        return m_flags;
    }

    const lldb_private::Flags&
    GetFlags () const
    {
        return m_flags;
    }

    bool
    HasForwardDeclForClangType (const lldb_private::CompilerType &clang_type);

    lldb_private::CompileUnit*
    GetCompUnitForDWARFCompUnit(DWARFCompileUnit* dwarf_cu,
                                uint32_t cu_idx = UINT32_MAX);

    lldb::user_id_t
    MakeUserID (dw_offset_t die_offset) const
    {
        return GetID() | die_offset;
    }

    size_t
    GetObjCMethodDIEOffsets (lldb_private::ConstString class_name,
                             DIEArray &method_die_offsets);

    bool
    Supports_DW_AT_APPLE_objc_complete_type (DWARFCompileUnit *cu);

    static DWARFDIE
    GetParentSymbolContextDIE(const DWARFDIE &die);

    virtual lldb::CompUnitSP
    ParseCompileUnit (DWARFCompileUnit* dwarf_cu, uint32_t cu_idx);

    virtual lldb_private::DWARFExpression::LocationListFormat
    GetLocationListFormat() const;

protected:
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb_private::Type *> DIEToTypePtr;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::VariableSP> DIEToVariableSP;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::clang_type_t> DIEToClangType;
    typedef llvm::DenseMap<lldb::clang_type_t, DIERef> ClangTypeToDIE;

    enum
    {
        flagsGotDebugAbbrevData     = (1 << 0),
        flagsGotDebugAddrData       = (1 << 1),
        flagsGotDebugArangesData    = (1 << 2),
        flagsGotDebugFrameData      = (1 << 3),
        flagsGotDebugInfoData       = (1 << 4),
        flagsGotDebugLineData       = (1 << 5),
        flagsGotDebugLocData        = (1 << 6),
        flagsGotDebugMacInfoData    = (1 << 7),
        flagsGotDebugPubNamesData   = (1 << 8),
        flagsGotDebugPubTypesData   = (1 << 9),
        flagsGotDebugRangesData     = (1 << 10),
        flagsGotDebugStrData        = (1 << 11),
        flagsGotDebugStrOffsetsData = (1 << 12),
        flagsGotAppleNamesData      = (1 << 13),
        flagsGotAppleTypesData      = (1 << 14),
        flagsGotAppleNamespacesData = (1 << 15),
        flagsGotAppleObjCData       = (1 << 16)
    };
    
    bool
    DeclContextMatchesThisSymbolFile (const lldb_private::CompilerDeclContext *decl_ctx);

    bool
    DIEInDeclContext (const lldb_private::CompilerDeclContext *parent_decl_ctx,
                      const DWARFDIE &die);

    DISALLOW_COPY_AND_ASSIGN (SymbolFileDWARF);

    virtual DWARFCompileUnit*
    GetDWARFCompileUnit (lldb_private::CompileUnit *comp_unit);

    DWARFCompileUnit*
    GetNextUnparsedDWARFCompileUnit (DWARFCompileUnit* prev_cu);

    bool
    GetFunction (const DWARFDIE &die,
                 lldb_private::SymbolContext& sc);

    lldb_private::Function *
    ParseCompileUnitFunction (const lldb_private::SymbolContext& sc,
                              const DWARFDIE &die);

    size_t
    ParseFunctionBlocks (const lldb_private::SymbolContext& sc,
                         lldb_private::Block *parent_block,
                         const DWARFDIE &die,
                         lldb::addr_t subprogram_low_pc,
                         uint32_t depth);

    size_t
    ParseTypes (const lldb_private::SymbolContext& sc,
                const DWARFDIE &die,
                bool parse_siblings,
                bool parse_children);

    lldb::TypeSP
    ParseType (const lldb_private::SymbolContext& sc,
               const DWARFDIE &die,
               bool *type_is_new);

    lldb_private::Type *
    ResolveTypeUID (const DWARFDIE &die,
                    bool assert_not_being_parsed);

    lldb::VariableSP
    ParseVariableDIE(const lldb_private::SymbolContext& sc,
                     const DWARFDIE &die,
                     const lldb::addr_t func_low_pc);

    size_t
    ParseVariables (const lldb_private::SymbolContext& sc,
                    const DWARFDIE &orig_die,
                    const lldb::addr_t func_low_pc,
                    bool parse_siblings,
                    bool parse_children,
                    lldb_private::VariableList* cc_variable_list = NULL);

    bool
    ClassOrStructIsVirtual (const DWARFDIE &die);

    // Given a die_offset, figure out the symbol context representing that die.
    bool
    ResolveFunction (const DIERef& die_ref,
                     bool include_inlines,
                     lldb_private::SymbolContextList& sc_list);

    bool
    ResolveFunction (const DWARFDIE &die,
                     bool include_inlines,
                     lldb_private::SymbolContextList& sc_list);

    void
    FindFunctions(const lldb_private::ConstString &name,
                  const NameToDIE &name_to_die,
                  bool include_inlines,
                  lldb_private::SymbolContextList& sc_list);

    void
    FindFunctions (const lldb_private::RegularExpression &regex,
                   const NameToDIE &name_to_die,
                   bool include_inlines,
                   lldb_private::SymbolContextList& sc_list);

    void
    FindFunctions (const lldb_private::RegularExpression &regex,
                   const DWARFMappedHash::MemoryTable &memory_table,
                   bool include_inlines,
                   lldb_private::SymbolContextList& sc_list);

    virtual lldb::TypeSP
    FindDefinitionTypeForDWARFDeclContext (const DWARFDeclContext &die_decl_ctx);

    lldb::TypeSP
    FindCompleteObjCDefinitionTypeForDIE (const DWARFDIE &die,
                                          const lldb_private::ConstString &type_name,
                                          bool must_be_implementation);

    lldb::TypeSP
    FindCompleteObjCDefinitionType (const lldb_private::ConstString &type_name,
                                    bool header_definition_ok);

    lldb_private::Symbol *
    GetObjCClassSymbol (const lldb_private::ConstString &objc_class_name);

    void
    ParseFunctions (const DIEArray &die_offsets,
                    bool include_inlines,
                    lldb_private::SymbolContextList& sc_list);

    lldb::TypeSP
    GetTypeForDIE (const DWARFDIE &die);

    void
    Index();
    
    void
    DumpIndexes();

    void
    SetDebugMapModule (const lldb::ModuleSP &module_sp)
    {
        m_debug_map_module_wp = module_sp;
    }

    SymbolFileDWARFDebugMap *
    GetDebugMapSymfile ();

    DWARFDIE
    FindBlockContainingSpecification (const DIERef& func_die_ref, dw_offset_t spec_block_die_offset);

    DWARFDIE
    FindBlockContainingSpecification (const DWARFDIE &die, dw_offset_t spec_block_die_offset);
    
    UniqueDWARFASTTypeMap &
    GetUniqueDWARFASTTypeMap ();
    
    bool
    UserIDMatches (lldb::user_id_t uid) const
    {
        const lldb::user_id_t high_uid = uid & 0xffffffff00000000ull;
        if (high_uid != 0 && GetID() != 0)
            return high_uid == GetID();
        return true;
    }
    
    bool
    DIEDeclContextsMatch (const DWARFDIE &die1,
                          const DWARFDIE &die2);
    
    bool
    ClassContainsSelector (const DWARFDIE &class_die,
                           const lldb_private::ConstString &selector);

    bool
    FixupAddress (lldb_private::Address &addr);

    typedef std::set<lldb_private::Type *> TypeSet;
    
    typedef struct {
        lldb_private::ConstString   m_name;
        lldb::ModuleSP              m_module_sp;
    } ClangModuleInfo;
    
    typedef std::map<uint64_t, ClangModuleInfo> ExternalTypeModuleMap;

    void
    GetTypes (const DWARFDIE &die,
              dw_offset_t min_die_offset,
              dw_offset_t max_die_offset,
              uint32_t type_mask,
              TypeSet &type_set);

    typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t, lldb_private::Variable *> GlobalVariableMap;

    GlobalVariableMap &
    GetGlobalAranges();
    
    void
    UpdateExternalModuleListIfNeeded();

    virtual DIEToTypePtr&
    GetDIEToType() { return m_die_to_type; }

    virtual DIEToVariableSP&
    GetDIEToVariable() { return m_die_to_variable_sp; }
    
    virtual DIEToClangType&
    GetForwardDeclDieToClangType() { return m_forward_decl_die_to_clang_type; }

    virtual ClangTypeToDIE&
    GetForwardDeclClangTypeToDie() { return m_forward_decl_clang_type_to_die; }

    lldb::ModuleWP                        m_debug_map_module_wp;
    SymbolFileDWARFDebugMap *             m_debug_map_symfile;
    lldb_private::Flags                   m_flags;
    lldb_private::DWARFDataExtractor      m_dwarf_data;
    lldb_private::DWARFDataExtractor      m_data_debug_abbrev;
    lldb_private::DWARFDataExtractor      m_data_debug_addr;
    lldb_private::DWARFDataExtractor      m_data_debug_aranges;
    lldb_private::DWARFDataExtractor      m_data_debug_frame;
    lldb_private::DWARFDataExtractor      m_data_debug_info;
    lldb_private::DWARFDataExtractor      m_data_debug_line;
    lldb_private::DWARFDataExtractor      m_data_debug_loc;
    lldb_private::DWARFDataExtractor      m_data_debug_ranges;
    lldb_private::DWARFDataExtractor      m_data_debug_str;
    lldb_private::DWARFDataExtractor      m_data_debug_str_offsets;
    lldb_private::DWARFDataExtractor      m_data_apple_names;
    lldb_private::DWARFDataExtractor      m_data_apple_types;
    lldb_private::DWARFDataExtractor      m_data_apple_namespaces;
    lldb_private::DWARFDataExtractor      m_data_apple_objc;

    // The unique pointer items below are generated on demand if and when someone accesses
    // them through a non const version of this class.
    std::unique_ptr<DWARFDebugAbbrev>     m_abbr;
    std::unique_ptr<DWARFDebugInfo>       m_info;
    std::unique_ptr<DWARFDebugLine>       m_line;
    std::unique_ptr<DWARFMappedHash::MemoryTable> m_apple_names_ap;
    std::unique_ptr<DWARFMappedHash::MemoryTable> m_apple_types_ap;
    std::unique_ptr<DWARFMappedHash::MemoryTable> m_apple_namespaces_ap;
    std::unique_ptr<DWARFMappedHash::MemoryTable> m_apple_objc_ap;
    std::unique_ptr<GlobalVariableMap>  m_global_aranges_ap;
    ExternalTypeModuleMap               m_external_type_modules;
    NameToDIE                           m_function_basename_index;  // All concrete functions
    NameToDIE                           m_function_fullname_index;  // All concrete functions
    NameToDIE                           m_function_method_index;    // All inlined functions
    NameToDIE                           m_function_selector_index;  // All method names for functions of classes
    NameToDIE                           m_objc_class_selectors_index; // Given a class name, find all selectors for the class
    NameToDIE                           m_global_index;             // Global and static variables
    NameToDIE                           m_type_index;               // All type DIE offsets
    NameToDIE                           m_namespace_index;          // All type DIE offsets
    bool                                m_indexed:1,
                                        m_using_apple_tables:1,
                                        m_fetched_external_modules:1;
    lldb_private::LazyBool              m_supports_DW_AT_APPLE_objc_complete_type;

    std::unique_ptr<DWARFDebugRanges>     m_ranges;
    UniqueDWARFASTTypeMap m_unique_ast_type_map;
    DIEToTypePtr m_die_to_type;
    DIEToVariableSP m_die_to_variable_sp;
    DIEToClangType m_forward_decl_die_to_clang_type;
    ClangTypeToDIE m_forward_decl_clang_type_to_die;
};

#endif  // SymbolFileDWARF_SymbolFileDWARF_h_
