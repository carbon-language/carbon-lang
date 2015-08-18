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
#include "clang/AST/CharUnits.h"
#include "clang/AST/ExternalASTSource.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Core/UniqueCStringMap.h"
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
    friend class DebugMapModule;
    friend class DWARFCompileUnit;
    friend class lldb_private::ClangASTContext;
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

    uint32_t        CalculateAbilities () override;
    void            InitializeObject() override;

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    uint32_t        GetNumCompileUnits() override;
    lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index) override;

    lldb::LanguageType ParseCompileUnitLanguage (const lldb_private::SymbolContext& sc) override;
    size_t          ParseCompileUnitFunctions (const lldb_private::SymbolContext& sc) override;
    bool            ParseCompileUnitLineTable (const lldb_private::SymbolContext& sc) override;
    bool            ParseCompileUnitSupportFiles (const lldb_private::SymbolContext& sc, lldb_private::FileSpecList& support_files) override;
    bool            ParseImportedModules (const lldb_private::SymbolContext &sc, std::vector<lldb_private::ConstString> &imported_modules) override;
    size_t          ParseFunctionBlocks (const lldb_private::SymbolContext& sc) override;
    size_t          ParseTypes (const lldb_private::SymbolContext& sc) override;
    size_t          ParseVariablesForContext (const lldb_private::SymbolContext& sc) override;

    lldb_private::Type* ResolveTypeUID(lldb::user_id_t type_uid) override;
    bool            CompleteType (lldb_private::CompilerType& clang_type) override;

    lldb_private::Type* ResolveType (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* type_die, bool assert_not_being_parsed = true);
    lldb_private::Type* GetCachedTypeForDIE (const DWARFDebugInfoEntry* type_die) const;
    void                ClearDIEBeingParsed (const DWARFDebugInfoEntry* type_die);
    clang::DeclContext* GetClangDeclContextContainingTypeUID (lldb::user_id_t type_uid) override;
    clang::DeclContext* GetClangDeclContextForTypeUID (const lldb_private::SymbolContext &sc, lldb::user_id_t type_uid) override;

    uint32_t        ResolveSymbolContext (const lldb_private::Address& so_addr, uint32_t resolve_scope, lldb_private::SymbolContext& sc) override;
    uint32_t        ResolveSymbolContext (const lldb_private::FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, lldb_private::SymbolContextList& sc_list) override;
    uint32_t        FindGlobalVariables(const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, lldb_private::VariableList& variables) override;
    uint32_t        FindGlobalVariables(const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::VariableList& variables) override;
    uint32_t        FindFunctions(const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, uint32_t name_type_mask, bool include_inlines, bool append, lldb_private::SymbolContextList& sc_list) override;
    uint32_t        FindFunctions(const lldb_private::RegularExpression& regex, bool include_inlines, bool append, lldb_private::SymbolContextList& sc_list) override;
    uint32_t        FindTypes (const lldb_private::SymbolContext& sc, const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, lldb_private::TypeList& types) override;
    lldb_private::TypeList *
                            GetTypeList () override;
    size_t          GetTypes (lldb_private::SymbolContextScope *sc_scope,
                                      uint32_t type_mask,
                                      lldb_private::TypeList &type_list) override;

    lldb_private::ClangASTContext &
                    GetClangASTContext () override;

    lldb_private::TypeSystem *
                            GetTypeSystemForLanguage (lldb::LanguageType language) override;

    lldb_private::ClangNamespaceDecl
            FindNamespace (const lldb_private::SymbolContext& sc, 
                           const lldb_private::ConstString &name, 
                           const lldb_private::ClangNamespaceDecl *parent_namespace_decl) override;


    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    lldb_private::ConstString
    GetPluginName() override;

    uint32_t
    GetPluginVersion() override;

    const lldb_private::DWARFDataExtractor&     get_debug_abbrev_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_aranges_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_frame_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_info_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_line_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_loc_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_ranges_data ();
    const lldb_private::DWARFDataExtractor&     get_debug_str_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_names_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_types_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_namespaces_data ();
    const lldb_private::DWARFDataExtractor&     get_apple_objc_data ();


    DWARFDebugAbbrev*       DebugAbbrev();
    const DWARFDebugAbbrev* DebugAbbrev() const;

    DWARFDebugInfo*         DebugInfo();
    const DWARFDebugInfo*   DebugInfo() const;

    DWARFDebugRanges*       DebugRanges();
    const DWARFDebugRanges* DebugRanges() const;

    const lldb_private::DWARFDataExtractor&
    GetCachedSectionData (uint32_t got_flag, 
                          lldb::SectionType sect_type, 
                          lldb_private::DWARFDataExtractor &data);

    static bool
    SupportedVersion(uint16_t version);

    const DWARFDebugInfoEntry *
    GetDeclContextDIEContainingDIE (const DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die);

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
    GetCompUnitForDWARFCompUnit(DWARFCompileUnit* dwarf_cu, uint32_t cu_idx = UINT32_MAX);

    lldb::user_id_t
    MakeUserID (dw_offset_t die_offset) const
    {
        return GetID() | die_offset;
    }

    size_t
    GetObjCMethodDIEOffsets (lldb_private::ConstString class_name, DIEArray &method_die_offsets);

    bool
    Supports_DW_AT_APPLE_objc_complete_type (DWARFCompileUnit *cu);

    static const DWARFDebugInfoEntry *
    GetParentSymbolContextDIE(const DWARFDebugInfoEntry *child_die);

protected:

    enum
    {
        flagsGotDebugAbbrevData     = (1 << 0),
        flagsGotDebugArangesData    = (1 << 1),
        flagsGotDebugFrameData      = (1 << 2),
        flagsGotDebugInfoData       = (1 << 3),
        flagsGotDebugLineData       = (1 << 4),
        flagsGotDebugLocData        = (1 << 5),
        flagsGotDebugMacInfoData    = (1 << 6),
        flagsGotDebugPubNamesData   = (1 << 7),
        flagsGotDebugPubTypesData   = (1 << 8),
        flagsGotDebugRangesData     = (1 << 9),
        flagsGotDebugStrData        = (1 << 10),
        flagsGotAppleNamesData      = (1 << 11),
        flagsGotAppleTypesData      = (1 << 12),
        flagsGotAppleNamespacesData = (1 << 13),
        flagsGotAppleObjCData       = (1 << 14)
    };
    
    bool                    NamespaceDeclMatchesThisSymbolFile (const lldb_private::ClangNamespaceDecl *namespace_decl);

    DISALLOW_COPY_AND_ASSIGN (SymbolFileDWARF);
    lldb::CompUnitSP        ParseCompileUnit (DWARFCompileUnit* dwarf_cu, uint32_t cu_idx);
    DWARFCompileUnit*       GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit);
    DWARFCompileUnit*       GetNextUnparsedDWARFCompileUnit(DWARFCompileUnit* prev_cu);
    bool                    GetFunction (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* func_die, lldb_private::SymbolContext& sc);
    lldb_private::Function *        ParseCompileUnitFunction (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die);
    size_t                  ParseFunctionBlocks (const lldb_private::SymbolContext& sc,
                                                 lldb_private::Block *parent_block,
                                                 DWARFCompileUnit* dwarf_cu,
                                                 const DWARFDebugInfoEntry *die,
                                                 lldb::addr_t subprogram_low_pc,
                                                 uint32_t depth);
    size_t                  ParseTypes (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool parse_siblings, bool parse_children);
    lldb::TypeSP            ParseType (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool *type_is_new);
    lldb_private::Type*     ResolveTypeUID (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* die, bool assert_not_being_parsed);

    lldb::VariableSP        ParseVariableDIE(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *die,
                                const lldb::addr_t func_low_pc);

    size_t                  ParseVariables(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const lldb::addr_t func_low_pc,
                                const DWARFDebugInfoEntry *die,
                                bool parse_siblings,
                                bool parse_children,
                                lldb_private::VariableList* cc_variable_list = NULL);
    
    bool                    ClassOrStructIsVirtual (
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *parent_die);

                            // Given a die_offset, figure out the symbol context representing that die.
    bool                    ResolveFunction (dw_offset_t offset,
                                             DWARFCompileUnit *&dwarf_cu,
                                             bool include_inlines,
                                             lldb_private::SymbolContextList& sc_list);
                            
    bool                    ResolveFunction (DWARFCompileUnit *cu,
                                             const DWARFDebugInfoEntry *die,
                                             bool include_inlines,
                                             lldb_private::SymbolContextList& sc_list);

    bool                    FunctionDieMatchesPartialName (
                                const DWARFDebugInfoEntry* die,
                                const DWARFCompileUnit *dwarf_cu,
                                uint32_t name_type_mask,
                                const char *partial_name,
                                const char *base_name_start,
                                const char *base_name_end);

    void                    FindFunctions(
                                const lldb_private::ConstString &name, 
                                const NameToDIE &name_to_die,
                                bool include_inlines,
                                lldb_private::SymbolContextList& sc_list);

    void                    FindFunctions (
                                const lldb_private::RegularExpression &regex, 
                                const NameToDIE &name_to_die,
                                bool include_inlines,
                                lldb_private::SymbolContextList& sc_list);

    void                    FindFunctions (
                                const lldb_private::RegularExpression &regex, 
                                const DWARFMappedHash::MemoryTable &memory_table,
                                bool include_inlines,
                                lldb_private::SymbolContextList& sc_list);

    lldb::TypeSP            FindDefinitionTypeForDWARFDeclContext (
                                const DWARFDeclContext &die_decl_ctx);

    lldb::TypeSP            FindCompleteObjCDefinitionTypeForDIE (
                                const DWARFDebugInfoEntry *die, 
                                const lldb_private::ConstString &type_name,
                                bool must_be_implementation);

    lldb::TypeSP            FindCompleteObjCDefinitionType (const lldb_private::ConstString &type_name,
                                                            bool header_definition_ok);

    lldb_private::Symbol *  GetObjCClassSymbol (const lldb_private::ConstString &objc_class_name);

    void                    ParseFunctions (const DIEArray &die_offsets,
                                            bool include_inlines,
                                            lldb_private::SymbolContextList& sc_list);
    lldb::TypeSP            GetTypeForDIE (DWARFCompileUnit *cu, 
                                           const DWARFDebugInfoEntry* die);

    uint32_t                FindTypes(std::vector<dw_offset_t> die_offsets, uint32_t max_matches, lldb_private::TypeList& types);

    void                    Index();
    
    void                    DumpIndexes();

    void                    SetDebugMapModule (const lldb::ModuleSP &module_sp)
                            {
                                m_debug_map_module_wp = module_sp;
                            }
    
    SymbolFileDWARFDebugMap *
                            GetDebugMapSymfile ();

    const DWARFDebugInfoEntry *
                            FindBlockContainingSpecification (dw_offset_t func_die_offset, 
                                                              dw_offset_t spec_block_die_offset,
                                                              DWARFCompileUnit **dwarf_cu_handle);

    const DWARFDebugInfoEntry *
                            FindBlockContainingSpecification (DWARFCompileUnit* dwarf_cu,
                                                              const DWARFDebugInfoEntry *die,
                                                              dw_offset_t spec_block_die_offset,
                                                              DWARFCompileUnit **dwarf_cu_handle);
    
    UniqueDWARFASTTypeMap &
    GetUniqueDWARFASTTypeMap ();
    
    bool
    UserIDMatches (lldb::user_id_t uid) const
    {
        const lldb::user_id_t high_uid = uid & 0xffffffff00000000ull;
        if (high_uid)
            return high_uid == GetID();
        return true;
    }
    
    bool
    DIEDeclContextsMatch (DWARFCompileUnit* cu1, const DWARFDebugInfoEntry *die1,
                          DWARFCompileUnit* cu2, const DWARFDebugInfoEntry *die2);
    
    bool
    ClassContainsSelector (DWARFCompileUnit *dwarf_cu,
                           const DWARFDebugInfoEntry *class_die,
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
    GetTypes (DWARFCompileUnit* dwarf_cu,
              const DWARFDebugInfoEntry *die,
              dw_offset_t min_die_offset,
              dw_offset_t max_die_offset,
              uint32_t type_mask,
              TypeSet &type_set);

    typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t, lldb_private::Variable *> GlobalVariableMap;

    GlobalVariableMap &
    GetGlobalAranges();
    
    void
    UpdateExternalModuleListIfNeeded();

    lldb::ModuleWP                        m_debug_map_module_wp;
    SymbolFileDWARFDebugMap *             m_debug_map_symfile;
    lldb_private::Flags                   m_flags;
    lldb_private::DWARFDataExtractor      m_dwarf_data; 
    lldb_private::DWARFDataExtractor      m_data_debug_abbrev;
    lldb_private::DWARFDataExtractor      m_data_debug_aranges;
    lldb_private::DWARFDataExtractor      m_data_debug_frame;
    lldb_private::DWARFDataExtractor      m_data_debug_info;
    lldb_private::DWARFDataExtractor      m_data_debug_line;
    lldb_private::DWARFDataExtractor      m_data_debug_loc;
    lldb_private::DWARFDataExtractor      m_data_debug_ranges;
    lldb_private::DWARFDataExtractor      m_data_debug_str;
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
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb_private::Type *> DIEToTypePtr;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::VariableSP> DIEToVariableSP;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::clang_type_t> DIEToClangType;
    typedef llvm::DenseMap<lldb::clang_type_t, const DWARFDebugInfoEntry *> ClangTypeToDIE;
    DIEToTypePtr m_die_to_type;
    DIEToVariableSP m_die_to_variable_sp;
    DIEToClangType m_forward_decl_die_to_clang_type;
    ClangTypeToDIE m_forward_decl_clang_type_to_die;
};

#endif  // SymbolFileDWARF_SymbolFileDWARF_h_
