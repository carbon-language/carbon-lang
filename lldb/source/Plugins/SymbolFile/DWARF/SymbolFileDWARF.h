//===-- SymbolFileDWARF.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolFileDWARF_h_
#define liblldb_SymbolFileDWARF_h_

// C Includes
// C++ Includes
#include <list>
#include <memory>
#include <map>
#include <vector>

// Other libraries and framework includes
#include "clang/AST/ExternalASTSource.h"
#include "llvm/ADT/DenseMap.h"

#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolContext.h"

// Project includes
#include "DWARFDefines.h"
#include "NameToDIE.h"
#include "UniqueDWARFASTType.h"


//----------------------------------------------------------------------
// Forward Declarations for this DWARF plugin
//----------------------------------------------------------------------
class DWARFAbbreviationDeclaration;
class DWARFAbbreviationDeclarationSet;
class DWARFCompileUnit;
class DWARFDebugAbbrev;
class DWARFDebugAranges;
class DWARFDebugInfo;
class DWARFDebugInfoEntry;
class DWARFDebugLine;
class DWARFDebugPubnames;
class DWARFDebugRanges;
class DWARFDIECollection;
class DWARFFormValue;
class SymbolFileDWARFDebugMap;

class SymbolFileDWARF : public lldb_private::SymbolFile
{
public:
    friend class SymbolFileDWARFDebugMap;

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
                            SymbolFileDWARF(lldb_private::ObjectFile* ofile);
    virtual                 ~SymbolFileDWARF();

    virtual uint32_t        GetAbilities ();
    virtual void            InitializeObject();

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    virtual uint32_t        GetNumCompileUnits();
    virtual lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index);

    virtual size_t          ParseCompileUnitFunctions (const lldb_private::SymbolContext& sc);
    virtual bool            ParseCompileUnitLineTable (const lldb_private::SymbolContext& sc);
    virtual bool            ParseCompileUnitSupportFiles (const lldb_private::SymbolContext& sc, lldb_private::FileSpecList& support_files);
    virtual size_t          ParseFunctionBlocks (const lldb_private::SymbolContext& sc);
    virtual size_t          ParseTypes (const lldb_private::SymbolContext& sc);
    virtual size_t          ParseVariablesForContext (const lldb_private::SymbolContext& sc);

    virtual lldb_private::Type* ResolveTypeUID(lldb::user_id_t type_uid);
    virtual lldb::clang_type_t ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_opaque_type);

    virtual lldb_private::Type* ResolveType (DWARFCompileUnit* cu, const DWARFDebugInfoEntry* type_die, bool assert_not_being_parsed = true);
    virtual clang::DeclContext* GetClangDeclContextContainingTypeUID (lldb::user_id_t type_uid);
    virtual clang::DeclContext* GetClangDeclContextForTypeUID (const lldb_private::SymbolContext &sc, lldb::user_id_t type_uid);

    virtual uint32_t        ResolveSymbolContext (const lldb_private::Address& so_addr, uint32_t resolve_scope, lldb_private::SymbolContext& sc);
    virtual uint32_t        ResolveSymbolContext (const lldb_private::FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindGlobalVariables(const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindGlobalVariables(const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindFunctions(const lldb_private::ConstString &name, uint32_t name_type_mask, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindFunctions(const lldb_private::RegularExpression& regex, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindTypes (const lldb_private::SymbolContext& sc, const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::TypeList& types);
    virtual lldb_private::TypeList *
                            GetTypeList ();
    virtual lldb_private::ClangASTContext &
                            GetClangASTContext ();

    virtual lldb_private::ClangNamespaceDecl
            FindNamespace (const lldb_private::SymbolContext& sc, 
                           const lldb_private::ConstString &name);


    //------------------------------------------------------------------
    // ClangASTContext callbacks for external source lookups.
    //------------------------------------------------------------------
    static void
    CompleteTagDecl (void *baton, clang::TagDecl *);
    
    static void
    CompleteObjCInterfaceDecl (void *baton, clang::ObjCInterfaceDecl *);
    
    static void
    FindExternalVisibleDeclsByName (void *baton,
                                    const clang::DeclContext *DC,
                                    clang::DeclarationName Name,
                                    llvm::SmallVectorImpl <clang::NamedDecl *> *results);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

    // Approach 2 - count + accessor
    // Index compile units would scan the initial compile units and register
    // them with the module. This would only be done on demand if and only if
    // the compile units were needed.
    //virtual size_t        GetCompUnitCount() = 0;
    //virtual CompUnitSP    GetCompUnitAtIndex(size_t cu_idx) = 0;

    const lldb_private::DataExtractor&      get_debug_abbrev_data();
    const lldb_private::DataExtractor&      get_debug_frame_data();
    const lldb_private::DataExtractor&      get_debug_info_data();
    const lldb_private::DataExtractor&      get_debug_line_data();
    const lldb_private::DataExtractor&      get_debug_loc_data();
    const lldb_private::DataExtractor&      get_debug_ranges_data();
    const lldb_private::DataExtractor&      get_debug_str_data();

    DWARFDebugAbbrev*       DebugAbbrev();
    const DWARFDebugAbbrev* DebugAbbrev() const;

    DWARFDebugAranges*      DebugAranges();
    const DWARFDebugAranges*DebugAranges() const;

    DWARFDebugInfo*         DebugInfo();
    const DWARFDebugInfo*   DebugInfo() const;

    DWARFDebugRanges*       DebugRanges();
    const DWARFDebugRanges* DebugRanges() const;

    const lldb_private::DataExtractor&
    GetCachedSectionData (uint32_t got_flag, 
                          lldb::SectionType sect_type, 
                          lldb_private::DataExtractor &data);

    static bool
    SupportedVersion(uint16_t version);

    clang::DeclContext *
    GetClangDeclContextForDIE (const lldb_private::SymbolContext &sc, DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die);
    
    clang::DeclContext *
    GetClangDeclContextForDIEOffset (const lldb_private::SymbolContext &sc, dw_offset_t die_offset);
    
    clang::DeclContext *
    GetClangDeclContextContainingDIE (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die);

    clang::DeclContext *
    GetClangDeclContextContainingDIEOffset (dw_offset_t die_offset);
    
    void
    SearchDeclContext (const clang::DeclContext *decl_context, 
                       const char *name, 
                       llvm::SmallVectorImpl <clang::NamedDecl *> *results);
    
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
    HasForwardDeclForClangType (lldb::clang_type_t clang_type);

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
        flagsGotDebugStrData        = (1 << 10)
    };

    DISALLOW_COPY_AND_ASSIGN (SymbolFileDWARF);
    bool                    ParseCompileUnit (DWARFCompileUnit* cu, lldb::CompUnitSP& compile_unit_sp);
    DWARFCompileUnit*       GetDWARFCompileUnitForUID(lldb::user_id_t cu_uid);
    DWARFCompileUnit*       GetNextUnparsedDWARFCompileUnit(DWARFCompileUnit* prev_cu);
    lldb_private::CompileUnit*      GetCompUnitForDWARFCompUnit(DWARFCompileUnit* cu, uint32_t cu_idx = UINT32_MAX);
    bool                    GetFunction (DWARFCompileUnit* cu, const DWARFDebugInfoEntry* func_die, lldb_private::SymbolContext& sc);
    lldb_private::Function *        ParseCompileUnitFunction (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die);
    size_t                  ParseFunctionBlocks (const lldb_private::SymbolContext& sc,
                                                 lldb_private::Block *parent_block,
                                                 DWARFCompileUnit* dwarf_cu,
                                                 const DWARFDebugInfoEntry *die,
                                                 lldb::addr_t subprogram_low_pc,
                                                 bool parse_siblings,
                                                 bool parse_children);
    size_t                  ParseTypes (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool parse_siblings, bool parse_children);
    lldb::TypeSP            ParseType (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool *type_is_new);

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

    size_t                  ParseChildMembers(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *die,
                                lldb::clang_type_t class_clang_type,
                                const lldb::LanguageType class_language,
                                std::vector<clang::CXXBaseSpecifier *>& base_classes,
                                std::vector<int>& member_accessibilities,
                                DWARFDIECollection& member_function_dies,
                                lldb::AccessType &default_accessibility,
                                bool &is_a_class);

    size_t                  ParseChildParameters(
                                const lldb_private::SymbolContext& sc,
                                lldb::TypeSP& type_sp,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *parent_die,
                                bool skip_artificial,
                                bool &is_static,
                                lldb_private::TypeList* type_list,
                                std::vector<lldb::clang_type_t>& function_args,
                                std::vector<clang::ParmVarDecl*>& function_param_decls,
                                unsigned &type_quals);

    size_t                  ParseChildEnumerators(
                                const lldb_private::SymbolContext& sc,
                                lldb::clang_type_t enumerator_qual_type,
                                uint32_t enumerator_byte_size,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *enum_die);

    void                    ParseChildArrayInfo(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *parent_die,
                                int64_t& first_index,
                                std::vector<uint64_t>& element_orders,
                                uint32_t& byte_stride,
                                uint32_t& bit_stride);

    void                    FindFunctions(
                                const lldb_private::ConstString &name, 
                                const NameToDIE &name_to_die,
                                lldb_private::SymbolContextList& sc_list);

    void                    FindFunctions (
                                const lldb_private::RegularExpression &regex, 
                                const NameToDIE &name_to_die,
                                lldb_private::SymbolContextList& sc_list);

    lldb::TypeSP            FindDefinitionTypeForDIE (
                                DWARFCompileUnit* cu, 
                                const DWARFDebugInfoEntry *die, 
                                const lldb_private::ConstString &type_name);
    
    lldb::TypeSP            GetTypeForDIE (DWARFCompileUnit *cu, 
                                           const DWARFDebugInfoEntry* die);

    uint32_t                FindTypes(std::vector<dw_offset_t> die_offsets, uint32_t max_matches, lldb_private::TypeList& types);

    void                    Index();
    
    void                    DumpIndexes();

    void                    SetDebugMapSymfile (SymbolFileDWARFDebugMap *debug_map_symfile)
                            {
                                m_debug_map_symfile = debug_map_symfile;
                            }

    const DWARFDebugInfoEntry *
                            FindBlockContainingSpecification (dw_offset_t func_die_offset, 
                                                              dw_offset_t spec_block_die_offset,
                                                              DWARFCompileUnit **dwarf_cu_handle);

    const DWARFDebugInfoEntry *
                            FindBlockContainingSpecification (DWARFCompileUnit* dwarf_cu,
                                                              const DWARFDebugInfoEntry *die,
                                                              dw_offset_t spec_block_die_offset,
                                                              DWARFCompileUnit **dwarf_cu_handle);

    clang::NamespaceDecl *
    ResolveNamespaceDIE (DWARFCompileUnit *curr_cu, const DWARFDebugInfoEntry *die);
    
    UniqueDWARFASTTypeMap &
    GetUniqueDWARFASTTypeMap ();

    void                    LinkDeclContextToDIE (clang::DeclContext *decl_ctx,
                                                  const DWARFDebugInfoEntry *die)
                            {
                                m_die_to_decl_ctx[die] = decl_ctx;
                                m_decl_ctx_to_die[decl_ctx] = die;
                            }
    
    void
    ReportError (const char *format, ...);
    
    SymbolFileDWARFDebugMap *       m_debug_map_symfile;
    clang::TranslationUnitDecl *    m_clang_tu_decl;
    lldb_private::Flags             m_flags;
    lldb_private::DataExtractor     m_dwarf_data; 
    lldb_private::DataExtractor     m_data_debug_abbrev;
    lldb_private::DataExtractor     m_data_debug_frame;
    lldb_private::DataExtractor     m_data_debug_info;
    lldb_private::DataExtractor     m_data_debug_line;
    lldb_private::DataExtractor     m_data_debug_loc;
    lldb_private::DataExtractor     m_data_debug_ranges;
    lldb_private::DataExtractor     m_data_debug_str;

    // The auto_ptr items below are generated on demand if and when someone accesses
    // them through a non const version of this class.
    std::auto_ptr<DWARFDebugAbbrev>     m_abbr;
    std::auto_ptr<DWARFDebugAranges>    m_aranges;
    std::auto_ptr<DWARFDebugInfo>       m_info;
    std::auto_ptr<DWARFDebugLine>       m_line;
    NameToDIE                           m_function_basename_index;  // All concrete functions
    NameToDIE                           m_function_fullname_index;  // All concrete functions
    NameToDIE                           m_function_method_index;    // All inlined functions
    NameToDIE                           m_function_selector_index;  // All method names for functions of classes
    NameToDIE                           m_objc_class_selectors_index; // Given a class name, find all selectors for the class
    NameToDIE                           m_global_index;                 // Global and static variables
    NameToDIE                           m_type_index;                  // All type DIE offsets
    NameToDIE                           m_namespace_index;              // All type DIE offsets
    bool m_indexed:1,
         m_is_external_ast_source:1;

    std::auto_ptr<DWARFDebugRanges>     m_ranges;
    UniqueDWARFASTTypeMap m_unique_ast_type_map;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, clang::DeclContext *> DIEToDeclContextMap;
    typedef llvm::DenseMap<const clang::DeclContext *, const DWARFDebugInfoEntry *> DeclContextToDIEMap;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb_private::Type *> DIEToTypePtr;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::VariableSP> DIEToVariableSP;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::clang_type_t> DIEToClangType;
    typedef llvm::DenseMap<lldb::clang_type_t, const DWARFDebugInfoEntry *> ClangTypeToDIE;
    DIEToDeclContextMap m_die_to_decl_ctx;
    DeclContextToDIEMap m_decl_ctx_to_die;
    DIEToTypePtr m_die_to_type;
    DIEToVariableSP m_die_to_variable_sp;
    DIEToClangType m_forward_decl_die_to_clang_type;
    ClangTypeToDIE m_forward_decl_clang_type_to_die;
};

#endif  // liblldb_SymbolFileDWARF_h_
