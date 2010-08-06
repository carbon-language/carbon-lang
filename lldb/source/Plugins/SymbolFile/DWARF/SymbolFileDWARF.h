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
#include "llvm/ADT/DenseMap.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolContext.h"

// Project includes
#include "DWARFDefines.h"


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

    virtual lldb_private::Type*     ResolveTypeUID(lldb::user_id_t type_uid);
    virtual clang::DeclContext* GetClangDeclContextForTypeUID (lldb::user_id_t type_uid);

    virtual uint32_t        ResolveSymbolContext (const lldb_private::Address& so_addr, uint32_t resolve_scope, lldb_private::SymbolContext& sc);
    virtual uint32_t        ResolveSymbolContext (const lldb_private::FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindGlobalVariables(const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindGlobalVariables(const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindFunctions(const lldb_private::ConstString &name, uint32_t name_type_mask, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindFunctions(const lldb_private::RegularExpression& regex, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindTypes (const lldb_private::SymbolContext& sc, const lldb_private::ConstString &name, bool append, uint32_t max_matches, lldb_private::TypeList& types);
//  virtual uint32_t        FindTypes(const lldb_private::SymbolContext& sc, const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb::Type::Encoding encoding, lldb::user_id_t udt_uid, lldb_private::TypeList& types);


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

    // Approach 2 - count + accessor
    // Index compile units would scan the initial compile units and register
    // them with the module. This would only be done on demand if and only if
    // the compile units were needed.
    //virtual size_t        GetCompUnitCount() = 0;
    //virtual CompUnitSP    GetCompUnitAtIndex(size_t cu_idx) = 0;

    const lldb_private::DataExtractor&      get_debug_abbrev_data();
    const lldb_private::DataExtractor&      get_debug_aranges_data();
    const lldb_private::DataExtractor&      get_debug_frame_data();
    const lldb_private::DataExtractor&      get_debug_info_data();
    const lldb_private::DataExtractor&      get_debug_line_data();
    const lldb_private::DataExtractor&      get_debug_loc_data();
    const lldb_private::DataExtractor&      get_debug_macinfo_data();
    const lldb_private::DataExtractor&      get_debug_pubnames_data();
    const lldb_private::DataExtractor&      get_debug_pubtypes_data();
    const lldb_private::DataExtractor&      get_debug_ranges_data();
    const lldb_private::DataExtractor&      get_debug_str_data();

    DWARFDebugAbbrev*       DebugAbbrev();
    const DWARFDebugAbbrev* DebugAbbrev() const;

    DWARFDebugAranges*      DebugAranges();
    const DWARFDebugAranges*DebugAranges() const;

    DWARFDebugInfo*         DebugInfo();
    const DWARFDebugInfo*   DebugInfo() const;

//  These shouldn't be used unless we want to dump the DWARF line tables.
//  DWARFDebugLine*         DebugLine();
//  const DWARFDebugLine*   DebugLine() const;

//    DWARFDebugPubnames*     DebugPubnames();
//    const DWARFDebugPubnames* DebugPubnames() const;
//
//    DWARFDebugPubnames*     DebugPubBaseTypes();
//    const DWARFDebugPubnames* DebugPubBaseTypes() const;
//
    DWARFDebugPubnames*     DebugPubtypes();
//    const DWARFDebugPubnames* DebugPubtypes() const;

    DWARFDebugRanges*       DebugRanges();
    const DWARFDebugRanges* DebugRanges() const;

    const lldb_private::DataExtractor&
    GetCachedSectionData (uint32_t got_flag, lldb::SectionType sect_type, lldb_private::DataExtractor &data);

    static bool
    SupportedVersion(uint16_t version);

    clang::DeclContext *
    GetClangDeclContextForDIE (const DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die);

    clang::DeclContext *
    GetClangDeclContextForDIEOffset (dw_offset_t die_offset);

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
        // True if this is a .o file used when resolving a N_OSO entry with
        // debug maps.
        flagsDWARFIsOSOForDebugMap  = (1 << 16)
    };

    DISALLOW_COPY_AND_ASSIGN (SymbolFileDWARF);
    bool                    ParseCompileUnit(DWARFCompileUnit* cu, lldb::CompUnitSP& compile_unit_sp);
    DWARFCompileUnit*       GetDWARFCompileUnitForUID(lldb::user_id_t cu_uid);
    DWARFCompileUnit*       GetNextUnparsedDWARFCompileUnit(DWARFCompileUnit* prev_cu);
    lldb_private::CompileUnit*      GetCompUnitForDWARFCompUnit(DWARFCompileUnit* cu, uint32_t cu_idx = UINT_MAX);
    bool                    GetFunction (DWARFCompileUnit* cu, const DWARFDebugInfoEntry* func_die, lldb_private::SymbolContext& sc);
    lldb_private::Function *        ParseCompileUnitFunction (const lldb_private::SymbolContext& sc, const DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die);
    size_t                  ParseFunctionBlocks (const lldb_private::SymbolContext& sc,
                                                 lldb::user_id_t parentBlockID,
                                                 const DWARFCompileUnit* dwarf_cu,
                                                 const DWARFDebugInfoEntry *die,
                                                 lldb::addr_t subprogram_low_pc,
                                                 bool parse_siblings,
                                                 bool parse_children);
    size_t                  ParseTypes (const lldb_private::SymbolContext& sc, const DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool parse_siblings, bool parse_children);
    lldb::TypeSP            ParseType (const lldb_private::SymbolContext& sc, const DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool &type_is_new);

    lldb::VariableSP        ParseVariableDIE(
                                const lldb_private::SymbolContext& sc,
                                const DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *die);

    size_t                  ParseVariables(
                                const lldb_private::SymbolContext& sc,
                                const DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *die,
                                bool parse_siblings,
                                bool parse_children,
                                lldb_private::VariableList* cc_variable_list = NULL);

    size_t                  ParseChildMembers(
                                const lldb_private::SymbolContext& sc,
                                lldb::TypeSP& type_sp,
                                const DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *die,
                                void *class_clang_type,
                                const lldb::LanguageType class_language,
                                std::vector<clang::CXXBaseSpecifier *>& base_classes,
                                std::vector<int>& member_accessibilities,
                                lldb::AccessType &default_accessibility,
                                bool &is_a_class);

    size_t                  ParseChildParameters(
                                const lldb_private::SymbolContext& sc,
                                lldb::TypeSP& type_sp,
                                const DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *parent_die,
                                lldb_private::TypeList* type_list,
                                std::vector<void *>& function_args,
                                std::vector<clang::ParmVarDecl*>& function_param_decls);

    size_t                  ParseChildEnumerators(
                                const lldb_private::SymbolContext& sc,
                                lldb::TypeSP& type_sp,
                                void *enumerator_qual_type,
                                uint32_t enumerator_byte_size,
                                const DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *enum_die);

    void                    ParseChildArrayInfo(
                                const lldb_private::SymbolContext& sc,
                                const DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *parent_die,
                                int64_t& first_index,
                                std::vector<uint64_t>& element_orders,
                                uint32_t& byte_stride,
                                uint32_t& bit_stride);

    void                    FindFunctions(
                                const lldb_private::ConstString &name, 
                                lldb_private::UniqueCStringMap<dw_offset_t> &name_to_die,
                                lldb_private::SymbolContextList& sc_list);

    lldb_private::Type*     GetUniquedTypeForDIEOffset(dw_offset_t type_die_offset, lldb::TypeSP& owning_type_sp, int32_t child_type, uint32_t idx, bool safe);
    lldb::TypeSP            GetTypeForDIE(DWARFCompileUnit *cu, const DWARFDebugInfoEntry* die, lldb::TypeSP& owning_type_sp, int32_t child_type, uint32_t idx);
    uint32_t                FindTypes(std::vector<dw_offset_t> die_offsets, uint32_t max_matches, lldb_private::TypeList& types);

    void                    Index();

    lldb_private::Flags             m_flags;
    lldb_private::DataExtractor     m_dwarf_data; 
    lldb_private::DataExtractor     m_data_debug_abbrev;
    lldb_private::DataExtractor     m_data_debug_aranges;
    lldb_private::DataExtractor     m_data_debug_frame;
    lldb_private::DataExtractor     m_data_debug_info;
    lldb_private::DataExtractor     m_data_debug_line;
    lldb_private::DataExtractor     m_data_debug_loc;
    lldb_private::DataExtractor     m_data_debug_macinfo;
    lldb_private::DataExtractor     m_data_debug_pubnames;
    lldb_private::DataExtractor     m_data_debug_pubtypes;
    lldb_private::DataExtractor     m_data_debug_ranges;
    lldb_private::DataExtractor     m_data_debug_str;

    // The auto_ptr items below are generated on demand if and when someone accesses
    // them through a non const version of this class.
    std::auto_ptr<DWARFDebugAbbrev>     m_abbr;
    std::auto_ptr<DWARFDebugAranges>    m_aranges;
    std::auto_ptr<DWARFDebugInfo>       m_info;
    std::auto_ptr<DWARFDebugLine>       m_line;
    lldb_private::UniqueCStringMap<dw_offset_t> m_base_name_to_function_die; // All concrete functions
    lldb_private::UniqueCStringMap<dw_offset_t> m_full_name_to_function_die; // All concrete functions
    lldb_private::UniqueCStringMap<dw_offset_t> m_method_name_to_function_die;  // All inlined functions
    lldb_private::UniqueCStringMap<dw_offset_t> m_selector_name_to_function_die;   // All method names for functions of classes
    lldb_private::UniqueCStringMap<dw_offset_t> m_name_to_global_die;   // Global and static variables
    lldb_private::UniqueCStringMap<dw_offset_t> m_name_to_type_die;     // All type DIE offsets
    bool m_indexed;

//    std::auto_ptr<DWARFDebugPubnames>   m_pubnames;
//    std::auto_ptr<DWARFDebugPubnames>   m_pubbasetypes; // Just like m_pubtypes, but for DW_TAG_base_type DIEs
    std::auto_ptr<DWARFDebugPubnames>   m_pubtypes;
    std::auto_ptr<DWARFDebugRanges>     m_ranges;

    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, clang::DeclContext *> DIEToDeclContextMap;
    DIEToDeclContextMap m_die_to_decl_ctx;
    
//  TypeFixupColl   m_type_fixups;
//  std::vector<Type*> m_indirect_fixups;

//#define LLDB_SYMBOL_FILE_DWARF_SHRINK_TEST 1
#if defined(LLDB_SYMBOL_FILE_DWARF_SHRINK_TEST)

    typedef std::map<FileSpec, DWARFDIECollection> FSToDIES;
    void ShrinkDSYM(CompileUnit *dc_cu, DWARFCompileUnit *dw_cu, const FileSpec& cu_fspec, const FileSpec& base_types_cu_fspec, FSToDIES& fs_to_dies, const DWARFDebugInfoEntry *die);
#endif
};

#endif  // liblldb_SymbolFileDWARF_h_
