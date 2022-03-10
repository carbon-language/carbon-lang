//===-- SymbolFileDWARFDebugMap.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARFDEBUGMAP_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARFDEBUGMAP_H

#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Utility/RangeMap.h"
#include "llvm/Support/Chrono.h"
#include <bitset>
#include <map>
#include <vector>

#include "UniqueDWARFASTType.h"

class SymbolFileDWARF;
class DWARFDebugAranges;
class DWARFDeclContext;

class SymbolFileDWARFDebugMap : public lldb_private::SymbolFile {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SymbolFile::isA(ClassID);
  }
  static bool classof(const SymbolFile *obj) { return obj->isA(&ID); }
  /// \}

  // Static Functions
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "dwarf-debugmap"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::SymbolFile *
  CreateInstance(lldb::ObjectFileSP objfile_sp);

  // Constructors and Destructors
  SymbolFileDWARFDebugMap(lldb::ObjectFileSP objfile_sp);
  ~SymbolFileDWARFDebugMap() override;

  uint32_t CalculateAbilities() override;
  void InitializeObject() override;

  // Compile Unit function calls
  lldb::LanguageType
  ParseLanguage(lldb_private::CompileUnit &comp_unit) override;
  lldb_private::XcodeSDK
  ParseXcodeSDK(lldb_private::CompileUnit &comp_unit) override;
  size_t ParseFunctions(lldb_private::CompileUnit &comp_unit) override;
  bool ParseLineTable(lldb_private::CompileUnit &comp_unit) override;
  bool ParseDebugMacros(lldb_private::CompileUnit &comp_unit) override;

  bool ForEachExternalModule(
      lldb_private::CompileUnit &, llvm::DenseSet<lldb_private::SymbolFile *> &,
      llvm::function_ref<bool(lldb_private::Module &)>) override;

  bool ParseSupportFiles(lldb_private::CompileUnit &comp_unit,
                         lldb_private::FileSpecList &support_files) override;

  bool ParseIsOptimized(lldb_private::CompileUnit &comp_unit) override;

  size_t ParseTypes(lldb_private::CompileUnit &comp_unit) override;

  bool ParseImportedModules(
      const lldb_private::SymbolContext &sc,
      std::vector<lldb_private::SourceModule> &imported_modules) override;
  size_t ParseBlocksRecursive(lldb_private::Function &func) override;
  size_t
  ParseVariablesForContext(const lldb_private::SymbolContext &sc) override;

  lldb_private::Type *ResolveTypeUID(lldb::user_id_t type_uid) override;
  llvm::Optional<ArrayInfo> GetDynamicArrayInfoForUID(
      lldb::user_id_t type_uid,
      const lldb_private::ExecutionContext *exe_ctx) override;

  lldb_private::CompilerDeclContext
  GetDeclContextForUID(lldb::user_id_t uid) override;
  lldb_private::CompilerDeclContext
  GetDeclContextContainingUID(lldb::user_id_t uid) override;
  void
  ParseDeclsForContext(lldb_private::CompilerDeclContext decl_ctx) override;

  bool CompleteType(lldb_private::CompilerType &compiler_type) override;
  uint32_t ResolveSymbolContext(const lldb_private::Address &so_addr,
                                lldb::SymbolContextItem resolve_scope,
                                lldb_private::SymbolContext &sc) override;
  uint32_t ResolveSymbolContext(
      const lldb_private::SourceLocationSpec &src_location_spec,
      lldb::SymbolContextItem resolve_scope,
      lldb_private::SymbolContextList &sc_list) override;
  void
  FindGlobalVariables(lldb_private::ConstString name,
                      const lldb_private::CompilerDeclContext &parent_decl_ctx,
                      uint32_t max_matches,
                      lldb_private::VariableList &variables) override;
  void FindGlobalVariables(const lldb_private::RegularExpression &regex,
                           uint32_t max_matches,
                           lldb_private::VariableList &variables) override;
  void FindFunctions(lldb_private::ConstString name,
                     const lldb_private::CompilerDeclContext &parent_decl_ctx,
                     lldb::FunctionNameType name_type_mask,
                     bool include_inlines,
                     lldb_private::SymbolContextList &sc_list) override;
  void FindFunctions(const lldb_private::RegularExpression &regex,
                     bool include_inlines,
                     lldb_private::SymbolContextList &sc_list) override;
  void
  FindTypes(lldb_private::ConstString name,
            const lldb_private::CompilerDeclContext &parent_decl_ctx,
            uint32_t max_matches,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            lldb_private::TypeMap &types) override;
  void
  FindTypes(llvm::ArrayRef<lldb_private::CompilerContext> context,
            lldb_private::LanguageSet languages,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            lldb_private::TypeMap &types) override;
  lldb_private::CompilerDeclContext FindNamespace(
      lldb_private::ConstString name,
      const lldb_private::CompilerDeclContext &parent_decl_ctx) override;
  void GetTypes(lldb_private::SymbolContextScope *sc_scope,
                lldb::TypeClass type_mask,
                lldb_private::TypeList &type_list) override;
  std::vector<std::unique_ptr<lldb_private::CallEdge>>
  ParseCallEdgesInFunction(lldb_private::UserID func_id) override;

  void DumpClangAST(lldb_private::Stream &s) override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // Statistics overrides.
  lldb_private::ModuleList GetDebugInfoModules() override;

protected:
  enum { kHaveInitializedOSOs = (1 << 0), kNumFlags };

  friend class DebugMapModule;
  friend class DWARFASTParserClang;
  friend class DWARFCompileUnit;
  friend class SymbolFileDWARF;
  struct OSOInfo {
    lldb::ModuleSP module_sp;

    OSOInfo() : module_sp() {}
  };

  typedef std::shared_ptr<OSOInfo> OSOInfoSP;

  typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t,
                                        lldb::addr_t>
      FileRangeMap;

  // Class specific types
  struct CompileUnitInfo {
    lldb_private::FileSpec so_file;
    lldb_private::ConstString oso_path;
    llvm::sys::TimePoint<> oso_mod_time;
    OSOInfoSP oso_sp;
    lldb::CompUnitSP compile_unit_sp;
    uint32_t first_symbol_index = UINT32_MAX;
    uint32_t last_symbol_index = UINT32_MAX;
    uint32_t first_symbol_id = UINT32_MAX;
    uint32_t last_symbol_id = UINT32_MAX;
    FileRangeMap file_range_map;
    bool file_range_map_valid = false;

    CompileUnitInfo()
        : so_file(), oso_path(), oso_mod_time(), oso_sp(), compile_unit_sp(),

          file_range_map() {}

    const FileRangeMap &GetFileRangeMap(SymbolFileDWARFDebugMap *exe_symfile);
  };

  // Protected Member Functions
  void InitOSO();

  uint32_t CalculateNumCompileUnits() override;
  lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index) override;

  static uint32_t GetOSOIndexFromUserID(lldb::user_id_t uid) {
    return (uint32_t)((uid >> 32ull) - 1ull);
  }

  static SymbolFileDWARF *GetSymbolFileAsSymbolFileDWARF(SymbolFile *sym_file);

  bool GetFileSpecForSO(uint32_t oso_idx, lldb_private::FileSpec &file_spec);

  CompileUnitInfo *GetCompUnitInfo(const lldb_private::SymbolContext &sc);
  CompileUnitInfo *GetCompUnitInfo(const lldb_private::CompileUnit &comp_unit);

  size_t GetCompUnitInfosForModule(const lldb_private::Module *oso_module,
                                   std::vector<CompileUnitInfo *> &cu_infos);

  lldb_private::Module *
  GetModuleByCompUnitInfo(CompileUnitInfo *comp_unit_info);

  lldb_private::Module *GetModuleByOSOIndex(uint32_t oso_idx);

  lldb_private::ObjectFile *
  GetObjectFileByCompUnitInfo(CompileUnitInfo *comp_unit_info);

  lldb_private::ObjectFile *GetObjectFileByOSOIndex(uint32_t oso_idx);

  uint32_t GetCompUnitInfoIndex(const CompileUnitInfo *comp_unit_info);

  SymbolFileDWARF *GetSymbolFile(const lldb_private::SymbolContext &sc);
  SymbolFileDWARF *GetSymbolFile(const lldb_private::CompileUnit &comp_unit);

  SymbolFileDWARF *GetSymbolFileByCompUnitInfo(CompileUnitInfo *comp_unit_info);

  SymbolFileDWARF *GetSymbolFileByOSOIndex(uint32_t oso_idx);

  // If closure returns "false", iteration continues.  If it returns
  // "true", iteration terminates.
  void ForEachSymbolFile(std::function<bool(SymbolFileDWARF *)> closure) {
    for (uint32_t oso_idx = 0, num_oso_idxs = m_compile_unit_infos.size();
         oso_idx < num_oso_idxs; ++oso_idx) {
      if (SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex(oso_idx)) {
        if (closure(oso_dwarf))
          return;
      }
    }
  }

  CompileUnitInfo *GetCompileUnitInfoForSymbolWithIndex(uint32_t symbol_idx,
                                                        uint32_t *oso_idx_ptr);

  CompileUnitInfo *GetCompileUnitInfoForSymbolWithID(lldb::user_id_t symbol_id,
                                                     uint32_t *oso_idx_ptr);

  static int
  SymbolContainsSymbolWithIndex(uint32_t *symbol_idx_ptr,
                                const CompileUnitInfo *comp_unit_info);

  static int SymbolContainsSymbolWithID(lldb::user_id_t *symbol_idx_ptr,
                                        const CompileUnitInfo *comp_unit_info);

  void PrivateFindGlobalVariables(
      lldb_private::ConstString name,
      const lldb_private::CompilerDeclContext &parent_decl_ctx,
      const std::vector<uint32_t> &name_symbol_indexes, uint32_t max_matches,
      lldb_private::VariableList &variables);

  void SetCompileUnit(SymbolFileDWARF *oso_dwarf,
                      const lldb::CompUnitSP &cu_sp);

  lldb::CompUnitSP GetCompileUnit(SymbolFileDWARF *oso_dwarf);

  CompileUnitInfo *GetCompileUnitInfo(SymbolFileDWARF *oso_dwarf);

  lldb::TypeSP
  FindDefinitionTypeForDWARFDeclContext(const DWARFDeclContext &die_decl_ctx);

  bool Supports_DW_AT_APPLE_objc_complete_type(SymbolFileDWARF *skip_dwarf_oso);

  lldb::TypeSP FindCompleteObjCDefinitionTypeForDIE(
      const DWARFDIE &die, lldb_private::ConstString type_name,
      bool must_be_implementation);

  UniqueDWARFASTTypeMap &GetUniqueDWARFASTTypeMap() {
    return m_unique_ast_type_map;
  }

  // OSOEntry
  class OSOEntry {
  public:
    OSOEntry() = default;

    OSOEntry(uint32_t exe_sym_idx, lldb::addr_t oso_file_addr)
        : m_exe_sym_idx(exe_sym_idx), m_oso_file_addr(oso_file_addr) {}

    uint32_t GetExeSymbolIndex() const { return m_exe_sym_idx; }

    bool operator<(const OSOEntry &rhs) const {
      return m_exe_sym_idx < rhs.m_exe_sym_idx;
    }

    lldb::addr_t GetOSOFileAddress() const { return m_oso_file_addr; }

    void SetOSOFileAddress(lldb::addr_t oso_file_addr) {
      m_oso_file_addr = oso_file_addr;
    }

  protected:
    uint32_t m_exe_sym_idx = UINT32_MAX;
    lldb::addr_t m_oso_file_addr = LLDB_INVALID_ADDRESS;
  };

  typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t, OSOEntry>
      DebugMap;

  // Member Variables
  std::bitset<kNumFlags> m_flags;
  std::vector<CompileUnitInfo> m_compile_unit_infos;
  std::vector<uint32_t> m_func_indexes; // Sorted by address
  std::vector<uint32_t> m_glob_indexes;
  std::map<std::pair<lldb_private::ConstString, llvm::sys::TimePoint<>>,
           OSOInfoSP>
      m_oso_map;
  UniqueDWARFASTTypeMap m_unique_ast_type_map;
  lldb_private::LazyBool m_supports_DW_AT_APPLE_objc_complete_type;
  DebugMap m_debug_map;

  // When an object file from the debug map gets parsed in
  // SymbolFileDWARF, it needs to tell the debug map about the object
  // files addresses by calling this function once for each N_FUN,
  // N_GSYM and N_STSYM and after all entries in the debug map have
  // been matched up, FinalizeOSOFileRanges() should be called.
  bool AddOSOFileRange(CompileUnitInfo *cu_info, lldb::addr_t exe_file_addr,
                       lldb::addr_t exe_byte_size, lldb::addr_t oso_file_addr,
                       lldb::addr_t oso_byte_size);

  // Called after calling AddOSOFileRange() for each object file debug
  // map entry to finalize the info for the unlinked compile unit.
  void FinalizeOSOFileRanges(CompileUnitInfo *cu_info);

  /// Convert \a addr from a .o file address, to an executable address.
  ///
  /// \param[in] addr
  ///     A section offset address from a .o file
  ///
  /// \return
  ///     Returns true if \a addr was converted to be an executable
  ///     section/offset address, false otherwise.
  bool LinkOSOAddress(lldb_private::Address &addr);

  /// Convert a .o file "file address" to an executable "file address".
  ///
  /// \param[in] oso_symfile
  ///     The DWARF symbol file that contains \a oso_file_addr
  ///
  /// \param[in] oso_file_addr
  ///     A .o file "file address" to convert.
  ///
  /// \return
  ///     LLDB_INVALID_ADDRESS if \a oso_file_addr is not in the
  ///     linked executable, otherwise a valid "file address" from the
  ///     linked executable that contains the debug map.
  lldb::addr_t LinkOSOFileAddress(SymbolFileDWARF *oso_symfile,
                                  lldb::addr_t oso_file_addr);

  /// Given a line table full of lines with "file addresses" that are
  /// for a .o file represented by \a oso_symfile, link a new line table
  /// and return it.
  ///
  /// \param[in] oso_symfile
  ///     The DWARF symbol file that produced the \a line_table
  ///
  /// \param[in] line_table
  ///     A pointer to the line table.
  ///
  /// \return
  ///     Returns a valid line table full of linked addresses, or NULL
  ///     if none of the line table addresses exist in the main
  ///     executable.
  lldb_private::LineTable *
  LinkOSOLineTable(SymbolFileDWARF *oso_symfile,
                   lldb_private::LineTable *line_table);

  size_t AddOSOARanges(SymbolFileDWARF *dwarf2Data,
                       DWARFDebugAranges *debug_aranges);
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARFDEBUGMAP_H
