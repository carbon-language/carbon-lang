//===-- SymbolFileOnDemand.h ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_SYMBOLFILEONDEMAND_H
#define LLDB_SYMBOL_SYMBOLFILEONDEMAND_H

#include <mutex>
#include <vector>

#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Flags.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// SymbolFileOnDemand wraps an actual SymbolFile by providing
/// on demand symbol parsing/indexing to improve performance.
/// By default SymbolFileOnDemand will skip load the underlying
/// symbols. Any client can on demand hydrate the underlying
/// SymbolFile via SymbolFile::SetLoadDebugInfoEnabled().
class SymbolFileOnDemand : public lldb_private::SymbolFile {
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

  SymbolFileOnDemand(std::unique_ptr<SymbolFile> &&symbol_file);
  virtual ~SymbolFileOnDemand() override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return "ondemand"; }

  bool GetLoadDebugInfoEnabled() override { return m_debug_info_enabled; }

  void SetLoadDebugInfoEnabled() override;

  uint32_t GetNumCompileUnits() override;
  lldb::CompUnitSP GetCompileUnitAtIndex(uint32_t idx) override;

  SymbolFile *GetBackingSymbolFile() override { return m_sym_file_impl.get(); }

  uint32_t CalculateAbilities() override;

  std::recursive_mutex &GetModuleMutex() const override;

  lldb::LanguageType
  ParseLanguage(lldb_private::CompileUnit &comp_unit) override;

  lldb_private::XcodeSDK
  ParseXcodeSDK(lldb_private::CompileUnit &comp_unit) override;

  void InitializeObject() override;

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

  bool CompleteType(lldb_private::CompilerType &compiler_type) override;

  lldb_private::CompilerDecl GetDeclForUID(lldb::user_id_t uid) override;

  lldb_private::CompilerDeclContext
  GetDeclContextForUID(lldb::user_id_t uid) override;

  lldb_private::CompilerDeclContext
  GetDeclContextContainingUID(lldb::user_id_t uid) override;

  void
  ParseDeclsForContext(lldb_private::CompilerDeclContext decl_ctx) override;

  uint32_t ResolveSymbolContext(const lldb_private::Address &so_addr,
                                lldb::SymbolContextItem resolve_scope,
                                lldb_private::SymbolContext &sc) override;

  uint32_t ResolveSymbolContext(
      const lldb_private::SourceLocationSpec &src_location_spec,
      lldb::SymbolContextItem resolve_scope,
      lldb_private::SymbolContextList &sc_list) override;

  void Dump(lldb_private::Stream &s) override;
  void DumpClangAST(lldb_private::Stream &s) override;

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

  void GetMangledNamesForFunction(
      const std::string &scope_qualified_name,
      std::vector<lldb_private::ConstString> &mangled_names) override;

  void
  FindTypes(lldb_private::ConstString name,
            const lldb_private::CompilerDeclContext &parent_decl_ctx,
            uint32_t max_matches,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            lldb_private::TypeMap &types) override;

  void FindTypes(llvm::ArrayRef<lldb_private::CompilerContext> pattern,
                 lldb_private::LanguageSet languages,
                 llvm::DenseSet<SymbolFile *> &searched_symbol_files,
                 lldb_private::TypeMap &types) override;

  void GetTypes(lldb_private::SymbolContextScope *sc_scope,
                lldb::TypeClass type_mask,
                lldb_private::TypeList &type_list) override;

  llvm::Expected<lldb_private::TypeSystem &>
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  lldb_private::CompilerDeclContext FindNamespace(
      lldb_private::ConstString name,
      const lldb_private::CompilerDeclContext &parent_decl_ctx) override;

  std::vector<std::unique_ptr<lldb_private::CallEdge>>
  ParseCallEdgesInFunction(UserID func_id) override;

  lldb::UnwindPlanSP
  GetUnwindPlan(const Address &address,
                const RegisterInfoResolver &resolver) override;

  llvm::Expected<lldb::addr_t> GetParameterStackSize(Symbol &symbol) override;

  void PreloadSymbols() override;

  uint64_t GetDebugInfoSize() override;
  lldb_private::StatsDuration::Duration GetDebugInfoParseTime() override;
  lldb_private::StatsDuration::Duration GetDebugInfoIndexTime() override;

  uint32_t GetAbilities() override;

  Symtab *GetSymtab() override { return m_sym_file_impl->GetSymtab(); }

  ObjectFile *GetObjectFile() override {
    return m_sym_file_impl->GetObjectFile();
  }
  const ObjectFile *GetObjectFile() const override {
    return m_sym_file_impl->GetObjectFile();
  }
  ObjectFile *GetMainObjectFile() override {
    return m_sym_file_impl->GetMainObjectFile();
  }

  void SectionFileAddressesChanged() override {
    return m_sym_file_impl->SectionFileAddressesChanged();
  }

  bool GetDebugInfoIndexWasLoadedFromCache() const override {
    return m_sym_file_impl->GetDebugInfoIndexWasLoadedFromCache();
  }
  void SetDebugInfoIndexWasLoadedFromCache() override {
    m_sym_file_impl->SetDebugInfoIndexWasLoadedFromCache();
  }
  bool GetDebugInfoIndexWasSavedToCache() const override {
    return m_sym_file_impl->GetDebugInfoIndexWasSavedToCache();
  }
  void SetDebugInfoIndexWasSavedToCache() override {
    m_sym_file_impl->SetDebugInfoIndexWasSavedToCache();
  }

private:
  Log *GetLog() const { return ::lldb_private::GetLog(LLDBLog::OnDemand); }

  ConstString GetSymbolFileName() {
    return GetObjectFile()->GetFileSpec().GetFilename();
  }

private:
  bool m_debug_info_enabled = false;
  bool m_preload_symbols = false;
  std::unique_ptr<SymbolFile> m_sym_file_impl;
};
} // namespace lldb_private

#endif // LLDB_SYMBOL_SYMBOLFILEONDEMAND_H
