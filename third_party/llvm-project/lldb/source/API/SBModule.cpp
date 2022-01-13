//===-- SBModule.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBModule.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBModuleSpec.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

SBModule::SBModule() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBModule);
}

SBModule::SBModule(const lldb::ModuleSP &module_sp) : m_opaque_sp(module_sp) {}

SBModule::SBModule(const SBModuleSpec &module_spec) : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR(SBModule, (const lldb::SBModuleSpec &), module_spec);

  ModuleSP module_sp;
  Status error = ModuleList::GetSharedModule(
      *module_spec.m_opaque_up, module_sp, nullptr, nullptr, nullptr);
  if (module_sp)
    SetSP(module_sp);
}

SBModule::SBModule(const SBModule &rhs) : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBModule, (const lldb::SBModule &), rhs);
}

SBModule::SBModule(lldb::SBProcess &process, lldb::addr_t header_addr)
    : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR(SBModule, (lldb::SBProcess &, lldb::addr_t), process,
                          header_addr);

  ProcessSP process_sp(process.GetSP());
  if (process_sp) {
    m_opaque_sp = process_sp->ReadModuleFromMemory(FileSpec(), header_addr);
    if (m_opaque_sp) {
      Target &target = process_sp->GetTarget();
      bool changed = false;
      m_opaque_sp->SetLoadAddress(target, 0, true, changed);
      target.GetImages().Append(m_opaque_sp);
    }
  }
}

const SBModule &SBModule::operator=(const SBModule &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBModule &, SBModule, operator=,
                     (const lldb::SBModule &), rhs);

  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return LLDB_RECORD_RESULT(*this);
}

SBModule::~SBModule() = default;

bool SBModule::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBModule, IsValid);
  return this->operator bool();
}
SBModule::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBModule, operator bool);

  return m_opaque_sp.get() != nullptr;
}

void SBModule::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBModule, Clear);

  m_opaque_sp.reset();
}

SBFileSpec SBModule::GetFileSpec() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFileSpec, SBModule, GetFileSpec);

  SBFileSpec file_spec;
  ModuleSP module_sp(GetSP());
  if (module_sp)
    file_spec.SetFileSpec(module_sp->GetFileSpec());

  return LLDB_RECORD_RESULT(file_spec);
}

lldb::SBFileSpec SBModule::GetPlatformFileSpec() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFileSpec, SBModule,
                                   GetPlatformFileSpec);

  SBFileSpec file_spec;
  ModuleSP module_sp(GetSP());
  if (module_sp)
    file_spec.SetFileSpec(module_sp->GetPlatformFileSpec());

  return LLDB_RECORD_RESULT(file_spec);
}

bool SBModule::SetPlatformFileSpec(const lldb::SBFileSpec &platform_file) {
  LLDB_RECORD_METHOD(bool, SBModule, SetPlatformFileSpec,
                     (const lldb::SBFileSpec &), platform_file);

  bool result = false;

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    module_sp->SetPlatformFileSpec(*platform_file);
    result = true;
  }

  return result;
}

lldb::SBFileSpec SBModule::GetRemoteInstallFileSpec() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFileSpec, SBModule,
                             GetRemoteInstallFileSpec);

  SBFileSpec sb_file_spec;
  ModuleSP module_sp(GetSP());
  if (module_sp)
    sb_file_spec.SetFileSpec(module_sp->GetRemoteInstallFileSpec());
  return LLDB_RECORD_RESULT(sb_file_spec);
}

bool SBModule::SetRemoteInstallFileSpec(lldb::SBFileSpec &file) {
  LLDB_RECORD_METHOD(bool, SBModule, SetRemoteInstallFileSpec,
                     (lldb::SBFileSpec &), file);

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    module_sp->SetRemoteInstallFileSpec(file.ref());
    return true;
  }
  return false;
}

const uint8_t *SBModule::GetUUIDBytes() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const uint8_t *, SBModule, GetUUIDBytes);

  const uint8_t *uuid_bytes = nullptr;
  ModuleSP module_sp(GetSP());
  if (module_sp)
    uuid_bytes = module_sp->GetUUID().GetBytes().data();

  return uuid_bytes;
}

const char *SBModule::GetUUIDString() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBModule, GetUUIDString);

  const char *uuid_cstr = nullptr;
  ModuleSP module_sp(GetSP());
  if (module_sp) {
    // We are going to return a "const char *" value through the public API, so
    // we need to constify it so it gets added permanently the string pool and
    // then we don't need to worry about the lifetime of the string as it will
    // never go away once it has been put into the ConstString string pool
    uuid_cstr = ConstString(module_sp->GetUUID().GetAsString()).GetCString();
  }

  if (uuid_cstr && uuid_cstr[0]) {
    return uuid_cstr;
  }

  return nullptr;
}

bool SBModule::operator==(const SBModule &rhs) const {
  LLDB_RECORD_METHOD_CONST(bool, SBModule, operator==, (const lldb::SBModule &),
                           rhs);

  if (m_opaque_sp)
    return m_opaque_sp.get() == rhs.m_opaque_sp.get();
  return false;
}

bool SBModule::operator!=(const SBModule &rhs) const {
  LLDB_RECORD_METHOD_CONST(bool, SBModule, operator!=, (const lldb::SBModule &),
                           rhs);

  if (m_opaque_sp)
    return m_opaque_sp.get() != rhs.m_opaque_sp.get();
  return false;
}

ModuleSP SBModule::GetSP() const { return m_opaque_sp; }

void SBModule::SetSP(const ModuleSP &module_sp) { m_opaque_sp = module_sp; }

SBAddress SBModule::ResolveFileAddress(lldb::addr_t vm_addr) {
  LLDB_RECORD_METHOD(lldb::SBAddress, SBModule, ResolveFileAddress,
                     (lldb::addr_t), vm_addr);

  lldb::SBAddress sb_addr;
  ModuleSP module_sp(GetSP());
  if (module_sp) {
    Address addr;
    if (module_sp->ResolveFileAddress(vm_addr, addr))
      sb_addr.ref() = addr;
  }
  return LLDB_RECORD_RESULT(sb_addr);
}

SBSymbolContext
SBModule::ResolveSymbolContextForAddress(const SBAddress &addr,
                                         uint32_t resolve_scope) {
  LLDB_RECORD_METHOD(lldb::SBSymbolContext, SBModule,
                     ResolveSymbolContextForAddress,
                     (const lldb::SBAddress &, uint32_t), addr, resolve_scope);

  SBSymbolContext sb_sc;
  ModuleSP module_sp(GetSP());
  SymbolContextItem scope = static_cast<SymbolContextItem>(resolve_scope);
  if (module_sp && addr.IsValid())
    module_sp->ResolveSymbolContextForAddress(addr.ref(), scope, *sb_sc);
  return LLDB_RECORD_RESULT(sb_sc);
}

bool SBModule::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBModule, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    module_sp->GetDescription(strm.AsRawOstream());
  } else
    strm.PutCString("No value");

  return true;
}

uint32_t SBModule::GetNumCompileUnits() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBModule, GetNumCompileUnits);

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    return module_sp->GetNumCompileUnits();
  }
  return 0;
}

SBCompileUnit SBModule::GetCompileUnitAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBCompileUnit, SBModule, GetCompileUnitAtIndex,
                     (uint32_t), index);

  SBCompileUnit sb_cu;
  ModuleSP module_sp(GetSP());
  if (module_sp) {
    CompUnitSP cu_sp = module_sp->GetCompileUnitAtIndex(index);
    sb_cu.reset(cu_sp.get());
  }
  return LLDB_RECORD_RESULT(sb_cu);
}

SBSymbolContextList SBModule::FindCompileUnits(const SBFileSpec &sb_file_spec) {
  LLDB_RECORD_METHOD(lldb::SBSymbolContextList, SBModule, FindCompileUnits,
                     (const lldb::SBFileSpec &), sb_file_spec);

  SBSymbolContextList sb_sc_list;
  const ModuleSP module_sp(GetSP());
  if (sb_file_spec.IsValid() && module_sp) {
    module_sp->FindCompileUnits(*sb_file_spec, *sb_sc_list);
  }
  return LLDB_RECORD_RESULT(sb_sc_list);
}

static Symtab *GetUnifiedSymbolTable(const lldb::ModuleSP &module_sp) {
  if (module_sp)
    return module_sp->GetSymtab();
  return nullptr;
}

size_t SBModule::GetNumSymbols() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBModule, GetNumSymbols);

  ModuleSP module_sp(GetSP());
  if (Symtab *symtab = GetUnifiedSymbolTable(module_sp))
    return symtab->GetNumSymbols();
  return 0;
}

SBSymbol SBModule::GetSymbolAtIndex(size_t idx) {
  LLDB_RECORD_METHOD(lldb::SBSymbol, SBModule, GetSymbolAtIndex, (size_t), idx);

  SBSymbol sb_symbol;
  ModuleSP module_sp(GetSP());
  Symtab *symtab = GetUnifiedSymbolTable(module_sp);
  if (symtab)
    sb_symbol.SetSymbol(symtab->SymbolAtIndex(idx));
  return LLDB_RECORD_RESULT(sb_symbol);
}

lldb::SBSymbol SBModule::FindSymbol(const char *name,
                                    lldb::SymbolType symbol_type) {
  LLDB_RECORD_METHOD(lldb::SBSymbol, SBModule, FindSymbol,
                     (const char *, lldb::SymbolType), name, symbol_type);

  SBSymbol sb_symbol;
  if (name && name[0]) {
    ModuleSP module_sp(GetSP());
    Symtab *symtab = GetUnifiedSymbolTable(module_sp);
    if (symtab)
      sb_symbol.SetSymbol(symtab->FindFirstSymbolWithNameAndType(
          ConstString(name), symbol_type, Symtab::eDebugAny,
          Symtab::eVisibilityAny));
  }
  return LLDB_RECORD_RESULT(sb_symbol);
}

lldb::SBSymbolContextList SBModule::FindSymbols(const char *name,
                                                lldb::SymbolType symbol_type) {
  LLDB_RECORD_METHOD(lldb::SBSymbolContextList, SBModule, FindSymbols,
                     (const char *, lldb::SymbolType), name, symbol_type);

  SBSymbolContextList sb_sc_list;
  if (name && name[0]) {
    ModuleSP module_sp(GetSP());
    Symtab *symtab = GetUnifiedSymbolTable(module_sp);
    if (symtab) {
      std::vector<uint32_t> matching_symbol_indexes;
      symtab->FindAllSymbolsWithNameAndType(ConstString(name), symbol_type,
                                            matching_symbol_indexes);
      const size_t num_matches = matching_symbol_indexes.size();
      if (num_matches) {
        SymbolContext sc;
        sc.module_sp = module_sp;
        SymbolContextList &sc_list = *sb_sc_list;
        for (size_t i = 0; i < num_matches; ++i) {
          sc.symbol = symtab->SymbolAtIndex(matching_symbol_indexes[i]);
          if (sc.symbol)
            sc_list.Append(sc);
        }
      }
    }
  }
  return LLDB_RECORD_RESULT(sb_sc_list);
}

size_t SBModule::GetNumSections() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBModule, GetNumSections);

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    // Give the symbol vendor a chance to add to the unified section list.
    module_sp->GetSymbolFile();
    SectionList *section_list = module_sp->GetSectionList();
    if (section_list)
      return section_list->GetSize();
  }
  return 0;
}

SBSection SBModule::GetSectionAtIndex(size_t idx) {
  LLDB_RECORD_METHOD(lldb::SBSection, SBModule, GetSectionAtIndex, (size_t),
                     idx);

  SBSection sb_section;
  ModuleSP module_sp(GetSP());
  if (module_sp) {
    // Give the symbol vendor a chance to add to the unified section list.
    module_sp->GetSymbolFile();
    SectionList *section_list = module_sp->GetSectionList();

    if (section_list)
      sb_section.SetSP(section_list->GetSectionAtIndex(idx));
  }
  return LLDB_RECORD_RESULT(sb_section);
}

lldb::SBSymbolContextList SBModule::FindFunctions(const char *name,
                                                  uint32_t name_type_mask) {
  LLDB_RECORD_METHOD(lldb::SBSymbolContextList, SBModule, FindFunctions,
                     (const char *, uint32_t), name, name_type_mask);

  lldb::SBSymbolContextList sb_sc_list;
  ModuleSP module_sp(GetSP());
  if (name && module_sp) {

    ModuleFunctionSearchOptions function_options;
    function_options.include_symbols = true;
    function_options.include_inlines = true;
    FunctionNameType type = static_cast<FunctionNameType>(name_type_mask);
    module_sp->FindFunctions(ConstString(name), CompilerDeclContext(), type,
                             function_options, *sb_sc_list);
  }
  return LLDB_RECORD_RESULT(sb_sc_list);
}

SBValueList SBModule::FindGlobalVariables(SBTarget &target, const char *name,
                                          uint32_t max_matches) {
  LLDB_RECORD_METHOD(lldb::SBValueList, SBModule, FindGlobalVariables,
                     (lldb::SBTarget &, const char *, uint32_t), target, name,
                     max_matches);

  SBValueList sb_value_list;
  ModuleSP module_sp(GetSP());
  if (name && module_sp) {
    VariableList variable_list;
    module_sp->FindGlobalVariables(ConstString(name), CompilerDeclContext(),
                                   max_matches, variable_list);
    for (const VariableSP &var_sp : variable_list) {
      lldb::ValueObjectSP valobj_sp;
      TargetSP target_sp(target.GetSP());
      valobj_sp = ValueObjectVariable::Create(target_sp.get(), var_sp);
      if (valobj_sp)
        sb_value_list.Append(SBValue(valobj_sp));
    }
  }

  return LLDB_RECORD_RESULT(sb_value_list);
}

lldb::SBValue SBModule::FindFirstGlobalVariable(lldb::SBTarget &target,
                                                const char *name) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBModule, FindFirstGlobalVariable,
                     (lldb::SBTarget &, const char *), target, name);

  SBValueList sb_value_list(FindGlobalVariables(target, name, 1));
  if (sb_value_list.IsValid() && sb_value_list.GetSize() > 0)
    return LLDB_RECORD_RESULT(sb_value_list.GetValueAtIndex(0));
  return LLDB_RECORD_RESULT(SBValue());
}

lldb::SBType SBModule::FindFirstType(const char *name_cstr) {
  LLDB_RECORD_METHOD(lldb::SBType, SBModule, FindFirstType, (const char *),
                     name_cstr);

  SBType sb_type;
  ModuleSP module_sp(GetSP());
  if (name_cstr && module_sp) {
    SymbolContext sc;
    const bool exact_match = false;
    ConstString name(name_cstr);

    sb_type = SBType(module_sp->FindFirstType(sc, name, exact_match));

    if (!sb_type.IsValid()) {
      auto type_system_or_err =
          module_sp->GetTypeSystemForLanguage(eLanguageTypeC);
      if (auto err = type_system_or_err.takeError()) {
        llvm::consumeError(std::move(err));
        return LLDB_RECORD_RESULT(SBType());
      }
      sb_type = SBType(type_system_or_err->GetBuiltinTypeByName(name));
    }
  }
  return LLDB_RECORD_RESULT(sb_type);
}

lldb::SBType SBModule::GetBasicType(lldb::BasicType type) {
  LLDB_RECORD_METHOD(lldb::SBType, SBModule, GetBasicType, (lldb::BasicType),
                     type);

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    auto type_system_or_err =
        module_sp->GetTypeSystemForLanguage(eLanguageTypeC);
    if (auto err = type_system_or_err.takeError()) {
      llvm::consumeError(std::move(err));
    } else {
      return LLDB_RECORD_RESULT(
          SBType(type_system_or_err->GetBasicTypeFromAST(type)));
    }
  }
  return LLDB_RECORD_RESULT(SBType());
}

lldb::SBTypeList SBModule::FindTypes(const char *type) {
  LLDB_RECORD_METHOD(lldb::SBTypeList, SBModule, FindTypes, (const char *),
                     type);

  SBTypeList retval;

  ModuleSP module_sp(GetSP());
  if (type && module_sp) {
    TypeList type_list;
    const bool exact_match = false;
    ConstString name(type);
    llvm::DenseSet<SymbolFile *> searched_symbol_files;
    module_sp->FindTypes(name, exact_match, UINT32_MAX, searched_symbol_files,
                         type_list);

    if (type_list.Empty()) {
      auto type_system_or_err =
          module_sp->GetTypeSystemForLanguage(eLanguageTypeC);
      if (auto err = type_system_or_err.takeError()) {
        llvm::consumeError(std::move(err));
      } else {
        CompilerType compiler_type =
            type_system_or_err->GetBuiltinTypeByName(name);
        if (compiler_type)
          retval.Append(SBType(compiler_type));
      }
    } else {
      for (size_t idx = 0; idx < type_list.GetSize(); idx++) {
        TypeSP type_sp(type_list.GetTypeAtIndex(idx));
        if (type_sp)
          retval.Append(SBType(type_sp));
      }
    }
  }
  return LLDB_RECORD_RESULT(retval);
}

lldb::SBType SBModule::GetTypeByID(lldb::user_id_t uid) {
  LLDB_RECORD_METHOD(lldb::SBType, SBModule, GetTypeByID, (lldb::user_id_t),
                     uid);

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    if (SymbolFile *symfile = module_sp->GetSymbolFile()) {
      Type *type_ptr = symfile->ResolveTypeUID(uid);
      if (type_ptr)
        return LLDB_RECORD_RESULT(SBType(type_ptr->shared_from_this()));
    }
  }
  return LLDB_RECORD_RESULT(SBType());
}

lldb::SBTypeList SBModule::GetTypes(uint32_t type_mask) {
  LLDB_RECORD_METHOD(lldb::SBTypeList, SBModule, GetTypes, (uint32_t),
                     type_mask);

  SBTypeList sb_type_list;

  ModuleSP module_sp(GetSP());
  if (!module_sp)
    return LLDB_RECORD_RESULT(sb_type_list);
  SymbolFile *symfile = module_sp->GetSymbolFile();
  if (!symfile)
    return LLDB_RECORD_RESULT(sb_type_list);

  TypeClass type_class = static_cast<TypeClass>(type_mask);
  TypeList type_list;
  symfile->GetTypes(nullptr, type_class, type_list);
  sb_type_list.m_opaque_up->Append(type_list);
  return LLDB_RECORD_RESULT(sb_type_list);
}

SBSection SBModule::FindSection(const char *sect_name) {
  LLDB_RECORD_METHOD(lldb::SBSection, SBModule, FindSection, (const char *),
                     sect_name);

  SBSection sb_section;

  ModuleSP module_sp(GetSP());
  if (sect_name && module_sp) {
    // Give the symbol vendor a chance to add to the unified section list.
    module_sp->GetSymbolFile();
    SectionList *section_list = module_sp->GetSectionList();
    if (section_list) {
      ConstString const_sect_name(sect_name);
      SectionSP section_sp(section_list->FindSectionByName(const_sect_name));
      if (section_sp) {
        sb_section.SetSP(section_sp);
      }
    }
  }
  return LLDB_RECORD_RESULT(sb_section);
}

lldb::ByteOrder SBModule::GetByteOrder() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::ByteOrder, SBModule, GetByteOrder);

  ModuleSP module_sp(GetSP());
  if (module_sp)
    return module_sp->GetArchitecture().GetByteOrder();
  return eByteOrderInvalid;
}

const char *SBModule::GetTriple() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBModule, GetTriple);

  ModuleSP module_sp(GetSP());
  if (module_sp) {
    std::string triple(module_sp->GetArchitecture().GetTriple().str());
    // Unique the string so we don't run into ownership issues since the const
    // strings put the string into the string pool once and the strings never
    // comes out
    ConstString const_triple(triple.c_str());
    return const_triple.GetCString();
  }
  return nullptr;
}

uint32_t SBModule::GetAddressByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBModule, GetAddressByteSize);

  ModuleSP module_sp(GetSP());
  if (module_sp)
    return module_sp->GetArchitecture().GetAddressByteSize();
  return sizeof(void *);
}

uint32_t SBModule::GetVersion(uint32_t *versions, uint32_t num_versions) {
  LLDB_RECORD_METHOD(uint32_t, SBModule, GetVersion, (uint32_t *, uint32_t),
                     versions, num_versions);

  llvm::VersionTuple version;
  if (ModuleSP module_sp = GetSP())
    version = module_sp->GetVersion();
  uint32_t result = 0;
  if (!version.empty())
    ++result;
  if (version.getMinor())
    ++result;
  if (version.getSubminor())
    ++result;

  if (!versions)
    return result;

  if (num_versions > 0)
    versions[0] = version.empty() ? UINT32_MAX : version.getMajor();
  if (num_versions > 1)
    versions[1] = version.getMinor().getValueOr(UINT32_MAX);
  if (num_versions > 2)
    versions[2] = version.getSubminor().getValueOr(UINT32_MAX);
  for (uint32_t i = 3; i < num_versions; ++i)
    versions[i] = UINT32_MAX;
  return result;
}

lldb::SBFileSpec SBModule::GetSymbolFileSpec() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFileSpec, SBModule,
                                   GetSymbolFileSpec);

  lldb::SBFileSpec sb_file_spec;
  ModuleSP module_sp(GetSP());
  if (module_sp) {
    if (SymbolFile *symfile = module_sp->GetSymbolFile())
      sb_file_spec.SetFileSpec(symfile->GetObjectFile()->GetFileSpec());
  }
  return LLDB_RECORD_RESULT(sb_file_spec);
}

lldb::SBAddress SBModule::GetObjectFileHeaderAddress() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBAddress, SBModule,
                                   GetObjectFileHeaderAddress);

  lldb::SBAddress sb_addr;
  ModuleSP module_sp(GetSP());
  if (module_sp) {
    ObjectFile *objfile_ptr = module_sp->GetObjectFile();
    if (objfile_ptr)
      sb_addr.ref() = objfile_ptr->GetBaseAddress();
  }
  return LLDB_RECORD_RESULT(sb_addr);
}

lldb::SBAddress SBModule::GetObjectFileEntryPointAddress() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBAddress, SBModule,
                                   GetObjectFileEntryPointAddress);

  lldb::SBAddress sb_addr;
  ModuleSP module_sp(GetSP());
  if (module_sp) {
    ObjectFile *objfile_ptr = module_sp->GetObjectFile();
    if (objfile_ptr)
      sb_addr.ref() = objfile_ptr->GetEntryPointAddress();
  }
  return LLDB_RECORD_RESULT(sb_addr);
}

uint32_t SBModule::GetNumberAllocatedModules() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(uint32_t, SBModule,
                                    GetNumberAllocatedModules);

  return Module::GetNumberAllocatedModules();
}

void SBModule::GarbageCollectAllocatedModules() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(void, SBModule,
                                    GarbageCollectAllocatedModules);

  const bool mandatory = false;
  ModuleList::RemoveOrphanSharedModules(mandatory);
}

namespace lldb_private {
namespace repro {

template <> void RegisterMethods<SBModule>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBModule, ());
  LLDB_REGISTER_CONSTRUCTOR(SBModule, (const lldb::SBModuleSpec &));
  LLDB_REGISTER_CONSTRUCTOR(SBModule, (const lldb::SBModule &));
  LLDB_REGISTER_CONSTRUCTOR(SBModule, (lldb::SBProcess &, lldb::addr_t));
  LLDB_REGISTER_METHOD(const lldb::SBModule &, SBModule, operator=,
                       (const lldb::SBModule &));
  LLDB_REGISTER_METHOD_CONST(bool, SBModule, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBModule, operator bool, ());
  LLDB_REGISTER_METHOD(void, SBModule, Clear, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBModule, GetFileSpec, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBModule, GetPlatformFileSpec,
                             ());
  LLDB_REGISTER_METHOD(bool, SBModule, SetPlatformFileSpec,
                       (const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModule, GetRemoteInstallFileSpec,
                       ());
  LLDB_REGISTER_METHOD(bool, SBModule, SetRemoteInstallFileSpec,
                       (lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD_CONST(const char *, SBModule, GetUUIDString, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBModule, operator==,
                             (const lldb::SBModule &));
  LLDB_REGISTER_METHOD_CONST(bool, SBModule, operator!=,
                             (const lldb::SBModule &));
  LLDB_REGISTER_METHOD(lldb::SBAddress, SBModule, ResolveFileAddress,
                       (lldb::addr_t));
  LLDB_REGISTER_METHOD(lldb::SBSymbolContext, SBModule,
                       ResolveSymbolContextForAddress,
                       (const lldb::SBAddress &, uint32_t));
  LLDB_REGISTER_METHOD(bool, SBModule, GetDescription, (lldb::SBStream &));
  LLDB_REGISTER_METHOD(uint32_t, SBModule, GetNumCompileUnits, ());
  LLDB_REGISTER_METHOD(lldb::SBCompileUnit, SBModule, GetCompileUnitAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBModule, FindCompileUnits,
                       (const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD(size_t, SBModule, GetNumSymbols, ());
  LLDB_REGISTER_METHOD(lldb::SBSymbol, SBModule, GetSymbolAtIndex, (size_t));
  LLDB_REGISTER_METHOD(lldb::SBSymbol, SBModule, FindSymbol,
                       (const char *, lldb::SymbolType));
  LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBModule, FindSymbols,
                       (const char *, lldb::SymbolType));
  LLDB_REGISTER_METHOD(size_t, SBModule, GetNumSections, ());
  LLDB_REGISTER_METHOD(lldb::SBSection, SBModule, GetSectionAtIndex, (size_t));
  LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBModule, FindFunctions,
                       (const char *, uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBValueList, SBModule, FindGlobalVariables,
                       (lldb::SBTarget &, const char *, uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBModule, FindFirstGlobalVariable,
                       (lldb::SBTarget &, const char *));
  LLDB_REGISTER_METHOD(lldb::SBType, SBModule, FindFirstType, (const char *));
  LLDB_REGISTER_METHOD(lldb::SBType, SBModule, GetBasicType, (lldb::BasicType));
  LLDB_REGISTER_METHOD(lldb::SBTypeList, SBModule, FindTypes, (const char *));
  LLDB_REGISTER_METHOD(lldb::SBType, SBModule, GetTypeByID, (lldb::user_id_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeList, SBModule, GetTypes, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBSection, SBModule, FindSection, (const char *));
  LLDB_REGISTER_METHOD(lldb::ByteOrder, SBModule, GetByteOrder, ());
  LLDB_REGISTER_METHOD(const char *, SBModule, GetTriple, ());
  LLDB_REGISTER_METHOD(uint32_t, SBModule, GetAddressByteSize, ());
  LLDB_REGISTER_METHOD(uint32_t, SBModule, GetVersion, (uint32_t *, uint32_t));
  LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBModule, GetSymbolFileSpec, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBModule,
                             GetObjectFileHeaderAddress, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBModule,
                             GetObjectFileEntryPointAddress, ());
  LLDB_REGISTER_STATIC_METHOD(uint32_t, SBModule, GetNumberAllocatedModules,
                              ());
  LLDB_REGISTER_STATIC_METHOD(void, SBModule, GarbageCollectAllocatedModules,
                              ());
}

} // namespace repro
} // namespace lldb_private
