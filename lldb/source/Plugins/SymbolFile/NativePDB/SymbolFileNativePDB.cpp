//===-- SymbolFileNativePDB.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileNativePDB.h"

#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/Language/CPlusPlus/MSVCUndecoratedNameParser.h"
#include "Plugins/ObjectFile/PDB/ObjectFilePDB.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamBuffer.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Utility/Log.h"

#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/RecordName.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecordHelpers.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Demangle/MicrosoftDemangle.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "DWARFLocationExpression.h"
#include "PdbAstBuilder.h"
#include "PdbSymUid.h"
#include "PdbUtil.h"
#include "UdtRecordCompleter.h"

using namespace lldb;
using namespace lldb_private;
using namespace npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

char SymbolFileNativePDB::ID;

static lldb::LanguageType TranslateLanguage(PDB_Lang lang) {
  switch (lang) {
  case PDB_Lang::Cpp:
    return lldb::LanguageType::eLanguageTypeC_plus_plus;
  case PDB_Lang::C:
    return lldb::LanguageType::eLanguageTypeC;
  case PDB_Lang::Swift:
    return lldb::LanguageType::eLanguageTypeSwift;
  default:
    return lldb::LanguageType::eLanguageTypeUnknown;
  }
}

static std::unique_ptr<PDBFile>
loadMatchingPDBFile(std::string exe_path, llvm::BumpPtrAllocator &allocator) {
  // Try to find a matching PDB for an EXE.
  using namespace llvm::object;
  auto expected_binary = createBinary(exe_path);

  // If the file isn't a PE/COFF executable, fail.
  if (!expected_binary) {
    llvm::consumeError(expected_binary.takeError());
    return nullptr;
  }
  OwningBinary<Binary> binary = std::move(*expected_binary);

  // TODO: Avoid opening the PE/COFF binary twice by reading this information
  // directly from the lldb_private::ObjectFile.
  auto *obj = llvm::dyn_cast<llvm::object::COFFObjectFile>(binary.getBinary());
  if (!obj)
    return nullptr;
  const llvm::codeview::DebugInfo *pdb_info = nullptr;

  // If it doesn't have a debug directory, fail.
  llvm::StringRef pdb_file;
  if (llvm::Error e = obj->getDebugPDBInfo(pdb_info, pdb_file)) {
    consumeError(std::move(e));
    return nullptr;
  }

  // If the file doesn't exist, perhaps the path specified at build time
  // doesn't match the PDB's current location, so check the location of the
  // executable.
  if (!FileSystem::Instance().Exists(pdb_file)) {
    const auto exe_dir = FileSpec(exe_path).CopyByRemovingLastPathComponent();
    const auto pdb_name = FileSpec(pdb_file).GetFilename().GetCString();
    pdb_file = exe_dir.CopyByAppendingPathComponent(pdb_name).GetCString();
  }

  // If the file is not a PDB or if it doesn't have a matching GUID, fail.
  auto pdb = ObjectFilePDB::loadPDBFile(std::string(pdb_file), allocator);
  if (!pdb)
    return nullptr;

  auto expected_info = pdb->getPDBInfoStream();
  if (!expected_info) {
    llvm::consumeError(expected_info.takeError());
    return nullptr;
  }
  llvm::codeview::GUID guid;
  memcpy(&guid, pdb_info->PDB70.Signature, 16);

  if (expected_info->getGuid() != guid)
    return nullptr;
  return pdb;
}

static bool IsFunctionPrologue(const CompilandIndexItem &cci,
                               lldb::addr_t addr) {
  // FIXME: Implement this.
  return false;
}

static bool IsFunctionEpilogue(const CompilandIndexItem &cci,
                               lldb::addr_t addr) {
  // FIXME: Implement this.
  return false;
}

static llvm::StringRef GetSimpleTypeName(SimpleTypeKind kind) {
  switch (kind) {
  case SimpleTypeKind::Boolean128:
  case SimpleTypeKind::Boolean16:
  case SimpleTypeKind::Boolean32:
  case SimpleTypeKind::Boolean64:
  case SimpleTypeKind::Boolean8:
    return "bool";
  case SimpleTypeKind::Byte:
  case SimpleTypeKind::UnsignedCharacter:
    return "unsigned char";
  case SimpleTypeKind::NarrowCharacter:
    return "char";
  case SimpleTypeKind::SignedCharacter:
  case SimpleTypeKind::SByte:
    return "signed char";
  case SimpleTypeKind::Character16:
    return "char16_t";
  case SimpleTypeKind::Character32:
    return "char32_t";
  case SimpleTypeKind::Complex80:
  case SimpleTypeKind::Complex64:
  case SimpleTypeKind::Complex32:
    return "complex";
  case SimpleTypeKind::Float128:
  case SimpleTypeKind::Float80:
    return "long double";
  case SimpleTypeKind::Float64:
    return "double";
  case SimpleTypeKind::Float32:
    return "float";
  case SimpleTypeKind::Float16:
    return "single";
  case SimpleTypeKind::Int128:
    return "__int128";
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::Int64Quad:
    return "int64_t";
  case SimpleTypeKind::Int32:
    return "int";
  case SimpleTypeKind::Int16:
    return "short";
  case SimpleTypeKind::UInt128:
    return "unsigned __int128";
  case SimpleTypeKind::UInt64:
  case SimpleTypeKind::UInt64Quad:
    return "uint64_t";
  case SimpleTypeKind::HResult:
    return "HRESULT";
  case SimpleTypeKind::UInt32:
    return "unsigned";
  case SimpleTypeKind::UInt16:
  case SimpleTypeKind::UInt16Short:
    return "unsigned short";
  case SimpleTypeKind::Int32Long:
    return "long";
  case SimpleTypeKind::UInt32Long:
    return "unsigned long";
  case SimpleTypeKind::Void:
    return "void";
  case SimpleTypeKind::WideCharacter:
    return "wchar_t";
  default:
    return "";
  }
}

static bool IsClassRecord(TypeLeafKind kind) {
  switch (kind) {
  case LF_STRUCTURE:
  case LF_CLASS:
  case LF_INTERFACE:
    return true;
  default:
    return false;
  }
}

void SymbolFileNativePDB::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
}

void SymbolFileNativePDB::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

void SymbolFileNativePDB::DebuggerInitialize(Debugger &debugger) {}

llvm::StringRef SymbolFileNativePDB::GetPluginDescriptionStatic() {
  return "Microsoft PDB debug symbol cross-platform file reader.";
}

SymbolFile *SymbolFileNativePDB::CreateInstance(ObjectFileSP objfile_sp) {
  return new SymbolFileNativePDB(std::move(objfile_sp));
}

SymbolFileNativePDB::SymbolFileNativePDB(ObjectFileSP objfile_sp)
    : SymbolFile(std::move(objfile_sp)) {}

SymbolFileNativePDB::~SymbolFileNativePDB() = default;

uint32_t SymbolFileNativePDB::CalculateAbilities() {
  uint32_t abilities = 0;
  if (!m_objfile_sp)
    return 0;

  if (!m_index) {
    // Lazily load and match the PDB file, but only do this once.
    PDBFile *pdb_file;
    if (auto *pdb = llvm::dyn_cast<ObjectFilePDB>(m_objfile_sp.get())) {
      pdb_file = &pdb->GetPDBFile();
    } else {
      m_file_up = loadMatchingPDBFile(m_objfile_sp->GetFileSpec().GetPath(),
                                      m_allocator);
      pdb_file = m_file_up.get();
    }

    if (!pdb_file)
      return 0;

    auto expected_index = PdbIndex::create(pdb_file);
    if (!expected_index) {
      llvm::consumeError(expected_index.takeError());
      return 0;
    }
    m_index = std::move(*expected_index);
  }
  if (!m_index)
    return 0;

  // We don't especially have to be precise here.  We only distinguish between
  // stripped and not stripped.
  abilities = kAllAbilities;

  if (m_index->dbi().isStripped())
    abilities &= ~(Blocks | LocalVariables);
  return abilities;
}

void SymbolFileNativePDB::InitializeObject() {
  m_obj_load_address = m_objfile_sp->GetModule()
                           ->GetObjectFile()
                           ->GetBaseAddress()
                           .GetFileAddress();
  m_index->SetLoadAddress(m_obj_load_address);
  m_index->ParseSectionContribs();

  auto ts_or_err = m_objfile_sp->GetModule()->GetTypeSystemForLanguage(
      lldb::eLanguageTypeC_plus_plus);
  if (auto err = ts_or_err.takeError()) {
    LLDB_LOG_ERROR(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_SYMBOLS),
                   std::move(err), "Failed to initialize");
  } else {
    ts_or_err->SetSymbolFile(this);
    auto *clang = llvm::cast_or_null<TypeSystemClang>(&ts_or_err.get());
    lldbassert(clang);
    m_ast = std::make_unique<PdbAstBuilder>(*m_objfile_sp, *m_index, *clang);
  }
}

uint32_t SymbolFileNativePDB::CalculateNumCompileUnits() {
  const DbiModuleList &modules = m_index->dbi().modules();
  uint32_t count = modules.getModuleCount();
  if (count == 0)
    return count;

  // The linker can inject an additional "dummy" compilation unit into the
  // PDB. Ignore this special compile unit for our purposes, if it is there.
  // It is always the last one.
  DbiModuleDescriptor last = modules.getModuleDescriptor(count - 1);
  if (last.getModuleName() == "* Linker *")
    --count;
  return count;
}

Block &SymbolFileNativePDB::CreateBlock(PdbCompilandSymId block_id) {
  CompilandIndexItem *cii = m_index->compilands().GetCompiland(block_id.modi);
  CVSymbol sym = cii->m_debug_stream.readSymbolAtOffset(block_id.offset);
  CompUnitSP comp_unit = GetOrCreateCompileUnit(*cii);
  lldb::user_id_t opaque_block_uid = toOpaqueUid(block_id);
  BlockSP child_block = std::make_shared<Block>(opaque_block_uid);

  switch (sym.kind()) {
  case S_GPROC32:
  case S_LPROC32: {
    // This is a function.  It must be global.  Creating the Function entry
    // for it automatically creates a block for it.
    FunctionSP func = GetOrCreateFunction(block_id, *comp_unit);
    Block &block = func->GetBlock(false);
    if (block.GetNumRanges() == 0)
      block.AddRange(Block::Range(0, func->GetAddressRange().GetByteSize()));
    return block;
  }
  case S_BLOCK32: {
    // This is a block.  Its parent is either a function or another block.  In
    // either case, its parent can be viewed as a block (e.g. a function
    // contains 1 big block.  So just get the parent block and add this block
    // to it.
    BlockSym block(static_cast<SymbolRecordKind>(sym.kind()));
    cantFail(SymbolDeserializer::deserializeAs<BlockSym>(sym, block));
    lldbassert(block.Parent != 0);
    PdbCompilandSymId parent_id(block_id.modi, block.Parent);
    Block &parent_block = GetOrCreateBlock(parent_id);
    parent_block.AddChild(child_block);
    m_ast->GetOrCreateBlockDecl(block_id);
    m_blocks.insert({opaque_block_uid, child_block});
    break;
  }
  case S_INLINESITE: {
    // This ensures line table is parsed first so we have inline sites info.
    comp_unit->GetLineTable();

    std::shared_ptr<InlineSite> inline_site = m_inline_sites[opaque_block_uid];
    Block &parent_block = GetOrCreateBlock(inline_site->parent_id);
    parent_block.AddChild(child_block);

    // Copy ranges from InlineSite to Block.
    for (size_t i = 0; i < inline_site->ranges.GetSize(); ++i) {
      auto *entry = inline_site->ranges.GetEntryAtIndex(i);
      child_block->AddRange(
          Block::Range(entry->GetRangeBase(), entry->GetByteSize()));
    }
    child_block->FinalizeRanges();

    // Get the inlined function callsite info.
    Declaration &decl = inline_site->inline_function_info->GetDeclaration();
    Declaration &callsite = inline_site->inline_function_info->GetCallSite();
    child_block->SetInlinedFunctionInfo(
        inline_site->inline_function_info->GetName().GetCString(), nullptr,
        &decl, &callsite);
    m_blocks.insert({opaque_block_uid, child_block});
    break;
  }
  default:
    lldbassert(false && "Symbol is not a block!");
  }

  return *child_block;
}

lldb::FunctionSP SymbolFileNativePDB::CreateFunction(PdbCompilandSymId func_id,
                                                     CompileUnit &comp_unit) {
  const CompilandIndexItem *cci =
      m_index->compilands().GetCompiland(func_id.modi);
  lldbassert(cci);
  CVSymbol sym_record = cci->m_debug_stream.readSymbolAtOffset(func_id.offset);

  lldbassert(sym_record.kind() == S_LPROC32 || sym_record.kind() == S_GPROC32);
  SegmentOffsetLength sol = GetSegmentOffsetAndLength(sym_record);

  auto file_vm_addr = m_index->MakeVirtualAddress(sol.so);
  if (file_vm_addr == LLDB_INVALID_ADDRESS || file_vm_addr == 0)
    return nullptr;

  AddressRange func_range(file_vm_addr, sol.length,
                          comp_unit.GetModule()->GetSectionList());
  if (!func_range.GetBaseAddress().IsValid())
    return nullptr;

  ProcSym proc(static_cast<SymbolRecordKind>(sym_record.kind()));
  cantFail(SymbolDeserializer::deserializeAs<ProcSym>(sym_record, proc));
  if (proc.FunctionType == TypeIndex::None())
    return nullptr;
  TypeSP func_type = GetOrCreateType(proc.FunctionType);
  if (!func_type)
    return nullptr;

  PdbTypeSymId sig_id(proc.FunctionType, false);
  Mangled mangled(proc.Name);
  FunctionSP func_sp = std::make_shared<Function>(
      &comp_unit, toOpaqueUid(func_id), toOpaqueUid(sig_id), mangled,
      func_type.get(), func_range);

  comp_unit.AddFunction(func_sp);

  m_ast->GetOrCreateFunctionDecl(func_id);

  return func_sp;
}

CompUnitSP
SymbolFileNativePDB::CreateCompileUnit(const CompilandIndexItem &cci) {
  lldb::LanguageType lang =
      cci.m_compile_opts ? TranslateLanguage(cci.m_compile_opts->getLanguage())
                         : lldb::eLanguageTypeUnknown;

  LazyBool optimized = eLazyBoolNo;
  if (cci.m_compile_opts && cci.m_compile_opts->hasOptimizations())
    optimized = eLazyBoolYes;

  llvm::SmallString<64> source_file_name =
      m_index->compilands().GetMainSourceFile(cci);
  FileSpec fs(source_file_name);

  CompUnitSP cu_sp =
      std::make_shared<CompileUnit>(m_objfile_sp->GetModule(), nullptr, fs,
                                    toOpaqueUid(cci.m_id), lang, optimized);

  SetCompileUnitAtIndex(cci.m_id.modi, cu_sp);
  return cu_sp;
}

lldb::TypeSP SymbolFileNativePDB::CreateModifierType(PdbTypeSymId type_id,
                                                     const ModifierRecord &mr,
                                                     CompilerType ct) {
  TpiStream &stream = m_index->tpi();

  std::string name;
  if (mr.ModifiedType.isSimple())
    name = std::string(GetSimpleTypeName(mr.ModifiedType.getSimpleKind()));
  else
    name = computeTypeName(stream.typeCollection(), mr.ModifiedType);
  Declaration decl;
  lldb::TypeSP modified_type = GetOrCreateType(mr.ModifiedType);

  return std::make_shared<Type>(toOpaqueUid(type_id), this, ConstString(name),
                                modified_type->GetByteSize(nullptr), nullptr,
                                LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                                ct, Type::ResolveState::Full);
}

lldb::TypeSP
SymbolFileNativePDB::CreatePointerType(PdbTypeSymId type_id,
                                       const llvm::codeview::PointerRecord &pr,
                                       CompilerType ct) {
  TypeSP pointee = GetOrCreateType(pr.ReferentType);
  if (!pointee)
    return nullptr;

  if (pr.isPointerToMember()) {
    MemberPointerInfo mpi = pr.getMemberInfo();
    GetOrCreateType(mpi.ContainingType);
  }

  Declaration decl;
  return std::make_shared<Type>(toOpaqueUid(type_id), this, ConstString(),
                                pr.getSize(), nullptr, LLDB_INVALID_UID,
                                Type::eEncodingIsUID, decl, ct,
                                Type::ResolveState::Full);
}

lldb::TypeSP SymbolFileNativePDB::CreateSimpleType(TypeIndex ti,
                                                   CompilerType ct) {
  uint64_t uid = toOpaqueUid(PdbTypeSymId(ti, false));
  if (ti == TypeIndex::NullptrT()) {
    Declaration decl;
    return std::make_shared<Type>(
        uid, this, ConstString("std::nullptr_t"), 0, nullptr, LLDB_INVALID_UID,
        Type::eEncodingIsUID, decl, ct, Type::ResolveState::Full);
  }

  if (ti.getSimpleMode() != SimpleTypeMode::Direct) {
    TypeSP direct_sp = GetOrCreateType(ti.makeDirect());
    uint32_t pointer_size = 0;
    switch (ti.getSimpleMode()) {
    case SimpleTypeMode::FarPointer32:
    case SimpleTypeMode::NearPointer32:
      pointer_size = 4;
      break;
    case SimpleTypeMode::NearPointer64:
      pointer_size = 8;
      break;
    default:
      // 128-bit and 16-bit pointers unsupported.
      return nullptr;
    }
    Declaration decl;
    return std::make_shared<Type>(
        uid, this, ConstString(), pointer_size, nullptr, LLDB_INVALID_UID,
        Type::eEncodingIsUID, decl, ct, Type::ResolveState::Full);
  }

  if (ti.getSimpleKind() == SimpleTypeKind::NotTranslated)
    return nullptr;

  size_t size = GetTypeSizeForSimpleKind(ti.getSimpleKind());
  llvm::StringRef type_name = GetSimpleTypeName(ti.getSimpleKind());

  Declaration decl;
  return std::make_shared<Type>(uid, this, ConstString(type_name), size,
                                nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID,
                                decl, ct, Type::ResolveState::Full);
}

static std::string GetUnqualifiedTypeName(const TagRecord &record) {
  if (!record.hasUniqueName()) {
    MSVCUndecoratedNameParser parser(record.Name);
    llvm::ArrayRef<MSVCUndecoratedNameSpecifier> specs = parser.GetSpecifiers();

    return std::string(specs.back().GetBaseName());
  }

  llvm::ms_demangle::Demangler demangler;
  StringView sv(record.UniqueName.begin(), record.UniqueName.size());
  llvm::ms_demangle::TagTypeNode *ttn = demangler.parseTagUniqueName(sv);
  if (demangler.Error)
    return std::string(record.Name);

  llvm::ms_demangle::IdentifierNode *idn =
      ttn->QualifiedName->getUnqualifiedIdentifier();
  return idn->toString();
}

lldb::TypeSP
SymbolFileNativePDB::CreateClassStructUnion(PdbTypeSymId type_id,
                                            const TagRecord &record,
                                            size_t size, CompilerType ct) {

  std::string uname = GetUnqualifiedTypeName(record);

  // FIXME: Search IPI stream for LF_UDT_MOD_SRC_LINE.
  Declaration decl;
  return std::make_shared<Type>(toOpaqueUid(type_id), this, ConstString(uname),
                                size, nullptr, LLDB_INVALID_UID,
                                Type::eEncodingIsUID, decl, ct,
                                Type::ResolveState::Forward);
}

lldb::TypeSP SymbolFileNativePDB::CreateTagType(PdbTypeSymId type_id,
                                                const ClassRecord &cr,
                                                CompilerType ct) {
  return CreateClassStructUnion(type_id, cr, cr.getSize(), ct);
}

lldb::TypeSP SymbolFileNativePDB::CreateTagType(PdbTypeSymId type_id,
                                                const UnionRecord &ur,
                                                CompilerType ct) {
  return CreateClassStructUnion(type_id, ur, ur.getSize(), ct);
}

lldb::TypeSP SymbolFileNativePDB::CreateTagType(PdbTypeSymId type_id,
                                                const EnumRecord &er,
                                                CompilerType ct) {
  std::string uname = GetUnqualifiedTypeName(er);

  Declaration decl;
  TypeSP underlying_type = GetOrCreateType(er.UnderlyingType);

  return std::make_shared<lldb_private::Type>(
      toOpaqueUid(type_id), this, ConstString(uname),
      underlying_type->GetByteSize(nullptr), nullptr, LLDB_INVALID_UID,
      lldb_private::Type::eEncodingIsUID, decl, ct,
      lldb_private::Type::ResolveState::Forward);
}

TypeSP SymbolFileNativePDB::CreateArrayType(PdbTypeSymId type_id,
                                            const ArrayRecord &ar,
                                            CompilerType ct) {
  TypeSP element_type = GetOrCreateType(ar.ElementType);

  Declaration decl;
  TypeSP array_sp = std::make_shared<lldb_private::Type>(
      toOpaqueUid(type_id), this, ConstString(), ar.Size, nullptr,
      LLDB_INVALID_UID, lldb_private::Type::eEncodingIsUID, decl, ct,
      lldb_private::Type::ResolveState::Full);
  array_sp->SetEncodingType(element_type.get());
  return array_sp;
}


TypeSP SymbolFileNativePDB::CreateFunctionType(PdbTypeSymId type_id,
                                               const MemberFunctionRecord &mfr,
                                               CompilerType ct) {
  Declaration decl;
  return std::make_shared<lldb_private::Type>(
      toOpaqueUid(type_id), this, ConstString(), 0, nullptr, LLDB_INVALID_UID,
      lldb_private::Type::eEncodingIsUID, decl, ct,
      lldb_private::Type::ResolveState::Full);
}

TypeSP SymbolFileNativePDB::CreateProcedureType(PdbTypeSymId type_id,
                                                const ProcedureRecord &pr,
                                                CompilerType ct) {
  Declaration decl;
  return std::make_shared<lldb_private::Type>(
      toOpaqueUid(type_id), this, ConstString(), 0, nullptr, LLDB_INVALID_UID,
      lldb_private::Type::eEncodingIsUID, decl, ct,
      lldb_private::Type::ResolveState::Full);
}

TypeSP SymbolFileNativePDB::CreateType(PdbTypeSymId type_id, CompilerType ct) {
  if (type_id.index.isSimple())
    return CreateSimpleType(type_id.index, ct);

  TpiStream &stream = type_id.is_ipi ? m_index->ipi() : m_index->tpi();
  CVType cvt = stream.getType(type_id.index);

  if (cvt.kind() == LF_MODIFIER) {
    ModifierRecord modifier;
    llvm::cantFail(
        TypeDeserializer::deserializeAs<ModifierRecord>(cvt, modifier));
    return CreateModifierType(type_id, modifier, ct);
  }

  if (cvt.kind() == LF_POINTER) {
    PointerRecord pointer;
    llvm::cantFail(
        TypeDeserializer::deserializeAs<PointerRecord>(cvt, pointer));
    return CreatePointerType(type_id, pointer, ct);
  }

  if (IsClassRecord(cvt.kind())) {
    ClassRecord cr;
    llvm::cantFail(TypeDeserializer::deserializeAs<ClassRecord>(cvt, cr));
    return CreateTagType(type_id, cr, ct);
  }

  if (cvt.kind() == LF_ENUM) {
    EnumRecord er;
    llvm::cantFail(TypeDeserializer::deserializeAs<EnumRecord>(cvt, er));
    return CreateTagType(type_id, er, ct);
  }

  if (cvt.kind() == LF_UNION) {
    UnionRecord ur;
    llvm::cantFail(TypeDeserializer::deserializeAs<UnionRecord>(cvt, ur));
    return CreateTagType(type_id, ur, ct);
  }

  if (cvt.kind() == LF_ARRAY) {
    ArrayRecord ar;
    llvm::cantFail(TypeDeserializer::deserializeAs<ArrayRecord>(cvt, ar));
    return CreateArrayType(type_id, ar, ct);
  }

  if (cvt.kind() == LF_PROCEDURE) {
    ProcedureRecord pr;
    llvm::cantFail(TypeDeserializer::deserializeAs<ProcedureRecord>(cvt, pr));
    return CreateProcedureType(type_id, pr, ct);
  }
  if (cvt.kind() == LF_MFUNCTION) {
    MemberFunctionRecord mfr;
    llvm::cantFail(TypeDeserializer::deserializeAs<MemberFunctionRecord>(cvt, mfr));
    return CreateFunctionType(type_id, mfr, ct);
  }

  return nullptr;
}

TypeSP SymbolFileNativePDB::CreateAndCacheType(PdbTypeSymId type_id) {
  // If they search for a UDT which is a forward ref, try and resolve the full
  // decl and just map the forward ref uid to the full decl record.
  llvm::Optional<PdbTypeSymId> full_decl_uid;
  if (IsForwardRefUdt(type_id, m_index->tpi())) {
    auto expected_full_ti =
        m_index->tpi().findFullDeclForForwardRef(type_id.index);
    if (!expected_full_ti)
      llvm::consumeError(expected_full_ti.takeError());
    else if (*expected_full_ti != type_id.index) {
      full_decl_uid = PdbTypeSymId(*expected_full_ti, false);

      // It's possible that a lookup would occur for the full decl causing it
      // to be cached, then a second lookup would occur for the forward decl.
      // We don't want to create a second full decl, so make sure the full
      // decl hasn't already been cached.
      auto full_iter = m_types.find(toOpaqueUid(*full_decl_uid));
      if (full_iter != m_types.end()) {
        TypeSP result = full_iter->second;
        // Map the forward decl to the TypeSP for the full decl so we can take
        // the fast path next time.
        m_types[toOpaqueUid(type_id)] = result;
        return result;
      }
    }
  }

  PdbTypeSymId best_decl_id = full_decl_uid ? *full_decl_uid : type_id;

  clang::QualType qt = m_ast->GetOrCreateType(best_decl_id);

  TypeSP result = CreateType(best_decl_id, m_ast->ToCompilerType(qt));
  if (!result)
    return nullptr;

  uint64_t best_uid = toOpaqueUid(best_decl_id);
  m_types[best_uid] = result;
  // If we had both a forward decl and a full decl, make both point to the new
  // type.
  if (full_decl_uid)
    m_types[toOpaqueUid(type_id)] = result;

  return result;
}

TypeSP SymbolFileNativePDB::GetOrCreateType(PdbTypeSymId type_id) {
  // We can't use try_emplace / overwrite here because the process of creating
  // a type could create nested types, which could invalidate iterators.  So
  // we have to do a 2-phase lookup / insert.
  auto iter = m_types.find(toOpaqueUid(type_id));
  if (iter != m_types.end())
    return iter->second;

  TypeSP type = CreateAndCacheType(type_id);
  if (type)
    GetTypeList().Insert(type);
  return type;
}

VariableSP SymbolFileNativePDB::CreateGlobalVariable(PdbGlobalSymId var_id) {
  CVSymbol sym = m_index->symrecords().readRecord(var_id.offset);
  if (sym.kind() == S_CONSTANT)
    return CreateConstantSymbol(var_id, sym);

  lldb::ValueType scope = eValueTypeInvalid;
  TypeIndex ti;
  llvm::StringRef name;
  lldb::addr_t addr = 0;
  uint16_t section = 0;
  uint32_t offset = 0;
  bool is_external = false;
  switch (sym.kind()) {
  case S_GDATA32:
    is_external = true;
    LLVM_FALLTHROUGH;
  case S_LDATA32: {
    DataSym ds(sym.kind());
    llvm::cantFail(SymbolDeserializer::deserializeAs<DataSym>(sym, ds));
    ti = ds.Type;
    scope = (sym.kind() == S_GDATA32) ? eValueTypeVariableGlobal
                                      : eValueTypeVariableStatic;
    name = ds.Name;
    section = ds.Segment;
    offset = ds.DataOffset;
    addr = m_index->MakeVirtualAddress(ds.Segment, ds.DataOffset);
    break;
  }
  case S_GTHREAD32:
    is_external = true;
    LLVM_FALLTHROUGH;
  case S_LTHREAD32: {
    ThreadLocalDataSym tlds(sym.kind());
    llvm::cantFail(
        SymbolDeserializer::deserializeAs<ThreadLocalDataSym>(sym, tlds));
    ti = tlds.Type;
    name = tlds.Name;
    section = tlds.Segment;
    offset = tlds.DataOffset;
    addr = m_index->MakeVirtualAddress(tlds.Segment, tlds.DataOffset);
    scope = eValueTypeVariableThreadLocal;
    break;
  }
  default:
    llvm_unreachable("unreachable!");
  }

  CompUnitSP comp_unit;
  llvm::Optional<uint16_t> modi = m_index->GetModuleIndexForVa(addr);
  if (modi) {
    CompilandIndexItem &cci = m_index->compilands().GetOrCreateCompiland(*modi);
    comp_unit = GetOrCreateCompileUnit(cci);
  }

  Declaration decl;
  PdbTypeSymId tid(ti, false);
  SymbolFileTypeSP type_sp =
      std::make_shared<SymbolFileType>(*this, toOpaqueUid(tid));
  Variable::RangeList ranges;

  m_ast->GetOrCreateVariableDecl(var_id);

  DWARFExpression location = MakeGlobalLocationExpression(
      section, offset, GetObjectFile()->GetModule());

  std::string global_name("::");
  global_name += name;
  bool artificial = false;
  bool location_is_constant_data = false;
  bool static_member = false;
  VariableSP var_sp = std::make_shared<Variable>(
      toOpaqueUid(var_id), name.str().c_str(), global_name.c_str(), type_sp,
      scope, comp_unit.get(), ranges, &decl, location, is_external, artificial,
      location_is_constant_data, static_member);

  return var_sp;
}

lldb::VariableSP
SymbolFileNativePDB::CreateConstantSymbol(PdbGlobalSymId var_id,
                                          const CVSymbol &cvs) {
  TpiStream &tpi = m_index->tpi();
  ConstantSym constant(cvs.kind());

  llvm::cantFail(SymbolDeserializer::deserializeAs<ConstantSym>(cvs, constant));
  std::string global_name("::");
  global_name += constant.Name;
  PdbTypeSymId tid(constant.Type, false);
  SymbolFileTypeSP type_sp =
      std::make_shared<SymbolFileType>(*this, toOpaqueUid(tid));

  Declaration decl;
  Variable::RangeList ranges;
  ModuleSP module = GetObjectFile()->GetModule();
  DWARFExpression location = MakeConstantLocationExpression(
      constant.Type, tpi, constant.Value, module);

  bool external = false;
  bool artificial = false;
  bool location_is_constant_data = true;
  bool static_member = false;
  VariableSP var_sp = std::make_shared<Variable>(
      toOpaqueUid(var_id), constant.Name.str().c_str(), global_name.c_str(),
      type_sp, eValueTypeVariableGlobal, module.get(), ranges, &decl, location,
      external, artificial, location_is_constant_data, static_member);
  return var_sp;
}

VariableSP
SymbolFileNativePDB::GetOrCreateGlobalVariable(PdbGlobalSymId var_id) {
  auto emplace_result = m_global_vars.try_emplace(toOpaqueUid(var_id), nullptr);
  if (emplace_result.second)
    emplace_result.first->second = CreateGlobalVariable(var_id);

  return emplace_result.first->second;
}

lldb::TypeSP SymbolFileNativePDB::GetOrCreateType(TypeIndex ti) {
  return GetOrCreateType(PdbTypeSymId(ti, false));
}

FunctionSP SymbolFileNativePDB::GetOrCreateFunction(PdbCompilandSymId func_id,
                                                    CompileUnit &comp_unit) {
  auto emplace_result = m_functions.try_emplace(toOpaqueUid(func_id), nullptr);
  if (emplace_result.second)
    emplace_result.first->second = CreateFunction(func_id, comp_unit);

  return emplace_result.first->second;
}

CompUnitSP
SymbolFileNativePDB::GetOrCreateCompileUnit(const CompilandIndexItem &cci) {

  auto emplace_result =
      m_compilands.try_emplace(toOpaqueUid(cci.m_id), nullptr);
  if (emplace_result.second)
    emplace_result.first->second = CreateCompileUnit(cci);

  lldbassert(emplace_result.first->second);
  return emplace_result.first->second;
}

Block &SymbolFileNativePDB::GetOrCreateBlock(PdbCompilandSymId block_id) {
  auto iter = m_blocks.find(toOpaqueUid(block_id));
  if (iter != m_blocks.end())
    return *iter->second;

  return CreateBlock(block_id);
}

void SymbolFileNativePDB::ParseDeclsForContext(
    lldb_private::CompilerDeclContext decl_ctx) {
  clang::DeclContext *context = m_ast->FromCompilerDeclContext(decl_ctx);
  if (!context)
    return;
  m_ast->ParseDeclsForContext(*context);
}

lldb::CompUnitSP SymbolFileNativePDB::ParseCompileUnitAtIndex(uint32_t index) {
  if (index >= GetNumCompileUnits())
    return CompUnitSP();
  lldbassert(index < UINT16_MAX);
  if (index >= UINT16_MAX)
    return nullptr;

  CompilandIndexItem &item = m_index->compilands().GetOrCreateCompiland(index);

  return GetOrCreateCompileUnit(item);
}

lldb::LanguageType SymbolFileNativePDB::ParseLanguage(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  PdbSymUid uid(comp_unit.GetID());
  lldbassert(uid.kind() == PdbSymUidKind::Compiland);

  CompilandIndexItem *item =
      m_index->compilands().GetCompiland(uid.asCompiland().modi);
  lldbassert(item);
  if (!item->m_compile_opts)
    return lldb::eLanguageTypeUnknown;

  return TranslateLanguage(item->m_compile_opts->getLanguage());
}

void SymbolFileNativePDB::AddSymbols(Symtab &symtab) {}

size_t SymbolFileNativePDB::ParseFunctions(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  PdbSymUid uid{comp_unit.GetID()};
  lldbassert(uid.kind() == PdbSymUidKind::Compiland);
  uint16_t modi = uid.asCompiland().modi;
  CompilandIndexItem &cii = m_index->compilands().GetOrCreateCompiland(modi);

  size_t count = comp_unit.GetNumFunctions();
  const CVSymbolArray &syms = cii.m_debug_stream.getSymbolArray();
  for (auto iter = syms.begin(); iter != syms.end(); ++iter) {
    if (iter->kind() != S_LPROC32 && iter->kind() != S_GPROC32)
      continue;

    PdbCompilandSymId sym_id{modi, iter.offset()};

    FunctionSP func = GetOrCreateFunction(sym_id, comp_unit);
  }

  size_t new_count = comp_unit.GetNumFunctions();
  lldbassert(new_count >= count);
  return new_count - count;
}

static bool NeedsResolvedCompileUnit(uint32_t resolve_scope) {
  // If any of these flags are set, we need to resolve the compile unit.
  uint32_t flags = eSymbolContextCompUnit;
  flags |= eSymbolContextVariable;
  flags |= eSymbolContextFunction;
  flags |= eSymbolContextBlock;
  flags |= eSymbolContextLineEntry;
  return (resolve_scope & flags) != 0;
}

uint32_t SymbolFileNativePDB::ResolveSymbolContext(
    const Address &addr, SymbolContextItem resolve_scope, SymbolContext &sc) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  uint32_t resolved_flags = 0;
  lldb::addr_t file_addr = addr.GetFileAddress();

  if (NeedsResolvedCompileUnit(resolve_scope)) {
    llvm::Optional<uint16_t> modi = m_index->GetModuleIndexForVa(file_addr);
    if (!modi)
      return 0;
    CompUnitSP cu_sp = GetCompileUnitAtIndex(modi.getValue());
    if (!cu_sp)
      return 0;

    sc.comp_unit = cu_sp.get();
    resolved_flags |= eSymbolContextCompUnit;
  }

  if (resolve_scope & eSymbolContextFunction ||
      resolve_scope & eSymbolContextBlock) {
    lldbassert(sc.comp_unit);
    std::vector<SymbolAndUid> matches = m_index->FindSymbolsByVa(file_addr);
    // Search the matches in reverse.  This way if there are multiple matches
    // (for example we are 3 levels deep in a nested scope) it will find the
    // innermost one first.
    for (const auto &match : llvm::reverse(matches)) {
      if (match.uid.kind() != PdbSymUidKind::CompilandSym)
        continue;

      PdbCompilandSymId csid = match.uid.asCompilandSym();
      CVSymbol cvs = m_index->ReadSymbolRecord(csid);
      PDB_SymType type = CVSymToPDBSym(cvs.kind());
      if (type != PDB_SymType::Function && type != PDB_SymType::Block)
        continue;
      if (type == PDB_SymType::Function) {
        sc.function = GetOrCreateFunction(csid, *sc.comp_unit).get();
        Block &block = sc.function->GetBlock(true);
        addr_t func_base =
            sc.function->GetAddressRange().GetBaseAddress().GetFileAddress();
        addr_t offset = file_addr - func_base;
        sc.block = block.FindInnermostBlockByOffset(offset);
      }

      if (type == PDB_SymType::Block) {
        sc.block = &GetOrCreateBlock(csid);
        sc.function = sc.block->CalculateSymbolContextFunction();
      }
      if (sc.function)
        resolved_flags |= eSymbolContextFunction;
      if (sc.block)
        resolved_flags |= eSymbolContextBlock;
      break;
    }
  }

  if (resolve_scope & eSymbolContextLineEntry) {
    lldbassert(sc.comp_unit);
    if (auto *line_table = sc.comp_unit->GetLineTable()) {
      if (line_table->FindLineEntryByAddress(addr, sc.line_entry))
        resolved_flags |= eSymbolContextLineEntry;
    }
  }

  return resolved_flags;
}

uint32_t SymbolFileNativePDB::ResolveSymbolContext(
    const SourceLocationSpec &src_location_spec,
    lldb::SymbolContextItem resolve_scope, SymbolContextList &sc_list) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  const uint32_t prev_size = sc_list.GetSize();
  if (resolve_scope & eSymbolContextCompUnit) {
    for (uint32_t cu_idx = 0, num_cus = GetNumCompileUnits(); cu_idx < num_cus;
         ++cu_idx) {
      CompileUnit *cu = ParseCompileUnitAtIndex(cu_idx).get();
      if (!cu)
        continue;

      bool file_spec_matches_cu_file_spec = FileSpec::Match(
          src_location_spec.GetFileSpec(), cu->GetPrimaryFile());
      if (file_spec_matches_cu_file_spec) {
        cu->ResolveSymbolContext(src_location_spec, resolve_scope, sc_list);
        break;
      }
    }
  }
  return sc_list.GetSize() - prev_size;
}

bool SymbolFileNativePDB::ParseLineTable(CompileUnit &comp_unit) {
  // Unfortunately LLDB is set up to parse the entire compile unit line table
  // all at once, even if all it really needs is line info for a specific
  // function.  In the future it would be nice if it could set the sc.m_function
  // member, and we could only get the line info for the function in question.
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  PdbSymUid cu_id(comp_unit.GetID());
  lldbassert(cu_id.kind() == PdbSymUidKind::Compiland);
  uint16_t modi = cu_id.asCompiland().modi;
  CompilandIndexItem *cii = m_index->compilands().GetCompiland(modi);
  lldbassert(cii);

  // Parse DEBUG_S_LINES subsections first, then parse all S_INLINESITE records
  // in this CU. Add line entries into the set first so that if there are line
  // entries with same addres, the later is always more accurate than the
  // former.
  std::set<LineTable::Entry, LineTableEntryComparator> line_set;

  // This is basically a copy of the .debug$S subsections from all original COFF
  // object files merged together with address relocations applied.  We are
  // looking for all DEBUG_S_LINES subsections.
  for (const DebugSubsectionRecord &dssr :
       cii->m_debug_stream.getSubsectionsArray()) {
    if (dssr.kind() != DebugSubsectionKind::Lines)
      continue;

    DebugLinesSubsectionRef lines;
    llvm::BinaryStreamReader reader(dssr.getRecordData());
    if (auto EC = lines.initialize(reader)) {
      llvm::consumeError(std::move(EC));
      return false;
    }

    const LineFragmentHeader *lfh = lines.header();
    uint64_t virtual_addr =
        m_index->MakeVirtualAddress(lfh->RelocSegment, lfh->RelocOffset);

    for (const LineColumnEntry &group : lines) {
      llvm::Expected<uint32_t> file_index_or_err =
          GetFileIndex(*cii, group.NameIndex);
      if (!file_index_or_err)
        continue;
      uint32_t file_index = file_index_or_err.get();
      lldbassert(!group.LineNumbers.empty());
      CompilandIndexItem::GlobalLineTable::Entry line_entry(
          LLDB_INVALID_ADDRESS, 0);
      for (const LineNumberEntry &entry : group.LineNumbers) {
        LineInfo cur_info(entry.Flags);

        if (cur_info.isAlwaysStepInto() || cur_info.isNeverStepInto())
          continue;

        uint64_t addr = virtual_addr + entry.Offset;

        bool is_statement = cur_info.isStatement();
        bool is_prologue = IsFunctionPrologue(*cii, addr);
        bool is_epilogue = IsFunctionEpilogue(*cii, addr);

        uint32_t lno = cur_info.getStartLine();

        line_set.emplace(addr, lno, 0, file_index, is_statement, false,
                         is_prologue, is_epilogue, false);

        if (line_entry.GetRangeBase() != LLDB_INVALID_ADDRESS) {
          line_entry.SetRangeEnd(addr);
          cii->m_global_line_table.Append(line_entry);
        }
        line_entry.SetRangeBase(addr);
        line_entry.data = {file_index, lno};
      }
      LineInfo last_line(group.LineNumbers.back().Flags);
      line_set.emplace(virtual_addr + lfh->CodeSize, last_line.getEndLine(), 0,
                       file_index, false, false, false, false, true);

      if (line_entry.GetRangeBase() != LLDB_INVALID_ADDRESS) {
        line_entry.SetRangeEnd(virtual_addr + lfh->CodeSize);
        cii->m_global_line_table.Append(line_entry);
      }
    }
  }

  cii->m_global_line_table.Sort();

  // Parse all S_INLINESITE in this CU.
  const CVSymbolArray &syms = cii->m_debug_stream.getSymbolArray();
  for (auto iter = syms.begin(); iter != syms.end();) {
    if (iter->kind() != S_LPROC32 && iter->kind() != S_GPROC32) {
      ++iter;
      continue;
    }

    uint32_t record_offset = iter.offset();
    CVSymbol func_record =
        cii->m_debug_stream.readSymbolAtOffset(record_offset);
    SegmentOffsetLength sol = GetSegmentOffsetAndLength(func_record);
    addr_t file_vm_addr = m_index->MakeVirtualAddress(sol.so);
    AddressRange func_range(file_vm_addr, sol.length,
                            comp_unit.GetModule()->GetSectionList());
    Address func_base = func_range.GetBaseAddress();
    PdbCompilandSymId func_id{modi, record_offset};

    // Iterate all S_INLINESITEs in the function.
    auto parse_inline_sites = [&](SymbolKind kind, PdbCompilandSymId id) {
      if (kind != S_INLINESITE)
        return false;

      ParseInlineSite(id, func_base);

      for (const auto &line_entry :
           m_inline_sites[toOpaqueUid(id)]->line_entries) {
        // If line_entry is not terminal entry, remove previous line entry at
        // the same address and insert new one. Terminal entry inside an inline
        // site might not be terminal entry for its parent.
        if (!line_entry.is_terminal_entry)
          line_set.erase(line_entry);
        line_set.insert(line_entry);
      }
      // No longer useful after adding to line_set.
      m_inline_sites[toOpaqueUid(id)]->line_entries.clear();
      return true;
    };
    ParseSymbolArrayInScope(func_id, parse_inline_sites);
    // Jump to the end of the function record.
    iter = syms.at(getScopeEndOffset(func_record));
  }

  cii->m_global_line_table.Clear();

  // Add line entries in line_set to line_table.
  auto line_table = std::make_unique<LineTable>(&comp_unit);
  std::unique_ptr<LineSequence> sequence(
      line_table->CreateLineSequenceContainer());
  for (const auto &line_entry : line_set) {
    line_table->AppendLineEntryToSequence(
        sequence.get(), line_entry.file_addr, line_entry.line,
        line_entry.column, line_entry.file_idx,
        line_entry.is_start_of_statement, line_entry.is_start_of_basic_block,
        line_entry.is_prologue_end, line_entry.is_epilogue_begin,
        line_entry.is_terminal_entry);
  }
  line_table->InsertSequence(sequence.get());

  if (line_table->GetSize() == 0)
    return false;

  comp_unit.SetLineTable(line_table.release());
  return true;
}

bool SymbolFileNativePDB::ParseDebugMacros(CompileUnit &comp_unit) {
  // PDB doesn't contain information about macros
  return false;
}

llvm::Expected<uint32_t>
SymbolFileNativePDB::GetFileIndex(const CompilandIndexItem &cii,
                                  uint32_t file_id) {
  auto index_iter = m_file_indexes.find(file_id);
  if (index_iter != m_file_indexes.end())
    return index_iter->getSecond();
  const auto &checksums = cii.m_strings.checksums().getArray();
  const auto &strings = cii.m_strings.strings();
  // Indices in this structure are actually offsets of records in the
  // DEBUG_S_FILECHECKSUMS subsection.  Those entries then have an index
  // into the global PDB string table.
  auto iter = checksums.at(file_id);
  if (iter == checksums.end())
    return llvm::make_error<RawError>(raw_error_code::no_entry);

  llvm::Expected<llvm::StringRef> efn = strings.getString(iter->FileNameOffset);
  if (!efn) {
    return efn.takeError();
  }

  // LLDB wants the index of the file in the list of support files.
  auto fn_iter = llvm::find(cii.m_file_list, *efn);
  lldbassert(fn_iter != cii.m_file_list.end());
  m_file_indexes[file_id] = std::distance(cii.m_file_list.begin(), fn_iter);
  return m_file_indexes[file_id];
}

bool SymbolFileNativePDB::ParseSupportFiles(CompileUnit &comp_unit,
                                            FileSpecList &support_files) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  PdbSymUid cu_id(comp_unit.GetID());
  lldbassert(cu_id.kind() == PdbSymUidKind::Compiland);
  CompilandIndexItem *cci =
      m_index->compilands().GetCompiland(cu_id.asCompiland().modi);
  lldbassert(cci);

  for (llvm::StringRef f : cci->m_file_list) {
    FileSpec::Style style =
        f.startswith("/") ? FileSpec::Style::posix : FileSpec::Style::windows;
    FileSpec spec(f, style);
    support_files.Append(spec);
  }
  return true;
}

bool SymbolFileNativePDB::ParseImportedModules(
    const SymbolContext &sc, std::vector<SourceModule> &imported_modules) {
  // PDB does not yet support module debug info
  return false;
}

void SymbolFileNativePDB::ParseInlineSite(PdbCompilandSymId id,
                                          Address func_addr) {
  lldb::user_id_t opaque_uid = toOpaqueUid(id);
  if (m_inline_sites.find(opaque_uid) != m_inline_sites.end())
    return;

  addr_t func_base = func_addr.GetFileAddress();
  CompilandIndexItem *cii = m_index->compilands().GetCompiland(id.modi);
  CVSymbol sym = cii->m_debug_stream.readSymbolAtOffset(id.offset);
  CompUnitSP comp_unit = GetOrCreateCompileUnit(*cii);

  InlineSiteSym inline_site(static_cast<SymbolRecordKind>(sym.kind()));
  cantFail(SymbolDeserializer::deserializeAs<InlineSiteSym>(sym, inline_site));
  PdbCompilandSymId parent_id(id.modi, inline_site.Parent);

  std::shared_ptr<InlineSite> inline_site_sp =
      std::make_shared<InlineSite>(parent_id);

  // Get the inlined function declaration info.
  auto iter = cii->m_inline_map.find(inline_site.Inlinee);
  if (iter == cii->m_inline_map.end())
    return;
  InlineeSourceLine inlinee_line = iter->second;

  const FileSpecList &files = comp_unit->GetSupportFiles();
  FileSpec decl_file;
  llvm::Expected<uint32_t> file_index_or_err =
      GetFileIndex(*cii, inlinee_line.Header->FileID);
  if (!file_index_or_err)
    return;
  uint32_t decl_file_idx = file_index_or_err.get();
  decl_file = files.GetFileSpecAtIndex(decl_file_idx);
  uint32_t decl_line = inlinee_line.Header->SourceLineNum;
  std::unique_ptr<Declaration> decl_up =
      std::make_unique<Declaration>(decl_file, decl_line);

  // Parse range and line info.
  uint32_t code_offset = 0;
  int32_t line_offset = 0;
  bool has_base = false;
  bool is_new_line_offset = false;

  bool is_start_of_statement = false;
  // The first instruction is the prologue end.
  bool is_prologue_end = true;

  auto change_code_offset = [&](uint32_t code_delta) {
    if (has_base) {
      inline_site_sp->ranges.Append(RangeSourceLineVector::Entry(
          code_offset, code_delta, decl_line + line_offset));
      is_prologue_end = false;
      is_start_of_statement = false;
    } else {
      is_start_of_statement = true;
    }
    has_base = true;
    code_offset += code_delta;

    if (is_new_line_offset) {
      LineTable::Entry line_entry(func_base + code_offset,
                                  decl_line + line_offset, 0, decl_file_idx,
                                  true, false, is_prologue_end, false, false);
      inline_site_sp->line_entries.push_back(line_entry);
      is_new_line_offset = false;
    }
  };
  auto change_code_length = [&](uint32_t length) {
    inline_site_sp->ranges.Append(RangeSourceLineVector::Entry(
        code_offset, length, decl_line + line_offset));
    has_base = false;

    LineTable::Entry end_line_entry(func_base + code_offset + length,
                                    decl_line + line_offset, 0, decl_file_idx,
                                    false, false, false, false, true);
    inline_site_sp->line_entries.push_back(end_line_entry);
  };
  auto change_line_offset = [&](int32_t line_delta) {
    line_offset += line_delta;
    if (has_base) {
      LineTable::Entry line_entry(
          func_base + code_offset, decl_line + line_offset, 0, decl_file_idx,
          is_start_of_statement, false, is_prologue_end, false, false);
      inline_site_sp->line_entries.push_back(line_entry);
    } else {
      // Add line entry in next call to change_code_offset.
      is_new_line_offset = true;
    }
  };

  for (auto &annot : inline_site.annotations()) {
    switch (annot.OpCode) {
    case BinaryAnnotationsOpCode::CodeOffset:
    case BinaryAnnotationsOpCode::ChangeCodeOffset:
    case BinaryAnnotationsOpCode::ChangeCodeOffsetBase:
      change_code_offset(annot.U1);
      break;
    case BinaryAnnotationsOpCode::ChangeLineOffset:
      change_line_offset(annot.S1);
      break;
    case BinaryAnnotationsOpCode::ChangeCodeLength:
      change_code_length(annot.U1);
      code_offset += annot.U1;
      break;
    case BinaryAnnotationsOpCode::ChangeCodeOffsetAndLineOffset:
      change_code_offset(annot.U1);
      change_line_offset(annot.S1);
      break;
    case BinaryAnnotationsOpCode::ChangeCodeLengthAndCodeOffset:
      change_code_offset(annot.U2);
      change_code_length(annot.U1);
      break;
    default:
      break;
    }
  }

  inline_site_sp->ranges.Sort();
  inline_site_sp->ranges.CombineConsecutiveEntriesWithEqualData();

  // Get the inlined function callsite info.
  std::unique_ptr<Declaration> callsite_up;
  if (!inline_site_sp->ranges.IsEmpty()) {
    auto *entry = inline_site_sp->ranges.GetEntryAtIndex(0);
    addr_t base_offset = entry->GetRangeBase();
    if (cii->m_debug_stream.readSymbolAtOffset(parent_id.offset).kind() ==
        S_INLINESITE) {
      // Its parent is another inline site, lookup parent site's range vector
      // for callsite line.
      ParseInlineSite(parent_id, func_base);
      std::shared_ptr<InlineSite> parent_site =
          m_inline_sites[toOpaqueUid(parent_id)];
      FileSpec &parent_decl_file =
          parent_site->inline_function_info->GetDeclaration().GetFile();
      if (auto *parent_entry =
              parent_site->ranges.FindEntryThatContains(base_offset)) {
        callsite_up =
            std::make_unique<Declaration>(parent_decl_file, parent_entry->data);
      }
    } else {
      // Its parent is a function, lookup global line table for callsite.
      if (auto *entry = cii->m_global_line_table.FindEntryThatContains(
              func_base + base_offset)) {
        const FileSpec &callsite_file =
            files.GetFileSpecAtIndex(entry->data.first);
        callsite_up =
            std::make_unique<Declaration>(callsite_file, entry->data.second);
      }
    }
  }

  // Get the inlined function name.
  CVType inlinee_cvt = m_index->ipi().getType(inline_site.Inlinee);
  std::string inlinee_name;
  if (inlinee_cvt.kind() == LF_MFUNC_ID) {
    MemberFuncIdRecord mfr;
    cantFail(
        TypeDeserializer::deserializeAs<MemberFuncIdRecord>(inlinee_cvt, mfr));
    LazyRandomTypeCollection &types = m_index->tpi().typeCollection();
    inlinee_name.append(std::string(types.getTypeName(mfr.ClassType)));
    inlinee_name.append("::");
    inlinee_name.append(mfr.getName().str());
  } else if (inlinee_cvt.kind() == LF_FUNC_ID) {
    FuncIdRecord fir;
    cantFail(TypeDeserializer::deserializeAs<FuncIdRecord>(inlinee_cvt, fir));
    TypeIndex parent_idx = fir.getParentScope();
    if (!parent_idx.isNoneType()) {
      LazyRandomTypeCollection &ids = m_index->ipi().typeCollection();
      inlinee_name.append(std::string(ids.getTypeName(parent_idx)));
      inlinee_name.append("::");
    }
    inlinee_name.append(fir.getName().str());
  }
  inline_site_sp->inline_function_info = std::make_shared<InlineFunctionInfo>(
      inlinee_name.c_str(), llvm::StringRef(), decl_up.get(),
      callsite_up.get());

  m_inline_sites[opaque_uid] = inline_site_sp;
}

size_t SymbolFileNativePDB::ParseBlocksRecursive(Function &func) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  PdbCompilandSymId func_id = PdbSymUid(func.GetID()).asCompilandSym();
  // After we iterate through inline sites inside the function, we already get
  // all the info needed, removing from the map to save memory.
  std::set<uint64_t> remove_uids;
  auto parse_blocks = [&](SymbolKind kind, PdbCompilandSymId id) {
    if (kind == S_GPROC32 || kind == S_LPROC32 || kind == S_BLOCK32 ||
        kind == S_INLINESITE) {
      GetOrCreateBlock(id);
      if (kind == S_INLINESITE)
        remove_uids.insert(toOpaqueUid(id));
      return true;
    }
    return false;
  };
  size_t count = ParseSymbolArrayInScope(func_id, parse_blocks);
  for (uint64_t uid : remove_uids) {
    m_inline_sites.erase(uid);
  }
  return count;
}

size_t SymbolFileNativePDB::ParseSymbolArrayInScope(
    PdbCompilandSymId parent_id,
    llvm::function_ref<bool(SymbolKind, PdbCompilandSymId)> fn) {
  CompilandIndexItem *cii = m_index->compilands().GetCompiland(parent_id.modi);
  CVSymbolArray syms =
      cii->m_debug_stream.getSymbolArrayForScope(parent_id.offset);

  size_t count = 1;
  for (auto iter = syms.begin(); iter != syms.end(); ++iter) {
    PdbCompilandSymId child_id(parent_id.modi, iter.offset());
    if (fn(iter->kind(), child_id))
      ++count;
  }

  return count;
}

void SymbolFileNativePDB::DumpClangAST(Stream &s) { m_ast->Dump(s); }

void SymbolFileNativePDB::FindGlobalVariables(
    ConstString name, const CompilerDeclContext &parent_decl_ctx,
    uint32_t max_matches, VariableList &variables) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  using SymbolAndOffset = std::pair<uint32_t, llvm::codeview::CVSymbol>;

  std::vector<SymbolAndOffset> results = m_index->globals().findRecordsByName(
      name.GetStringRef(), m_index->symrecords());
  for (const SymbolAndOffset &result : results) {
    VariableSP var;
    switch (result.second.kind()) {
    case SymbolKind::S_GDATA32:
    case SymbolKind::S_LDATA32:
    case SymbolKind::S_GTHREAD32:
    case SymbolKind::S_LTHREAD32:
    case SymbolKind::S_CONSTANT: {
      PdbGlobalSymId global(result.first, false);
      var = GetOrCreateGlobalVariable(global);
      variables.AddVariable(var);
      break;
    }
    default:
      continue;
    }
  }
}

void SymbolFileNativePDB::FindFunctions(
    ConstString name, const CompilerDeclContext &parent_decl_ctx,
    FunctionNameType name_type_mask, bool include_inlines,
    SymbolContextList &sc_list) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // For now we only support lookup by method name or full name.
  if (!(name_type_mask & eFunctionNameTypeFull ||
        name_type_mask & eFunctionNameTypeMethod))
    return;

  using SymbolAndOffset = std::pair<uint32_t, llvm::codeview::CVSymbol>;

  std::vector<SymbolAndOffset> matches = m_index->globals().findRecordsByName(
      name.GetStringRef(), m_index->symrecords());
  for (const SymbolAndOffset &match : matches) {
    if (match.second.kind() != S_PROCREF && match.second.kind() != S_LPROCREF)
      continue;
    ProcRefSym proc(match.second.kind());
    cantFail(SymbolDeserializer::deserializeAs<ProcRefSym>(match.second, proc));

    if (!IsValidRecord(proc))
      continue;

    CompilandIndexItem &cci =
        m_index->compilands().GetOrCreateCompiland(proc.modi());
    SymbolContext sc;

    sc.comp_unit = GetOrCreateCompileUnit(cci).get();
    PdbCompilandSymId func_id(proc.modi(), proc.SymOffset);
    sc.function = GetOrCreateFunction(func_id, *sc.comp_unit).get();

    sc_list.Append(sc);
  }
}

void SymbolFileNativePDB::FindFunctions(const RegularExpression &regex,
                                        bool include_inlines,
                                        SymbolContextList &sc_list) {}

void SymbolFileNativePDB::FindTypes(
    ConstString name, const CompilerDeclContext &parent_decl_ctx,
    uint32_t max_matches, llvm::DenseSet<SymbolFile *> &searched_symbol_files,
    TypeMap &types) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (!name)
    return;

  searched_symbol_files.clear();
  searched_symbol_files.insert(this);

  // There is an assumption 'name' is not a regex
  FindTypesByName(name.GetStringRef(), max_matches, types);
}

void SymbolFileNativePDB::FindTypes(
    llvm::ArrayRef<CompilerContext> pattern, LanguageSet languages,
    llvm::DenseSet<SymbolFile *> &searched_symbol_files, TypeMap &types) {}

void SymbolFileNativePDB::FindTypesByName(llvm::StringRef name,
                                          uint32_t max_matches,
                                          TypeMap &types) {

  std::vector<TypeIndex> matches = m_index->tpi().findRecordsByName(name);
  if (max_matches > 0 && max_matches < matches.size())
    matches.resize(max_matches);

  for (TypeIndex ti : matches) {
    TypeSP type = GetOrCreateType(ti);
    if (!type)
      continue;

    types.Insert(type);
  }
}

size_t SymbolFileNativePDB::ParseTypes(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  // Only do the full type scan the first time.
  if (m_done_full_type_scan)
    return 0;

  const size_t old_count = GetTypeList().GetSize();
  LazyRandomTypeCollection &types = m_index->tpi().typeCollection();

  // First process the entire TPI stream.
  for (auto ti = types.getFirst(); ti; ti = types.getNext(*ti)) {
    TypeSP type = GetOrCreateType(*ti);
    if (type)
      (void)type->GetFullCompilerType();
  }

  // Next look for S_UDT records in the globals stream.
  for (const uint32_t gid : m_index->globals().getGlobalsTable()) {
    PdbGlobalSymId global{gid, false};
    CVSymbol sym = m_index->ReadSymbolRecord(global);
    if (sym.kind() != S_UDT)
      continue;

    UDTSym udt = llvm::cantFail(SymbolDeserializer::deserializeAs<UDTSym>(sym));
    bool is_typedef = true;
    if (IsTagRecord(PdbTypeSymId{udt.Type, false}, m_index->tpi())) {
      CVType cvt = m_index->tpi().getType(udt.Type);
      llvm::StringRef name = CVTagRecord::create(cvt).name();
      if (name == udt.Name)
        is_typedef = false;
    }

    if (is_typedef)
      GetOrCreateTypedef(global);
  }

  const size_t new_count = GetTypeList().GetSize();

  m_done_full_type_scan = true;

  return new_count - old_count;
}

size_t
SymbolFileNativePDB::ParseVariablesForCompileUnit(CompileUnit &comp_unit,
                                                  VariableList &variables) {
  PdbSymUid sym_uid(comp_unit.GetID());
  lldbassert(sym_uid.kind() == PdbSymUidKind::Compiland);
  return 0;
}

VariableSP SymbolFileNativePDB::CreateLocalVariable(PdbCompilandSymId scope_id,
                                                    PdbCompilandSymId var_id,
                                                    bool is_param) {
  ModuleSP module = GetObjectFile()->GetModule();
  Block &block = GetOrCreateBlock(scope_id);
  VariableInfo var_info =
      GetVariableLocationInfo(*m_index, var_id, block, module);
  if (!var_info.location || !var_info.ranges)
    return nullptr;

  CompilandIndexItem *cii = m_index->compilands().GetCompiland(var_id.modi);
  CompUnitSP comp_unit_sp = GetOrCreateCompileUnit(*cii);
  TypeSP type_sp = GetOrCreateType(var_info.type);
  std::string name = var_info.name.str();
  Declaration decl;
  SymbolFileTypeSP sftype =
      std::make_shared<SymbolFileType>(*this, type_sp->GetID());

  ValueType var_scope =
      is_param ? eValueTypeVariableArgument : eValueTypeVariableLocal;
  bool external = false;
  bool artificial = false;
  bool location_is_constant_data = false;
  bool static_member = false;
  VariableSP var_sp = std::make_shared<Variable>(
      toOpaqueUid(var_id), name.c_str(), name.c_str(), sftype, var_scope,
      comp_unit_sp.get(), *var_info.ranges, &decl, *var_info.location, external,
      artificial, location_is_constant_data, static_member);

  if (!is_param)
    m_ast->GetOrCreateVariableDecl(scope_id, var_id);

  m_local_variables[toOpaqueUid(var_id)] = var_sp;
  return var_sp;
}

VariableSP SymbolFileNativePDB::GetOrCreateLocalVariable(
    PdbCompilandSymId scope_id, PdbCompilandSymId var_id, bool is_param) {
  auto iter = m_local_variables.find(toOpaqueUid(var_id));
  if (iter != m_local_variables.end())
    return iter->second;

  return CreateLocalVariable(scope_id, var_id, is_param);
}

TypeSP SymbolFileNativePDB::CreateTypedef(PdbGlobalSymId id) {
  CVSymbol sym = m_index->ReadSymbolRecord(id);
  lldbassert(sym.kind() == SymbolKind::S_UDT);

  UDTSym udt = llvm::cantFail(SymbolDeserializer::deserializeAs<UDTSym>(sym));

  TypeSP target_type = GetOrCreateType(udt.Type);

  (void)m_ast->GetOrCreateTypedefDecl(id);

  Declaration decl;
  return std::make_shared<lldb_private::Type>(
      toOpaqueUid(id), this, ConstString(udt.Name),
      target_type->GetByteSize(nullptr), nullptr, target_type->GetID(),
      lldb_private::Type::eEncodingIsTypedefUID, decl,
      target_type->GetForwardCompilerType(),
      lldb_private::Type::ResolveState::Forward);
}

TypeSP SymbolFileNativePDB::GetOrCreateTypedef(PdbGlobalSymId id) {
  auto iter = m_types.find(toOpaqueUid(id));
  if (iter != m_types.end())
    return iter->second;

  return CreateTypedef(id);
}

size_t SymbolFileNativePDB::ParseVariablesForBlock(PdbCompilandSymId block_id) {
  Block &block = GetOrCreateBlock(block_id);

  size_t count = 0;

  CompilandIndexItem *cii = m_index->compilands().GetCompiland(block_id.modi);
  CVSymbol sym = cii->m_debug_stream.readSymbolAtOffset(block_id.offset);
  uint32_t params_remaining = 0;
  switch (sym.kind()) {
  case S_GPROC32:
  case S_LPROC32: {
    ProcSym proc(static_cast<SymbolRecordKind>(sym.kind()));
    cantFail(SymbolDeserializer::deserializeAs<ProcSym>(sym, proc));
    CVType signature = m_index->tpi().getType(proc.FunctionType);
    ProcedureRecord sig;
    cantFail(TypeDeserializer::deserializeAs<ProcedureRecord>(signature, sig));
    params_remaining = sig.getParameterCount();
    break;
  }
  case S_BLOCK32:
    break;
  case S_INLINESITE:
    // TODO: Handle inline site case.
    return 0;
  default:
    lldbassert(false && "Symbol is not a block!");
    return 0;
  }

  VariableListSP variables = block.GetBlockVariableList(false);
  if (!variables) {
    variables = std::make_shared<VariableList>();
    block.SetVariableList(variables);
  }

  CVSymbolArray syms = limitSymbolArrayToScope(
      cii->m_debug_stream.getSymbolArray(), block_id.offset);

  // Skip the first record since it's a PROC32 or BLOCK32, and there's
  // no point examining it since we know it's not a local variable.
  syms.drop_front();
  auto iter = syms.begin();
  auto end = syms.end();

  while (iter != end) {
    uint32_t record_offset = iter.offset();
    CVSymbol variable_cvs = *iter;
    PdbCompilandSymId child_sym_id(block_id.modi, record_offset);
    ++iter;

    // If this is a block, recurse into its children and then skip it.
    if (variable_cvs.kind() == S_BLOCK32) {
      uint32_t block_end = getScopeEndOffset(variable_cvs);
      count += ParseVariablesForBlock(child_sym_id);
      iter = syms.at(block_end);
      continue;
    }

    bool is_param = params_remaining > 0;
    VariableSP variable;
    switch (variable_cvs.kind()) {
    case S_REGREL32:
    case S_REGISTER:
    case S_LOCAL:
      variable = GetOrCreateLocalVariable(block_id, child_sym_id, is_param);
      if (is_param)
        --params_remaining;
      if (variable)
        variables->AddVariableIfUnique(variable);
      break;
    default:
      break;
    }
  }

  // Pass false for set_children, since we call this recursively so that the
  // children will call this for themselves.
  block.SetDidParseVariables(true, false);

  return count;
}

size_t SymbolFileNativePDB::ParseVariablesForContext(const SymbolContext &sc) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  lldbassert(sc.function || sc.comp_unit);

  VariableListSP variables;
  if (sc.block) {
    PdbSymUid block_id(sc.block->GetID());

    size_t count = ParseVariablesForBlock(block_id.asCompilandSym());
    return count;
  }

  if (sc.function) {
    PdbSymUid block_id(sc.function->GetID());

    size_t count = ParseVariablesForBlock(block_id.asCompilandSym());
    return count;
  }

  if (sc.comp_unit) {
    variables = sc.comp_unit->GetVariableList(false);
    if (!variables) {
      variables = std::make_shared<VariableList>();
      sc.comp_unit->SetVariableList(variables);
    }
    return ParseVariablesForCompileUnit(*sc.comp_unit, *variables);
  }

  llvm_unreachable("Unreachable!");
}

CompilerDecl SymbolFileNativePDB::GetDeclForUID(lldb::user_id_t uid) {
  if (auto decl = m_ast->GetOrCreateDeclForUid(uid))
    return decl.getValue();
  else
    return CompilerDecl();
}

CompilerDeclContext
SymbolFileNativePDB::GetDeclContextForUID(lldb::user_id_t uid) {
  clang::DeclContext *context =
      m_ast->GetOrCreateDeclContextForUid(PdbSymUid(uid));
  if (!context)
    return {};

  return m_ast->ToCompilerDeclContext(*context);
}

CompilerDeclContext
SymbolFileNativePDB::GetDeclContextContainingUID(lldb::user_id_t uid) {
  clang::DeclContext *context = m_ast->GetParentDeclContext(PdbSymUid(uid));
  return m_ast->ToCompilerDeclContext(*context);
}

Type *SymbolFileNativePDB::ResolveTypeUID(lldb::user_id_t type_uid) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  auto iter = m_types.find(type_uid);
  // lldb should not be passing us non-sensical type uids.  the only way it
  // could have a type uid in the first place is if we handed it out, in which
  // case we should know about the type.  However, that doesn't mean we've
  // instantiated it yet.  We can vend out a UID for a future type.  So if the
  // type doesn't exist, let's instantiate it now.
  if (iter != m_types.end())
    return &*iter->second;

  PdbSymUid uid(type_uid);
  lldbassert(uid.kind() == PdbSymUidKind::Type);
  PdbTypeSymId type_id = uid.asTypeSym();
  if (type_id.index.isNoneType())
    return nullptr;

  TypeSP type_sp = CreateAndCacheType(type_id);
  return &*type_sp;
}

llvm::Optional<SymbolFile::ArrayInfo>
SymbolFileNativePDB::GetDynamicArrayInfoForUID(
    lldb::user_id_t type_uid, const lldb_private::ExecutionContext *exe_ctx) {
  return llvm::None;
}


bool SymbolFileNativePDB::CompleteType(CompilerType &compiler_type) {
  clang::QualType qt =
      clang::QualType::getFromOpaquePtr(compiler_type.GetOpaqueQualType());

  return m_ast->CompleteType(qt);
}

void SymbolFileNativePDB::GetTypes(lldb_private::SymbolContextScope *sc_scope,
                                   TypeClass type_mask,
                                   lldb_private::TypeList &type_list) {}

CompilerDeclContext
SymbolFileNativePDB::FindNamespace(ConstString name,
                                   const CompilerDeclContext &parent_decl_ctx) {
  return {};
}

llvm::Expected<TypeSystem &>
SymbolFileNativePDB::GetTypeSystemForLanguage(lldb::LanguageType language) {
  auto type_system_or_err =
      m_objfile_sp->GetModule()->GetTypeSystemForLanguage(language);
  if (type_system_or_err) {
    type_system_or_err->SetSymbolFile(this);
  }
  return type_system_or_err;
}

uint64_t SymbolFileNativePDB::GetDebugInfoSize() {
  // PDB files are a separate file that contains all debug info.
  return m_index->pdb().getFileSize();
}

