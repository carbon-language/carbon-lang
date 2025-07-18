// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_cpp.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Lookup.h"
#include "common/check.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/class.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Generates C++ file contents to #include all requested imports.
static auto GenerateCppIncludesHeaderCode(
    Context& context, llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> std::string {
  std::string code;
  llvm::raw_string_ostream code_stream(code);
  for (const Parse::Tree::PackagingNames& import : imports) {
    // Add a line marker directive pointing at the location of the `import Cpp`
    // declaration in the Carbon source file. This will cause Clang's
    // diagnostics machinery to track and report the location in Carbon code
    // where the import was written.
    auto token = context.parse_tree().node_token(import.node_id);
    code_stream << "# " << context.tokens().GetLineNumber(token) << " \""
                << FormatEscaped(context.tokens().source().filename())
                << "\"\n";

    code_stream << "#include \""
                << FormatEscaped(
                       context.string_literal_values().Get(import.library_id))
                << "\"\n";
  }
  return code;
}

// Adds the name to the scope with the given `inst_id`, if the `inst_id` is not
// `None`.
static auto AddNameToScope(Context& context, SemIR::NameScopeId scope_id,
                           SemIR::NameId name_id, SemIR::InstId inst_id)
    -> void {
  if (inst_id.has_value()) {
    context.name_scopes().AddRequiredName(scope_id, name_id, inst_id);
  }
}

// Maps a Clang name to a Carbon `NameId`.
static auto AddIdentifierName(Context& context, llvm::StringRef name)
    -> SemIR::NameId {
  return SemIR::NameId::ForIdentifier(context.identifiers().Add(name));
}

// Adds the given source location and an `ImportIRInst` referring to it in
// `ImportIRId::Cpp`.
static auto AddImportIRInst(Context& context,
                            clang::SourceLocation clang_source_loc)
    -> SemIR::ImportIRInstId {
  SemIR::ClangSourceLocId clang_source_loc_id =
      context.sem_ir().clang_source_locs().Add(clang_source_loc);
  return context.import_ir_insts().Add(
      SemIR::ImportIRInst(clang_source_loc_id));
}

namespace {

// Used to convert Clang diagnostics to Carbon diagnostics.
class CarbonClangDiagnosticConsumer : public clang::DiagnosticConsumer {
 public:
  // Creates an instance with the location that triggers calling Clang.
  // `context` must not be null.
  explicit CarbonClangDiagnosticConsumer(Context* context,
                                         clang::CompilerInvocation* invocation)
      : context_(context), invocation_(invocation) {}

  // Generates a Carbon warning for each Clang warning and a Carbon error for
  // each Clang error or fatal.
  auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                        const clang::Diagnostic& info) -> void override {
    DiagnosticConsumer::HandleDiagnostic(diag_level, info);

    SemIR::ImportIRInstId clang_import_ir_inst_id =
        AddImportIRInst(*context_, info.getLocation());

    llvm::SmallString<256> message;
    info.FormatDiagnostic(message);

    if (!info.hasSourceManager()) {
      // If we don't have a source manager, we haven't actually started
      // compiling yet, and this is an error from the driver or early in the
      // frontend. Pass it on directly.
      CARBON_CHECK(info.getLocation().isInvalid());
      diagnostic_infos_.push_back({.level = diag_level,
                                   .import_ir_inst_id = clang_import_ir_inst_id,
                                   .message = message.str().str()});
      return;
    }

    RawStringOstream diagnostics_stream;
    clang::TextDiagnostic text_diagnostic(diagnostics_stream,
                                          invocation_->getLangOpts(),
                                          invocation_->getDiagnosticOpts());
    text_diagnostic.emitDiagnostic(
        clang::FullSourceLoc(info.getLocation(), info.getSourceManager()),
        diag_level, message, info.getRanges(), info.getFixItHints());

    std::string diagnostics_str = diagnostics_stream.TakeStr();

    diagnostic_infos_.push_back({.level = diag_level,
                                 .import_ir_inst_id = clang_import_ir_inst_id,
                                 .message = diagnostics_str});
  }

  // Outputs Carbon diagnostics based on the collected Clang diagnostics. Must
  // be called after the AST is set in the context.
  auto EmitDiagnostics() -> void {
    for (const ClangDiagnosticInfo& info : diagnostic_infos_) {
      switch (info.level) {
        case clang::DiagnosticsEngine::Ignored:
        case clang::DiagnosticsEngine::Note:
        case clang::DiagnosticsEngine::Remark: {
          context_->TODO(
              SemIR::LocId(info.import_ir_inst_id),
              llvm::formatv(
                  "Unsupported: C++ diagnostic level for diagnostic\n{0}",
                  info.message));
          break;
        }
        case clang::DiagnosticsEngine::Warning:
        case clang::DiagnosticsEngine::Error:
        case clang::DiagnosticsEngine::Fatal: {
          CARBON_DIAGNOSTIC(CppInteropParseWarning, Warning, "{0}",
                            std::string);
          CARBON_DIAGNOSTIC(CppInteropParseError, Error, "{0}", std::string);
          context_->emitter().Emit(
              SemIR::LocId(info.import_ir_inst_id),
              info.level == clang::DiagnosticsEngine::Warning
                  ? CppInteropParseWarning
                  : CppInteropParseError,
              info.message);
          break;
        }
      }
    }
  }

 private:
  // The type-checking context in which we're running Clang.
  Context* context_;

  // The compiler invocation that is producing the diagnostics.
  clang::CompilerInvocation* invocation_;

  // Information on a Clang diagnostic that can be converted to a Carbon
  // diagnostic.
  struct ClangDiagnosticInfo {
    // The Clang diagnostic level.
    clang::DiagnosticsEngine::Level level;

    // The ID of the ImportIR instruction referring to the Clang source
    // location.
    SemIR::ImportIRInstId import_ir_inst_id;

    // The Clang diagnostic textual message.
    std::string message;
  };

  // Collects the information for all Clang diagnostics to be converted to
  // Carbon diagnostics after the context has been initialized with the Clang
  // AST.
  llvm::SmallVector<ClangDiagnosticInfo> diagnostic_infos_;
};

}  // namespace

// Returns an AST for the C++ imports and a bool that represents whether
// compilation errors where encountered or the generated AST is null due to an
// error. Sets the AST in the context's `sem_ir`.
// TODO: Consider to always have a (non-null) AST.
static auto GenerateAst(Context& context,
                        llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                        std::shared_ptr<clang::CompilerInvocation> invocation)
    -> std::pair<std::unique_ptr<clang::ASTUnit>, bool> {
  // Build a diagnostics engine.
  CarbonClangDiagnosticConsumer diagnostics_consumer(&context,
                                                     invocation.get());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
      clang::CompilerInstance::createDiagnostics(
          *fs, invocation->getDiagnosticOpts(), &diagnostics_consumer,
          /*ShouldOwnClient=*/false));

  // Extract the input from the frontend invocation and make sure it makes
  // sense.
  const auto& inputs = invocation->getFrontendOpts().Inputs;
  CARBON_CHECK(inputs.size() == 1 &&
               inputs[0].getKind().getLanguage() == clang::Language::CXX &&
               inputs[0].getKind().getFormat() == clang::InputKind::Source);
  llvm::StringRef file_name = inputs[0].getFile();

  // Remap the imports file name to the corresponding `#include`s.
  // TODO: Modify the frontend options to specify this memory buffer as input
  // instead of remapping the file.
  std::string includes = GenerateCppIncludesHeaderCode(context, imports);
  auto includes_buffer = llvm::MemoryBuffer::getMemBuffer(includes, file_name);
  invocation->getPreprocessorOpts().addRemappedFile(file_name,
                                                    includes_buffer.get());

  // Create the AST unit.
  auto ast = clang::ASTUnit::LoadFromCompilerInvocation(
      invocation, std::make_shared<clang::PCHContainerOperations>(), nullptr,
      diags, new clang::FileManager(invocation->getFileSystemOpts(), fs));

  // Remove link to the diagnostics consumer before its destruction.
  ast->getDiagnostics().setClient(nullptr);

  // Remove remapped file before its underlying storage is destroyed.
  invocation->getPreprocessorOpts().clearRemappedFiles();

  // Attach the AST to SemIR. This needs to be done before we can emit any
  // diagnostics, so their locations can be properly interpreted by our
  // diagnostics machinery.
  context.sem_ir().set_cpp_ast(ast.get());

  // Emit any diagnostics we queued up while building the AST.
  diagnostics_consumer.EmitDiagnostics();

  return {std::move(ast), !ast || diagnostics_consumer.getNumErrors() > 0};
}

// Adds a namespace for the `Cpp` import and returns its `NameScopeId`.
static auto AddNamespace(Context& context, PackageNameId cpp_package_id,
                         llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> SemIR::NameScopeId {
  auto& import_cpps = context.sem_ir().import_cpps();
  import_cpps.Reserve(imports.size());
  for (const Parse::Tree::PackagingNames& import : imports) {
    import_cpps.Add({.node_id = context.parse_tree().As<Parse::ImportDeclId>(
                         import.node_id),
                     .library_id = import.library_id});
  }

  return AddImportNamespaceToScope(
             context,
             GetSingletonType(context, SemIR::NamespaceType::TypeInstId),
             SemIR::NameId::ForPackageName(cpp_package_id),
             SemIR::NameScopeId::Package,
             /*diagnose_duplicate_namespace=*/false,
             [&]() {
               return AddInst<SemIR::ImportCppDecl>(
                   context,
                   context.parse_tree().As<Parse::ImportDeclId>(
                       imports.front().node_id),
                   {});
             })
      .add_result.name_scope_id;
}

auto ImportCppFiles(Context& context,
                    llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                    std::shared_ptr<clang::CompilerInvocation> invocation)
    -> std::unique_ptr<clang::ASTUnit> {
  if (imports.empty()) {
    return nullptr;
  }

  CARBON_CHECK(!context.sem_ir().cpp_ast());

  PackageNameId package_id = imports.front().package_id;
  CARBON_CHECK(
      llvm::all_of(imports, [&](const Parse::Tree::PackagingNames& import) {
        return import.package_id == package_id;
      }));
  auto name_scope_id = AddNamespace(context, package_id, imports);

  auto [generated_ast, ast_has_error] =
      GenerateAst(context, imports, fs, std::move(invocation));

  SemIR::NameScope& name_scope = context.name_scopes().Get(name_scope_id);
  name_scope.set_is_closed_import(true);
  name_scope.set_clang_decl_context_id(context.sem_ir().clang_decls().Add(
      {.decl = generated_ast->getASTContext().getTranslationUnitDecl(),
       .inst_id = name_scope.inst_id()}));

  if (ast_has_error) {
    name_scope.set_has_error();
  }

  return std::move(generated_ast);
}

// Look ups the given name in the Clang AST in a specific scope. Returns the
// lookup result if lookup was successful.
static auto ClangLookup(Context& context, SemIR::NameScopeId scope_id,
                        SemIR::NameId name_id)
    -> std::optional<clang::LookupResult> {
  std::optional<llvm::StringRef> name =
      context.names().GetAsStringIfIdentifier(name_id);
  if (!name) {
    // Special names never exist in C++ code.
    return std::nullopt;
  }

  clang::ASTUnit* ast = context.sem_ir().cpp_ast();
  CARBON_CHECK(ast);
  clang::Sema& sema = ast->getSema();

  clang::LookupResult lookup(
      sema,
      clang::DeclarationNameInfo(
          clang::DeclarationName(
              sema.getPreprocessor().getIdentifierInfo(*name)),
          clang::SourceLocation()),
      clang::Sema::LookupNameKind::LookupOrdinaryName);
  // TODO: Diagnose on access and return the `AccessKind` for storage. We'll
  // probably need a dedicated `DiagnosticConsumer` because
  // `TextDiagnosticPrinter` assumes we're processing a C++ source file.
  lookup.suppressDiagnostics();

  auto scope_clang_decl_context_id =
      context.name_scopes().Get(scope_id).clang_decl_context_id();
  bool found = sema.LookupQualifiedName(
      lookup,
      clang::dyn_cast<clang::DeclContext>(context.sem_ir()
                                              .clang_decls()
                                              .Get(scope_clang_decl_context_id)
                                              .decl));

  if (!found) {
    return std::nullopt;
  }

  return lookup;
}

// Returns whether `decl` already mapped to an instruction.
static auto IsClangDeclImported(const Context& context, clang::Decl* decl)
    -> bool {
  return context.sem_ir().clang_decls().Lookup(decl).has_value();
}

// If `decl` already mapped to an instruction, returns that instruction.
// Otherwise returns `None`.
static auto LookupClangDeclInstId(const Context& context, clang::Decl* decl)
    -> SemIR::InstId {
  const auto& clang_decls = context.sem_ir().clang_decls();
  if (auto context_clang_decl_id = clang_decls.Lookup(decl);
      context_clang_decl_id.has_value()) {
    return clang_decls.Get(context_clang_decl_id).inst_id;
  }
  return SemIR::InstId::None;
}

// Returns the parent of the given declaration. Skips declaration types we
// ignore.
static auto GetParentDecl(clang::Decl* clang_decl) -> clang::Decl* {
  clang::DeclContext* decl_context = clang_decl->getDeclContext();
  while (llvm::isa<clang::LinkageSpecDecl>(decl_context)) {
    decl_context = decl_context->getParent();
  }
  return llvm::cast<clang::Decl>(decl_context);
}

// Returns the given declaration's parent scope. Assumes the parent declaration
// was already imported.
static auto GetParentNameScopeId(Context& context, clang::Decl* clang_decl)
    -> SemIR::NameScopeId {
  SemIR::InstId parent_inst_id =
      LookupClangDeclInstId(context, GetParentDecl(clang_decl));
  CARBON_CHECK(parent_inst_id.has_value());

  CARBON_KIND_SWITCH(context.insts().Get(parent_inst_id)) {
    case CARBON_KIND(SemIR::ClassDecl class_decl): {
      return context.classes().Get(class_decl.class_id).scope_id;
    }
    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      return context.interfaces().Get(interface_decl.interface_id).scope_id;
    }
    case CARBON_KIND(SemIR::Namespace namespace_inst): {
      return namespace_inst.name_scope_id;
    }
    default: {
      CARBON_FATAL("Unexpected parent instruction kind");
    }
  }
}

// Imports a namespace declaration from Clang to Carbon. If successful, returns
// the new Carbon namespace declaration `InstId`. If the declaration was already
// imported, returns the mapped instruction.
static auto ImportNamespaceDecl(Context& context,
                                clang::NamespaceDecl* clang_decl)
    -> SemIR::InstId {
  // Check if the declaration is already mapped.
  if (SemIR::InstId existing_inst_id =
          LookupClangDeclInstId(context, clang_decl);
      existing_inst_id.has_value()) {
    return existing_inst_id;
  }
  auto result = AddImportNamespace(
      context, GetSingletonType(context, SemIR::NamespaceType::TypeInstId),
      AddIdentifierName(context, clang_decl->getName()),
      GetParentNameScopeId(context, clang_decl),
      /*import_id=*/SemIR::InstId::None);
  context.name_scopes()
      .Get(result.name_scope_id)
      .set_clang_decl_context_id(context.sem_ir().clang_decls().Add(
          {.decl = clang_decl, .inst_id = result.inst_id}));
  return result.inst_id;
}

// Creates a class declaration for the given class name in the given scope.
// Returns the `InstId` for the declaration.
static auto BuildClassDecl(Context& context, SemIR::NameScopeId parent_scope_id,
                           SemIR::NameId name_id)
    -> std::tuple<SemIR::ClassId, SemIR::InstId> {
  // Add the class declaration.
  auto class_decl = SemIR::ClassDecl{.type_id = SemIR::TypeType::TypeId,
                                     .class_id = SemIR::ClassId::None,
                                     .decl_block_id = SemIR::InstBlockId::None};
  // TODO: Consider setting a proper location.
  auto class_decl_id = AddPlaceholderInstInNoBlock(
      context, SemIR::LocIdAndInst::NoLoc(class_decl));
  context.imports().push_back(class_decl_id);

  SemIR::Class class_info = {
      {.name_id = name_id,
       .parent_scope_id = parent_scope_id,
       .generic_id = SemIR::GenericId::None,
       .first_param_node_id = Parse::NodeId::None,
       .last_param_node_id = Parse::NodeId::None,
       .pattern_block_id = SemIR::InstBlockId::None,
       .implicit_param_patterns_id = SemIR::InstBlockId::None,
       .param_patterns_id = SemIR::InstBlockId::None,
       .is_extern = false,
       .extern_library_id = SemIR::LibraryNameId::None,
       .non_owning_decl_id = SemIR::InstId::None,
       .first_owning_decl_id = class_decl_id},
      {// `.self_type_id` depends on the ClassType, so is set below.
       .self_type_id = SemIR::TypeId::None,
       // TODO: Support Dynamic classes.
       // TODO: Support Final classes.
       .inheritance_kind = SemIR::Class::Base}};

  class_decl.class_id = context.classes().Add(class_info);

  // Write the class ID into the ClassDecl.
  ReplaceInstBeforeConstantUse(context, class_decl_id, class_decl);

  SetClassSelfType(context, class_decl.class_id);

  return {class_decl.class_id, class_decl_id};
}

// Creates a class definition based on the information in the given Clang
// declaration, which is assumed to be for a class definition. Returns the new
// class id and instruction id.
static auto BuildClassDefinition(Context& context,
                                 clang::CXXRecordDecl* clang_decl)
    -> std::tuple<SemIR::ClassId, SemIR::InstId> {
  auto [class_id, class_inst_id] =
      BuildClassDecl(context, GetParentNameScopeId(context, clang_decl),
                     AddIdentifierName(context, clang_decl->getName()));
  auto& class_info = context.classes().Get(class_id);
  StartClassDefinition(context, class_info, class_inst_id);

  context.name_scopes()
      .Get(class_info.scope_id)
      .set_clang_decl_context_id(context.sem_ir().clang_decls().Add(
          {.decl = clang_decl, .inst_id = class_inst_id}));

  return {class_id, class_inst_id};
}

// Mark the given `Decl` as failed in `clang_decls`.
static auto MarkFailedDecl(Context& context, clang::Decl* clang_decl) {
  context.sem_ir().clang_decls().Add(
      {.decl = clang_decl, .inst_id = SemIR::ErrorInst::InstId});
}

// Imports a record declaration from Clang to Carbon. If successful, returns
// the new Carbon class declaration `InstId`.
// TODO: Change `clang_decl` to `const &` when lookup is using `clang::DeclID`
// and we don't need to store the decl for lookup context.
static auto ImportCXXRecordDecl(Context& context, SemIR::LocId loc_id,
                                clang::CXXRecordDecl* clang_decl)
    -> SemIR::InstId {
  clang::CXXRecordDecl* clang_def = clang_decl->getDefinition();
  if (!clang_def) {
    context.TODO(loc_id,
                 "Unsupported: Record declarations without a definition");
    MarkFailedDecl(context, clang_decl);
    return SemIR::ErrorInst::InstId;
  }

  if (clang_def->isDynamicClass()) {
    context.TODO(loc_id, "Unsupported: Dynamic Class");
    MarkFailedDecl(context, clang_decl);
    return SemIR::ErrorInst::InstId;
  }

  if (clang_def->isUnion() && !clang_def->fields().empty()) {
    context.TODO(loc_id, "Unsupported: Non-empty union");
    MarkFailedDecl(context, clang_decl);
    return SemIR::ErrorInst::InstId;
  }

  auto [class_id, class_def_id] = BuildClassDefinition(context, clang_def);

  // The class type is now fully defined. Compute its object representation.
  ComputeClassObjectRepr(context,
                         // TODO: Consider having a proper location here.
                         Parse::ClassDefinitionId::None, class_id,
                         // TODO: Set fields.
                         /*field_decls=*/{},
                         // TODO: Set vtable.
                         /*vtable_contents=*/{},
                         // TODO: Set block.
                         /*body=*/{});

  return class_def_id;
}

// Creates an integer type of the given size.
static auto MakeIntType(Context& context, IntId size_id) -> TypeExpr {
  // TODO: Fill in a location for the type once available.
  auto type_inst_id = MakeIntTypeLiteral(context, Parse::NodeId::None,
                                         SemIR::IntKind::Signed, size_id);
  return ExprAsType(context, Parse::NodeId::None, type_inst_id);
}

// Maps a C++ builtin type to a Carbon type.
// TODO: Support more builtin types.
static auto MapBuiltinType(Context& context, const clang::BuiltinType& type)
    -> TypeExpr {
  // TODO: Refactor to avoid duplication.
  switch (type.getKind()) {
    case clang::BuiltinType::Short:
      if (context.ast_context().getTypeSize(&type) == 16) {
        return MakeIntType(context, context.ints().Add(16));
      }
      break;
    case clang::BuiltinType::Int:
      if (context.ast_context().getTypeSize(&type) == 32) {
        return MakeIntType(context, context.ints().Add(32));
      }
      break;
    default:
      break;
  }
  return {.inst_id = SemIR::TypeInstId::None, .type_id = SemIR::TypeId::None};
}

// Maps a C++ record type to a Carbon type.
// TODO: Support more record types.
static auto MapRecordType(Context& context, SemIR::LocId loc_id,
                          const clang::RecordType& type) -> TypeExpr {
  auto* record_decl = clang::dyn_cast<clang::CXXRecordDecl>(type.getDecl());
  if (!record_decl) {
    return {.inst_id = SemIR::TypeInstId::None, .type_id = SemIR::TypeId::None};
  }

  // Check if the declaration is already mapped.
  SemIR::InstId record_inst_id = LookupClangDeclInstId(context, record_decl);
  if (!record_inst_id.has_value()) {
    record_inst_id = ImportCXXRecordDecl(context, loc_id, record_decl);
  }
  SemIR::TypeInstId record_type_inst_id =
      context.types().GetAsTypeInstId(record_inst_id);
  return {
      .inst_id = record_type_inst_id,
      .type_id = context.types().GetTypeIdForTypeInstId(record_type_inst_id)};
}

// Maps a C++ type that is not a wrapper type such as a pointer to a Carbon
// type.
// TODO: Support more types.
static auto MapNonWrapperType(Context& context, SemIR::LocId loc_id,
                              clang::QualType type) -> TypeExpr {
  if (const auto* builtin_type = type->getAs<clang::BuiltinType>()) {
    return MapBuiltinType(context, *builtin_type);
  }

  if (const auto* record_type = type->getAs<clang::RecordType>()) {
    return MapRecordType(context, loc_id, *record_type);
  }

  CARBON_CHECK(!type.hasQualifiers() && !type->isPointerType(),
               "Should not see wrapper types here");

  return {.inst_id = SemIR::TypeInstId::None, .type_id = SemIR::TypeId::None};
}

// Maps a qualified C++ type to a Carbon type.
static auto MapQualifiedType(Context& context, SemIR::LocId loc_id,
                             clang::QualType type, TypeExpr type_expr)
    -> TypeExpr {
  auto quals = type.getQualifiers();

  if (quals.hasConst()) {
    auto type_id = GetConstType(context, type_expr.inst_id);
    type_expr = {.inst_id = context.types().GetInstId(type_id),
                 .type_id = type_id};
    quals.removeConst();
  }

  // TODO: Support other qualifiers.
  if (!quals.empty()) {
    context.TODO(loc_id, llvm::formatv("Unsupported: qualified type: {0}",
                                       type.getAsString()));
    return {.inst_id = SemIR::ErrorInst::TypeInstId,
            .type_id = SemIR::ErrorInst::TypeId};
  }

  return type_expr;
}

// Maps a C++ pointer type to a Carbon pointer type.
static auto MapPointerType(Context& context, SemIR::LocId loc_id,
                           clang::QualType type, TypeExpr pointee_type_expr)
    -> TypeExpr {
  CARBON_CHECK(type->isPointerType());

  if (auto nullability = type->getNullability();
      !nullability.has_value() ||
      *nullability != clang::NullabilityKind::NonNull) {
    context.TODO(loc_id, llvm::formatv("Unsupported: nullable pointer: {0}",
                                       type.getAsString()));
    return {.inst_id = SemIR::ErrorInst::TypeInstId,
            .type_id = SemIR::ErrorInst::TypeId};
  }

  SemIR::TypeId pointer_type_id =
      GetPointerType(context, pointee_type_expr.inst_id);
  return {.inst_id = context.types().GetInstId(pointer_type_id),
          .type_id = pointer_type_id};
}

// Maps a C++ type to a Carbon type. `type` should not be canonicalized because
// we check for pointer nullability and nullability will be lost by
// canonicalization.
static auto MapType(Context& context, SemIR::LocId loc_id, clang::QualType type)
    -> TypeExpr {
  // Unwrap any type modifiers and wrappers.
  llvm::SmallVector<clang::QualType> wrapper_types;
  while (true) {
    clang::QualType orig_type = type;
    if (type.hasQualifiers()) {
      type = type.getUnqualifiedType();
    } else if (type->isPointerType()) {
      type = type->getPointeeType();
    } else {
      break;
    }
    wrapper_types.push_back(orig_type);
  }

  auto mapped = MapNonWrapperType(context, loc_id, type);

  for (auto wrapper : llvm::reverse(wrapper_types)) {
    if (!mapped.inst_id.has_value() ||
        mapped.type_id == SemIR::ErrorInst::TypeId) {
      break;
    }

    if (wrapper.hasQualifiers()) {
      mapped = MapQualifiedType(context, loc_id, wrapper, mapped);
    } else if (wrapper->isPointerType()) {
      mapped = MapPointerType(context, loc_id, wrapper, mapped);
    } else {
      CARBON_FATAL("Unexpected wrapper type {0}", wrapper.getAsString());
    }
  }

  return mapped;
}

// Returns a block for the implicit parameters of the given function
// declaration. Because function templates are not yet supported, this currently
// only contains the `self` parameter. On error, produces a diagnostic and
// returns None.
static auto MakeImplicitParamPatternsBlockId(
    Context& context, SemIR::LocId loc_id,
    const clang::FunctionDecl& clang_decl) -> SemIR::InstBlockId {
  const auto* method_decl = dyn_cast<clang::CXXMethodDecl>(&clang_decl);
  if (!method_decl || method_decl->isStatic()) {
    return SemIR::InstBlockId::Empty;
  }

  // Build a `self` parameter from the object parameter.
  BeginSubpattern(context);

  // Perform some special-case mapping for the object parameter:
  //
  //  - If it's a const reference to T, produce a by-value `self: T` parameter.
  //  - If it's a non-const reference to T, produce an `addr self: T*`
  //    parameter.
  //  - Otherwise, map it directly, which will currently fail for `&&`-qualified
  //    methods.
  //
  // TODO: Some of this mapping should be performed for all parameters.
  clang::QualType param_type =
      method_decl->getFunctionObjectParameterReferenceType();
  bool addr_self = false;
  if (param_type->isLValueReferenceType()) {
    param_type = param_type.getNonReferenceType();
    if (param_type.isConstQualified()) {
      // TODO: Consider only doing this if `const` is the only qualifier. For
      // now, any other qualifier will fail when mapping the type.
      auto split_type = param_type.getSplitUnqualifiedType();
      split_type.Quals.removeConst();
      param_type = method_decl->getASTContext().getQualifiedType(split_type);
    } else {
      addr_self = true;
    }
  }

  auto [type_inst_id, type_id] = MapType(context, loc_id, param_type);
  SemIR::ExprRegionId type_expr_region_id =
      EndSubpatternAsExpr(context, type_inst_id);

  if (!type_id.has_value()) {
    context.TODO(loc_id,
                 llvm::formatv("Unsupported: object parameter type: {0}",
                               param_type.getAsString()));
    return SemIR::InstBlockId::None;
  }

  if (addr_self) {
    type_id = GetPointerType(context, type_inst_id);
  }

  SemIR::InstId pattern_id =
      // TODO: Fill in a location once available.
      AddBindingPattern(context, SemIR::LocId::None, SemIR::NameId::SelfValue,
                        type_id, type_expr_region_id, /*is_generic*/ false,
                        /*is_template*/ false)
          .pattern_id;

  // TODO: Fill in a location once available.
  pattern_id = AddPatternInst<SemIR::ValueParamPattern>(
      context, SemIR::LocId::None,
      {.type_id = context.insts().Get(pattern_id).type_id(),
       .subpattern_id = pattern_id,
       .index = SemIR::CallParamIndex::None});

  // If we're building `addr self: Self*`, do that now.
  if (addr_self) {
    // TODO: Fill in a location once available.
    pattern_id = AddPatternInst<SemIR::AddrPattern>(
        context, SemIR::LocId::None,
        {.type_id = GetPatternType(context, SemIR::AutoType::TypeId),
         .inner_id = pattern_id});
  }

  return context.inst_blocks().Add({pattern_id});
}

// Returns a block id for the explicit parameters of the given function
// declaration. If the function declaration has no parameters, it returns
// `SemIR::InstBlockId::Empty`. In the case of an unsupported parameter type, it
// produces an error and returns `SemIR::InstBlockId::None`.
// TODO: Consider refactoring to extract and reuse more logic from
// `HandleAnyBindingPattern()`.
static auto MakeParamPatternsBlockId(Context& context, SemIR::LocId loc_id,
                                     const clang::FunctionDecl& clang_decl)
    -> SemIR::InstBlockId {
  if (clang_decl.parameters().empty()) {
    return SemIR::InstBlockId::Empty;
  }
  llvm::SmallVector<SemIR::InstId> params;
  params.reserve(clang_decl.parameters().size());
  for (const clang::ParmVarDecl* param : clang_decl.parameters()) {
    // TODO: Get the parameter type from the function, not from the
    // `ParmVarDecl`. The type of the `ParmVarDecl` is the type within the
    // function, and isn't in general the same as the type that's exposed to
    // callers. In particular, the parameter type exposed to callers will never
    // be cv-qualified.
    clang::QualType param_type = param->getType();

    // Mark the start of a region of insts, needed for the type expression
    // created later with the call of `EndSubpatternAsExpr()`.
    BeginSubpattern(context);
    auto [type_inst_id, type_id] = MapType(context, loc_id, param_type);
    // Type expression of the binding pattern - a single-entry/single-exit
    // region that allows control flow in the type expression e.g. fn F(x: if C
    // then i32 else i64).
    SemIR::ExprRegionId type_expr_region_id =
        EndSubpatternAsExpr(context, type_inst_id);

    if (!type_id.has_value()) {
      context.TODO(loc_id, llvm::formatv("Unsupported: parameter type: {0}",
                                         param_type.getAsString()));
      return SemIR::InstBlockId::None;
    }

    llvm::StringRef param_name = param->getName();
    SemIR::NameId name_id =
        param_name.empty()
            // Translate an unnamed parameter to an underscore to
            // match Carbon's naming of unnamed/unused function params.
            ? SemIR::NameId::Underscore
            : AddIdentifierName(context, param_name);

    // TODO: Fix this once templates are supported.
    bool is_template = false;
    // TODO: Fix this once generics are supported.
    bool is_generic = false;
    SemIR::InstId binding_pattern_id =
        // TODO: Fill in a location once available.
        AddBindingPattern(context, SemIR::LocId::None, name_id, type_id,
                          type_expr_region_id, is_generic, is_template)
            .pattern_id;
    SemIR::InstId var_pattern_id = AddPatternInst(
        context,
        // TODO: Fill in a location once available.
        SemIR::LocIdAndInst::NoLoc(SemIR::ValueParamPattern(
            {.type_id = context.insts().Get(binding_pattern_id).type_id(),
             .subpattern_id = binding_pattern_id,
             .index = SemIR::CallParamIndex::None})));
    params.push_back(var_pattern_id);
  }
  return context.inst_blocks().Add(params);
}

// Returns the return type of the given function declaration. In case of an
// unsupported return type, it produces a diagnostic and returns
// `SemIR::ErrorInst::InstId`.
// TODO: Support more return types.
static auto GetReturnType(Context& context, SemIR::LocId loc_id,
                          const clang::FunctionDecl* clang_decl)
    -> SemIR::InstId {
  clang::QualType ret_type = clang_decl->getReturnType();
  if (ret_type->isVoidType()) {
    return SemIR::InstId::None;
  }

  auto [type_inst_id, type_id] = MapType(context, loc_id, ret_type);
  if (!type_inst_id.has_value()) {
    context.TODO(loc_id, llvm::formatv("Unsupported: return type: {0}",
                                       ret_type.getAsString()));
    return SemIR::ErrorInst::InstId;
  }
  auto pattern_type_id = GetPatternType(context, type_id);
  SemIR::InstId return_slot_pattern_id = AddPatternInst(
      // TODO: Fill in a location for the return type once available.
      context,
      SemIR::LocIdAndInst::NoLoc(SemIR::ReturnSlotPattern(
          {.type_id = pattern_type_id, .type_inst_id = type_inst_id})));
  SemIR::InstId param_pattern_id = AddPatternInst(
      // TODO: Fill in a location for the return type once available.
      context, SemIR::LocIdAndInst::NoLoc(SemIR::OutParamPattern(
                   {.type_id = pattern_type_id,
                    .subpattern_id = return_slot_pattern_id,
                    .index = SemIR::CallParamIndex::None})));
  return param_pattern_id;
}

namespace {
// Represents the parameter patterns block id, the return slot pattern id and
// the call parameters block id for a function declaration.
struct FunctionParamsInsts {
  SemIR::InstBlockId implicit_param_patterns_id;
  SemIR::InstBlockId param_patterns_id;
  SemIR::InstId return_slot_pattern_id;
  SemIR::InstBlockId call_params_id;
};
}  // namespace

// Creates a block containing the parameter pattern instructions for the
// explicit parameters, a parameter pattern instruction for the return type and
// a block containing the call parameters of the function. Emits a callee
// pattern-match for the explicit parameter patterns and the return slot pattern
// to create the Call parameters instructions block. Currently the implicit
// parameter patterns are not taken into account. Returns the parameter patterns
// block id, the return slot pattern id, and the call parameters block id.
// Produces a diagnostic and returns `std::nullopt` if the function declaration
// has an unsupported parameter type.
static auto CreateFunctionParamsInsts(Context& context, SemIR::LocId loc_id,
                                      const clang::FunctionDecl* clang_decl)
    -> std::optional<FunctionParamsInsts> {
  if (isa<clang::CXXConstructorDecl, clang::CXXDestructorDecl>(clang_decl)) {
    context.TODO(loc_id, "Unsupported: Constructor/Destructor");
    return std::nullopt;
  }

  auto implicit_param_patterns_id =
      MakeImplicitParamPatternsBlockId(context, loc_id, *clang_decl);
  if (!implicit_param_patterns_id.has_value()) {
    return std::nullopt;
  }
  auto param_patterns_id =
      MakeParamPatternsBlockId(context, loc_id, *clang_decl);
  if (!param_patterns_id.has_value()) {
    return std::nullopt;
  }
  auto return_slot_pattern_id = GetReturnType(context, loc_id, clang_decl);
  if (SemIR::ErrorInst::InstId == return_slot_pattern_id) {
    return std::nullopt;
  }

  auto call_params_id =
      CalleePatternMatch(context, implicit_param_patterns_id, param_patterns_id,
                         return_slot_pattern_id);

  return {{.implicit_param_patterns_id = implicit_param_patterns_id,
           .param_patterns_id = param_patterns_id,
           .return_slot_pattern_id = return_slot_pattern_id,
           .call_params_id = call_params_id}};
}

// Imports a function declaration from Clang to Carbon. If successful, returns
// the new Carbon function declaration `InstId`. If the declaration was already
// imported, returns the mapped instruction.
static auto ImportFunctionDecl(Context& context, SemIR::LocId loc_id,
                               clang::FunctionDecl* clang_decl)
    -> SemIR::InstId {
  // Check if the declaration is already mapped.
  if (SemIR::InstId existing_inst_id =
          LookupClangDeclInstId(context, clang_decl);
      existing_inst_id.has_value()) {
    return existing_inst_id;
  }

  if (clang_decl->isVariadic()) {
    context.TODO(loc_id, "Unsupported: Variadic function");
    MarkFailedDecl(context, clang_decl);
    return SemIR::ErrorInst::InstId;
  }
  if (clang_decl->getTemplatedKind() ==
      clang::FunctionDecl::TK_FunctionTemplate) {
    context.TODO(loc_id, "Unsupported: Template function");
    MarkFailedDecl(context, clang_decl);
    return SemIR::ErrorInst::InstId;
  }

  context.scope_stack().PushForDeclName();
  context.inst_block_stack().Push();
  context.pattern_block_stack().Push();

  auto function_params_insts =
      CreateFunctionParamsInsts(context, loc_id, clang_decl);

  auto pattern_block_id = context.pattern_block_stack().Pop();
  auto decl_block_id = context.inst_block_stack().Pop();
  context.scope_stack().Pop();

  if (!function_params_insts.has_value()) {
    MarkFailedDecl(context, clang_decl);
    return SemIR::ErrorInst::InstId;
  }

  auto function_decl = SemIR::FunctionDecl{
      SemIR::TypeId::None, SemIR::FunctionId::None, decl_block_id};
  auto decl_id =
      AddPlaceholderInstInNoBlock(context, Parse::NodeId::None, function_decl);
  context.imports().push_back(decl_id);

  auto function_info = SemIR::Function{
      {.name_id = AddIdentifierName(context, clang_decl->getName()),
       .parent_scope_id = GetParentNameScopeId(context, clang_decl),
       .generic_id = SemIR::GenericId::None,
       .first_param_node_id = Parse::NodeId::None,
       .last_param_node_id = Parse::NodeId::None,
       .pattern_block_id = pattern_block_id,
       .implicit_param_patterns_id =
           function_params_insts->implicit_param_patterns_id,
       .param_patterns_id = function_params_insts->param_patterns_id,
       .is_extern = false,
       .extern_library_id = SemIR::LibraryNameId::None,
       .non_owning_decl_id = SemIR::InstId::None,
       .first_owning_decl_id = decl_id,
       .definition_id = SemIR::InstId::None},
      {.call_params_id = function_params_insts->call_params_id,
       .return_slot_pattern_id = function_params_insts->return_slot_pattern_id,
       .virtual_modifier = SemIR::FunctionFields::VirtualModifier::None,
       .self_param_id = FindSelfPattern(
           context, function_params_insts->implicit_param_patterns_id),
       .clang_decl_id = context.sem_ir().clang_decls().Add(
           {.decl = clang_decl, .inst_id = decl_id})}};

  function_decl.function_id = context.functions().Add(function_info);

  function_decl.type_id = GetFunctionType(context, function_decl.function_id,
                                          SemIR::SpecificId::None);

  ReplaceInstBeforeConstantUse(context, decl_id, function_decl);

  return decl_id;
}

// Returns all decls that need to be imported before importing the given type.
static auto GetDependentUnimportedTypeDecls(const Context& context,
                                            clang::QualType type)
    -> llvm::SmallVector<clang::Decl*> {
  while (true) {
    type = type.getCanonicalType();
    if (type->isPointerType() || type->isReferenceType()) {
      type = type->getPointeeType();
    } else if (const clang::ArrayType* array_type =
                   type->getAsArrayTypeUnsafe()) {
      type = array_type->getElementType();
    } else {
      break;
    }
  }

  type = type.getUnqualifiedType();

  if (const auto* record_type = type->getAs<clang::RecordType>()) {
    if (auto* record_decl =
            clang::dyn_cast<clang::CXXRecordDecl>(record_type->getDecl())) {
      if (!IsClangDeclImported(context, record_decl)) {
        return {record_decl};
      }
    }
  }

  return {};
}

// Returns all decls that need to be imported before importing the given
// function.
static auto GetDependentUnimportedFunctionDecls(
    const Context& context, const clang::FunctionDecl& clang_decl)
    -> llvm::SmallVector<clang::Decl*> {
  llvm::SmallVector<clang::Decl*> decls;
  for (const auto* param : clang_decl.parameters()) {
    llvm::append_range(
        decls, GetDependentUnimportedTypeDecls(context, param->getType()));
  }
  llvm::append_range(decls, GetDependentUnimportedTypeDecls(
                                context, clang_decl.getReturnType()));
  return decls;
}

// Returns all decls that need to be imported before importing the given
// declaration.
static auto GetDependentUnimportedDecls(const Context& context,
                                        clang::Decl* clang_decl)
    -> llvm::SmallVector<clang::Decl*> {
  llvm::SmallVector<clang::Decl*> decls;
  if (auto* parent_decl = GetParentDecl(clang_decl);
      !IsClangDeclImported(context, parent_decl)) {
    decls.push_back(parent_decl);
  }

  if (auto* clang_function_decl = clang_decl->getAsFunction()) {
    llvm::append_range(decls, GetDependentUnimportedFunctionDecls(
                                  context, *clang_function_decl));
  } else if (auto* type_decl = clang::dyn_cast<clang::TypeDecl>(clang_decl)) {
    llvm::append_range(
        decls,
        GetDependentUnimportedTypeDecls(
            context, type_decl->getASTContext().getTypeDeclType(type_decl)));
  }

  return decls;
}

// Imports a declaration from Clang to Carbon. If successful, returns the
// instruction for the new Carbon declaration. Assumes all dependencies have
// already been imported.
static auto ImportDeclAfterDependencies(Context& context, SemIR::LocId loc_id,
                                        clang::Decl* clang_decl)
    -> SemIR::InstId {
  if (auto* clang_function_decl = clang_decl->getAsFunction()) {
    return ImportFunctionDecl(context, loc_id, clang_function_decl);
  }
  if (auto* clang_namespace_decl =
          clang::dyn_cast<clang::NamespaceDecl>(clang_decl)) {
    return ImportNamespaceDecl(context, clang_namespace_decl);
  }
  if (auto* type_decl = clang::dyn_cast<clang::TypeDecl>(clang_decl)) {
    auto type = type_decl->getASTContext().getTypeDeclType(type_decl);
    auto type_inst_id = MapType(context, loc_id, type).inst_id;
    if (!type_inst_id.has_value()) {
      context.TODO(loc_id, llvm::formatv("Unsupported: Type declaration: {0}",
                                         type.getAsString()));
      return SemIR::ErrorInst::InstId;
    }
    return type_inst_id;
  }

  context.TODO(loc_id, llvm::formatv("Unsupported: Declaration type {0}",
                                     clang_decl->getDeclKindName())
                           .str());
  return SemIR::InstId::None;
}

// Imports a declaration from Clang to Carbon. If successful, returns the
// instruction for the new Carbon declaration. All unimported dependencies would
// be imported first.
static auto ImportDeclAndDependencies(Context& context, SemIR::LocId loc_id,
                                      clang::Decl* clang_decl)
    -> SemIR::InstId {
  // Collect dependencies.
  llvm::SetVector<clang::Decl*> clang_decls;
  clang_decls.insert(clang_decl);
  for (size_t i = 0; i < clang_decls.size(); ++i) {
    auto dependent_decls = GetDependentUnimportedDecls(context, clang_decls[i]);
    for (clang::Decl* dependent_decl : dependent_decls) {
      clang_decls.insert(dependent_decl);
    }
  }

  // Import dependencies in reverse order.
  auto inst_id = SemIR::InstId::None;
  do {
    inst_id = ImportDeclAfterDependencies(context, loc_id,
                                          clang_decls.pop_back_val());
  } while (inst_id.has_value() && !clang_decls.empty());

  return inst_id;
}

// Imports a `clang::NamedDecl` into Carbon and adds that name into the
// `NameScope`.
static auto ImportNameDeclIntoScope(Context& context, SemIR::LocId loc_id,
                                    SemIR::NameScopeId scope_id,
                                    SemIR::NameId name_id,
                                    clang::NamedDecl* clang_decl)
    -> SemIR::InstId {
  SemIR::InstId inst_id =
      ImportDeclAndDependencies(context, loc_id, clang_decl);
  AddNameToScope(context, scope_id, name_id, inst_id);
  return inst_id;
}

auto ImportNameFromCpp(Context& context, SemIR::LocId loc_id,
                       SemIR::NameScopeId scope_id, SemIR::NameId name_id)
    -> SemIR::InstId {
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCppNameLookup, Note,
                          "in `Cpp` name lookup for `{0}`", SemIR::NameId);
        builder.Note(loc_id, InCppNameLookup, name_id);
      });

  auto lookup = ClangLookup(context, scope_id, name_id);
  if (!lookup) {
    return SemIR::InstId::None;
  }

  if (!lookup->isSingleResult()) {
    context.TODO(loc_id,
                 llvm::formatv("Unsupported: Lookup succeeded but couldn't "
                               "find a single result; LookupResultKind: {0}",
                               static_cast<int>(lookup->getResultKind()))
                     .str());
    context.name_scopes().AddRequiredName(scope_id, name_id,
                                          SemIR::ErrorInst::InstId);
    return SemIR::ErrorInst::InstId;
  }

  return ImportNameDeclIntoScope(context, loc_id, scope_id, name_id,
                                 lookup->getFoundDecl());
}

}  // namespace Carbon::Check
