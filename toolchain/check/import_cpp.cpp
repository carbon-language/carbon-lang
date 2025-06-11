// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_cpp.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Tooling/Tooling.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/check/class.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/import.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/node_ids.h"
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
    code_stream << "#include \""
                << FormatEscaped(
                       context.string_literal_values().Get(import.library_id))
                << "\"\n";
  }
  return code;
}

namespace {

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

// Used to convert Clang diagnostics to Carbon diagnostics.
class CarbonClangDiagnosticConsumer : public clang::DiagnosticConsumer {
 public:
  // Creates an instance with the location that triggers calling Clang.
  // `context` must not be null.
  explicit CarbonClangDiagnosticConsumer(Context* context)
      : context_(context) {}

  // Generates a Carbon warning for each Clang warning and a Carbon error for
  // each Clang error or fatal.
  auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                        const clang::Diagnostic& info) -> void override {
    DiagnosticConsumer::HandleDiagnostic(diag_level, info);

    SemIR::ImportIRInstId clang_import_ir_inst_id =
        AddImportIRInst(*context_, info.getLocation());

    llvm::SmallString<256> message;
    info.FormatDiagnostic(message);

    RawStringOstream diagnostics_stream;
    // TODO: Consider allowing setting `LangOptions` or use
    // `ASTContext::getLangOptions()`.
    clang::LangOptions lang_options;
    // TODO: Consider allowing setting `DiagnosticOptions` or use
    // `ASTUnit::getDiagnostics().getLangOptions().getDiagnosticOptions()`.
    clang::DiagnosticOptions diagnostic_options;
    clang::TextDiagnostic text_diagnostic(diagnostics_stream, lang_options,
                                          diagnostic_options);
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
          // TODO: Parse the message to select the relevant C++ import and add
          // that information to the location.
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
static auto GenerateAst(Context& context, llvm::StringRef importing_file_path,
                        llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                        llvm::StringRef target)
    -> std::pair<std::unique_ptr<clang::ASTUnit>, bool> {
  CarbonClangDiagnosticConsumer diagnostics_consumer(&context);

  // TODO: Share compilation flags with ClangRunner.
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      GenerateCppIncludesHeaderCode(context, imports),
      // Parse C++ (and not C).
      {
          "-x",
          "c++",
          // Propagate the target to Clang.
          "-target",
          target.str(),
          // Require PIE. Note its default is configurable in Clang.
          "-fPIE",
      },
      (importing_file_path + ".generated.cpp_imports.h").str(), "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
      clang::tooling::getClangStripDependencyFileAdjuster(),
      clang::tooling::FileContentMappings(), &diagnostics_consumer, fs);
  // Remove link to the diagnostics consumer before its deletion.
  ast->getDiagnostics().setClient(nullptr);

  // In order to emit diagnostics, we need the AST.
  context.sem_ir().set_cpp_ast(ast.get());
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

auto ImportCppFiles(Context& context, llvm::StringRef importing_file_path,
                    llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                    llvm::StringRef target) -> std::unique_ptr<clang::ASTUnit> {
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
      GenerateAst(context, importing_file_path, imports, fs, target);

  SemIR::NameScope& name_scope = context.name_scopes().Get(name_scope_id);
  name_scope.set_is_closed_import(true);
  name_scope.set_clang_decl_context_id(context.sem_ir().clang_decls().Add(
      generated_ast->getASTContext().getTranslationUnitDecl()));

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

  bool found = sema.LookupQualifiedName(
      lookup,
      clang::dyn_cast<clang::DeclContext>(context.sem_ir().clang_decls().Get(
          context.name_scopes().Get(scope_id).clang_decl_context_id())));

  if (!found) {
    return std::nullopt;
  }

  return lookup;
}

// Creates an integer type of the given size.
static auto MakeIntType(Context& context, IntId size_id) -> TypeExpr {
  // TODO: Fill in a location for the type once available.
  auto type_inst_id = MakeIntTypeLiteral(context, Parse::NodeId::None,
                                         SemIR::IntKind::Signed, size_id);
  return ExprAsType(context, Parse::NodeId::None, type_inst_id);
}

// Maps a C++ type to a Carbon type.
// TODO: Support more types.
static auto MapType(Context& context, clang::QualType type) -> TypeExpr {
  const auto* builtin_type = dyn_cast<clang::BuiltinType>(type);
  if (!builtin_type) {
    return {.inst_id = SemIR::ErrorInst::TypeInstId,
            .type_id = SemIR::ErrorInst::TypeId};
  }
  // TODO: Refactor to avoid duplication.
  switch (builtin_type->getKind()) {
    case clang::BuiltinType::Short:
      if (context.ast_context().getTypeSize(type) == 16) {
        return MakeIntType(context, context.ints().Add(16));
      }
      break;
    case clang::BuiltinType::Int:
      if (context.ast_context().getTypeSize(type) == 32) {
        return MakeIntType(context, context.ints().Add(32));
      }
      break;
    default:
      break;
  }
  return {.inst_id = SemIR::ErrorInst::TypeInstId,
          .type_id = SemIR::ErrorInst::TypeId};
}

// Returns a block id for the explicit parameters of the given function
// declaration. If the function declaration has no parameters, it returns
// `SemIR::InstBlockId::Empty`. In the case of an unsupported parameter type, it
// returns `SemIR::InstBlockId::None`.
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
    clang::QualType param_type = param->getType().getCanonicalType();

    // Mark the start of a region of insts, needed for the type expression
    // created later with the call of `EndSubpatternAsExpr()`.
    BeginSubpattern(context);
    auto [type_inst_id, type_id] = MapType(context, param_type);
    // Type expression of the binding pattern - a single-entry/single-exit
    // region that allows control flow in the type expression e.g. fn F(x: if C
    // then i32 else i64).
    SemIR::ExprRegionId type_expr_region_id =
        EndSubpatternAsExpr(context, type_inst_id);

    if (type_id == SemIR::ErrorInst::TypeId) {
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
            : SemIR::NameId::ForIdentifier(
                  context.sem_ir().identifiers().Add(param_name));

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
// unsupported return type, it returns `SemIR::ErrorInst::InstId`.
// TODO: Support more return types.
static auto GetReturnType(Context& context, SemIR::LocId loc_id,
                          const clang::FunctionDecl* clang_decl)
    -> SemIR::InstId {
  clang::QualType ret_type = clang_decl->getReturnType().getCanonicalType();
  if (ret_type->isVoidType()) {
    return SemIR::InstId::None;
  }

  auto [type_inst_id, type_id] = MapType(context, ret_type);
  if (type_id == SemIR::ErrorInst::TypeId) {
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
// Returns `std::nullopt` if the function declaration has an unsupported
// parameter type.
static auto CreateFunctionParamsInsts(Context& context, SemIR::LocId loc_id,
                                      const clang::FunctionDecl* clang_decl)
    -> std::optional<FunctionParamsInsts> {
  auto param_patterns_id =
      MakeParamPatternsBlockId(context, loc_id, *clang_decl);
  if (!param_patterns_id.has_value()) {
    return std::nullopt;
  }
  auto return_slot_pattern_id = GetReturnType(context, loc_id, clang_decl);
  if (SemIR::ErrorInst::InstId == return_slot_pattern_id) {
    return std::nullopt;
  }
  // TODO: Add support for implicit parameters.
  auto call_params_id = CalleePatternMatch(
      context, /*implicit_param_patterns_id=*/SemIR::InstBlockId::None,
      param_patterns_id, return_slot_pattern_id);
  return {{.param_patterns_id = param_patterns_id,
           .return_slot_pattern_id = return_slot_pattern_id,
           .call_params_id = call_params_id}};
}

// Imports a function declaration from Clang to Carbon. If successful, returns
// the new Carbon function declaration `InstId`.
static auto ImportFunctionDecl(Context& context, SemIR::LocId loc_id,
                               SemIR::NameScopeId scope_id,
                               SemIR::NameId name_id,
                               clang::FunctionDecl* clang_decl)
    -> SemIR::InstId {
  if (clang_decl->isVariadic()) {
    context.TODO(loc_id, "Unsupported: Variadic function");
    return SemIR::ErrorInst::InstId;
  }
  if (!clang_decl->isGlobal()) {
    context.TODO(loc_id, "Unsupported: Non-global function");
    return SemIR::ErrorInst::InstId;
  }
  if (clang_decl->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) {
    context.TODO(loc_id, "Unsupported: Template function");
    return SemIR::ErrorInst::InstId;
  }

  context.inst_block_stack().Push();
  context.pattern_block_stack().Push();

  auto function_params_insts =
      CreateFunctionParamsInsts(context, loc_id, clang_decl);

  auto pattern_block_id = context.pattern_block_stack().Pop();
  auto decl_block_id = context.inst_block_stack().Pop();

  if (!function_params_insts.has_value()) {
    return SemIR::ErrorInst::InstId;
  }

  auto function_decl = SemIR::FunctionDecl{
      SemIR::TypeId::None, SemIR::FunctionId::None, decl_block_id};
  auto decl_id =
      AddPlaceholderInstInNoBlock(context, Parse::NodeId::None, function_decl);
  context.imports().push_back(decl_id);

  auto function_info = SemIR::Function{
      {.name_id = name_id,
       .parent_scope_id = scope_id,
       .generic_id = SemIR::GenericId::None,
       .first_param_node_id = Parse::NodeId::None,
       .last_param_node_id = Parse::NodeId::None,
       .pattern_block_id = pattern_block_id,
       .implicit_param_patterns_id = SemIR::InstBlockId::Empty,
       .param_patterns_id = function_params_insts->param_patterns_id,
       .is_extern = false,
       .extern_library_id = SemIR::LibraryNameId::None,
       .non_owning_decl_id = SemIR::InstId::None,
       .first_owning_decl_id = decl_id,
       .definition_id = SemIR::InstId::None},
      {.call_params_id = function_params_insts->call_params_id,
       .return_slot_pattern_id = function_params_insts->return_slot_pattern_id,
       .virtual_modifier = SemIR::FunctionFields::VirtualModifier::None,
       .self_param_id = SemIR::InstId::None,
       .clang_decl_id = context.sem_ir().clang_decls().Add(clang_decl)}};

  function_decl.function_id = context.functions().Add(function_info);

  function_decl.type_id = GetFunctionType(context, function_decl.function_id,
                                          SemIR::SpecificId::None);

  ReplaceInstBeforeConstantUse(context, decl_id, function_decl);

  return decl_id;
}

// Imports a namespace declaration from Clang to Carbon. If successful, returns
// the new Carbon namespace declaration `InstId`.
static auto ImportNamespaceDecl(Context& context,
                                SemIR::NameScopeId parent_scope_id,
                                SemIR::NameId name_id,
                                clang::NamespaceDecl* clang_decl)
    -> SemIR::InstId {
  auto result = AddImportNamespace(
      context, GetSingletonType(context, SemIR::NamespaceType::TypeInstId),
      name_id, parent_scope_id, /*import_id=*/SemIR::InstId::None);
  context.name_scopes()
      .Get(result.name_scope_id)
      .set_clang_decl_context_id(
          context.sem_ir().clang_decls().Add(clang_decl));
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

// Creates a class definition for the given class name in the given scope based
// on the information in the given Clang declaration. Returns the `InstId` for
// the declaration, which is assumed to be for a class definition. Returns the
// new class id and instruction id.
static auto BuildClassDefinition(Context& context,
                                 SemIR::NameScopeId parent_scope_id,
                                 SemIR::NameId name_id,
                                 clang::CXXRecordDecl* clang_decl)
    -> std::tuple<SemIR::ClassId, SemIR::InstId> {
  auto [class_id, class_decl_id] =
      BuildClassDecl(context, parent_scope_id, name_id);
  auto& class_info = context.classes().Get(class_id);
  StartClassDefinition(context, class_info, class_decl_id);

  context.name_scopes()
      .Get(class_info.scope_id)
      .set_clang_decl_context_id(
          context.sem_ir().clang_decls().Add(clang_decl));

  return {class_id, class_decl_id};
}

// Imports a record declaration from Clang to Carbon. If successful, returns
// the new Carbon class declaration `InstId`.
// TODO: Change `clang_decl` to `const &` when lookup is using `clang::DeclID`
// and we don't need to store the decl for lookup context.
static auto ImportCXXRecordDecl(Context& context, SemIR::LocId loc_id,
                                SemIR::NameScopeId parent_scope_id,
                                SemIR::NameId name_id,
                                clang::CXXRecordDecl* clang_decl)
    -> SemIR::InstId {
  clang::CXXRecordDecl* clang_def = clang_decl->getDefinition();
  if (!clang_def) {
    context.TODO(loc_id,
                 "Unsupported: Record declarations without a definition");
    return SemIR::ErrorInst::InstId;
  }

  if (clang_def->isDynamicClass()) {
    context.TODO(loc_id, "Unsupported: Dynamic Class");
    return SemIR::ErrorInst::InstId;
  }

  auto [class_id, class_def_id] =
      BuildClassDefinition(context, parent_scope_id, name_id, clang_def);

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

// Imports a declaration from Clang to Carbon. If successful, returns the
// instruction for the new Carbon declaration.
static auto ImportNameDecl(Context& context, SemIR::LocId loc_id,
                           SemIR::NameScopeId scope_id, SemIR::NameId name_id,
                           clang::NamedDecl* clang_decl) -> SemIR::InstId {
  if (auto* clang_function_decl =
          clang::dyn_cast<clang::FunctionDecl>(clang_decl)) {
    return ImportFunctionDecl(context, loc_id, scope_id, name_id,
                              clang_function_decl);
  }
  if (auto* clang_namespace_decl =
          clang::dyn_cast<clang::NamespaceDecl>(clang_decl)) {
    return ImportNamespaceDecl(context, scope_id, name_id,
                               clang_namespace_decl);
  }
  if (auto* clang_record_decl =
          clang::dyn_cast<clang::CXXRecordDecl>(clang_decl)) {
    return ImportCXXRecordDecl(context, loc_id, scope_id, name_id,
                               clang_record_decl);
  }

  context.TODO(loc_id, llvm::formatv("Unsupported: Declaration type {0}",
                                     clang_decl->getDeclKindName())
                           .str());
  return SemIR::InstId::None;
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
    return SemIR::ErrorInst::InstId;
  }

  return ImportNameDecl(context, loc_id, scope_id, name_id,
                        lookup->getFoundDecl());
}

}  // namespace Carbon::Check
