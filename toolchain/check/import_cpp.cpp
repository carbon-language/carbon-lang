// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_cpp.h"

#include <memory>
#include <optional>
#include <string>

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Sema/Lookup.h"
#include "clang/Tooling/Tooling.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/import.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"

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

// Returns an AST for the C++ imports and a bool that represents whether
// compilation errors where encountered or the generated AST is null due to an
// error.
// TODO: Consider to always have a (non-null) AST.
static auto GenerateAst(Context& context, llvm::StringRef importing_file_path,
                        llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs)
    -> std::pair<std::unique_ptr<clang::ASTUnit>, bool> {
  // TODO: Use all import locations by referring each Clang diagnostic to the
  // relevant import.
  SemIRLoc loc = imports.back().node_id;

  std::string diagnostics_str;
  llvm::raw_string_ostream diagnostics_stream(diagnostics_str);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options(
      new clang::DiagnosticOptions());
  clang::TextDiagnosticPrinter diagnostics_consumer(diagnostics_stream,
                                                    diagnostic_options.get());
  // TODO: Share compilation flags with ClangRunner.
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      GenerateCppIncludesHeaderCode(context, imports),
      // Parse C++ (and not C)
      {"-x", "c++"}, (importing_file_path + ".generated.cpp_imports.h").str(),
      "clang-tool", std::make_shared<clang::PCHContainerOperations>(),
      clang::tooling::getClangStripDependencyFileAdjuster(),
      clang::tooling::FileContentMappings(), &diagnostics_consumer, fs);
  // TODO: Implement and use a DynamicRecursiveASTVisitor to traverse the AST.
  int num_errors = diagnostics_consumer.getNumErrors();
  int num_warnings = diagnostics_consumer.getNumWarnings();
  int num_imports = imports.size();
  if (num_errors > 0) {
    // TODO: Remove the warnings part when there are no warnings.
    CARBON_DIAGNOSTIC(
        CppInteropParseError, Error,
        "{0} error{0:s} and {1} warning{1:s} in {2} `Cpp` import{2:s}:\n{3}",
        IntAsSelect, IntAsSelect, IntAsSelect, std::string);
    context.emitter().Emit(loc, CppInteropParseError, num_errors, num_warnings,
                           num_imports, diagnostics_str);
  } else if (num_warnings > 0) {
    CARBON_DIAGNOSTIC(CppInteropParseWarning, Warning,
                      "{0} warning{0:s} in `Cpp` {1} import{1:s}:\n{2}",
                      IntAsSelect, IntAsSelect, std::string);
    context.emitter().Emit(loc, CppInteropParseWarning, num_warnings,
                           num_imports, diagnostics_str);
  }
  return {std::move(ast), !ast || num_errors > 0};
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
             GetSingletonType(context, SemIR::NamespaceType::SingletonInstId),
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
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs)
    -> std::unique_ptr<clang::ASTUnit> {
  if (imports.empty()) {
    return nullptr;
  }

  CARBON_CHECK(!context.sem_ir().cpp_ast());

  auto [generated_ast, ast_has_error] =
      GenerateAst(context, importing_file_path, imports, fs);

  PackageNameId package_id = imports.front().package_id;
  CARBON_CHECK(
      llvm::all_of(imports, [&](const Parse::Tree::PackagingNames& import) {
        return import.package_id == package_id;
      }));
  auto name_scope_id = AddNamespace(context, package_id, imports);
  SemIR::NameScope& name_scope = context.name_scopes().Get(name_scope_id);
  name_scope.set_is_closed_import(true);
  name_scope.set_cpp_decl_context(
      generated_ast->getASTContext().getTranslationUnitDecl());

  context.sem_ir().set_cpp_ast(generated_ast.get());

  if (ast_has_error) {
    name_scope.set_has_error();
  }

  return std::move(generated_ast);
}

// Look ups the given name in the Clang AST in a specific scope. Returns the
// lookup result if lookup was successful.
static auto ClangLookup(Context& context, SemIR::LocId loc_id,
                        SemIR::NameScopeId scope_id, SemIR::NameId name_id)
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

  bool found = sema.LookupQualifiedName(
      lookup, context.name_scopes().Get(scope_id).cpp_decl_context());

  if (lookup.isClassLookup()) {
    // TODO: To support class lookup, also return the AccessKind for storage.
    context.TODO(loc_id, "Unsupported: Lookup in Class");
    return std::nullopt;
  }

  if (!found) {
    return std::nullopt;
  }

  return lookup;
}

// Imports a function declaration from Clang to Carbon. If successful, returns
// the new Carbon function declaration `InstId`.
static auto ImportFunctionDecl(Context& context, SemIR::LocId loc_id,
                               SemIR::NameScopeId scope_id,
                               SemIR::NameId name_id,
                               const clang::FunctionDecl* clang_decl)
    -> SemIR::InstId {
  if (clang_decl->isVariadic()) {
    context.TODO(loc_id, "Unsupported: Variadic function");
    return SemIR::ErrorInst::SingletonInstId;
  }
  if (!clang_decl->isGlobal()) {
    context.TODO(loc_id, "Unsupported: Non-global function");
    return SemIR::ErrorInst::SingletonInstId;
  }
  if (clang_decl->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) {
    context.TODO(loc_id, "Unsupported: Template function");
    return SemIR::ErrorInst::SingletonInstId;
  }
  if (!clang_decl->param_empty()) {
    context.TODO(loc_id, "Unsupported: Function with parameters");
    return SemIR::ErrorInst::SingletonInstId;
  }
  if (!clang_decl->getReturnType()->isVoidType()) {
    context.TODO(loc_id, "Unsupported: Function with non-void return type");
    return SemIR::ErrorInst::SingletonInstId;
  }

  auto function_decl = SemIR::FunctionDecl{
      SemIR::TypeId::None, SemIR::FunctionId::None, SemIR::InstBlockId::Empty};
  auto decl_id = AddPlaceholderInst(
      context, SemIR::LocIdAndInst(Parse::NodeId::None, function_decl));

  auto function_info = SemIR::Function{
      {.name_id = name_id,
       .parent_scope_id = scope_id,
       .generic_id = SemIR::GenericId::None,
       .first_param_node_id = Parse::NodeId::None,
       .last_param_node_id = Parse::NodeId::None,
       .pattern_block_id = SemIR::InstBlockId::Empty,
       .implicit_param_patterns_id = SemIR::InstBlockId::Empty,
       .param_patterns_id = SemIR::InstBlockId::Empty,
       .is_extern = false,
       .extern_library_id = SemIR::LibraryNameId::None,
       .non_owning_decl_id = SemIR::InstId::None,
       .first_owning_decl_id = decl_id,
       .definition_id = SemIR::InstId::None},
      {.call_params_id = SemIR::InstBlockId::Empty,
       .return_slot_pattern_id = SemIR::InstId::None,
       .virtual_modifier = SemIR::FunctionFields::VirtualModifier::None,
       .self_param_id = SemIR::InstId::None,
       .cpp_decl = clang_decl}};

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
      context, GetSingletonType(context, SemIR::NamespaceType::SingletonInstId),
      name_id, parent_scope_id, /*import_id=*/SemIR::InstId::None);
  context.name_scopes()
      .Get(result.name_scope_id)
      .set_cpp_decl_context(clang_decl);
  return result.inst_id;
}

// Imports a declaration from Clang to Carbon. If successful, returns the
// instruction for the new Carbon declaration.
static auto ImportNameDecl(Context& context, SemIR::LocId loc_id,
                           SemIR::NameScopeId scope_id, SemIR::NameId name_id,
                           clang::NamedDecl* clang_decl) -> SemIR::InstId {
  if (const auto* clang_function_decl =
          clang::dyn_cast<clang::FunctionDecl>(clang_decl)) {
    return ImportFunctionDecl(context, loc_id, scope_id, name_id,
                              clang_function_decl);
  }
  if (auto* clang_namespace_decl =
          clang::dyn_cast<clang::NamespaceDecl>(clang_decl)) {
    return ImportNamespaceDecl(context, scope_id, name_id,
                               clang_namespace_decl);
  }

  context.TODO(loc_id, llvm::formatv("Unsupported: Declaration type {0}",
                                     clang_decl->getDeclKindName())
                           .str());
  return SemIR::InstId::None;
}

auto ImportNameFromCpp(Context& context, SemIR::LocId loc_id,
                       SemIR::NameScopeId scope_id, SemIR::NameId name_id)
    -> SemIR::InstId {
  auto lookup = ClangLookup(context, loc_id, scope_id, name_id);
  if (!lookup) {
    return SemIR::InstId::None;
  }

  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCppNameLookup, Note,
                          "in `Cpp` name lookup for `{0}`", SemIR::NameId);
        builder.Note(loc_id, InCppNameLookup, name_id);
      });

  if (!lookup->isSingleResult()) {
    context.TODO(loc_id,
                 llvm::formatv("Unsupported: Lookup succeeded but couldn't "
                               "find a single result; LookupResultKind: {0}",
                               lookup->getResultKind())
                     .str());
    return SemIR::ErrorInst::SingletonInstId;
  }

  return ImportNameDecl(context, loc_id, scope_id, name_id,
                        lookup->getFoundDecl());
}

}  // namespace Carbon::Check
