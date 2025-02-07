// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_cpp.h"

#include <memory>
#include <optional>
#include <string>

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/format_providers.h"
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

// Returns an AST for the C++ imports.
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
      GenerateCppIncludesHeaderCode(context, imports), {},
      (importing_file_path + ".generated.cpp_imports.h").str(), "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
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
static auto AddNamespace(Context& context, IdentifierId cpp_package_id,
                         llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> SemIR::NameScopeId {
  // TODO: Extract and deudplicate common logic with import.cpp.
  auto name_id = SemIR::NameId::ForIdentifier(cpp_package_id);

  Parse::ImportDeclId first_import_node_id = imports.front().node_id;

  SemIR::InstId import_cpp_id =
      context.AddInst<SemIR::ImportCppDecl>(first_import_node_id, {});

  auto& import_cpps = context.sem_ir().import_cpps();
  import_cpps.Reserve(imports.size());
  for (const Parse::Tree::PackagingNames& import : imports) {
    import_cpps.Add(
        {.node_id = import.node_id, .library_id = import.library_id});
  }

  auto namespace_inst = SemIR::Namespace{
      .type_id =
          context.GetSingletonType(SemIR::NamespaceType::SingletonInstId),
      .name_scope_id = SemIR::NameScopeId::None,
      .import_id = import_cpp_id};

  auto namespace_inst_and_loc =
      // TODO: Check that this actually is an `AnyNamespaceId`.
      SemIR::LocIdAndInst(Parse::AnyNamespaceId(first_import_node_id),
                          namespace_inst);

  SemIR::InstId namespace_id =
      context.AddPlaceholderInstInNoBlock(namespace_inst_and_loc);
  context.import_ref_ids().push_back(namespace_id);
  SemIR::NameScope& package_name_scope =
      context.name_scopes().Get(SemIR::NameScopeId::Package);
  auto [added, entry_id] = package_name_scope.LookupOrAdd(
      name_id, namespace_id, SemIR::AccessKind::Public);
  CARBON_CHECK(added ||
               !package_name_scope.GetEntry(entry_id).result.is_poisoned());

  namespace_inst.name_scope_id = context.name_scopes().Add(
      namespace_id, name_id, SemIR::NameScopeId::Package);
  context.ReplaceInstBeforeConstantUse(namespace_id, namespace_inst);

  context.name_scopes()
      .Get(namespace_inst.name_scope_id)
      .set_is_closed_import(true);

  return namespace_inst.name_scope_id;
}

auto ImportCppFiles(Context& context, llvm::StringRef importing_file_path,
                    llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs)
    -> void {
  if (imports.empty()) {
    return;
  }

  auto [ast, ast_has_error] =
      GenerateAst(context, importing_file_path, imports, fs);

  IdentifierId package_id = imports.front().package_id;
  CARBON_CHECK(
      llvm::all_of(imports, [&](const Parse::Tree::PackagingNames& import) {
        return import.package_id == package_id;
      }));
  auto name_scope_id = AddNamespace(context, package_id, imports);

  if (ast_has_error) {
    context.name_scopes().Get(name_scope_id).set_has_error();
  }
}

}  // namespace Carbon::Check
