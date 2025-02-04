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

namespace Carbon::Check {

// Generates C++ file contents to #include all requested imports.
static auto GenerateCppIncludesHeaderCode(
    llvm::ArrayRef<std::pair<llvm::StringRef, SemIRLoc>> imports)
    -> std::string {
  std::string code;
  llvm::raw_string_ostream code_stream(code);
  for (const auto& [path, _] : imports) {
    code_stream << "#include \"" << FormatEscaped(path) << "\"\n";
  }
  return code;
}

static auto GenerateAst(
    Context& context, llvm::StringRef importing_file_path,
    llvm::ArrayRef<std::pair<llvm::StringRef, SemIRLoc>> imports,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs)
    -> std::optional<std::unique_ptr<clang::ASTUnit>> {
  // TODO: Use all import locations by referring each Clang diagnostic to the
  // relevant import.
  SemIRLoc loc = imports.back().second;

  std::string diagnostics_str;
  llvm::raw_string_ostream diagnostics_stream(diagnostics_str);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options(
      new clang::DiagnosticOptions());
  clang::TextDiagnosticPrinter diagnostics_consumer(diagnostics_stream,
                                                    diagnostic_options.get());
  // TODO: Share compilation flags with ClangRunner.
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      GenerateCppIncludesHeaderCode(imports), {},
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
    return std::nullopt;
  }
  if (num_warnings > 0) {
    CARBON_DIAGNOSTIC(CppInteropParseWarning, Warning,
                      "{0} warning{0:s} in `Cpp` {1} import{1:s}:\n{2}",
                      IntAsSelect, IntAsSelect, std::string);
    context.emitter().Emit(loc, CppInteropParseWarning, num_warnings,
                           num_imports, diagnostics_str);
  }
  return ast;
}

static auto AddNamespace(Context& context, IdentifierId cpp_package_id,
                         SemIR::InstId first_import_decl_id) -> void {
  // TODO: Extract and deudplicate common logic with import.cpp.
  auto name_id = SemIR::NameId::ForIdentifier(cpp_package_id);
  SemIR::TypeId namespace_type_id =
      context.GetSingletonType(SemIR::NamespaceType::SingletonInstId);

  auto [inserted, entry_id] =
      context.name_scopes()
          .Get(SemIR::NameScopeId::Package)
          .LookupOrAdd(
              name_id,
              // This InstId is temporary and would be overridden if used.
              first_import_decl_id, SemIR::AccessKind::Public);
  CARBON_CHECK(inserted);

  auto import_loc_id = context.insts().GetLocId(first_import_decl_id);

  auto namespace_inst =
      SemIR::Namespace{.type_id = namespace_type_id,
                       .name_scope_id = SemIR::NameScopeId::None,
                       .import_id = first_import_decl_id};

  auto namespace_inst_and_loc =
      import_loc_id.is_import_ir_inst_id()
          ? context.MakeImportedLocAndInst(import_loc_id.import_ir_inst_id(),
                                           namespace_inst)
          // TODO: Check that this actually is an `AnyNamespaceId`.
          : SemIR::LocIdAndInst(Parse::AnyNamespaceId(import_loc_id.node_id()),
                                namespace_inst);

  auto namespace_id =
      context.AddPlaceholderInstInNoBlock(namespace_inst_and_loc);

  context.import_ref_ids().push_back(namespace_id);
  namespace_inst.name_scope_id = context.name_scopes().Add(
      namespace_id, name_id, SemIR::NameScopeId::Package);
  context.ReplaceInstBeforeConstantUse(namespace_id, namespace_inst);

  // Note we have to get the parent scope freshly, creating the imported
  // namespace may invalidate any pointers into `name_scopes()`.
  context.name_scopes()
      .Get(SemIR::NameScopeId::Package)
      .GetEntry(entry_id)
      .result = SemIR::ScopeLookupResult::MakeFound(namespace_id,
                                                    SemIR::AccessKind::Public);

  auto& scope = context.name_scopes().Get(namespace_inst.name_scope_id);
  scope.set_is_closed_import(true);
}

auto ImportCppFiles(
    Context& context, llvm::StringRef importing_file_path,
    IdentifierId cpp_package_id, SemIR::InstId first_import_decl_id,
    llvm::ArrayRef<std::pair<llvm::StringRef, SemIRLoc>> imports,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs) -> void {
  size_t num_imports = imports.size();
  if (num_imports == 0) {
    return;
  }

  auto ast = GenerateAst(context, importing_file_path, imports, fs);
  if (!ast) {
    return;
  }

  AddNamespace(context, cpp_package_id, first_import_decl_id);
}

}  // namespace Carbon::Check
