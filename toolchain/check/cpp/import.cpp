// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/import.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/AST/VTableBuilder.h"
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
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/cpp/custom_type_mapping.h"
#include "toolchain/check/cpp/thunk.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/cpp_overload_set.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Add a line marker directive pointing at the location of the `import Cpp`
// declaration in the Carbon source file. This will cause Clang's diagnostics
// machinery to track and report the location in Carbon code where the import
// was written.
static auto GenerateLineMarker(Context& context, llvm::raw_ostream& out,
                               int line) {
  out << "# " << line << " \""
      << FormatEscaped(context.tokens().source().filename()) << "\"\n";
}

// Generates C++ file contents to #include all requested imports.
static auto GenerateCppIncludesHeaderCode(
    Context& context, llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> std::string {
  std::string code;
  llvm::raw_string_ostream code_stream(code);
  for (const Parse::Tree::PackagingNames& import : imports) {
    if (import.inline_body_id.has_value()) {
      // Expand `import Cpp inline "code";` directly into the specified code.
      auto code_token = context.parse_tree().node_token(import.inline_body_id);

      // Compute the line number on which the C++ code starts. Usually the code
      // is specified as a block string literal and starts on the line after the
      // start of the string token.
      // TODO: Determine if this is a block string literal without calling
      // `GetTokenText`, which re-lexes the string.
      int line = context.tokens().GetLineNumber(code_token);
      if (context.tokens().GetTokenText(code_token).contains('\n')) {
        ++line;
      }

      GenerateLineMarker(context, code_stream, line);
      code_stream << context.string_literal_values().Get(
                         context.tokens().GetStringLiteralValue(code_token))
                  << "\n";
      // TODO: Inject a clang pragma here to produce an error if there are
      // unclosed scopes at the end of this inline C++ fragment.
    } else {
      // Translate `import Cpp library "foo.h";` into `#include "foo.h"`.
      GenerateLineMarker(context, code_stream,
                         context.tokens().GetLineNumber(
                             context.parse_tree().node_token(import.node_id)));
      code_stream << "#include \""
                  << FormatEscaped(
                         context.string_literal_values().Get(import.library_id))
                  << "\"\n";
    }
  }

  // Inject a declaration of placement operator new, because the code we
  // generate in thunks depends on it for placement new expressions. Clang has
  // special-case logic for lowering a new-expression using this, so a
  // definition is not required.
  // TODO: This is a hack. We should be able to directly generate Clang AST to
  // construct objects in-place without this.
  // TODO: Once we can rely on libc++ being available, consider including
  // `<__new/placement_new_delete.h>` instead.
  code_stream << R"(# 1 "<carbon-internal>"
#undef constexpr
#if __cplusplus > 202302L
constexpr
#endif
#undef void
#undef operator
#undef new
void* operator new(__SIZE_TYPE__, void*)
#if __cplusplus < 201103L
#undef throw
throw()
#else
#undef noexcept
noexcept
#endif
;
)";

  return code;
}

// Adds the name to the scope with the given `access_kind` and `inst_id`.
// `inst_id` must have a value.
static auto AddNameToScope(Context& context, SemIR::NameScopeId scope_id,
                           SemIR::NameId name_id, SemIR::AccessKind access_kind,
                           SemIR::InstId inst_id) -> void {
  CARBON_CHECK(inst_id.has_value());
  context.name_scopes().Get(scope_id).AddRequired(
      {.name_id = name_id,
       .result = SemIR::ScopeLookupResult::MakeFound(inst_id, access_kind)});
}

// Maps a Clang name to a Carbon `NameId`.
static auto AddIdentifierName(Context& context, llvm::StringRef name)
    -> SemIR::NameId {
  return SemIR::NameId::ForIdentifier(context.identifiers().Add(name));
}

// Adds the given source location and an `ImportIRInst` referring to it in
// `ImportIRId::Cpp`.
static auto AddImportIRInst(SemIR::File& file,
                            clang::SourceLocation clang_source_loc)
    -> SemIR::ImportIRInstId {
  SemIR::ClangSourceLocId clang_source_loc_id =
      file.clang_source_locs().Add(clang_source_loc);
  return file.import_ir_insts().Add(SemIR::ImportIRInst(clang_source_loc_id));
}

namespace {

// Used to convert Clang diagnostics to Carbon diagnostics.
//
// Handling of Clang notes is a little subtle: as far as Clang is concerned,
// notes are separate diagnostics, not connected to the error or warning that
// precedes them. But in Carbon's diagnostics system, notes are part of the
// enclosing diagnostic. To handle this, we buffer Clang diagnostics until we
// reach a point where we know we're not in the middle of a diagnostic, and then
// emit a diagnostic along with all of its notes. This is triggered when adding
// or removing a Carbon context note, which could otherwise get attached to the
// wrong C++ diagnostics, and at the end of the Carbon program.
class CarbonClangDiagnosticConsumer : public clang::DiagnosticConsumer {
 public:
  // Creates an instance with the location that triggers calling Clang. The
  // `context` is not stored here, and the diagnostics consumer is expected to
  // outlive it.
  explicit CarbonClangDiagnosticConsumer(
      Context& context, std::shared_ptr<clang::CompilerInvocation> invocation)
      : sem_ir_(&context.sem_ir()),
        emitter_(&context.emitter()),
        invocation_(std::move(invocation)) {
    emitter_->AddFlushFn([this] { EmitDiagnostics(); });
  }

  ~CarbonClangDiagnosticConsumer() override {
    // Do not inspect `emitter_` here; it's typically destroyed before the
    // consumer is.
    // TODO: If Clang produces diagnostics after check finishes, they'll get
    // added to the list of pending diagnostics and never emitted.
    CARBON_CHECK(diagnostic_infos_.empty(),
                 "Missing flush before destroying diagnostic consumer");
  }

  // Generates a Carbon warning for each Clang warning and a Carbon error for
  // each Clang error or fatal.
  auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                        const clang::Diagnostic& info) -> void override {
    DiagnosticConsumer::HandleDiagnostic(diag_level, info);

    SemIR::ImportIRInstId clang_import_ir_inst_id =
        AddImportIRInst(*sem_ir_, info.getLocation());

    llvm::SmallString<256> message;
    info.FormatDiagnostic(message);

    // Render a code snippet including any highlighted ranges and fixit hints.
    // TODO: Also include the #include stack and macro expansion stack in the
    // diagnostic output in some way.
    RawStringOstream snippet_stream;
    if (!info.hasSourceManager()) {
      // If we don't have a source manager, this is an error from early in the
      // frontend. Don't produce a snippet.
      CARBON_CHECK(info.getLocation().isInvalid());
    } else {
      CodeContextRenderer(snippet_stream, invocation_->getLangOpts(),
                          invocation_->getDiagnosticOpts())
          .emitDiagnostic(
              clang::FullSourceLoc(info.getLocation(), info.getSourceManager()),
              diag_level, message, info.getRanges(), info.getFixItHints());
    }

    diagnostic_infos_.push_back({.level = diag_level,
                                 .import_ir_inst_id = clang_import_ir_inst_id,
                                 .message = message.str().str(),
                                 .snippet = snippet_stream.TakeStr()});
  }

  // Returns the diagnostic to use for a given Clang diagnostic level.
  static auto GetDiagnostic(clang::DiagnosticsEngine::Level level)
      -> const Diagnostics::DiagnosticBase<std::string>& {
    switch (level) {
      case clang::DiagnosticsEngine::Ignored: {
        CARBON_FATAL("Emitting an ignored diagnostic");
        break;
      }
      case clang::DiagnosticsEngine::Note: {
        CARBON_DIAGNOSTIC(CppInteropParseNote, Note, "{0}", std::string);
        return CppInteropParseNote;
      }
      case clang::DiagnosticsEngine::Remark:
      case clang::DiagnosticsEngine::Warning: {
        // TODO: Add a distinct Remark level to Carbon diagnostics, and stop
        // mapping remarks to warnings.
        CARBON_DIAGNOSTIC(CppInteropParseWarning, Warning, "{0}", std::string);
        return CppInteropParseWarning;
      }
      case clang::DiagnosticsEngine::Error:
      case clang::DiagnosticsEngine::Fatal: {
        CARBON_DIAGNOSTIC(CppInteropParseError, Error, "{0}", std::string);
        return CppInteropParseError;
      }
    }
  }

  // Outputs Carbon diagnostics based on the collected Clang diagnostics. Must
  // be called after the AST is set in the context.
  auto EmitDiagnostics() -> void {
    CARBON_CHECK(sem_ir_->clang_ast_unit(),
                 "Attempted to emit diagnostics before the AST Unit is loaded");

    for (size_t i = 0; i != diagnostic_infos_.size(); ++i) {
      const ClangDiagnosticInfo& info = diagnostic_infos_[i];
      auto builder = emitter_->Build(SemIR::LocId(info.import_ir_inst_id),
                                     GetDiagnostic(info.level), info.message);
      builder.OverrideSnippet(info.snippet);
      for (; i + 1 < diagnostic_infos_.size() &&
             diagnostic_infos_[i + 1].level == clang::DiagnosticsEngine::Note;
           ++i) {
        const ClangDiagnosticInfo& note_info = diagnostic_infos_[i + 1];
        builder
            .Note(SemIR::LocId(note_info.import_ir_inst_id),
                  GetDiagnostic(note_info.level), note_info.message)
            .OverrideSnippet(note_info.snippet);
      }
      // TODO: This will apply all current Carbon annotation functions. We
      // should instead track how Clang's context notes and Carbon's annotation
      // functions are interleaved, and interleave the notes in the same order.
      builder.Emit();
    }
    diagnostic_infos_.clear();
  }

 private:
  // A diagnostics renderer based on clang's TextDiagnostic that captures just
  // the code context (the snippet).
  class CodeContextRenderer : public clang::TextDiagnostic {
   public:
    using TextDiagnostic::TextDiagnostic;

    void emitDiagnosticMessage(
        clang::FullSourceLoc /*loc*/, clang::PresumedLoc /*ploc*/,
        clang::DiagnosticsEngine::Level /*level*/, llvm::StringRef /*message*/,
        llvm::ArrayRef<clang::CharSourceRange> /*ranges*/,
        clang::DiagOrStoredDiag /*info*/) override {}
    void emitDiagnosticLoc(
        clang::FullSourceLoc /*loc*/, clang::PresumedLoc /*ploc*/,
        clang::DiagnosticsEngine::Level /*level*/,
        llvm::ArrayRef<clang::CharSourceRange> /*ranges*/) override {}

    // emitCodeContext is inherited from clang::TextDiagnostic.

    void emitIncludeLocation(clang::FullSourceLoc /*loc*/,
                             clang::PresumedLoc /*ploc*/) override {}
    void emitImportLocation(clang::FullSourceLoc /*loc*/,
                            clang::PresumedLoc /*ploc*/,
                            llvm::StringRef /*module_name*/) override {}
    void emitBuildingModuleLocation(clang::FullSourceLoc /*loc*/,
                                    clang::PresumedLoc /*ploc*/,
                                    llvm::StringRef /*module_name*/) override {}

    // beginDiagnostic and endDiagnostic are inherited from
    // clang::TextDiagnostic in case it wants to do any setup / teardown work.
  };

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

    // The code snippet produced by clang.
    std::string snippet;
  };

  // The Carbon file that this C++ compilation is attached to.
  SemIR::File* sem_ir_;

  // The diagnostic emitter that we're emitting diagnostics into.
  DiagnosticEmitterBase* emitter_;

  // The compiler invocation that is producing the diagnostics.
  std::shared_ptr<clang::CompilerInvocation> invocation_;

  // Collects the information for all Clang diagnostics to be converted to
  // Carbon diagnostics after the context has been initialized with the Clang
  // AST.
  llvm::SmallVector<ClangDiagnosticInfo> diagnostic_infos_;
};

// A wrapper around a clang::CompilerInvocation that allows us to make a shallow
// copy of most of the invocation and only make a deep copy of the parts that we
// want to change.
//
// clang::CowCompilerInvocation almost allows this, but doesn't derive from
// CompilerInvocation or support shallow copies from a CompilerInvocation, so is
// not useful to us as we can't build an ASTUnit from it.
class ShallowCopyCompilerInvocation : public clang::CompilerInvocation {
 public:
  explicit ShallowCopyCompilerInvocation(
      const clang::CompilerInvocation& invocation) {
    shallow_copy_assign(invocation);

    // The preprocessor options are modified to hold a replacement includes
    // buffer, so make our own version of those options.
    PPOpts = std::make_shared<clang::PreprocessorOptions>(*PPOpts);
  }
};

}  // namespace

// Returns an AST for the C++ imports and a bool that represents whether
// compilation errors where encountered or the generated AST is null due to an
// error. Sets the AST in the context's `sem_ir`.
// TODO: Consider to always have a (non-null) AST.
static auto GenerateAst(
    Context& context, llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    std::shared_ptr<clang::CompilerInvocation> base_invocation)
    -> std::pair<std::unique_ptr<clang::ASTUnit>, bool> {
  auto invocation =
      std::make_shared<ShallowCopyCompilerInvocation>(*base_invocation);

  // Build a diagnostics engine.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
      clang::CompilerInstance::createDiagnostics(
          *fs, invocation->getDiagnosticOpts(),
          new CarbonClangDiagnosticConsumer(context, invocation),
          /*ShouldOwnClient=*/true));

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
  auto includes_buffer =
      llvm::MemoryBuffer::getMemBufferCopy(includes, file_name);
  invocation->getPreprocessorOpts().addRemappedFile(file_name,
                                                    includes_buffer.release());

  clang::DiagnosticErrorTrap trap(*diags);

  // Create the AST unit.
  auto ast = clang::ASTUnit::LoadFromCompilerInvocation(
      invocation, std::make_shared<clang::PCHContainerOperations>(), nullptr,
      diags, new clang::FileManager(invocation->getFileSystemOpts(), fs));

  // Attach the AST to SemIR. This needs to be done before we can emit any
  // diagnostics, so their locations can be properly interpreted by our
  // diagnostics machinery.
  context.sem_ir().set_clang_ast_unit(ast.get());

  // Emit any diagnostics we queued up while building the AST.
  context.emitter().Flush();

  return {std::move(ast), !ast || trap.hasErrorOccurred()};
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

  CARBON_CHECK(!context.sem_ir().clang_ast_unit());

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
  name_scope.set_clang_decl_context_id(context.clang_decls().Add(
      {.decl = generated_ast->getASTContext().getTranslationUnitDecl(),
       .inst_id = name_scope.inst_id()}));

  if (ast_has_error) {
    name_scope.set_has_error();
  }

  return std::move(generated_ast);
}

// Returns the Clang `DeclContext` for the given name scope. Return the
// translation unit decl if no scope is provided.
static auto GetDeclContext(Context& context, SemIR::NameScopeId scope_id)
    -> clang::DeclContext* {
  if (!scope_id.has_value()) {
    return context.ast_context().getTranslationUnitDecl();
  }
  auto scope_clang_decl_context_id =
      context.name_scopes().Get(scope_id).clang_decl_context_id();
  return dyn_cast<clang::DeclContext>(
      context.clang_decls().Get(scope_clang_decl_context_id).decl);
}

static auto ClangLookup(Context& context, SemIR::NameScopeId scope_id,
                        clang::DeclarationName name)
    -> std::optional<clang::LookupResult> {
  clang::Sema& sema = context.clang_sema();

  // TODO: Map the LocId of the lookup to a clang SourceLocation and provide it
  // here so that clang's diagnostics can point into the carbon code that uses
  // the name.
  clang::LookupResult lookup(
      sema, clang::DeclarationNameInfo(name, clang::SourceLocation()),
      clang::Sema::LookupNameKind::LookupOrdinaryName);

  bool found =
      sema.LookupQualifiedName(lookup, GetDeclContext(context, scope_id));

  if (!found) {
    return std::nullopt;
  }

  return lookup;
}

// Looks up for constructors in the class scope and returns the lookup result.
static auto ClangConstructorLookup(Context& context,
                                   SemIR::NameScopeId scope_id)
    -> clang::DeclContextLookupResult {
  const SemIR::NameScope& scope = context.name_scopes().Get(scope_id);

  clang::Sema& sema = context.clang_sema();
  clang::Decl* decl =
      context.clang_decls().Get(scope.clang_decl_context_id()).decl;
  return sema.LookupConstructors(cast<clang::CXXRecordDecl>(decl));
}

// Returns true if the given Clang declaration is the implicit injected class
// name within the class.
static auto IsDeclInjectedClassName(Context& context,
                                    SemIR::NameScopeId scope_id,
                                    SemIR::NameId name_id,
                                    const clang::NamedDecl* named_decl)
    -> bool {
  if (!named_decl->isImplicit()) {
    return false;
  }

  const auto* record_decl = dyn_cast<clang::CXXRecordDecl>(named_decl);
  if (!record_decl) {
    return false;
  }

  const SemIR::ClangDecl& clang_decl = context.clang_decls().Get(
      context.name_scopes().Get(scope_id).clang_decl_context_id());
  const auto* scope_record_decl = cast<clang::CXXRecordDecl>(clang_decl.decl);

  const clang::ASTContext& ast_context = context.ast_context();
  CARBON_CHECK(ast_context.getCanonicalTagType(scope_record_decl) ==
               ast_context.getCanonicalTagType(record_decl));

  auto class_decl = context.insts().GetAs<SemIR::ClassDecl>(clang_decl.inst_id);
  CARBON_CHECK(name_id == context.classes().Get(class_decl.class_id).name_id);
  return true;
}

// Returns a Clang DeclarationName for the given `NameId`.
static auto GetDeclarationName(Context& context, SemIR::NameId name_id)
    -> std::optional<clang::DeclarationName> {
  std::optional<llvm::StringRef> name =
      context.names().GetAsStringIfIdentifier(name_id);
  if (!name) {
    // Special names never exist in C++ code.
    return std::nullopt;
  }

  return clang::DeclarationName(
      context.clang_sema().getPreprocessor().getIdentifierInfo(*name));
}

// Looks up the given name in the Clang AST in a specific scope. Returns the
// lookup result if lookup was successful.
// TODO: Merge this with `ClangLookupDeclarationName`.
static auto ClangLookupName(Context& context, SemIR::NameScopeId scope_id,
                            SemIR::NameId name_id)
    -> std::optional<clang::LookupResult> {
  auto declaration_name = GetDeclarationName(context, name_id);
  if (!declaration_name) {
    return std::nullopt;
  }

  return ClangLookup(context, scope_id, *declaration_name);
}

// Returns whether `decl` already mapped to an instruction.
static auto IsClangDeclImported(Context& context, clang::Decl* decl) -> bool {
  return context.clang_decls().Lookup(decl->getCanonicalDecl()).has_value();
}

// If `decl` already mapped to an instruction, returns that instruction.
// Otherwise returns `None`.
static auto LookupClangDeclInstId(Context& context, clang::Decl* decl)
    -> SemIR::InstId {
  const auto& clang_decls = context.clang_decls();
  if (auto context_clang_decl_id = clang_decls.Lookup(decl->getCanonicalDecl());
      context_clang_decl_id.has_value()) {
    return clang_decls.Get(context_clang_decl_id).inst_id;
  }
  return SemIR::InstId::None;
}

// Returns the parent of the given declaration. Skips declaration types we
// ignore.
static auto GetParentDecl(clang::Decl* clang_decl) -> clang::Decl* {
  auto* parent_dc = clang_decl->getDeclContext();
  while (!parent_dc->isLookupContext()) {
    parent_dc = parent_dc->getParent();
  }
  return cast<clang::Decl>(parent_dc);
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
      .set_clang_decl_context_id(context.clang_decls().Add(
          {.decl = clang_decl->getCanonicalDecl(), .inst_id = result.inst_id}));
  return result.inst_id;
}

static auto ImportTypeAndDependencies(Context& context, SemIR::LocId loc_id,
                                      clang::QualType type) -> TypeExpr;

// Creates a class declaration for the given class name in the given scope.
// Returns the `InstId` for the declaration.
static auto BuildClassDecl(Context& context,
                           SemIR::ImportIRInstId import_ir_inst_id,
                           SemIR::NameScopeId parent_scope_id,
                           SemIR::NameId name_id)
    -> std::tuple<SemIR::ClassId, SemIR::TypeInstId> {
  // Add the class declaration.
  auto class_decl = SemIR::ClassDecl{.type_id = SemIR::TypeType::TypeId,
                                     .class_id = SemIR::ClassId::None,
                                     .decl_block_id = SemIR::InstBlockId::None};
  auto class_decl_id = AddPlaceholderInstInNoBlock(
      context,
      SemIR::LocIdAndInst::UncheckedLoc(import_ir_inst_id, class_decl));
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

  return {class_decl.class_id, context.types().GetAsTypeInstId(class_decl_id)};
}

// Imports a tag declaration from Clang to Carbon. This covers classes (which
// includes structs and unions) as well as enums. If successful, returns the new
// Carbon class declaration `InstId`.
static auto ImportTagDecl(Context& context, clang::TagDecl* clang_decl)
    -> SemIR::InstId {
  auto import_ir_inst_id =
      AddImportIRInst(context.sem_ir(), clang_decl->getLocation());

  auto [class_id, class_inst_id] = BuildClassDecl(
      context, import_ir_inst_id, GetParentNameScopeId(context, clang_decl),
      AddIdentifierName(context, clang_decl->getName()));

  // TODO: The caller does the same lookup. Avoid doing it twice.
  auto clang_decl_id = context.clang_decls().Add(
      {.decl = clang_decl->getCanonicalDecl(), .inst_id = class_inst_id});

  // Name lookup into the Carbon class looks in the C++ class definition.
  auto& class_info = context.classes().Get(class_id);
  class_info.scope_id = context.name_scopes().Add(
      class_inst_id, SemIR::NameId::None, class_info.parent_scope_id);
  context.name_scopes()
      .Get(class_info.scope_id)
      .set_clang_decl_context_id(clang_decl_id);

  return class_inst_id;
}

// Determines the Carbon inheritance kind to use for a C++ class definition.
static auto GetInheritanceKind(clang::CXXRecordDecl* class_def)
    -> SemIR::Class::InheritanceKind {
  if (class_def->isUnion()) {
    // Treat all unions as final classes to match their C++ semantics. While we
    // could support this, the author of a C++ union has no way to mark their
    // type as `final` to prevent it, and so we assume the intent was to
    // disallow inheritance.
    return SemIR::Class::Final;
  }

  if (class_def->hasAttr<clang::FinalAttr>()) {
    // The class is final in C++; don't allow Carbon types to derive from it.
    // Note that such a type might also be abstract in C++; we treat final as
    // taking precedence.
    //
    // We could also treat classes with a final destructor as being final, as
    // Clang does when determining whether a class is "effectively final", but
    // to keep our rules simpler we do not.
    return SemIR::Class::Final;
  }

  if (class_def->isAbstract()) {
    // If the class has any abstract members, it's abstract.
    return SemIR::Class::Abstract;
  }

  // Allow inheritance from any other C++ class type.
  return SemIR::Class::Base;
}

// Checks that the specified finished class definition is valid and builds and
// returns a corresponding complete type witness instruction.
static auto ImportClassObjectRepr(Context& context, SemIR::ClassId class_id,
                                  SemIR::ImportIRInstId import_ir_inst_id,
                                  SemIR::TypeInstId class_type_inst_id,
                                  const clang::CXXRecordDecl* clang_def)
    -> SemIR::TypeInstId {
  if (clang_def->isInvalidDecl()) {
    // Clang already diagnosed this error.
    return SemIR::ErrorInst::TypeInstId;
  }

  // For now, if the class is empty, produce an empty struct as the object
  // representation. This allows our tests to continue to pass while we don't
  // properly support initializing imported C++ classes.
  // TODO: Remove this.
  if (clang_def->isEmpty() && !clang_def->getNumBases()) {
    return context.types().GetAsTypeInstId(AddInst(
        context,
        MakeImportedLocIdAndInst(
            context, import_ir_inst_id,
            SemIR::StructType{.type_id = SemIR::TypeType::TypeId,
                              .fields_id = SemIR::StructTypeFieldsId::Empty})));
  }

  const auto& clang_layout =
      context.ast_context().getASTRecordLayout(clang_def);

  llvm::SmallVector<uint64_t> layout;
  llvm::SmallVector<SemIR::StructTypeField> fields;

  static_assert(SemIR::CustomLayoutId::SizeIndex == 0);
  layout.push_back(clang_layout.getSize().getQuantity());

  static_assert(SemIR::CustomLayoutId::AlignIndex == 1);
  layout.push_back(clang_layout.getAlignment().getQuantity());

  static_assert(SemIR::CustomLayoutId::FirstFieldIndex == 2);

  // TODO: Import vptr(s).

  // Import bases.
  for (const auto& base : clang_def->bases()) {
    CARBON_CHECK(!base.isVirtual(),
                 "Should not import definition for class with a virtual base");

    auto [base_type_inst_id, base_type_id] =
        ImportTypeAndDependencies(context, import_ir_inst_id, base.getType());
    if (!base_type_id.has_value()) {
      // TODO: If the base class's type can't be mapped, skip it.
      continue;
    }

    auto base_decl_id = AddInst(
        context,
        MakeImportedLocIdAndInst(
            context, import_ir_inst_id,
            SemIR::BaseDecl{.type_id = GetUnboundElementType(
                                context, class_type_inst_id, base_type_inst_id),
                            .base_type_inst_id = base_type_inst_id,
                            .index = SemIR::ElementIndex(fields.size())}));

    // If there's exactly one base class, treat it as a Carbon base class too.
    // TODO: Improve handling for the case where the class has multiple base
    // classes.
    if (clang_def->getNumBases() == 1) {
      auto& class_info = context.classes().Get(class_id);
      CARBON_CHECK(!class_info.base_id.has_value());
      class_info.base_id = base_decl_id;
    }

    auto* base_class = base.getType()->getAsCXXRecordDecl();
    CARBON_CHECK(base_class, "Base class {0} is not a class",
                 base.getType().getAsString());

    auto base_offset = base.isVirtual()
                           ? clang_layout.getVBaseClassOffset(base_class)
                           : clang_layout.getBaseClassOffset(base_class);
    layout.push_back(base_offset.getQuantity());
    fields.push_back(
        {.name_id = SemIR::NameId::Base, .type_inst_id = base_type_inst_id});
  }

  // Import fields.
  for (auto* decl : clang_def->decls()) {
    auto* field = dyn_cast<clang::FieldDecl>(decl);

    // Track the chain of fields from the class to this field. This chain is
    // only one element long unless the field is a member of an anonymous struct
    // or union.
    clang::NamedDecl* single_field_chain[1] = {field};
    llvm::ArrayRef<clang::NamedDecl*> chain = single_field_chain;

    // If this isn't a field, it might be an indirect field in an anonymous
    // struct or union.
    if (!field) {
      auto* indirect_field = dyn_cast<clang::IndirectFieldDecl>(decl);
      if (!indirect_field) {
        continue;
      }
      chain = indirect_field->chain();
      field = indirect_field->getAnonField();
    }

    if (field->isBitField()) {
      // TODO: Add a representation for named bitfield members.
      continue;
    }

    if (field->isAnonymousStructOrUnion()) {
      // Fields within an anonymous structure or union will be added via their
      // IndirectFieldDecls.
      continue;
    }

    auto field_name_id = AddIdentifierName(context, field->getName());
    auto [field_type_inst_id, field_type_id] =
        ImportTypeAndDependencies(context, import_ir_inst_id, field->getType());
    if (!field_type_inst_id.has_value()) {
      // TODO: For now, just skip over fields whose types we can't map.
      continue;
    }

    // Create a field now, as we know the index to use.
    // TODO: Consider doing this lazily instead.
    auto field_decl_id = AddInst(
        context, MakeImportedLocIdAndInst(
                     context, import_ir_inst_id,
                     SemIR::FieldDecl{
                         .type_id = GetUnboundElementType(
                             context, class_type_inst_id, field_type_inst_id),
                         .name_id = field_name_id,
                         .index = SemIR::ElementIndex(fields.size())}));
    context.clang_decls().Add(
        {.decl = decl->getCanonicalDecl(), .inst_id = field_decl_id});

    // Compute the offset to the field that appears directly in the class.
    uint64_t offset = clang_layout.getFieldOffset(
        cast<clang::FieldDecl>(chain.front())->getFieldIndex());

    // If this is an indirect field, walk the path and accumulate the offset to
    // the named field.
    for (auto* inner_decl : chain.drop_front()) {
      auto* inner_field = cast<clang::FieldDecl>(inner_decl);
      const auto& inner_layout =
          context.ast_context().getASTRecordLayout(inner_field->getParent());
      offset += inner_layout.getFieldOffset(inner_field->getFieldIndex());
    }

    layout.push_back(
        context.ast_context().toCharUnitsFromBits(offset).getQuantity());
    fields.push_back(
        {.name_id = field_name_id, .type_inst_id = field_type_inst_id});
  }

  // TODO: Add a field to prevent tail padding reuse if necessary.

  return AddTypeInst<SemIR::CustomLayoutType>(
      context, import_ir_inst_id,
      {.type_id = SemIR::TypeType::TypeId,
       .fields_id = context.struct_type_fields().Add(fields),
       .layout_id = context.custom_layouts().Add(layout)});
}

// Creates a Carbon class definition based on the information in the given Clang
// class declaration, which is assumed to be for a class definition.
static auto BuildClassDefinition(Context& context,
                                 SemIR::ImportIRInstId import_ir_inst_id,
                                 SemIR::ClassId class_id,
                                 SemIR::TypeInstId class_inst_id,
                                 clang::CXXRecordDecl* clang_def) -> void {
  auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(!class_info.has_definition_started());
  class_info.definition_id = class_inst_id;

  context.inst_block_stack().Push();

  class_info.inheritance_kind = GetInheritanceKind(clang_def);

  // Compute the class's object representation.
  auto object_repr_id = ImportClassObjectRepr(
      context, class_id, import_ir_inst_id, class_inst_id, clang_def);
  class_info.complete_type_witness_id = AddInst<SemIR::CompleteTypeWitness>(
      context, import_ir_inst_id,
      {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
       .object_repr_type_inst_id = object_repr_id});

  class_info.body_block_id = context.inst_block_stack().Pop();
}

// Computes and returns the Carbon type to use as the object representation of
// the given C++ enum type. This is a builtin int type matching the enum's
// representation.
static auto ImportEnumObjectRepresentation(
    Context& context, SemIR::ImportIRInstId import_ir_inst_id,
    clang::EnumDecl* enum_decl) -> SemIR::TypeInstId {
  auto int_type = enum_decl->getIntegerType();
  CARBON_CHECK(!int_type.isNull(), "incomplete enum type {0}",
               enum_decl->getNameAsString());

  auto int_kind = int_type->isSignedIntegerType() ? SemIR::IntKind::Signed
                                                  : SemIR::IntKind::Unsigned;
  auto bit_width_id = GetOrAddInst<SemIR::IntValue>(
      context, import_ir_inst_id,
      {.type_id = GetSingletonType(context, SemIR::IntLiteralType::TypeInstId),
       .int_id = context.ints().AddUnsigned(
           llvm::APInt(64, context.ast_context().getIntWidth(int_type)))});
  return context.types().GetAsTypeInstId(
      GetOrAddInst(context, SemIR::LocIdAndInst::NoLoc(SemIR::IntType{
                                .type_id = SemIR::TypeType::TypeId,
                                .int_kind = int_kind,
                                .bit_width_id = bit_width_id})));
}

// Creates a Carbon class definition based on the information in the given Clang
// enum declaration.
static auto BuildEnumDefinition(Context& context,
                                SemIR::ImportIRInstId import_ir_inst_id,
                                SemIR::ClassId class_id,
                                SemIR::TypeInstId class_inst_id,
                                clang::EnumDecl* enum_decl) -> void {
  auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(!class_info.has_definition_started());
  class_info.definition_id = class_inst_id;

  context.inst_block_stack().Push();

  // Don't allow inheritance from C++ enums, to match the behavior in C++.
  class_info.inheritance_kind = SemIR::Class::Final;

  // Compute the enum type's object representation. An enum is an adapter for
  // the corresponding builtin integer type.
  auto object_repr_id =
      ImportEnumObjectRepresentation(context, import_ir_inst_id, enum_decl);
  class_info.adapt_id = AddInst(
      context, SemIR::LocIdAndInst::UncheckedLoc(
                   import_ir_inst_id,
                   SemIR::AdaptDecl{.adapted_type_inst_id = object_repr_id}));
  class_info.complete_type_witness_id = AddInst<SemIR::CompleteTypeWitness>(
      context, import_ir_inst_id,
      {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
       .object_repr_type_inst_id = object_repr_id});

  class_info.body_block_id = context.inst_block_stack().Pop();
}

// Imports an enumerator declaration from Clang to Carbon.
static auto ImportEnumConstantDecl(Context& context,
                                   clang::EnumConstantDecl* enumerator_decl)
    -> SemIR::InstId {
  CARBON_CHECK(!IsClangDeclImported(context, enumerator_decl));

  // Find the enclosing enum type.
  auto type_inst_id = LookupClangDeclInstId(
      context, cast<clang::EnumDecl>(enumerator_decl->getDeclContext()));
  auto type_id = context.types().GetTypeIdForTypeInstId(type_inst_id);

  // Build a corresponding IntValue.
  auto int_id = context.ints().Add(enumerator_decl->getInitVal());
  auto loc_id =
      AddImportIRInst(context.sem_ir(), enumerator_decl->getLocation());
  auto inst_id = AddInstInNoBlock<SemIR::IntValue>(
      context, loc_id, {.type_id = type_id, .int_id = int_id});
  context.imports().push_back(inst_id);
  context.clang_decls().Add(
      {.decl = enumerator_decl->getCanonicalDecl(), .inst_id = inst_id});
  return inst_id;
}

// Mark the given `Decl` as failed in `clang_decls`.
static auto MarkFailedDecl(Context& context, clang::Decl* clang_decl) {
  context.clang_decls().Add({.decl = clang_decl->getCanonicalDecl(),
                             .inst_id = SemIR::ErrorInst::InstId});
}

// Creates an integer type of the given size.
static auto MakeIntType(Context& context, IntId size_id, bool is_signed)
    -> TypeExpr {
  auto type_inst_id = MakeIntTypeLiteral(
      context, Parse::NodeId::None,
      is_signed ? SemIR::IntKind::Signed : SemIR::IntKind::Unsigned, size_id);
  return ExprAsType(context, Parse::NodeId::None, type_inst_id);
}

// Maps a C++ builtin integer type to a Carbon type.
// TODO: Handle integer types that map to named aliases.
static auto MapBuiltinIntegerType(Context& context, SemIR::LocId loc_id,
                                  clang::QualType qual_type,
                                  const clang::BuiltinType& type) -> TypeExpr {
  clang::ASTContext& ast_context = context.ast_context();
  unsigned width = ast_context.getIntWidth(qual_type);
  bool is_signed = type.isSignedInteger();
  auto int_n_type = ast_context.getIntTypeForBitwidth(width, is_signed);
  if (ast_context.hasSameType(qual_type, int_n_type)) {
    TypeExpr type_expr =
        MakeIntType(context, context.ints().Add(width), is_signed);
    // Try to make sure integer types of 32 or 64 bits are complete so we can
    // check against them when deciding whether we need to generate a thunk.
    if (width == 32 || width == 64) {
      SemIR::TypeId type_id = type_expr.type_id;
      if (!context.types().IsComplete(type_id)) {
        TryToCompleteType(context, type_id, loc_id);
      }
    }
    return type_expr;
  }
  if (ast_context.hasSameType(qual_type, ast_context.CharTy)) {
    return ExprAsType(context, Parse::NodeId::None,
                      MakeCharTypeLiteral(context, Parse::NodeId::None));
  }
  return TypeExpr::None;
}

// Maps a C++ builtin type to a Carbon type.
// TODO: Support more builtin types.
static auto MapBuiltinType(Context& context, SemIR::LocId loc_id,
                           clang::QualType qual_type,
                           const clang::BuiltinType& type) -> TypeExpr {
  clang::ASTContext& ast_context = context.ast_context();
  if (type.isBooleanType()) {
    CARBON_CHECK(ast_context.hasSameType(qual_type, ast_context.BoolTy));
    return ExprAsType(context, Parse::NodeId::None,
                      context.types().GetInstId(GetSingletonType(
                          context, SemIR::BoolType::TypeInstId)));
  }
  if (type.isInteger()) {
    return MapBuiltinIntegerType(context, loc_id, qual_type, type);
  }
  if (type.isFloatingPoint()) {
    if (type.isFloat16Type() || type.isFloat32Type() || type.isDoubleType() ||
        type.isFloat128Type()) {
      return ExprAsType(
          context, Parse::NodeId::None,
          MakeFloatTypeLiteral(
              context, Parse::NodeId::None,
              context.ints().Add(ast_context.getTypeSize(qual_type))));
    }
    // TODO: Handle floating-point types that map to named aliases.
  }

  return TypeExpr::None;
}

// Determines whether record_decl is a C++ class that has a custom mapping into
// Carbon, and if so, returns the corresponding Carbon type. Otherwise returns
// None.
static auto LookupCustomRecordType(Context& context,
                                   const clang::CXXRecordDecl* record_decl)
    -> TypeExpr {
  switch (GetCustomCppTypeMapping(record_decl)) {
    case CustomCppTypeMapping::None:
      return TypeExpr::None;

    case CustomCppTypeMapping::Str:
      return MakeStringType(
          context,
          AddImportIRInst(context.sem_ir(), record_decl->getLocation()));
  }
}

// Maps a C++ tag type (class, struct, union, enum) to a Carbon type.
static auto MapTagType(Context& context, const clang::TagType& type)
    -> TypeExpr {
  auto* tag_decl = type.getOriginalDecl();
  CARBON_CHECK(tag_decl);

  // Check if the declaration is already mapped.
  SemIR::InstId tag_inst_id = LookupClangDeclInstId(context, tag_decl);
  if (!tag_inst_id.has_value()) {
    if (auto* record_decl = dyn_cast<clang::CXXRecordDecl>(tag_decl)) {
      auto custom_type = LookupCustomRecordType(context, record_decl);
      if (custom_type.inst_id.has_value()) {
        context.clang_decls().Add(
            {.decl = record_decl, .inst_id = custom_type.inst_id});
        return custom_type;
      }
    }

    tag_inst_id = ImportTagDecl(context, tag_decl);
  }
  SemIR::TypeInstId record_type_inst_id =
      context.types().GetAsTypeInstId(tag_inst_id);
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
    return MapBuiltinType(context, loc_id, type, *builtin_type);
  }

  if (const auto* tag_type = type->getAs<clang::TagType>()) {
    return MapTagType(context, *tag_type);
  }

  CARBON_CHECK(!type.hasQualifiers() && !type->isPointerType(),
               "Should not see wrapper types here");

  return TypeExpr::None;
}

// Maps a qualified C++ type to a Carbon type.
static auto MapQualifiedType(Context& context, clang::QualType type,
                             TypeExpr type_expr) -> TypeExpr {
  auto quals = type.getQualifiers();

  if (quals.hasConst()) {
    auto type_id = GetConstType(context, type_expr.inst_id);
    type_expr = {.inst_id = context.types().GetInstId(type_id),
                 .type_id = type_id};
    quals.removeConst();
  }

  // TODO: Support other qualifiers.
  if (!quals.empty()) {
    return TypeExpr::None;
  }

  return type_expr;
}

// Maps a C++ pointer type to a Carbon pointer type.
static auto MapPointerType(Context& context, clang::QualType type,
                           TypeExpr pointee_type_expr) -> TypeExpr {
  CARBON_CHECK(type->isPointerType());

  if (auto nullability = type->getNullability();
      !nullability.has_value() ||
      *nullability != clang::NullabilityKind::NonNull) {
    // If the type was produced by C++ template substitution, then we assume it
    // was deduced from a Carbon pointer type, so it's non-null.
    if (!type->getAs<clang::SubstTemplateTypeParmType>()) {
      // TODO: Support nullable pointers.
      return TypeExpr::None;
    }
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
      mapped = MapQualifiedType(context, wrapper, mapped);
    } else if (wrapper->isPointerType()) {
      mapped = MapPointerType(context, wrapper, mapped);
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
  if (!method_decl || method_decl->isStatic() ||
      isa<clang::CXXConstructorDecl>(clang_decl)) {
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

  // TODO: Fill in a location once available.
  auto pattern_id =
      addr_self ? AddAddrSelfParamPattern(context, SemIR::LocId::None,
                                          type_expr_region_id, type_inst_id)
                : AddSelfParamPattern(context, SemIR::LocId::None,
                                      type_expr_region_id, type_id);

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
  params.reserve(clang_decl.getNumNonObjectParams());
  for (unsigned i : llvm::seq(clang_decl.getNumNonObjectParams())) {
    const auto* param = clang_decl.getNonObjectParameter(i);
    // TODO: Get the parameter type from the function, not from the
    // `ParmVarDecl`. The type of the `ParmVarDecl` is the type within the
    // function, and isn't in general the same as the type that's exposed to
    // callers. In particular, the parameter type exposed to callers will never
    // be cv-qualified.
    clang::QualType param_type = param->getType();

    // We map `T&` parameters to `addr param: T*`, and `T&&` parameters to
    // `param: T`.
    // TODO: Revisit this and decide what we really want to do here.
    bool is_ref_param = param_type->isLValueReferenceType();
    param_type = param_type.getNonReferenceType();

    // Mark the start of a region of insts, needed for the type expression
    // created later with the call of `EndSubpatternAsExpr()`.
    BeginSubpattern(context);
    auto [orig_type_inst_id, type_id] = MapType(context, loc_id, param_type);
    // Type expression of the binding pattern - a single-entry/single-exit
    // region that allows control flow in the type expression e.g. fn F(x: if C
    // then i32 else i64).
    SemIR::ExprRegionId type_expr_region_id =
        EndSubpatternAsExpr(context, orig_type_inst_id);

    if (!type_id.has_value()) {
      context.TODO(loc_id, llvm::formatv("Unsupported: parameter type: {0}",
                                         param->getType().getAsString()));
      return SemIR::InstBlockId::None;
    }

    if (is_ref_param) {
      type_id = GetPointerType(context, orig_type_inst_id);
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
    SemIR::InstId pattern_id =
        // TODO: Fill in a location once available.
        AddBindingPattern(context, SemIR::LocId::None, name_id, type_id,
                          type_expr_region_id, is_generic, is_template)
            .pattern_id;
    pattern_id = AddPatternInst(
        context,
        // TODO: Fill in a location once available.
        SemIR::LocIdAndInst::NoLoc(SemIR::ValueParamPattern(
            {.type_id = context.insts().Get(pattern_id).type_id(),
             .subpattern_id = pattern_id,
             .index = SemIR::CallParamIndex::None})));
    if (is_ref_param) {
      pattern_id = AddPatternInst(
          context,
          // TODO: Fill in a location once available.
          SemIR::LocIdAndInst::NoLoc(SemIR::AddrPattern(
              {.type_id = GetPatternType(
                   context,
                   context.types().GetTypeIdForTypeInstId(orig_type_inst_id)),
               .inner_id = pattern_id})));
    }
    params.push_back(pattern_id);
  }
  return context.inst_blocks().Add(params);
}

// Returns the return `TypeExpr` of the given function declaration. In case of
// an unsupported return type, returns `SemIR::ErrorInst::InstId`. Constructors
// are treated as returning a class instance.
// TODO: Support more return types.
static auto GetReturnTypeExpr(Context& context, SemIR::LocId loc_id,
                              clang::FunctionDecl* clang_decl) -> TypeExpr {
  clang::QualType ret_type = clang_decl->getReturnType();
  if (!ret_type->isVoidType()) {
    TypeExpr mapped_type = MapType(context, loc_id, ret_type);
    if (!mapped_type.inst_id.has_value()) {
      context.TODO(loc_id, llvm::formatv("Unsupported: return type: {0}",
                                         ret_type.getAsString()));
      return {.inst_id = SemIR::ErrorInst::TypeInstId,
              .type_id = SemIR::ErrorInst::TypeId};
    }
    return mapped_type;
  }

  if (!isa<clang::CXXConstructorDecl>(clang_decl)) {
    // void.
    return TypeExpr::None;
  }

  // TODO: Make this a `PartialType`.
  SemIR::TypeInstId record_type_inst_id = context.types().GetAsTypeInstId(
      context.clang_decls()
          .Get(context.clang_decls().Lookup(
              cast<clang::Decl>(clang_decl->getParent())))
          .inst_id);
  return {
      .inst_id = record_type_inst_id,
      .type_id = context.types().GetTypeIdForTypeInstId(record_type_inst_id)};
}

// Returns the return pattern of the given function declaration. In case of an
// unsupported return type, it produces a diagnostic and returns
// `SemIR::ErrorInst::InstId`. Constructors are treated as returning a class
// instance.
static auto GetReturnPattern(Context& context, SemIR::LocId loc_id,
                             clang::FunctionDecl* clang_decl) -> SemIR::InstId {
  auto [type_inst_id, type_id] = GetReturnTypeExpr(context, loc_id, clang_decl);
  if (!type_inst_id.has_value()) {
    // void.
    return SemIR::InstId::None;
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
                                      clang::FunctionDecl* clang_decl)
    -> std::optional<FunctionParamsInsts> {
  if (isa<clang::CXXDestructorDecl>(clang_decl)) {
    context.TODO(loc_id, "Unsupported: Destructor");
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
  auto return_slot_pattern_id = GetReturnPattern(context, loc_id, clang_decl);
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

// Returns the Carbon function name for the given function.
static auto GetFunctionName(Context& context, clang::FunctionDecl* clang_decl)
    -> SemIR::NameId {
  switch (clang_decl->getDeclName().getNameKind()) {
    case clang::DeclarationName::CXXConstructorName: {
      return context.classes()
          .Get(context.insts()
                   .GetAs<SemIR::ClassDecl>(LookupClangDeclInstId(
                       context, cast<clang::Decl>(clang_decl->getParent())))
                   .class_id)
          .name_id;
    }

    case clang::DeclarationName::CXXOperatorName: {
      return SemIR::NameId::CppOperator;
    }

    default: {
      return AddIdentifierName(context, clang_decl->getName());
    }
  }
}

// Creates a `FunctionDecl` and a `Function` without C++ thunk information.
// Returns std::nullopt on failure. The given Clang declaration is assumed to:
// * Have not been imported before.
// * Be of supported type (ignoring parameters).
static auto ImportFunction(Context& context, SemIR::LocId loc_id,
                           clang::FunctionDecl* clang_decl)
    -> std::optional<SemIR::FunctionId> {
  context.scope_stack().PushForDeclName();
  context.inst_block_stack().Push();
  context.pattern_block_stack().Push();

  auto function_params_insts =
      CreateFunctionParamsInsts(context, loc_id, clang_decl);

  auto pattern_block_id = context.pattern_block_stack().Pop();
  auto decl_block_id = context.inst_block_stack().Pop();
  context.scope_stack().Pop();

  if (!function_params_insts.has_value()) {
    return std::nullopt;
  }

  auto function_decl = SemIR::FunctionDecl{
      SemIR::TypeId::None, SemIR::FunctionId::None, decl_block_id};
  auto decl_id =
      AddPlaceholderInstInNoBlock(context, Parse::NodeId::None, function_decl);
  context.imports().push_back(decl_id);

  auto virtual_modifier = SemIR::Function::VirtualModifier::None;
  int32_t virtual_index = -1;
  if (auto* method_decl = dyn_cast<clang::CXXMethodDecl>(clang_decl)) {
    if (method_decl->size_overridden_methods()) {
      virtual_modifier = SemIR::Function::VirtualModifier::Override;
    } else if (method_decl->isVirtual()) {
      virtual_modifier = SemIR::Function::VirtualModifier::Virtual;
    }
    if (virtual_modifier != SemIR::Function::VirtualModifier::None) {
      // TODO: Add support for Microsoft/non-Itanium vtables.
      virtual_index = dyn_cast<clang::ItaniumVTableContext>(
                          context.ast_context().getVTableContext())
                          ->getMethodVTableIndex(method_decl);
    }
  }
  auto function_info = SemIR::Function{
      {.name_id = GetFunctionName(context, clang_decl),
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
       .virtual_modifier = virtual_modifier,
       .virtual_index = virtual_index,
       .self_param_id = FindSelfPattern(
           context, function_params_insts->implicit_param_patterns_id),
       .clang_decl_id = context.clang_decls().Add(
           {.decl = clang_decl, .inst_id = decl_id})}};

  function_decl.function_id = context.functions().Add(function_info);
  function_decl.type_id = GetFunctionType(context, function_decl.function_id,
                                          SemIR::SpecificId::None);
  ReplaceInstBeforeConstantUse(context, decl_id, function_decl);
  return function_decl.function_id;
}

auto ImportCppFunctionDecl(Context& context, SemIR::LocId loc_id,
                           clang::FunctionDecl* clang_decl) -> SemIR::InstId {
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

  CARBON_CHECK(clang_decl->getFunctionType()->isFunctionProtoType(),
               "Not Prototype function (non-C++ code)");

  auto function_id = ImportFunction(context, loc_id, clang_decl);
  if (!function_id) {
    MarkFailedDecl(context, clang_decl);
    return SemIR::ErrorInst::InstId;
  }

  SemIR::Function& function_info = context.functions().Get(*function_id);
  if (IsCppThunkRequired(context, function_info)) {
    Diagnostics::AnnotationScope annotate_diagnostics(
        &context.emitter(), [&](auto& builder) {
          CARBON_DIAGNOSTIC(InCppThunk, Note,
                            "in thunk for C++ function used here");
          builder.Note(loc_id, InCppThunk);
        });

    clang::FunctionDecl* thunk_clang_decl =
        BuildCppThunk(context, function_info);
    if (thunk_clang_decl) {
      SemIR::FunctionId thunk_function_id =
          *ImportFunction(context, loc_id, thunk_clang_decl);
      SemIR::InstId thunk_function_decl_id =
          context.functions().Get(thunk_function_id).first_owning_decl_id;
      function_info.SetHasCppThunk(thunk_function_decl_id);
    }
  }

  return function_info.first_owning_decl_id;
}

namespace {
// An item to be imported in an import worklist.
// TODO: If worklists ever become particularly large, consider changing this
// to use a `PointerIntPair`.
struct ImportItem {
  // A declaration that we want to import.
  clang::Decl* decl;
  // Whether we have added `decl`'s dependencies to the worklist.
  bool added_dependencies;
};
// A worklist of declarations to import.
using ImportWorklist = llvm::SmallVector<ImportItem>;
}  // namespace

// Adds the given declaration to our list of declarations to import.
static auto AddDependentDecl(Context& context, clang::Decl* decl,
                             ImportWorklist& worklist) -> void {
  if (!IsClangDeclImported(context, decl)) {
    worklist.push_back({.decl = decl, .added_dependencies = false});
  }
}

// Finds all decls that need to be imported before importing the given type and
// adds them to the given set.
static auto AddDependentUnimportedTypeDecls(Context& context,
                                            clang::QualType type,
                                            ImportWorklist& worklist) -> void {
  while (true) {
    if (type->isPointerType() || type->isReferenceType()) {
      type = type->getPointeeType();
    } else if (const clang::ArrayType* array_type =
                   type->getAsArrayTypeUnsafe()) {
      type = array_type->getElementType();
    } else {
      break;
    }
  }

  if (const auto* tag_type = type->getAs<clang::TagType>()) {
    AddDependentDecl(context, tag_type->getOriginalDecl(), worklist);
  }
}

// Finds all decls that need to be imported before importing the given function
// and adds them to the given set.
static auto AddDependentUnimportedFunctionDecls(
    Context& context, const clang::FunctionDecl& clang_decl,
    ImportWorklist& worklist) -> void {
  for (const auto* param : clang_decl.parameters()) {
    AddDependentUnimportedTypeDecls(context, param->getType(), worklist);
  }
  AddDependentUnimportedTypeDecls(context, clang_decl.getReturnType(),
                                  worklist);
}

// Finds all decls that need to be imported before importing the given
// declaration and adds them to the given set.
static auto AddDependentUnimportedDecls(Context& context,
                                        clang::Decl* clang_decl,
                                        ImportWorklist& worklist) -> void {
  if (auto* clang_function_decl = clang_decl->getAsFunction()) {
    AddDependentUnimportedFunctionDecls(context, *clang_function_decl,
                                        worklist);
  } else if (auto* type_decl = dyn_cast<clang::TypeDecl>(clang_decl)) {
    if (!isa<clang::TagDecl>(clang_decl)) {
      AddDependentUnimportedTypeDecls(
          context, type_decl->getASTContext().getTypeDeclType(type_decl),
          worklist);
    }
  }
  if (!isa<clang::TranslationUnitDecl>(clang_decl)) {
    AddDependentDecl(context, GetParentDecl(clang_decl), worklist);
  }
}

static auto ImportVarDecl(Context& context, SemIR::LocId loc_id,
                          clang::VarDecl* var_decl) -> SemIR::InstId {
  if (SemIR::InstId existing_inst_id = LookupClangDeclInstId(context, var_decl);
      existing_inst_id.has_value()) {
    return existing_inst_id;
  }

  // Extract type and name.
  clang::QualType var_type = var_decl->getType();
  SemIR::TypeId var_type_id = MapType(context, loc_id, var_type).type_id;
  if (!var_type_id.has_value()) {
    context.TODO(loc_id, llvm::formatv("Unsupported: var type: {0}",
                                       var_type.getAsString()));
    return SemIR::ErrorInst::InstId;
  }
  SemIR::NameId var_name_id = AddIdentifierName(context, var_decl->getName());

  SemIR::VarStorage var_storage{.type_id = var_type_id,
                                .pattern_id = SemIR::InstId::None};
  // We can't use the convenience for `AddPlaceholderInstInNoBlock()` with typed
  // nodes because it doesn't support insts with cleanup.
  SemIR::InstId var_storage_inst_id =
      AddPlaceholderInstInNoBlock(context, {loc_id, var_storage});

  auto clang_decl_id = context.clang_decls().Add(
      {.decl = var_decl, .inst_id = var_storage_inst_id});

  // Entity name referring to a Clang decl for mangling.
  SemIR::EntityNameId entity_name_id =
      context.entity_names().AddSymbolicBindingName(
          var_name_id, GetParentNameScopeId(context, var_decl),
          SemIR::CompileTimeBindIndex::None, false, clang_decl_id);

  // Create `BindingPattern` and `VarPattern` in a `NameBindingDecl`.
  context.pattern_block_stack().Push();
  SemIR::TypeId pattern_type_id = GetPatternType(context, var_type_id);
  SemIR::InstId binding_pattern_inst_id = AddPatternInst<SemIR::BindingPattern>(
      context, loc_id,
      {.type_id = pattern_type_id, .entity_name_id = entity_name_id});
  var_storage.pattern_id = AddPatternInst<SemIR::VarPattern>(
      context, Parse::VariablePatternId::None,
      {.type_id = pattern_type_id, .subpattern_id = binding_pattern_inst_id});
  context.imports().push_back(AddInstInNoBlock<SemIR::NameBindingDecl>(
      context, loc_id,
      {.pattern_block_id = context.pattern_block_stack().Pop()}));

  // Finalize the `VarStorage` instruction.
  ReplaceInstBeforeConstantUse(context, var_storage_inst_id, var_storage);
  context.imports().push_back(var_storage_inst_id);

  return var_storage_inst_id;
}

// Imports a declaration from Clang to Carbon. Returns the instruction for the
// new Carbon declaration, which will be an ErrorInst on failure. Assumes all
// dependencies have already been imported.
static auto ImportDeclAfterDependencies(Context& context, SemIR::LocId loc_id,
                                        clang::Decl* clang_decl)
    -> SemIR::InstId {
  if (auto* clang_function_decl = clang_decl->getAsFunction()) {
    return ImportCppFunctionDecl(context, loc_id, clang_function_decl);
  }
  if (auto* clang_namespace_decl = dyn_cast<clang::NamespaceDecl>(clang_decl)) {
    return ImportNamespaceDecl(context, clang_namespace_decl);
  }
  if (auto* type_decl = dyn_cast<clang::TypeDecl>(clang_decl)) {
    auto type = clang_decl->getASTContext().getTypeDeclType(type_decl);
    auto type_inst_id = MapType(context, loc_id, type).inst_id;
    if (!type_inst_id.has_value()) {
      context.TODO(AddImportIRInst(context.sem_ir(), type_decl->getLocation()),
                   llvm::formatv("Unsupported: Type declaration: {0}",
                                 type.getAsString()));
      return SemIR::ErrorInst::InstId;
    }
    context.clang_decls().Add({.decl = clang_decl, .inst_id = type_inst_id});
    return type_inst_id;
  }
  if (isa<clang::FieldDecl, clang::IndirectFieldDecl>(clang_decl)) {
    // Usable fields get imported as a side effect of importing the class.
    if (SemIR::InstId existing_inst_id =
            LookupClangDeclInstId(context, clang_decl);
        existing_inst_id.has_value()) {
      return existing_inst_id;
    }
    context.TODO(AddImportIRInst(context.sem_ir(), clang_decl->getLocation()),
                 "Unsupported: field declaration has unhandled type or kind");
    return SemIR::ErrorInst::InstId;
  }
  if (auto* enum_const_decl = dyn_cast<clang::EnumConstantDecl>(clang_decl)) {
    return ImportEnumConstantDecl(context, enum_const_decl);
  }
  if (auto* var_decl = dyn_cast<clang::VarDecl>(clang_decl)) {
    return ImportVarDecl(context, loc_id, var_decl);
  }

  context.TODO(AddImportIRInst(context.sem_ir(), clang_decl->getLocation()),
               llvm::formatv("Unsupported: Declaration type {0}",
                             clang_decl->getDeclKindName()));
  return SemIR::ErrorInst::InstId;
}

// Attempts to import a set of declarations. Returns `false` if an error was
// produced, `true` otherwise.
static auto ImportDeclSet(Context& context, SemIR::LocId loc_id,
                          ImportWorklist& worklist) -> bool {
  // Walk the dependency graph in depth-first order, and import declarations
  // once we've imported all of their dependencies.
  while (!worklist.empty()) {
    auto& item = worklist.back();
    if (!item.added_dependencies) {
      // Skip items we've already imported. We checked this when initially
      // adding the item to the worklist, but it might have been added to the
      // worklist twice before the first time we visited it. For example, this
      // happens for `fn F(a: Cpp.T, b: Cpp.T)`.
      if (IsClangDeclImported(context, item.decl)) {
        worklist.pop_back();
        continue;
      }

      // First time visiting this declaration (preorder): add its dependencies
      // to the work list.
      item.added_dependencies = true;
      AddDependentUnimportedDecls(context, item.decl, worklist);
    } else {
      // Second time visiting this declaration (postorder): its dependencies are
      // already imported, so we can import it now.
      auto* decl = worklist.pop_back_val().decl;
      // Functions that are part of the overload set are imported at a later
      // point, once the overload resolution has selected the suitable function
      // for the call.
      if (decl->getAsFunction()) {
        continue;
      }
      auto inst_id = ImportDeclAfterDependencies(context, loc_id, decl);
      CARBON_CHECK(inst_id.has_value());
      if (inst_id == SemIR::ErrorInst::InstId) {
        return false;
      }
      CARBON_CHECK(IsClangDeclImported(context, decl));
    }
  }

  return true;
}

// Imports a declaration from Clang to Carbon. If successful, returns the
// instruction for the new Carbon declaration. All unimported dependencies are
// imported first.
static auto ImportDeclAndDependencies(Context& context, SemIR::LocId loc_id,
                                      clang::Decl* clang_decl)
    -> SemIR::InstId {
  // Collect dependencies by walking the dependency graph in depth-first order.
  ImportWorklist worklist;
  AddDependentDecl(context, clang_decl, worklist);
  if (!ImportDeclSet(context, loc_id, worklist)) {
    return SemIR::ErrorInst::InstId;
  }
  return LookupClangDeclInstId(context, clang_decl);
}

// Imports a type from Clang to Carbon. If successful, returns the imported
// TypeId. All unimported dependencies are imported first.
static auto ImportTypeAndDependencies(Context& context, SemIR::LocId loc_id,
                                      clang::QualType type) -> TypeExpr {
  // Collect dependencies by walking the dependency graph in depth-first order.
  ImportWorklist worklist;
  AddDependentUnimportedTypeDecls(context, type, worklist);
  if (!ImportDeclSet(context, loc_id, worklist)) {
    return {.inst_id = SemIR::ErrorInst::TypeInstId,
            .type_id = SemIR::ErrorInst::TypeId};
  }
  return MapType(context, loc_id, type);
}

// Maps `clang::AccessSpecifier` to `SemIR::AccessKind`.
static auto MapAccess(clang::AccessSpecifier access_specifier)
    -> SemIR::AccessKind {
  switch (access_specifier) {
    case clang::AS_public:
    case clang::AS_none:
      return SemIR::AccessKind::Public;
    case clang::AS_protected:
      return SemIR::AccessKind::Protected;
    case clang::AS_private:
      return SemIR::AccessKind::Private;
  }
}

// Imports a `clang::NamedDecl` into Carbon and adds that name into the
// `NameScope`.
static auto ImportNameDeclIntoScope(Context& context, SemIR::LocId loc_id,
                                    SemIR::NameScopeId scope_id,
                                    SemIR::NameId name_id,
                                    clang::NamedDecl* clang_decl,
                                    clang::AccessSpecifier access)
    -> SemIR::ScopeLookupResult {
  SemIR::InstId inst_id =
      ImportDeclAndDependencies(context, loc_id, clang_decl);
  if (!inst_id.has_value()) {
    return SemIR::ScopeLookupResult::MakeNotFound();
  }
  SemIR::AccessKind access_kind = MapAccess(access);
  AddNameToScope(context, scope_id, name_id, access_kind, inst_id);
  return SemIR::ScopeLookupResult::MakeWrappedLookupResult(inst_id,
                                                           access_kind);
}

// Returns true if the scope is the top `Cpp` scope.
static auto IsTopCppScope(Context& context, SemIR::NameScopeId scope_id)
    -> bool {
  const SemIR::NameScope& name_scope = context.name_scopes().Get(scope_id);
  CARBON_CHECK(name_scope.is_cpp_scope());
  return name_scope.parent_scope_id() == SemIR::NameScopeId::Package;
}

// For builtin names like `Cpp.long`, return the associated types.
static auto LookupBuiltinTypes(Context& context, SemIR::LocId loc_id,
                               SemIR::NameScopeId scope_id,
                               SemIR::NameId name_id) -> SemIR::InstId {
  if (!IsTopCppScope(context, scope_id)) {
    return SemIR::InstId::None;
  }

  auto name = context.names().GetAsStringIfIdentifier(name_id);
  if (!name) {
    return SemIR::InstId::None;
  }

  const clang::ASTContext& ast_context = context.ast_context();

  // List of types based on
  // https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p5448.md#details
  auto builtin_type =
      llvm::StringSwitch<clang::QualType>(*name)
          .Case("signed_char", ast_context.SignedCharTy)
          .Case("short", ast_context.ShortTy)
          .Case("int", ast_context.IntTy)
          .Case("long", ast_context.LongTy)
          .Case("long_long", ast_context.LongLongTy)
          .Case("unsigned_char", ast_context.UnsignedCharTy)
          .Case("unsigned_short", ast_context.UnsignedShortTy)
          .Case("unsigned_int", ast_context.UnsignedIntTy)
          .Case("unsigned_long", ast_context.UnsignedLongTy)
          .Case("unsigned_long_long", ast_context.UnsignedLongLongTy)
          .Case("float", ast_context.FloatTy)
          .Case("double", ast_context.DoubleTy)
          .Case("long_double", ast_context.LongDoubleTy)
          .Default(clang::QualType());
  if (builtin_type.isNull()) {
    return SemIR::InstId::None;
  }

  SemIR::InstId inst_id =
      MapNonWrapperType(context, loc_id, builtin_type).inst_id;
  if (!inst_id.has_value()) {
    context.TODO(loc_id, llvm::formatv("Unsupported: builtin type: {0}",
                                       builtin_type.getAsString()));
    return SemIR::ErrorInst::InstId;
  }
  return inst_id;
}

auto ImportCppOverloadSet(Context& context, SemIR::NameScopeId scope_id,
                          SemIR::NameId name_id,
                          const clang::UnresolvedSet<4>& overload_set)
    -> SemIR::InstId {
  SemIR::CppOverloadSetId overload_set_id = context.cpp_overload_sets().Add(
      SemIR::CppOverloadSet{.name_id = name_id,
                            .parent_scope_id = scope_id,
                            .candidate_functions = overload_set});

  auto overload_set_inst_id =
      // TODO: Add a location.
      AddInstInNoBlock<SemIR::CppOverloadSetValue>(
          context, Parse::NodeId::None,
          {.type_id = GetCppOverloadSetType(context, overload_set_id,
                                            SemIR::SpecificId::None),
           .overload_set_id = overload_set_id});

  context.imports().push_back(overload_set_inst_id);
  return overload_set_inst_id;
}

// Gets the access for an overloaded function set. Returns std::nullopt
// if the access is not the same for all functions in the overload set.
// TODO: Fix to support functions with different access levels.
static auto GetOverloadSetAccess(Context& context, SemIR::LocId loc_id,
                                 const clang::UnresolvedSet<4>& overload_set)
    -> std::optional<SemIR::AccessKind> {
  clang::AccessSpecifier access = overload_set.begin().getAccess();
  for (auto it = overload_set.begin() + 1; it != overload_set.end(); ++it) {
    if (it.getAccess() != access) {
      context.TODO(loc_id, "Unsupported: Overloaded set with mixed access");
      return std::nullopt;
    }
  }
  return MapAccess(access);
}

static auto ImportOverloadSetAndDependencies(
    Context& context, SemIR::LocId loc_id, SemIR::NameScopeId scope_id,
    SemIR::NameId name_id, const clang::UnresolvedSet<4>& overloaded_set)
    -> SemIR::InstId {
  ImportWorklist worklist;
  for (clang::NamedDecl* fn_decl : overloaded_set) {
    AddDependentDecl(context, fn_decl, worklist);
  }
  if (!ImportDeclSet(context, loc_id, worklist)) {
    return SemIR::ErrorInst::InstId;
  }
  return ImportCppOverloadSet(context, scope_id, name_id, overloaded_set);
}

// Imports an overloaded function set from Clang to Carbon and adds the
// name into the `NameScope`.
static auto ImportOverloadSetIntoScope(
    Context& context, SemIR::LocId loc_id, SemIR::NameScopeId scope_id,
    SemIR::NameId name_id, const clang::UnresolvedSet<4>& overload_set)
    -> SemIR::ScopeLookupResult {
  std::optional<SemIR::AccessKind> access_kind =
      GetOverloadSetAccess(context, loc_id, overload_set);
  if (!access_kind.has_value()) {
    return SemIR::ScopeLookupResult::MakeError();
  }

  SemIR::InstId inst_id = ImportOverloadSetAndDependencies(
      context, loc_id, scope_id, name_id, overload_set);
  AddNameToScope(context, scope_id, name_id, access_kind.value(), inst_id);
  return SemIR::ScopeLookupResult::MakeWrappedLookupResult(inst_id,
                                                           access_kind.value());
}

// TODO: Refactor this method.
// TODO: Do we need to import the dependences for all functions in the overload
// set?
auto ImportNameFromCpp(Context& context, SemIR::LocId loc_id,
                       SemIR::NameScopeId scope_id, SemIR::NameId name_id)
    -> SemIR::ScopeLookupResult {
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCppNameLookup, Note,
                          "in `Cpp` name lookup for `{0}`", SemIR::NameId);
        builder.Note(loc_id, InCppNameLookup, name_id);
      });

  if (auto class_decl = context.insts().TryGetAs<SemIR::ClassDecl>(
          context.name_scopes().Get(scope_id).inst_id());
      class_decl.has_value()) {
    if (!context.types().IsComplete(
            context.classes().Get(class_decl->class_id).self_type_id)) {
      return SemIR::ScopeLookupResult::MakeError();
    }
  }

  auto lookup = ClangLookupName(context, scope_id, name_id);
  if (!lookup) {
    SemIR::InstId builtin_inst_id =
        LookupBuiltinTypes(context, loc_id, scope_id, name_id);
    if (builtin_inst_id.has_value()) {
      AddNameToScope(context, scope_id, name_id, SemIR::AccessKind::Public,
                     builtin_inst_id);
      return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
          builtin_inst_id, SemIR::AccessKind::Public);
    }
    return SemIR::ScopeLookupResult::MakeNotFound();
  }

  // Access checks are performed separately by the Carbon name lookup logic.
  lookup->suppressAccessDiagnostics();

  if (lookup->isOverloadedResult() ||
      (lookup->isSingleResult() &&
       lookup->getFoundDecl()->isFunctionOrFunctionTemplate())) {
    clang::UnresolvedSet<4> overload_set;
    overload_set.append(lookup->begin(), lookup->end());
    return ImportOverloadSetIntoScope(context, loc_id, scope_id, name_id,
                                      overload_set);
  }

  if (!lookup->isSingleResult()) {
    // Clang will diagnose ambiguous lookup results for us.
    if (!lookup->isAmbiguous()) {
      context.TODO(loc_id,
                   llvm::formatv("Unsupported: Lookup succeeded but couldn't "
                                 "find a single result; LookupResultKind: {0}",
                                 static_cast<int>(lookup->getResultKind())));
    }
    context.name_scopes().AddRequiredName(scope_id, name_id,
                                          SemIR::ErrorInst::InstId);
    return SemIR::ScopeLookupResult::MakeError();
  }

  if (!IsDeclInjectedClassName(context, scope_id, name_id,
                               lookup->getFoundDecl())) {
    return ImportNameDeclIntoScope(context, loc_id, scope_id, name_id,
                                   lookup->getFoundDecl(),
                                   lookup->begin().getAccess());
  }

  clang::DeclContextLookupResult constructors_lookup =
      ClangConstructorLookup(context, scope_id);

  clang::UnresolvedSet<4> overload_set;
  for (clang::Decl* decl : constructors_lookup) {
    auto* constructor = cast<clang::CXXConstructorDecl>(decl);
    if (constructor->isDeleted() || constructor->isCopyOrMoveConstructor()) {
      continue;
    }
    overload_set.addDecl(constructor, constructor->getAccess());
  }
  if (overload_set.empty()) {
    return SemIR::ScopeLookupResult::MakeNotFound();
  }

  return ImportOverloadSetIntoScope(context, loc_id, scope_id, name_id,
                                    overload_set);
}

auto ImportClassDefinitionForClangDecl(Context& context, SemIR::LocId loc_id,
                                       SemIR::ClassId class_id,
                                       SemIR::ClangDeclId clang_decl_id)
    -> bool {
  clang::ASTUnit* ast = context.sem_ir().clang_ast_unit();
  CARBON_CHECK(ast);

  auto* clang_decl =
      cast<clang::TagDecl>(context.clang_decls().Get(clang_decl_id).decl);
  auto class_inst_id = context.types().GetAsTypeInstId(
      context.classes().Get(class_id).first_owning_decl_id);

  // TODO: Map loc_id into a clang location and use it for diagnostics if
  // instantiation fails, instead of annotating the diagnostic with another
  // location.
  clang::SourceLocation loc = clang_decl->getLocation();
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCppTypeCompletion, Note,
                          "while completing C++ type {0}", SemIR::TypeId);
        builder.Note(loc_id, InCppTypeCompletion,
                     context.classes().Get(class_id).self_type_id);
      });

  // Ask Clang whether the type is complete. This triggers template
  // instantiation if necessary.
  clang::DiagnosticErrorTrap trap(ast->getDiagnostics());
  if (!ast->getSema().isCompleteType(
          loc, context.ast_context().getCanonicalTagType(clang_decl))) {
    // Type is incomplete. Nothing more to do, but tell the caller if we
    // produced an error.
    return !trap.hasErrorOccurred();
  }

  auto import_ir_inst_id =
      context.insts().GetCanonicalLocId(class_inst_id).import_ir_inst_id();

  if (auto* class_decl = dyn_cast<clang::CXXRecordDecl>(clang_decl)) {
    auto* class_def = class_decl->getDefinition();
    CARBON_CHECK(class_def, "Complete type has no definition");

    if (class_def->getNumVBases()) {
      // TODO: Handle virtual bases. We don't actually know where they go in the
      // layout. We may also want to use a different size in the layout for
      // `partial C`, excluding the virtual base. It's also not entirely safe to
      // just skip over the virtual base, as the type we would construct would
      // have a misleading size. For now, treat a C++ class with vbases as
      // incomplete in Carbon.
      context.TODO(loc_id, "class with virtual bases");
      return false;
    }

    BuildClassDefinition(context, import_ir_inst_id, class_id, class_inst_id,
                         class_def);
  } else if (auto* enum_decl = dyn_cast<clang::EnumDecl>(clang_decl)) {
    BuildEnumDefinition(context, import_ir_inst_id, class_id, class_inst_id,
                        enum_decl);
  }
  return true;
}

}  // namespace Carbon::Check
