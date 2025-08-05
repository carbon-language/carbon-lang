// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_cpp.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
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
#include "toolchain/sem_ir/inst.h"
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
  // Creates an instance with the location that triggers calling Clang.
  // `context` must not be null.
  explicit CarbonClangDiagnosticConsumer(
      Context* context, std::shared_ptr<clang::CompilerInvocation> invocation)
      : context_(context), invocation_(std::move(invocation)) {
    context->emitter().AddFlushFn([this] { EmitDiagnostics(); });
  }

  ~CarbonClangDiagnosticConsumer() override {
    CARBON_CHECK(diagnostic_infos_.empty(),
                 "Missing flush before destroying diagnostic consumer");
  }

  // Generates a Carbon warning for each Clang warning and a Carbon error for
  // each Clang error or fatal.
  auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                        const clang::Diagnostic& info) -> void override {
    DiagnosticConsumer::HandleDiagnostic(diag_level, info);

    SemIR::ImportIRInstId clang_import_ir_inst_id =
        AddImportIRInst(*context_, info.getLocation());

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

  // Outputs Carbon diagnostics based on the collected Clang diagnostics. Must
  // be called after the AST is set in the context.
  auto EmitDiagnostics() -> void {
    CARBON_CHECK(context_->sem_ir().cpp_ast(),
                 "Attempted to emit diagnostics before the AST Unit is loaded");

    for (size_t i = 0; i != diagnostic_infos_.size(); ++i) {
      const ClangDiagnosticInfo& info = diagnostic_infos_[i];
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
          auto builder = context_->emitter().Build(
              SemIR::LocId(info.import_ir_inst_id),
              info.level == clang::DiagnosticsEngine::Warning
                  ? CppInteropParseWarning
                  : CppInteropParseError,
              info.message);
          builder.OverrideSnippet(info.snippet);
          for (;
               i + 1 < diagnostic_infos_.size() &&
               diagnostic_infos_[i + 1].level == clang::DiagnosticsEngine::Note;
               ++i) {
            const ClangDiagnosticInfo& note_info = diagnostic_infos_[i + 1];
            CARBON_DIAGNOSTIC(CppInteropParseNote, Note, "{0}", std::string);
            builder
                .Note(SemIR::LocId(note_info.import_ir_inst_id),
                      CppInteropParseNote, note_info.message)
                .OverrideSnippet(note_info.snippet);
          }
          // TODO: This will apply all current Carbon annotation functions. We
          // should instead track how Clang's context notes and Carbon's
          // annotation functions are interleaved, and interleave the notes in
          // the same order.
          builder.Emit();
          break;
        }
      }
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

  // The type-checking context in which we're running Clang.
  Context* context_;

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
          new CarbonClangDiagnosticConsumer(&context, invocation),
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
  context.sem_ir().set_cpp_ast(ast.get());

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

  // TODO: Map the LocId of the lookup to a clang SourceLocation and provide it
  // here so that clang's diagnostics can point into the carbon code that uses
  // the name.
  clang::LookupResult lookup(
      sema,
      clang::DeclarationNameInfo(
          clang::DeclarationName(
              sema.getPreprocessor().getIdentifierInfo(*name)),
          clang::SourceLocation()),
      clang::Sema::LookupNameKind::LookupOrdinaryName);

  auto scope_clang_decl_context_id =
      context.name_scopes().Get(scope_id).clang_decl_context_id();
  bool found = sema.LookupQualifiedName(
      lookup, dyn_cast<clang::DeclContext>(context.sem_ir()
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
  return context.sem_ir()
      .clang_decls()
      .Lookup(decl->getCanonicalDecl())
      .has_value();
}

// If `decl` already mapped to an instruction, returns that instruction.
// Otherwise returns `None`.
static auto LookupClangDeclInstId(const Context& context, clang::Decl* decl)
    -> SemIR::InstId {
  const auto& clang_decls = context.sem_ir().clang_decls();
  if (auto context_clang_decl_id = clang_decls.Lookup(decl->getCanonicalDecl());
      context_clang_decl_id.has_value()) {
    return clang_decls.Get(context_clang_decl_id).inst_id;
  }
  return SemIR::InstId::None;
}

// Returns the parent of the given declaration. Skips declaration types we
// ignore.
static auto GetParentDecl(clang::Decl* clang_decl) -> clang::Decl* {
  return cast<clang::Decl>(
      clang_decl->getDeclContext()->getNonTransparentContext());
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
          {.decl = clang_decl->getCanonicalDecl(), .inst_id = result.inst_id}));
  return result.inst_id;
}

static auto MapType(Context& context, SemIR::LocId loc_id, clang::QualType type)
    -> TypeExpr;

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

// Imports a record declaration from Clang to Carbon. If successful, returns
// the new Carbon class declaration `InstId`.
static auto ImportCXXRecordDecl(Context& context,
                                clang::CXXRecordDecl* clang_decl)
    -> SemIR::InstId {
  auto import_ir_inst_id = AddImportIRInst(context, clang_decl->getLocation());

  auto [class_id, class_inst_id] = BuildClassDecl(
      context, import_ir_inst_id, GetParentNameScopeId(context, clang_decl),
      AddIdentifierName(context, clang_decl->getName()));

  // TODO: The caller does the same lookup. Avoid doing it twice.
  auto clang_decl_id = context.sem_ir().clang_decls().Add(
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
        MapType(context, import_ir_inst_id, base.getType());
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
        MapType(context, import_ir_inst_id, field->getType());
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
    context.sem_ir().clang_decls().Add(
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

// Creates a class definition based on the information in the given Clang
// declaration, which is assumed to be for a class definition.
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

auto ImportCppClassDefinition(Context& context, SemIR::LocId loc_id,
                              SemIR::ClassId class_id,
                              SemIR::ClangDeclId clang_decl_id) -> bool {
  clang::ASTUnit* ast = context.sem_ir().cpp_ast();
  CARBON_CHECK(ast);

  auto* clang_decl = cast<clang::CXXRecordDecl>(
      context.sem_ir().clang_decls().Get(clang_decl_id).decl);
  auto class_inst_id = context.types().GetAsTypeInstId(
      context.classes().Get(class_id).first_owning_decl_id);

  // TODO: Map loc_id into a clang location and use it for diagnostics if
  // instantiation fails, instead of annotating the diagnostic with another
  // location.
  clang::SourceLocation loc = clang_decl->getLocation();
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCppTypeCompletion, Note,
                          "while completing C++ class type {0}", SemIR::TypeId);
        builder.Note(loc_id, InCppTypeCompletion,
                     context.classes().Get(class_id).self_type_id);
      });

  // Ask Clang whether the type is complete. This triggers template
  // instantiation if necessary.
  clang::DiagnosticErrorTrap trap(ast->getDiagnostics());
  if (!ast->getSema().isCompleteType(
          loc, context.ast_context().getRecordType(clang_decl))) {
    // Type is incomplete. Nothing more to do, but tell the caller if we
    // produced an error.
    return !trap.hasErrorOccurred();
  }

  clang::CXXRecordDecl* clang_def = clang_decl->getDefinition();
  CARBON_CHECK(clang_def, "Complete type has no definition");

  if (clang_def->getNumVBases()) {
    // TODO: Handle virtual bases. We don't actually know where they go in the
    // layout. We may also want to use a different size in the layout for
    // `partial C`, excluding the virtual base. It's also not entirely safe to
    // just skip over the virtual base, as the type we would construct would
    // have a misleading size. For now, treat a C++ class with vbases as
    // incomplete in Carbon.
    context.TODO(loc_id, "class with virtual bases");
    return false;
  }

  auto import_ir_inst_id =
      context.insts().GetCanonicalLocId(class_inst_id).import_ir_inst_id();
  BuildClassDefinition(context, import_ir_inst_id, class_id, class_inst_id,
                       clang_def);
  return true;
}

// Mark the given `Decl` as failed in `clang_decls`.
static auto MarkFailedDecl(Context& context, clang::Decl* clang_decl) {
  context.sem_ir().clang_decls().Add({.decl = clang_decl->getCanonicalDecl(),
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

// Maps a C++ builtin type to a Carbon type.
// TODO: Support more builtin types.
static auto MapBuiltinType(Context& context, clang::QualType qual_type,
                           const clang::BuiltinType& type) -> TypeExpr {
  clang::ASTContext& ast_context = context.ast_context();
  if (type.isBooleanType()) {
    CARBON_CHECK(ast_context.hasSameType(qual_type, ast_context.BoolTy));
    return ExprAsType(context, Parse::NodeId::None,
                      context.types().GetInstId(GetSingletonType(
                          context, SemIR::BoolType::TypeInstId)));
  }
  if (type.isInteger()) {
    auto width = ast_context.getIntWidth(qual_type);
    bool is_signed = type.isSignedInteger();
    auto int_n_type = ast_context.getIntTypeForBitwidth(width, is_signed);
    if (ast_context.hasSameType(qual_type, int_n_type)) {
      return MakeIntType(context, context.ints().Add(width), is_signed);
    }
    // TODO: Handle integer types that map to named aliases.
  } else if (type.isDoubleType()) {
    // TODO: Handle other floating point types when Carbon supports fN where N
    // != 64.
    CARBON_CHECK(ast_context.getTypeSize(qual_type) == 64);
    CARBON_CHECK(ast_context.hasSameType(qual_type, ast_context.DoubleTy));
    return ExprAsType(
        context, Parse::NodeId::None,
        MakeFloatTypeLiteral(context, Parse::NodeId::None,
                             SemIR::FloatKind::None, context.ints().Add(64)));
  }

  return {.inst_id = SemIR::TypeInstId::None, .type_id = SemIR::TypeId::None};
}

// Maps a C++ record type to a Carbon type.
static auto MapRecordType(Context& context, const clang::RecordType& type)
    -> TypeExpr {
  auto* record_decl = dyn_cast<clang::CXXRecordDecl>(type.getDecl());
  if (!record_decl) {
    return {.inst_id = SemIR::TypeInstId::None, .type_id = SemIR::TypeId::None};
  }

  // Check if the declaration is already mapped.
  SemIR::InstId record_inst_id = LookupClangDeclInstId(context, record_decl);
  if (!record_inst_id.has_value()) {
    record_inst_id = ImportCXXRecordDecl(context, record_decl);
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
static auto MapNonWrapperType(Context& context, clang::QualType type)
    -> TypeExpr {
  if (const auto* builtin_type = type->getAs<clang::BuiltinType>()) {
    return MapBuiltinType(context, type, *builtin_type);
  }

  if (const auto* record_type = type->getAs<clang::RecordType>()) {
    return MapRecordType(context, *record_type);
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

  auto mapped = MapNonWrapperType(context, type);

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

  if (auto* method_decl = dyn_cast<clang::CXXMethodDecl>(clang_decl)) {
    if (method_decl->isVirtual()) {
      context.TODO(loc_id, "Unsupported: Virtual function");
      MarkFailedDecl(context, clang_decl);
      return SemIR::ErrorInst::InstId;
    }
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

using DeclSet = llvm::SetVector<clang::Decl*>;

// Adds the given declaration to our list of declarations to import.
static auto AddDependentDecl(const Context& context, clang::Decl* decl,
                             DeclSet& decls) -> void {
  // TODO: Do we need to also add the parent of the declaration, recursively?
  if (!IsClangDeclImported(context, decl)) {
    decls.insert(decl);
  }
}

// Finds all decls that need to be imported before importing the given type and
// adds them to the given set.
static auto AddDependentUnimportedTypeDecls(const Context& context,
                                            clang::QualType type,
                                            DeclSet& decls) -> void {
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

  if (const auto* record_type = type->getAs<clang::RecordType>()) {
    AddDependentDecl(context, record_type->getDecl(), decls);
  }
}

// Finds all decls that need to be imported before importing the given function
// and adds them to the given set.
static auto AddDependentUnimportedFunctionDecls(
    const Context& context, const clang::FunctionDecl& clang_decl,
    DeclSet& decls) -> void {
  for (const auto* param : clang_decl.parameters()) {
    AddDependentUnimportedTypeDecls(context, param->getType(), decls);
  }
  AddDependentUnimportedTypeDecls(context, clang_decl.getReturnType(), decls);
}

// Finds all decls that need to be imported before importing the given
// declaration and adds them to the given set.
static auto AddDependentUnimportedDecls(const Context& context,
                                        clang::Decl* clang_decl, DeclSet& decls)
    -> void {
  if (auto* parent_decl = GetParentDecl(clang_decl)) {
    AddDependentDecl(context, parent_decl, decls);
  }

  if (auto* clang_function_decl = clang_decl->getAsFunction()) {
    AddDependentUnimportedFunctionDecls(context, *clang_function_decl, decls);
  } else if (auto* type_decl = dyn_cast<clang::TypeDecl>(clang_decl)) {
    AddDependentUnimportedTypeDecls(
        context, type_decl->getASTContext().getTypeDeclType(type_decl), decls);
  }
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
  if (auto* clang_namespace_decl = dyn_cast<clang::NamespaceDecl>(clang_decl)) {
    return ImportNamespaceDecl(context, clang_namespace_decl);
  }
  if (auto* type_decl = dyn_cast<clang::TypeDecl>(clang_decl)) {
    auto type = type_decl->getASTContext().getTypeDeclType(type_decl);
    auto type_inst_id = MapType(context, loc_id, type).inst_id;
    if (!type_inst_id.has_value()) {
      context.TODO(loc_id, llvm::formatv("Unsupported: Type declaration: {0}",
                                         type.getAsString()));
      return SemIR::ErrorInst::InstId;
    }
    return type_inst_id;
  }
  if (isa<clang::FieldDecl, clang::IndirectFieldDecl>(clang_decl)) {
    // Usable fields get imported as a side effect of importing the class.
    if (SemIR::InstId existing_inst_id =
            LookupClangDeclInstId(context, clang_decl);
        existing_inst_id.has_value()) {
      return existing_inst_id;
    }
    context.TODO(loc_id, "Unsupported: Unhandled kind of field declaration");
    return SemIR::InstId::None;
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
    AddDependentUnimportedDecls(context, clang_decls[i], clang_decls);
  }

  // Import dependencies in reverse order.
  auto inst_id = SemIR::InstId::None;
  for (clang::Decl* clang_decl_to_import : llvm::reverse(clang_decls)) {
    inst_id =
        ImportDeclAfterDependencies(context, loc_id, clang_decl_to_import);
    if (!inst_id.has_value()) {
      break;
    }
  }

  return inst_id;
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

auto ImportNameFromCpp(Context& context, SemIR::LocId loc_id,
                       SemIR::NameScopeId scope_id, SemIR::NameId name_id)
    -> SemIR::ScopeLookupResult {
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCppNameLookup, Note,
                          "in `Cpp` name lookup for `{0}`", SemIR::NameId);
        builder.Note(loc_id, InCppNameLookup, name_id);
      });

  auto lookup = ClangLookup(context, scope_id, name_id);
  if (!lookup) {
    return SemIR::ScopeLookupResult::MakeNotFound();
  }

  // Access checks are performed separately by the Carbon name lookup logic.
  lookup->suppressAccessDiagnostics();

  if (!lookup->isSingleResult()) {
    // Clang will diagnose ambiguous lookup results for us.
    if (!lookup->isAmbiguous()) {
      context.TODO(loc_id,
                   llvm::formatv("Unsupported: Lookup succeeded but couldn't "
                                 "find a single result; LookupResultKind: {0}",
                                 static_cast<int>(lookup->getResultKind()))
                       .str());
    }
    context.name_scopes().AddRequiredName(scope_id, name_id,
                                          SemIR::ErrorInst::InstId);
    return SemIR::ScopeLookupResult::MakeError();
  }

  return ImportNameDeclIntoScope(context, loc_id, scope_id, name_id,
                                 lookup->getFoundDecl(),
                                 lookup->begin().getAccess());
}

}  // namespace Carbon::Check
