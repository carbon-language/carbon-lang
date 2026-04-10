// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/generate_ast.h"

#include <memory>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/MultiplexExternalSemaSource.h"
#include "clang/Sema/Sema.h"
#include "common/check.h"
#include "common/map.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/cpp/export.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/cpp_file.h"

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

// Appends a line marker and the specified `code` to `out`, adjusting the
// `line` number if the `code_token` represents a block string literal.
static auto AppendInlineCode(Context& context, llvm::raw_ostream& out,
                             Lex::TokenIndex code_token, llvm::StringRef code)
    -> void {
  // Compute the line number on which the C++ code starts. Usually the code
  // is specified as a block string literal and starts on the line after the
  // start of the string token.
  // TODO: Determine if this is a block string literal without calling
  // `GetTokenText`, which re-lexes the string.
  int line = context.tokens().GetLineNumber(code_token);
  if (context.tokens().GetTokenText(code_token).contains('\n')) {
    ++line;
  }

  GenerateLineMarker(context, out, line);
  out << code << "\n";
}

// Generates C++ file contents to #include all requested imports.
static auto GenerateCppIncludesHeaderCode(
    Context& context, llvm::ArrayRef<Parse::Tree::PackagingNames> imports)
    -> std::string {
  RawStringOstream code_stream;
  for (const Parse::Tree::PackagingNames& import : imports) {
    if (import.inline_body_id.has_value()) {
      // Expand `import Cpp inline "code";` directly into the specified code.
      auto code_token = context.parse_tree().node_token(import.inline_body_id);
      AppendInlineCode(context, code_stream, code_token,
                       context.string_literal_values().Get(
                           context.tokens().GetStringLiteralValue(code_token)));
      // TODO: Inject a clang pragma here to produce an error if there are
      // unclosed scopes at the end of this inline C++ fragment.
    } else if (import.library_id.has_value()) {
      // Translate `import Cpp library "foo.h";` into `#include "foo.h"`.
      GenerateLineMarker(context, code_stream,
                         context.tokens().GetLineNumber(
                             context.parse_tree().node_token(import.node_id)));
      auto name = context.string_literal_values().Get(import.library_id);
      if (name.starts_with('<') && name.ends_with('>')) {
        code_stream << "#include <"
                    << FormatEscaped(name.drop_front().drop_back()) << ">\n";
      } else {
        code_stream << "#include \"" << FormatEscaped(name) << "\"\n";
      }
    }
  }
  return code_stream.TakeStr();
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
    CARBON_CHECK(
        sem_ir_->cpp_file(),
        "Attempted to emit C++ diagnostics before the C++ file is set");

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
   protected:
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

    // Make a deep copy of options that we modify.
    FrontendOpts = std::make_shared<clang::FrontendOptions>(*FrontendOpts);
    PPOpts = std::make_shared<clang::PreprocessorOptions>(*PPOpts);
  }
};

// Provides clang AST nodes representing Carbon SemIR entities.
class CarbonExternalASTSource : public clang::ExternalASTSource {
 public:
  explicit CarbonExternalASTSource(Context* context,
                                   clang::ASTContext* ast_context)
      : context_(context), ast_context_(ast_context) {}

  auto StartTranslationUnit(clang::ASTConsumer* consumer) -> void override;

  // Look up decls for `decl_name` inside `decl_context`, adding the decls to
  // `decl_context`. Returns true if any decls were added.
  auto FindExternalVisibleDeclsByName(
      const clang::DeclContext* decl_context, clang::DeclarationName decl_name,
      const clang::DeclContext* original_decl_context) -> bool override;

  auto CompleteType(clang::TagDecl* tag_decl) -> void override;

 private:
  // Map a Carbon entity to a Clang NamedDecl. Returns null if the entity cannot
  // currently be represented in C++.
  auto MapInstIdToClangDecl(clang::DeclContext& decl_context,
                            LookupResult lookup) -> clang::NamedDecl*;

  // Get a current best-effort location for the current position within C++
  // processing.
  auto GetCurrentCppLocId() -> SemIR::LocId {
    auto* cpp_context = context_->cpp_context();
    CARBON_CHECK(cpp_context);

    // Use the current token location when parsing.
    auto clang_source_loc = cpp_context->parser().getCurToken().getLocation();
    if (auto& code_synthesis_contexts =
            cpp_context->sema().CodeSynthesisContexts;
        !code_synthesis_contexts.empty()) {
      // Use the current point of instantiation during template instantiation.
      clang_source_loc = code_synthesis_contexts.back().PointOfInstantiation;
    }

    // TODO: Refactor with AddImportIRInst in import.cpp.
    SemIR::ClangSourceLocId clang_source_loc_id =
        context_->sem_ir().clang_source_locs().Add(clang_source_loc);
    return context_->import_ir_insts().Add(
        SemIR::ImportIRInst(clang_source_loc_id));
  }

  Check::Context* context_;
  clang::ASTContext* ast_context_;

  // Has the "Carbon" C++ namespace been created yet
  // (this could be replaced with `!scope_map_.empty()` if Carbon::Map supported
  // `empty()`)
  bool root_scope_initialized_ = false;
};

}  // namespace

void CarbonExternalASTSource::StartTranslationUnit(
    clang::ASTConsumer* /*Consumer*/) {
  auto& translation_unit = *ast_context_->getTranslationUnitDecl();
  // Mark the translation unit as having external storage so we get a query for
  // the `Carbon` namespace in the top level/translation unit scope.
  translation_unit.setHasExternalVisibleStorage();
}

// If the given name scope was produced by importing a C++ declaration, return
// the corresponding Clang declaration.
static auto GetClangDeclForScope(Context& context, SemIR::NameScopeId scope_id)
    -> clang::NamedDecl* {
  if (!scope_id.has_value()) {
    return nullptr;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  if (!scope.is_cpp_scope()) {
    return nullptr;
  }
  return dyn_cast<clang::NamedDecl>(
      context.clang_decls().Get(scope.clang_decl_context_id()).key.decl);
}

auto CarbonExternalASTSource::MapInstIdToClangDecl(
    clang::DeclContext& decl_context, LookupResult lookup)
    -> clang::NamedDecl* {
  auto target_inst_id = lookup.scope_result.target_inst_id();
  auto target_constant =
      context_->constant_values().GetConstantInstId(target_inst_id);
  auto target_inst = context_->insts().Get(target_constant);
  CARBON_KIND_SWITCH(target_inst) {
    case CARBON_KIND(SemIR::Namespace namespace_info): {
      if (auto* decl =
              GetClangDeclForScope(*context_, namespace_info.name_scope_id)) {
        return decl;
      }
      auto& name_scope =
          context_->name_scopes().Get(namespace_info.name_scope_id);
      auto* identifier_info =
          GetClangIdentifierInfo(*context_, name_scope.name_id());
      if (!identifier_info) {
        // TODO: Handle keyword package names like `Cpp` and `Core`. These can
        // be named from C++ via an alias.
        context_->TODO(SemIR::LocId(target_inst_id),
                       "interop with non-identifier package name");
        return nullptr;
      }
      // TODO: Don't immediately use the decl_context - build any intermediate
      // namespaces iteratively.
      // Eventually add a mapping and use that/populate it/keep it up to date.
      // decl_context could be prepopulated in that mapping and not passed
      // explicitly to MapInstIdToClangDecl.
      auto* namespace_decl = clang::NamespaceDecl::Create(
          *ast_context_, &decl_context, false, clang::SourceLocation(),
          clang::SourceLocation(), identifier_info, nullptr, false);
      auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(namespace_decl);
      context_->clang_decls().Add({.key = key, .inst_id = target_inst_id});
      namespace_decl->setHasExternalVisibleStorage();
      return namespace_decl;
    }
    case CARBON_KIND(SemIR::ClassType class_type): {
      const auto& class_info = context_->classes().Get(class_type.class_id);
      if (auto* decl = GetClangDeclForScope(*context_, class_info.scope_id)) {
        return decl;
      }
      auto* identifier_info =
          GetClangIdentifierInfo(*context_, class_info.name_id);
      CARBON_CHECK(identifier_info, "class with non-identifier name {0}",
                   class_info.name_id);
      // TODO: Check whether we've already mapped this class and if so, return
      // the prior mapping.
      auto* record_decl = clang::CXXRecordDecl::Create(
          *ast_context_, clang::TagTypeKind::Class, &decl_context,
          clang::SourceLocation(), clang::SourceLocation(), identifier_info);
      auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(record_decl);
      context_->clang_decls().Add({.key = key, .inst_id = target_inst_id});
      record_decl->setHasExternalLexicalStorage();
      record_decl->setHasExternalVisibleStorage();
      if (isa<clang::CXXRecordDecl>(decl_context)) {
        // TODO: Map Carbon access to C++ access.
        record_decl->setAccess(clang::AS_public);
      }
      return record_decl;
    }
    case SemIR::StructValue::Kind: {
      auto callee = GetCallee(context_->sem_ir(), target_constant);
      auto* callee_function = std::get_if<SemIR::CalleeFunction>(&callee);
      if (!callee_function) {
        return nullptr;
      }

      const SemIR::Function& function =
          context_->functions().Get(callee_function->function_id);
      if (function.clang_decl_id.has_value()) {
        return cast<clang::NamedDecl>(
            context_->clang_decls().Get(function.clang_decl_id).key.decl);
      }

      return GetReverseInteropFunctionDecl(
          *context_, SemIR::LocId(target_inst_id), decl_context,
          callee_function->function_id);
    }
    default:
      return nullptr;
  }
}

auto CarbonExternalASTSource::FindExternalVisibleDeclsByName(
    const clang::DeclContext* decl_context, clang::DeclarationName decl_name,
    const clang::DeclContext* /*OriginalDC*/) -> bool {
  if (decl_context->getDeclKind() == clang::Decl::Kind::TranslationUnit) {
    // If the context doesn't already have a mapping between C++ and Carbon,
    // check if this is the root mapping (for the "Carbon" namespace in the
    // translation unit scope) and if so, create that mapping.

    if (root_scope_initialized_) {
      return false;
    }

    static const llvm::StringLiteral carbon_namespace_name = "Carbon";
    if (auto* identifier = decl_name.getAsIdentifierInfo();
        !identifier || !identifier->isStr(carbon_namespace_name)) {
      return false;
    }

    // Build the top level 'Carbon' namespace
    auto& ast_context = decl_context->getParentASTContext();
    auto& mutable_tu_decl_context = *ast_context.getTranslationUnitDecl();
    auto* carbon_cpp_namespace = clang::NamespaceDecl::Create(
        ast_context, &mutable_tu_decl_context, false, clang::SourceLocation(),
        clang::SourceLocation(), &ast_context.Idents.get(carbon_namespace_name),
        nullptr, false);
    carbon_cpp_namespace->setHasExternalVisibleStorage();
    auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(carbon_cpp_namespace);
    context_->clang_decls().Add(
        {.key = key, .inst_id = SemIR::Namespace::PackageInstId});
    SetExternalVisibleDeclsForName(decl_context, decl_name,
                                   {carbon_cpp_namespace});
    root_scope_initialized_ = true;
    return true;
  }

  // Find the Carbon declaration corresponding to this Clang declaration.
  auto* decl = cast<clang::Decl>(
      const_cast<clang::DeclContext*>(decl_context->getPrimaryContext()));
  auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(decl);
  auto decl_id = context_->clang_decls().Lookup(key);
  CARBON_CHECK(
      decl_id.has_value(),
      "The DeclContext should already be associated with a Carbon InstId.");
  auto decl_context_inst_id = context_->clang_decls().Get(decl_id).inst_id;

  llvm::SmallVector<Check::LookupScope> lookup_scopes;

  // LocId::None seems fine here because we shouldn't produce any diagnostics
  // here - completeness should've been checked by clang before this point.
  if (!AppendLookupScopesForConstant(
          *context_, SemIR::LocId::None,
          context_->constant_values().Get(decl_context_inst_id),
          SemIR::ConstantId::None, &lookup_scopes)) {
    return false;
  }

  auto* identifier = decl_name.getAsIdentifierInfo();
  if (!identifier) {
    // Only supporting identifiers for now.
    return false;
  }

  auto name_id = AddIdentifierName(*context_, identifier->getName());

  // `required=false` so Carbon doesn't diagnose a failure, let Clang diagnose
  // it or even SFINAE.
  LookupResult result =
      LookupQualifiedName(*context_, SemIR::LocId::None, name_id, lookup_scopes,
                          /*required=*/false);
  if (!result.scope_result.is_found()) {
    return false;
  }

  // Map the found Carbon entity to a Clang NamedDecl.
  // TODO: Stop passing in the `DeclContext` here; the context in which we
  // performed the lookup that first found the Carbon declaration should not
  // affect the Clang declaration we produce.
  auto* clang_decl = MapInstIdToClangDecl(
      *const_cast<clang::DeclContext*>(decl_context), result);
  if (!clang_decl) {
    return false;
  }

  SetExternalVisibleDeclsForName(decl_context, decl_name, {clang_decl});
  return true;
}

auto CarbonExternalASTSource::CompleteType(clang::TagDecl* tag_decl) -> void {
  auto* class_decl = dyn_cast<clang::CXXRecordDecl>(tag_decl);
  if (!class_decl) {
    // TODO: If we start producing clang EnumTypes, we may have to handle them
    // here too.
    return;
  }

  auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(tag_decl->getFirstDecl());
  auto clang_decl_id = context_->clang_decls().Lookup(key);
  if (!clang_decl_id.has_value()) {
    return;
  }

  auto inst_id = context_->clang_decls().Get(clang_decl_id).inst_id;
  auto const_id = context_->constant_values().Get(inst_id);
  if (!const_id.has_value()) {
    return;
  }

  auto class_type =
      context_->constant_values().TryGetInstAs<SemIR::ClassType>(const_id);
  if (!class_type) {
    return;
  }

  auto class_type_id = context_->types().GetTypeIdForTypeConstantId(const_id);
  auto context_fn = [](DiagnosticContextBuilder& /*builder*/) -> void {};
  if (!RequireCompleteType(*context_, class_type_id, GetCurrentCppLocId(),
                           context_fn)) {
    return;
  }

  class_decl->startDefinition();
  // TODO: Import base class and fields, plus any special member functions that
  // affect class properties.
  class_decl->completeDefinition();
}

// Parses a sequence of top-level declarations and forms a corresponding
// representation in the Clang AST. Unlike clang::ParseAST, does not finish the
// translation unit when EOF is reached.
static auto ParseTopLevelDecls(clang::Parser& parser,
                               clang::ASTConsumer& consumer) -> void {
  // Don't allow C++20 module declarations in inline Cpp code fragments.
  auto module_import_state = clang::Sema::ModuleImportState::NotACXX20Module;

  // Parse top-level declarations until we see EOF. Do not parse EOF, as that
  // will cause the parser to end the translation unit prematurely.
  while (parser.getCurToken().isNot(clang::tok::eof)) {
    clang::Parser::DeclGroupPtrTy decl_group;
    bool eof = parser.ParseTopLevelDecl(decl_group, module_import_state);
    CARBON_CHECK(!eof, "Should not parse decls at EOF");
    if (decl_group && !consumer.HandleTopLevelDecl(decl_group.get())) {
      // If the consumer rejects the declaration, bail out of parsing.
      //
      // TODO: In this case, we shouldn't parse any more declarations even in
      // separate inline C++ fragments. But our current AST consumer only ever
      // returns true.
      break;
    }
  }
}

namespace {

// An action and a set of registered Clang callbacks used to generate an AST
// from a set of Cpp imports.
class GenerateASTAction : public clang::ASTFrontendAction {
 public:
  explicit GenerateASTAction(Context& context) : context_(&context) {}

 protected:
  auto CreateASTConsumer(clang::CompilerInstance& clang_instance,
                         llvm::StringRef /*file*/)
      -> std::unique_ptr<clang::ASTConsumer> override {
    auto& cpp_file = *context_->sem_ir().cpp_file();
    if (!cpp_file.llvm_context()) {
      return std::make_unique<clang::ASTConsumer>();
    }
    auto code_generator =
        std::unique_ptr<clang::CodeGenerator>(clang::CreateLLVMCodeGen(
            cpp_file.diagnostics(), context_->sem_ir().filename(),
            clang_instance.getVirtualFileSystemPtr(),
            clang_instance.getHeaderSearchOpts(),
            clang_instance.getPreprocessorOpts(),
            clang_instance.getCodeGenOpts(), *cpp_file.llvm_context()));
    cpp_file.SetCodeGenerator(code_generator.get());
    return code_generator;
  }

  auto BeginSourceFileAction(clang::CompilerInstance& /*clang_instance*/)
      -> bool override {
    // TODO: `clang.getPreprocessor().enableIncrementalProcessing();` to avoid
    // the TU scope getting torn down before we're done parsing macros.
    return true;
  }

  // Parse the imports and inline C++ fragments. This is notionally very similar
  // to `clang::ParseAST`, which `ASTFrontendAction::ExecuteAction` calls, but
  // this version doesn't parse C++20 modules and stops just before reaching the
  // end of the translation unit.
  auto ExecuteAction() -> void override {
    clang::CompilerInstance& clang_instance = getCompilerInstance();
    clang_instance.createSema(getTranslationUnitKind(),
                              /*CompletionConsumer=*/nullptr);

    auto parser_ptr = std::make_unique<clang::Parser>(
        clang_instance.getPreprocessor(), clang_instance.getSema(),
        /*SkipFunctionBodies=*/false);
    auto& parser = *parser_ptr;

    clang_instance.getPreprocessor().EnterMainSourceFile();
    if (auto* source = clang_instance.getASTContext().getExternalSource()) {
      source->StartTranslationUnit(&clang_instance.getASTConsumer());
    }

    parser.Initialize();
    clang_instance.getSema().ActOnStartOfTranslationUnit();

    context_->set_cpp_context(
        std::make_unique<CppContext>(clang_instance, std::move(parser_ptr)));

    ParseTopLevelDecls(parser, clang_instance.getASTConsumer());
  }

 private:
  Context* context_;
};

}  // namespace

auto GenerateAst(Context& context,
                 llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                 llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                 llvm::LLVMContext* llvm_context,
                 std::shared_ptr<clang::CompilerInvocation> base_invocation)
    -> bool {
  CARBON_CHECK(!context.cpp_context());
  CARBON_CHECK(!context.sem_ir().cpp_file());

  auto invocation =
      std::make_shared<ShallowCopyCompilerInvocation>(*base_invocation);

  // Ask Clang to not leak memory.
  invocation->getFrontendOpts().DisableFree = false;

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

  auto clang_instance_ptr =
      std::make_unique<clang::CompilerInstance>(invocation);
  auto& clang_instance = *clang_instance_ptr;
  context.sem_ir().set_cpp_file(std::make_unique<SemIR::CppFile>(
      std::move(clang_instance_ptr), llvm_context));

  clang_instance.setDiagnostics(diags);
  clang_instance.setVirtualFileSystem(fs);
  clang_instance.createFileManager();
  clang_instance.createSourceManager();
  if (!clang_instance.createTarget()) {
    return false;
  }

  GenerateASTAction action(context);
  if (!action.BeginSourceFile(clang_instance, inputs[0])) {
    return false;
  }

  auto& ast = clang_instance.getASTContext();
  // TODO: Clang's modules support is implemented as an ExternalASTSource
  // (ASTReader) and there's no multiplexing support for ExternalASTSources at
  // the moment - so registering CarbonExternalASTSource breaks Clang modules
  // support. Implement multiplexing support (possibly in Clang) to restore
  // modules functionality.
  ast.setExternalSource(
      llvm::makeIntrusiveRefCnt<CarbonExternalASTSource>(&context, &ast));

  if (llvm::Error error = action.Execute()) {
    // `Execute` currently never fails, but its contract allows it to.
    context.TODO(SemIR::LocId::None, "failed to execute clang action: " +
                                         llvm::toString(std::move(error)));
    return false;
  }

  // Flush any diagnostics. We know we're not part-way through emitting a
  // diagnostic now.
  context.emitter().Flush();

  return true;
}

auto InjectAstFromInlineCode(Context& context, SemIR::LocId loc_id,
                             llvm::StringRef source_code) -> void {
  auto* cpp_context = context.cpp_context();
  CARBON_CHECK(cpp_context);

  clang::Sema& sema = cpp_context->sema();
  clang::Preprocessor& preprocessor = sema.getPreprocessor();
  clang::Parser& parser = cpp_context->parser();

  RawStringOstream code_stream;
  AppendInlineCode(context, code_stream,
                   context.parse_tree().node_token(loc_id.node_id()),
                   source_code);

  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(code_stream.TakeStr(),
                                                     "<inline c++>");
  clang::FileID file_id =
      preprocessor.getSourceManager().createFileID(std::move(buffer));

  if (preprocessor.EnterSourceFile(file_id, nullptr, clang::SourceLocation())) {
    // Clang will have generated a suitable error. There's nothing more to do
    // here.
    return;
  }

  // The parser will typically have an EOF as its cached current token; consume
  // that so we can reach the newly-injected tokens.
  if (parser.getCurToken().is(clang::tok::eof)) {
    parser.ConsumeToken();
  }

  ParseTopLevelDecls(parser, sema.getASTConsumer());
}

auto FinishAst(Context& context) -> void {
  if (!context.cpp_context()) {
    return;
  }

  context.cpp_context()->sema().ActOnEndOfTranslationUnit();

  // We don't call FrontendAction::EndSourceFile, because that destroys the AST.
  context.set_cpp_context(nullptr);

  context.emitter().Flush();
}

}  // namespace Carbon::Check
