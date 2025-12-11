// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/generate_ast.h"

#include <memory>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
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

// An AST consumer that tracks top-level declarations so they can be handed off
// to code generation later.
class BufferingConsumer : public clang::ASTConsumer {
 public:
  explicit BufferingConsumer(SemIR::CppFile& file) : file_(&file) {}

  auto HandleTopLevelDecl(clang::DeclGroupRef decl_group) -> bool override {
    file_->decl_groups().push_back(decl_group);
    return true;
  }

 private:
  SemIR::CppFile* file_;
};

// An action and a set of registered Clang callbacks used to generate an AST
// from a set of Cpp imports.
class GenerateASTAction : public clang::ASTFrontendAction {
 public:
  explicit GenerateASTAction(Context& context) : context_(&context) {}

 protected:
  auto CreateASTConsumer(clang::CompilerInstance& /*clang_instance*/,
                         llvm::StringRef /*file*/)
      -> std::unique_ptr<clang::ASTConsumer> override {
    return std::make_unique<BufferingConsumer>(*context_->sem_ir().cpp_file());
  }

  auto BeginSourceFileAction(clang::CompilerInstance& /*clang_instance*/)
      -> bool override {
    // TODO: Consider creating an `ExternalSemaSource` here and attaching it to
    // the compilation.
    // TODO: `clang.getPreprocessor().enableIncrementalProcessing();` to avoid
    // the TU scope getting torn down before we're done parsing macros.
    return true;
  }

 private:
  Context* context_;
};

}  // namespace

auto GenerateAst(Context& context,
                 llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                 llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
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

  clang::DiagnosticErrorTrap trap(*diags);

  auto clang_instance_ptr =
      std::make_unique<clang::CompilerInstance>(invocation);
  auto& clang_instance = *clang_instance_ptr;
  context.sem_ir().set_cpp_file(
      std::make_unique<SemIR::CppFile>(std::move(clang_instance_ptr)));

  clang_instance.setDiagnostics(diags);
  clang_instance.setVirtualFileSystem(fs);
  clang_instance.createFileManager();
  clang_instance.createSourceManager();
  if (!clang_instance.createTarget()) {
    return false;
  }

  context.set_cpp_context(std::make_unique<CppContext>(
      std::make_unique<GenerateASTAction>(context)));

  if (!context.cpp_context()->action().BeginSourceFile(clang_instance,
                                                       inputs[0])) {
    return false;
  }

  if (llvm::Error error = context.cpp_context()->action().Execute()) {
    // `Execute` currently never fails, but its contract allows it to.
    context.TODO(SemIR::LocId::None, "failed to execute clang action: " +
                                         llvm::toString(std::move(error)));
    return false;
  }

  // Flush any diagnostics. We know we're not part-way through emitting a
  // diagnostic now.
  context.emitter().Flush();

  return !trap.hasErrorOccurred();
}

}  // namespace Carbon::Check
