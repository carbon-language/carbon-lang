// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/diagnostic_consumer.h"

#include <memory>
#include <string>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/check/context.h"
#include "toolchain/check/cpp/diagnostic_listener.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/emitter.h"

namespace Carbon::Check {

class CarbonClangDiagnosticConsumer;

// A diagnostic emitter that maps Clang SourceLocations to Carbon diagnostic
// locations.
class ClangLocDiagnosticEmitter
    : public Diagnostics::Emitter<clang::SourceLocation> {
 public:
  explicit ClangLocDiagnosticEmitter(Diagnostics::Consumer* consumer,
                                     const clang::SourceManager* source_manager)
      : Emitter(consumer), source_manager_(source_manager) {}

 protected:
  auto ConvertLoc(clang::SourceLocation loc, ContextFnT /*context_fn*/) const
      -> Diagnostics::ConvertedLoc override {
    Diagnostics::Loc result_loc;
    if (source_manager_ && loc.isValid()) {
      clang::PresumedLoc presumed_loc = source_manager_->getPresumedLoc(loc);
      if (presumed_loc.isValid()) {
        result_loc.filename = presumed_loc.getFilename();
        result_loc.line_number = presumed_loc.getLine();
        result_loc.column_number = presumed_loc.getColumn();
      }
    }
    return {.loc = result_loc, .last_byte_offset = -1};
  }

 private:
  const clang::SourceManager* source_manager_;
};

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

// A listener that emits Clang diagnostics directly to a provided Carbon
// Diagnostics::Consumer when no Carbon Context is active.
class FallbackDiagnosticListener : public CppDiagnosticListener {
 public:
  explicit FallbackDiagnosticListener(
      CarbonClangDiagnosticConsumer& clang_consumer,
      Diagnostics::Consumer& carbon_consumer)
      : CppDiagnosticListener(clang_consumer),
        carbon_consumer_(&carbon_consumer) {}

  auto EmitDiagnostics(llvm::ArrayRef<Diagnostic> diags) -> void override {
    if (diags.empty()) {
      return;
    }
    ClangLocDiagnosticEmitter emitter(carbon_consumer_,
                                      diags[0].source_manager);
    for (size_t i = 0; i != diags.size(); ++i) {
      const Diagnostic& info = diags[i];
      auto builder =
          emitter.Build(info.location, GetDiagnostic(info.level), info.message);
      builder.OverrideSnippet(info.snippet);
      for (; i + 1 < diags.size() &&
             diags[i + 1].level == clang::DiagnosticsEngine::Note;
           ++i) {
        const Diagnostic& note_info = diags[i + 1];
        builder
            .Note(note_info.location, GetDiagnostic(note_info.level),
                  note_info.message)
            .OverrideSnippet(note_info.snippet);
      }
      builder.Emit();
    }
  }

 private:
  Diagnostics::Consumer* carbon_consumer_;
};

// A listener that converts Clang diagnostics to Carbon diagnostics using a
// Carbon Context.
class ContextDiagnosticListener : public CppDiagnosticListener {
 public:
  explicit ContextDiagnosticListener(CarbonClangDiagnosticConsumer& consumer,
                                     Context& context)
      : CppDiagnosticListener(consumer), context_(&context) {}

  auto EmitDiagnostics(llvm::ArrayRef<Diagnostic> diags) -> void override {
    if (diags.empty()) {
      return;
    }
    CARBON_CHECK(
        context_->sem_ir().cpp_file(),
        "Attempted to emit C++ diagnostics before the C++ file is set");

    for (size_t i = 0; i != diags.size(); ++i) {
      const Diagnostic& info = diags[i];
      SemIR::ImportIRInstId import_ir_inst_id =
          AddImportIRInst(context_->sem_ir(), info.location);
      auto builder =
          context_->emitter().Build(SemIR::LocId(import_ir_inst_id),
                                    GetDiagnostic(info.level), info.message);
      builder.OverrideSnippet(info.snippet);
      for (; i + 1 < diags.size() &&
             diags[i + 1].level == clang::DiagnosticsEngine::Note;
           ++i) {
        const Diagnostic& note_info = diags[i + 1];
        SemIR::ImportIRInstId note_import_ir_inst_id =
            AddImportIRInst(context_->sem_ir(), note_info.location);
        builder
            .Note(SemIR::LocId(note_import_ir_inst_id),
                  GetDiagnostic(note_info.level), note_info.message)
            .OverrideSnippet(note_info.snippet);
      }
      // TODO: This will apply all current Carbon annotation functions. We
      // should instead track how Clang's context notes and Carbon's annotation
      // functions are interleaved, and interleave the notes in the same order.
      builder.Emit();
    }
  }

 private:
  Context* context_;
};

// Used to convert Clang diagnostics to Carbon diagnostics.
//
// Handling of Clang notes is a little subtle: as far as Clang is concerned,
// notes are separate diagnostics, not connected to the error or warning that
// precedes them. But in Carbon's diagnostics system, notes are part of the
// enclosing diagnostic. To handle this, listeners buffer Clang diagnostics
// until we reach a point where we know we're not in the middle of a diagnostic,
// and then emit a diagnostic along with all of its notes.
class CarbonClangDiagnosticConsumer : public clang::DiagnosticConsumer {
 public:
  explicit CarbonClangDiagnosticConsumer(
      Diagnostics::Consumer& consumer,
      std::shared_ptr<clang::CompilerInvocation> invocation)
      : fallback_listener_(*this, consumer),
        invocation_(std::move(invocation)) {}

  ~CarbonClangDiagnosticConsumer() override {
    CARBON_CHECK(diagnostic_infos_.empty(), "Missing flush before destruction");
    CARBON_CHECK(listeners_.size() == 1,
                 "Diagnostic listeners were not properly popped");
  }

  // Pushes a listener onto the stack. Diagnostics will be forwarded to the
  // innermost listener.
  auto PushListener(CppDiagnosticListener* listener) -> void {
    CARBON_CHECK(diagnostic_infos_.empty(),
                 "Missing flush before pushing listener");
    CARBON_CHECK(listener);
    listeners_.push_back(listener);
  }

  // Pops a listener from the stack.
  auto PopListener(CppDiagnosticListener* listener) -> void {
    CARBON_CHECK(diagnostic_infos_.empty(),
                 "Missing flush before popping listener");
    CARBON_CHECK(!listeners_.empty() && listeners_.back() == listener,
                 "Popping unexpected diagnostic listener");
    listeners_.pop_back();
  }

  // Flushes the innermost listener.
  auto Flush() -> void {
    if (diagnostic_infos_.empty()) {
      return;
    }
    CARBON_CHECK(!listeners_.empty(), "No diagnostic listeners registered");
    listeners_.back()->EmitDiagnostics(diagnostic_infos_);
    diagnostic_infos_.clear();
  }

  auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                        const clang::Diagnostic& info) -> void override {
    DiagnosticConsumer::HandleDiagnostic(diag_level, info);

    if (diag_level != clang::DiagnosticsEngine::Note &&
        !diagnostic_infos_.empty()) {
      Flush();
    }

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

    const clang::SourceManager* source_manager =
        info.hasSourceManager() ? &info.getSourceManager() : nullptr;
    diagnostic_infos_.push_back({.level = diag_level,
                                 .location = info.getLocation(),
                                 .source_manager = source_manager,
                                 .message = message.str().str(),
                                 .snippet = snippet_stream.TakeStr()});
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

  llvm::SmallVector<CppDiagnosticListener::Diagnostic> diagnostic_infos_;
  llvm::SmallVector<CppDiagnosticListener*, 2> listeners_;
  FallbackDiagnosticListener fallback_listener_;
  std::shared_ptr<clang::CompilerInvocation> invocation_;
};

CppDiagnosticListener::CppDiagnosticListener(
    CarbonClangDiagnosticConsumer& consumer)
    : consumer_(&consumer) {
  consumer_->PushListener(this);
}

CppDiagnosticListener::~CppDiagnosticListener() {
  consumer_->PopListener(this);
}

auto MakeDiagnosticConsumer(
    Diagnostics::Consumer& consumer,
    std::shared_ptr<clang::CompilerInvocation> invocation)
    -> std::unique_ptr<clang::DiagnosticConsumer> {
  return std::make_unique<CarbonClangDiagnosticConsumer>(consumer,
                                                         std::move(invocation));
}

auto FlushDiagnosticConsumer(clang::DiagnosticConsumer& consumer) -> void {
  static_cast<CarbonClangDiagnosticConsumer&>(consumer).Flush();
}

auto MakeContextDiagnosticListener(clang::DiagnosticConsumer& consumer,
                                   Context& context)
    -> std::unique_ptr<CppDiagnosticListener> {
  auto* clang_consumer = static_cast<CarbonClangDiagnosticConsumer*>(&consumer);
  auto listener =
      std::make_unique<ContextDiagnosticListener>(*clang_consumer, context);
  context.emitter().AddFlushFn([clang_consumer] { clang_consumer->Flush(); });
  return listener;
}

}  // namespace Carbon::Check
