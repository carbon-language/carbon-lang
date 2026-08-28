// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/diagnostic_consumer.h"

#include <memory>
#include <string>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/Lexer.h"
#include "common/check.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/check/context.h"
#include "toolchain/check/cpp/diagnostic_listener.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/sem_ir/diagnostic_loc_converter.h"

namespace Carbon::Check {

class CarbonClangDiagnosticConsumer;

// A diagnostic emitter that maps Clang source ranges to Carbon diagnostic
// locations.
class ClangLocDiagnosticEmitter
    : public Diagnostics::Emitter<clang::CharSourceRange> {
 public:
  explicit ClangLocDiagnosticEmitter(Diagnostics::Consumer* consumer,
                                     const clang::SourceManager* source_manager)
      : Emitter(consumer), source_manager_(source_manager) {}

 protected:
  auto ConvertLoc(clang::CharSourceRange range, ContextFnT /*context_fn*/) const
      -> Diagnostics::ConvertedLoc override {
    Diagnostics::Loc result_loc;
    if (source_manager_) {
      result_loc = SemIR::ConvertClangRangeToLoc(*source_manager_, range);
    }
    return {.loc = result_loc, .last_byte_offset = -1};
  }

 private:
  const clang::SourceManager* source_manager_;
};

// Returns the label saying what Clang suggests be done to a range.
//
// TODO: Carry fix-its as data -- span, replacement, confidence -- in a form
// Carbon's own diagnostics can share, rather than wording the edit into a
// label here, and render both as a unified diff following GCC's example
// rather than Clang's. See future work in
// /toolchain/docs/diagnostics_rendering.md.
static auto GetFixItLabel(const CppDiagnosticListener::FixIt& fix_it)
    -> const Diagnostics::LabelBase<std::string>& {
  if (fix_it.text.empty()) {
    CARBON_DIAGNOSTIC_LABEL(CppInteropFixItRemoval, Info, "{0}", std::string);
    return CppInteropFixItRemoval;
  }
  if (fix_it.range.getBegin() == fix_it.range.getEnd()) {
    CARBON_DIAGNOSTIC_LABEL(CppInteropFixItInsertion, Info, "{0}", std::string);
    return CppInteropFixItInsertion;
  }

  CARBON_DIAGNOSTIC_LABEL(CppInteropFixItReplacement, Info, "{0}", std::string);
  return CppInteropFixItReplacement;
}

// Returns what a fix-it label says. Clang words these as an edit to make, so
// they read as an instruction rather than as a description of the code.
static auto GetFixItText(const CppDiagnosticListener::FixIt& fix_it)
    -> std::string {
  if (fix_it.text.empty()) {
    return "remove this";
  }
  if (fix_it.range.getBegin() == fix_it.range.getEnd()) {
    return "insert `" + fix_it.text + "` here";
  }
  return "replace with `" + fix_it.text + "`";
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
      // A note explains the diagnostic it follows and is attached to that one
      // as a label. One can still lead a buffer -- a flush can land between an
      // error and its trailing notes, and Clang can emit a stray note -- and
      // however wrong that is of Clang, it must not be why a compiler dies
      // while reporting a problem.
      CARBON_DIAGNOSTIC(CppInteropStrayNote, Warning, "{0}", std::string);
      return CppInteropStrayNote;
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

// Returns whether `range` holds `loc`, which needs both to be spelled in the
// same file: two offsets from different files are unordered, and comparing them
// would report a containment that isn't one.
static auto Contains(const clang::SourceManager& source_manager,
                     clang::CharSourceRange range, clang::SourceLocation loc)
    -> bool {
  if (range.isInvalid()) {
    return false;
  }
  auto [file_id, offset] = source_manager.getDecomposedSpellingLoc(loc);
  auto [begin_file_id, begin] =
      source_manager.getDecomposedSpellingLoc(range.getBegin());
  auto [end_file_id, end] =
      source_manager.getDecomposedSpellingLoc(range.getEnd());
  return begin_file_id == file_id && end_file_id == file_id &&
         begin <= offset && offset < end;
}

// Returns whether `diag_id` is a Clang diagnostic whose text is written as
// `<what was considered>: <why it was not viable>`, with its first range
// marking the source that second half is about.
//
// These are the overload-resolution candidate notes, and each one is listed
// rather than being recognized by shape. Splitting on the first `: ` in any
// note that happens to carry a range would cut the wrong messages in the wrong
// place: a type or a path in the text supplies its own colon.
//
// TODO: Try to do this upstream instead. Clang has the structure here and
// loses it: `DiagnoseBadConversion` and its siblings know which half is the
// candidate and which is the reason, and know which argument the reason is
// about, and then format both halves into one string. A note carrying its two
// halves separately would serve Clang's own rendering as well, and would leave
// nothing here to take apart. Until then this list has to be revisited
// whenever the wording of one of these notes changes.
static auto SplitsIntoCandidateAndReason(unsigned diag_id) -> bool {
  switch (diag_id) {
    case clang::diag::note_ovl_candidate_arity:
    case clang::diag::note_ovl_candidate_arity_one:
    case clang::diag::note_ovl_candidate_bad_base_to_derived_conv:
    case clang::diag::note_ovl_candidate_bad_conv:
    case clang::diag::note_ovl_candidate_bad_conv_incomplete:
    case clang::diag::note_ovl_candidate_bad_cvr:
    case clang::diag::note_ovl_candidate_bad_cvr_this:
    case clang::diag::note_ovl_candidate_bad_list_argument:
    case clang::diag::note_ovl_candidate_bad_value_category:
      return true;
    default:
      return false;
  }
}

// Returns the parameter list of the declaration named at `loc`, which is what
// a note marks when it has nothing of its own to mark.
//
// Clang leaves a function's parameter-list range empty when the function
// declares no parameters, so a note about how many arguments it takes marks
// nothing even though the list is written. The list is the parentheses after
// the name, holding at most `void`, and it is where the count comes from.
//
// Returns an invalid range where those parentheses aren't right there: the name
// may be an implicit function's, which has no list at all, or be followed by
// template arguments, and reaching further would mark something that isn't the
// list.
static auto FindEmptyParameterList(const clang::SourceManager& source_manager,
                                   const clang::LangOptions& lang_opts,
                                   clang::SourceLocation loc)
    -> clang::CharSourceRange {
  if (loc.isInvalid() || loc.isMacroID()) {
    return clang::CharSourceRange();
  }
  // `operator()`'s name holds parentheses of its own, and the parameter list
  // is past them; the token after the name would be the name's `(`, not the
  // list's.
  clang::Token name;
  if (clang::Lexer::getRawToken(loc, name, source_manager, lang_opts,
                                /*IgnoreWhiteSpace=*/true) ||
      (name.is(clang::tok::raw_identifier) &&
       name.getRawIdentifier() == "operator")) {
    return clang::CharSourceRange();
  }
  auto open = clang::Lexer::findNextToken(loc, source_manager, lang_opts);
  if (!open || !open->is(clang::tok::l_paren)) {
    return clang::CharSourceRange();
  }
  auto close = clang::Lexer::findNextToken(open->getLocation(), source_manager,
                                           lang_opts);
  // An empty list is written as `()` or as `(void)`. The lexing here is raw, so
  // `void` arrives as an identifier rather than as the keyword it is.
  if (close && close->is(clang::tok::raw_identifier) &&
      close->getRawIdentifier() == "void") {
    close = clang::Lexer::findNextToken(close->getLocation(), source_manager,
                                        lang_opts);
  }
  if (!close || !close->is(clang::tok::r_paren)) {
    return clang::CharSourceRange();
  }
  return clang::CharSourceRange::getCharRange(open->getLocation(),
                                              close->getEndLoc());
}

// Returns the label to attach for a Clang note, which says more about the
// diagnostic it follows rather than reporting a problem of its own.
static auto GetNoteLabel() -> const Diagnostics::LabelBase<std::string>& {
  CARBON_DIAGNOSTIC_LABEL(CppInteropParseNote, Info, "{0}", std::string);
  return CppInteropParseNote;
}

// Returns the label to attach for what a note says about the source it marks,
// as opposed to the declaration the note names.
static auto GetNoteReasonLabel() -> const Diagnostics::LabelBase<std::string>& {
  CARBON_DIAGNOSTIC_LABEL(CppInteropParseNoteReason, Info, "{0}", std::string);
  return CppInteropParseNoteReason;
}

// Attaches what `info` marks: the ranges Clang would underline, which say
// nothing the message doesn't and so only mark, and the changes it suggests,
// which explain rather than state and so carry words.
//
// A range belonging to the leading diagnostic is part of the problem; one
// belonging to a note explains it, the same as the note itself does.
//
// `to_loc` maps a Clang range onto whatever locations the builder takes.
template <typename BuilderT, typename ToLocT>
static auto AttachMarks(BuilderT& builder,
                        const CppDiagnosticListener::Diagnostic& info,
                        bool is_note, ToLocT to_loc) -> void {
  for (clang::CharSourceRange range : info.ranges) {
    builder.Attach(to_loc(range), is_note
                                      ? Diagnostics::LabelCategory::Info
                                      : Diagnostics::LabelCategory::Primary);
  }
  for (const CppDiagnosticListener::FixIt& fix_it : info.fix_its) {
    builder.Attach(to_loc(fix_it.range), GetFixItLabel(fix_it),
                   GetFixItText(fix_it));
  }
}

// Attaches `note_info`, a note trailing the diagnostic being built: its words
// on the declaration it names, its reason half -- when it split into one --
// on the source that half is about, and then whatever it marks.
template <typename BuilderT, typename ToLocT>
static auto AttachNote(BuilderT& builder,
                       const CppDiagnosticListener::Diagnostic& note_info,
                       ToLocT to_loc) -> void {
  builder.Attach(to_loc(note_info.location), GetNoteLabel(), note_info.message);
  if (!note_info.reason.empty()) {
    builder.Attach(to_loc(note_info.reason_range), GetNoteReasonLabel(),
                   note_info.reason);
  }
  AttachMarks(builder, note_info, /*is_note=*/true, to_loc);
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
    auto as_range = [](clang::CharSourceRange range) { return range; };
    for (size_t i = 0; i != diags.size(); ++i) {
      const Diagnostic& info = diags[i];
      auto builder =
          emitter.Build(info.location, GetDiagnostic(info.level), info.message);
      AttachMarks(builder, info, /*is_note=*/false, as_range);
      for (; i + 1 < diags.size() &&
             diags[i + 1].level == clang::DiagnosticsEngine::Note;
           ++i) {
        AttachNote(builder, diags[i + 1], as_range);
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
      auto as_loc_id = [&](clang::CharSourceRange range) {
        return SemIR::LocId(AddImportIRInst(context_->sem_ir(), range));
      };
      AttachMarks(builder, info, /*is_note=*/false, as_loc_id);
      for (; i + 1 < diags.size() &&
             diags[i + 1].level == clang::DiagnosticsEngine::Note;
           ++i) {
        AttachNote(builder, diags[i + 1], as_loc_id);
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

    // What Clang would underline and what it suggests changing, kept as ranges
    // rather than as text: the toolchain marks and labels them itself, so that
    // a diagnostic from C++ reads like one from Carbon. The `#include` and
    // macro-expansion stacks reach the location as `included from` and
    // `expanded from macro defined at` steps, drawn like any other path a
    // location was reached by.
    const clang::SourceManager* source_manager =
        info.hasSourceManager() ? &info.getSourceManager() : nullptr;

    // Clang's ranges reach to the start of their last token rather than past
    // it, so they are widened here. Everything downstream then measures a range
    // by subtracting its ends, and an empty one is an insertion point rather
    // than a range that happens to name one token.
    auto as_char_range = [&](clang::CharSourceRange range) {
      if (!source_manager || range.isCharRange()) {
        return range;
      }
      return clang::Lexer::getAsCharRange(range, *source_manager,
                                          invocation_->getLangOpts());
    };
    llvm::SmallVector<clang::CharSourceRange, 2> ranges;
    for (clang::CharSourceRange range : info.getRanges()) {
      ranges.push_back(as_char_range(range));
    }
    llvm::SmallVector<CppDiagnosticListener::FixIt, 0> fix_its;
    for (const clang::FixItHint& hint : info.getFixItHints()) {
      // A hint copying code from elsewhere carries it in `InsertFromRange`,
      // which nothing here reads; presented as the removal its empty
      // `CodeToInsert` looks like, it would instruct the opposite edit.
      if (hint.isNull() || hint.InsertFromRange.isValid()) {
        continue;
      }
      // Widening fails for a token range ending inside a macro; Clang's own
      // rendering drops such a fix-it too.
      clang::CharSourceRange range = as_char_range(hint.RemoveRange);
      if (range.isInvalid()) {
        continue;
      }
      fix_its.push_back({.range = range, .text = hint.CodeToInsert});
    }

    // A Clang location names a token, and marking the whole of it says more
    // than marking the column it starts in. Not for a location inside a macro:
    // the raw lexer measures the token at the expansion site while the range
    // renders in spelling coordinates, and a wrong extent is worse than a
    // point.
    clang::SourceLocation begin = info.getLocation();
    clang::CharSourceRange location =
        clang::CharSourceRange::getCharRange(begin, begin);
    if (source_manager && begin.isValid() && !begin.isMacroID()) {
      unsigned length = clang::Lexer::MeasureTokenLength(
          begin, *source_manager, invocation_->getLangOpts());
      location = clang::CharSourceRange::getCharRange(
          begin, begin.getLocWithOffset(length));
    }

    // Clang draws a range holding the caret as part of the caret's own mark
    // rather than beside it -- the `~~~~` of a `^~~~~` -- so such a range is
    // what the message is about rather than something else it points at. The
    // rest stay marks of their own.
    if (source_manager && begin.isValid()) {
      auto* found = llvm::find_if(ranges, [&](clang::CharSourceRange range) {
        return Contains(*source_manager, range, begin);
      });
      if (found != ranges.end()) {
        location = *found;
        ranges.erase(found);
      }
    }

    if (!source_manager) {
      // Without a source manager this is an error from early in the frontend,
      // and there is nothing to mark.
      CARBON_CHECK(info.getLocation().isInvalid());
      ranges.clear();
      fix_its.clear();
    }

    // A note that says one thing about the declaration it names and another
    // about the source it marks becomes a label on each, rather than one
    // sentence hung off the name. Which notes are written that way is stated
    // by `SplitsIntoCandidateAndReason` rather than guessed at from the text.
    llvm::StringRef head = message.str();
    llvm::StringRef reason;
    clang::CharSourceRange reason_range;
    if (source_manager && SplitsIntoCandidateAndReason(info.getID()) &&
        !ranges.empty()) {
      if (ranges.front().isInvalid()) {
        ranges.front() = FindEmptyParameterList(
            *source_manager, invocation_->getLangOpts(), begin);
      }
      auto [before, after] = head.split(": ");
      if (ranges.front().isValid() && !after.empty()) {
        head = before;
        reason = after;
        reason_range = ranges.front();
        ranges.erase(ranges.begin());
      }
    }
    llvm::erase_if(
        ranges, [](clang::CharSourceRange range) { return range.isInvalid(); });

    diagnostic_infos_.push_back({.level = diag_level,
                                 .location = location,
                                 .source_manager = source_manager,
                                 .message = head.str(),
                                 .reason = reason.str(),
                                 .reason_range = reason_range,
                                 .ranges = std::move(ranges),
                                 .fix_its = std::move(fix_its)});
  }

 private:
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
