// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/diagnostics/diagnostic_kind.h"

namespace Carbon {

enum class DiagnosticLevel : int8_t {
  // A note, not indicating an error on its own, but possibly providing context
  // for an error.
  Note,
  // A warning diagnostic, indicating a likely problem with the program.
  Warning,
  // An error diagnostic, indicating that the program is not valid.
  Error,
};

// Provides a definition of a diagnostic. For example:
//   CARBON_DIAGNOSTIC(MyDiagnostic, Error, "Invalid code!");
//   CARBON_DIAGNOSTIC(MyDiagnostic, Warning, "Found {0}, expected {1}.",
//                     std::string, std::string);
//
// Arguments are passed to llvm::formatv; see:
// https://llvm.org/doxygen/FormatVariadic_8h_source.html
//
// See `DiagnosticEmitter::Emit` for comments about argument lifetimes.
#define CARBON_DIAGNOSTIC(DiagnosticName, Level, Format, ...) \
  static constexpr auto DiagnosticName =                      \
      ::Carbon::Internal::DiagnosticBase<__VA_ARGS__>(        \
          ::Carbon::DiagnosticKind::DiagnosticName,           \
          ::Carbon::DiagnosticLevel::Level, Format)

// A location for a diagnostic in a file. The lifetime of a DiagnosticLocation
// is required to be less than SourceBuffer that it refers to due to the
// contained file_name and line references.
struct DiagnosticLocation {
  // Name of the file or buffer that this diagnostic refers to.
  llvm::StringRef file_name;
  // A reference to the line of the error.
  llvm::StringRef line;
  // 1-based line number.
  int32_t line_number = -1;
  // 1-based column number.
  int32_t column_number = -1;
  // A location can represent a range of text if set to >1 value.
  int32_t length = 1;
};

// A message composing a diagnostic. This may be the main message, but can also
// be notes providing more information.
struct DiagnosticMessage {
  explicit DiagnosticMessage(
      DiagnosticKind kind, DiagnosticLocation location,
      llvm::StringLiteral format, llvm::SmallVector<llvm::Any> format_args,
      std::function<std::string(const DiagnosticMessage&)> format_fn)
      : kind(kind),
        location(location),
        format(format),
        format_args(std::move(format_args)),
        format_fn(std::move(format_fn)) {}

  // The diagnostic's kind.
  DiagnosticKind kind;

  // The calculated location of the diagnostic.
  DiagnosticLocation location;

  // The diagnostic's format string. This, along with format_args, will be
  // passed to format_fn.
  llvm::StringLiteral format;

  // A list of format arguments.
  //
  // These may be used by non-standard consumers to inspect diagnostic details
  // without needing to parse the formatted string; however, it should be
  // understood that diagnostic formats are subject to change and the llvm::Any
  // offers limited compile-time type safety. Integration tests are required.
  llvm::SmallVector<llvm::Any> format_args;

  // Returns the formatted string. By default, this uses llvm::formatv.
  std::function<std::string(const DiagnosticMessage&)> format_fn;
};

// An instance of a single error or warning.  Information about the diagnostic
// can be recorded into it for more complex consumers.
struct Diagnostic {
  // The diagnostic's level.
  DiagnosticLevel level;

  // The main error or warning.
  DiagnosticMessage message;

  // Notes that add context or supplemental information to the diagnostic.
  llvm::SmallVector<DiagnosticMessage> notes;
};

// Receives diagnostics as they are emitted.
class DiagnosticConsumer {
 public:
  virtual ~DiagnosticConsumer() = default;

  // Handle a diagnostic.
  //
  // This relies on moves of the Diagnostic. At present, diagnostics are
  // allocated on the stack, so their lifetime is that of HandleDiagnostic.
  // However, SortingDiagnosticConsumer needs a longer lifetime, until all
  // diagnostics have been produced. As a consequence, it needs to either copy
  // or move the Diagnostic, and right now we're moving due to the overhead of
  // notes.
  //
  // At present, there is no persistent storage of diagnostics because IDEs
  // would be fine with diagnostics being printed immediately and discarded,
  // without SortingDiagnosticConsumer. If this becomes a performance issue, we
  // may want to investigate alternative ownership models that address both IDE
  // and CLI user needs.
  virtual auto HandleDiagnostic(Diagnostic diagnostic) -> void = 0;

  // Flushes any buffered input.
  virtual auto Flush() -> void {}
};

// An interface that can translate some representation of a location into a
// diagnostic location.
//
// TODO: Revisit this once the diagnostics machinery is more complete and see
// if we can turn it into a `std::function`.
template <typename LocationT>
class DiagnosticLocationTranslator {
 public:
  virtual ~DiagnosticLocationTranslator() = default;

  virtual auto GetLocation(LocationT loc) -> DiagnosticLocation = 0;
};

namespace Internal {

// Use the DIAGNOSTIC macro to instantiate this.
// This stores static information about a diagnostic category.
template <typename... Args>
struct DiagnosticBase {
  explicit constexpr DiagnosticBase(DiagnosticKind kind, DiagnosticLevel level,
                                    llvm::StringLiteral format)
      : Kind(kind), Level(level), Format(format) {
    static_assert((... && !std::is_same_v<Args, llvm::StringRef>),
                  "Use std::string or llvm::StringLiteral for diagnostics to "
                  "avoid lifetime issues.");
  }

  // Calls formatv with the diagnostic's arguments.
  auto FormatFn(const DiagnosticMessage& message) const -> std::string {
    return FormatFnImpl(message, std::make_index_sequence<sizeof...(Args)>());
  };

  // The diagnostic's kind.
  DiagnosticKind Kind;
  // The diagnostic's level.
  DiagnosticLevel Level;
  // The diagnostic's format for llvm::formatv.
  llvm::StringLiteral Format;

 private:
  // Handles the cast of llvm::Any to Args types for formatv.
  // TODO: Custom formatting can be provided with an format_provider, but that
  // affects all formatv calls. Consider replacing formatv with a custom call
  // that allows diagnostic-specific formatting.
  template <std::size_t... N>
  inline auto FormatFnImpl(const DiagnosticMessage& message,
                           std::index_sequence<N...> /*indices*/) const
      -> std::string {
    assert(message.format_args.size() == sizeof...(Args));
    return llvm::formatv(message.format.data(),
                         llvm::any_cast<Args>(message.format_args[N])...);
  }
};

// Disable type deduction based on `args`; the type of `diagnostic_base`
// determines the diagnostic's parameter types.
template <typename Arg>
using NoTypeDeduction = std::common_type_t<Arg>;

}  // namespace Internal

template <typename LocationT, typename AnnotateFn>
class DiagnosticAnnotationScope;

// Manages the creation of reports, the testing if diagnostics are enabled, and
// the collection of reports.
//
// This class is parameterized by a location type, allowing different
// diagnostic clients to provide location information in whatever form is most
// convenient for them, such as a position within a buffer when lexing, a token
// when parsing, or a parse tree node when type-checking, and to allow unit
// tests to be decoupled from any concrete location representation.
template <typename LocationT>
class DiagnosticEmitter {
 public:
  // A builder-pattern type to provide a fluent interface for constructing
  // a more complex diagnostic. See `DiagnosticEmitter::Build` for the
  // expected usage.
  // This is nodiscard to protect against accidentally building a diagnostic
  // without emitting it.
  class [[nodiscard]] DiagnosticBuilder {
   public:
    // DiagnosticBuilder is move-only and cannot be copied.
    DiagnosticBuilder(DiagnosticBuilder&&) noexcept = default;
    auto operator=(DiagnosticBuilder&&) noexcept
        -> DiagnosticBuilder& = default;

    // Adds a note diagnostic attached to the main diagnostic being built.
    // The API mirrors the main emission API: `DiagnosticEmitter::Emit`.
    // For the expected usage see the builder API: `DiagnosticEmitter::Build`.
    template <typename... Args>
    auto Note(LocationT location,
              const Internal::DiagnosticBase<Args...>& diagnostic_base,
              Internal::NoTypeDeduction<Args>... args) -> DiagnosticBuilder& {
      CARBON_CHECK(diagnostic_base.Level == DiagnosticLevel::Note)
          << static_cast<int>(diagnostic_base.Level);
      diagnostic_.notes.push_back(MakeMessage(
          emitter_, location, diagnostic_base, {llvm::Any(args)...}));
      return *this;
    }

    // Emits the built diagnostic and its attached notes.
    // For the expected usage see the builder API: `DiagnosticEmitter::Build`.
    template <typename... Args>
    auto Emit() -> void {
      for (auto* annotator : emitter_->annotators_) {
        annotator->Annotate(*this);
      }
      emitter_->consumer_->HandleDiagnostic(std::move(diagnostic_));
    }

   private:
    friend class DiagnosticEmitter<LocationT>;

    template <typename... Args>
    explicit DiagnosticBuilder(
        DiagnosticEmitter<LocationT>* emitter, LocationT location,
        const Internal::DiagnosticBase<Args...>& diagnostic_base,
        llvm::SmallVector<llvm::Any> args)
        : emitter_(emitter),
          diagnostic_(
              {.level = diagnostic_base.Level,
               .message = MakeMessage(emitter, location, diagnostic_base,
                                      std::move(args))}) {
      CARBON_CHECK(diagnostic_base.Level != DiagnosticLevel::Note);
    }

    template <typename... Args>
    static auto MakeMessage(
        DiagnosticEmitter<LocationT>* emitter, LocationT location,
        const Internal::DiagnosticBase<Args...>& diagnostic_base,
        llvm::SmallVector<llvm::Any> args) -> DiagnosticMessage {
      return DiagnosticMessage(
          diagnostic_base.Kind, emitter->translator_->GetLocation(location),
          diagnostic_base.Format, std::move(args),
          [&diagnostic_base](const DiagnosticMessage& message) -> std::string {
            return diagnostic_base.FormatFn(message);
          });
    }

    DiagnosticEmitter<LocationT>* emitter_;
    Diagnostic diagnostic_;
  };

  // The `translator` and `consumer` are required to outlive the diagnostic
  // emitter.
  explicit DiagnosticEmitter(
      DiagnosticLocationTranslator<LocationT>& translator,
      DiagnosticConsumer& consumer)
      : translator_(&translator), consumer_(&consumer) {}
  ~DiagnosticEmitter() = default;

  // Emits an error.
  //
  // When passing arguments, they may be buffered. As a consequence, lifetimes
  // may outlive the `Emit` call.
  template <typename... Args>
  auto Emit(LocationT location,
            const Internal::DiagnosticBase<Args...>& diagnostic_base,
            Internal::NoTypeDeduction<Args>... args) -> void {
    DiagnosticBuilder(this, location, diagnostic_base, {llvm::Any(args)...})
        .Emit();
  }

  // A fluent interface for building a diagnostic and attaching notes for added
  // context or information. For example:
  //
  //   emitter_.Build(location1, MyDiagnostic)
  //     .Note(location2, MyDiagnosticNote)
  //     .Emit();
  template <typename... Args>
  auto Build(LocationT location,
             const Internal::DiagnosticBase<Args...>& diagnostic_base,
             Internal::NoTypeDeduction<Args>... args) -> DiagnosticBuilder {
    return DiagnosticBuilder(this, location, diagnostic_base,
                             {llvm::Any(args)...});
  }

 private:
  // Base class for scopes in which we perform diagnostic annotation, such as
  // adding notes with contextual information.
  class DiagnosticAnnotationScopeBase {
   public:
    virtual auto Annotate(DiagnosticBuilder& builder) -> void = 0;

    DiagnosticAnnotationScopeBase(const DiagnosticAnnotationScopeBase&) =
        delete;
    auto operator=(const DiagnosticAnnotationScopeBase&)
        -> DiagnosticAnnotationScopeBase& = delete;

   protected:
    explicit DiagnosticAnnotationScopeBase(DiagnosticEmitter* emitter)
        : emitter_(emitter) {
      emitter_->annotators_.push_back(this);
    }
    ~DiagnosticAnnotationScopeBase() {
      CARBON_CHECK(emitter_->annotators_.back() == this);
      emitter_->annotators_.pop_back();
    }

   private:
    DiagnosticEmitter* emitter_;
  };

  template <typename LocT, typename AnnotateFn>
  friend class DiagnosticAnnotationScope;

  DiagnosticLocationTranslator<LocationT>* translator_;
  DiagnosticConsumer* consumer_;
  llvm::SmallVector<DiagnosticAnnotationScopeBase*> annotators_;
};

class StreamDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit StreamDiagnosticConsumer(llvm::raw_ostream& stream)
      : stream_(&stream) {}

  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    std::string prefix;
    if (diagnostic.level == DiagnosticLevel::Error) {
      prefix = "ERROR: ";
    }
    Print(diagnostic.message, prefix);
    for (const auto& note : diagnostic.notes) {
      Print(note);
    }
  }
  auto Print(const DiagnosticMessage& message, llvm::StringRef prefix = "")
      -> void {
    *stream_ << message.location.file_name;
    if (message.location.line_number > 0) {
      *stream_ << ":" << message.location.line_number;
      if (message.location.column_number > 0) {
        *stream_ << ":" << message.location.column_number;
      }
    }
    *stream_ << ": " << prefix << message.format_fn(message) << "\n";
    if (message.location.column_number > 0) {
      *stream_ << message.location.line << "\n";
      stream_->indent(message.location.column_number - 1);
      *stream_ << "^";
      int underline_length = std::max(0, message.location.length - 1);
      // We want to ensure that we don't underline past the end of the line in
      // case of a multiline token.
      // TODO: revisit this once we can reference multiple ranges on multiple
      // lines in a single diagnostic message.
      underline_length = std::min(
          underline_length, static_cast<int32_t>(message.location.line.size()) -
                                message.location.column_number);
      for (int i = 0; i < underline_length; ++i) {
        *stream_ << "~";
      }
      *stream_ << "\n";
    }
  }

 private:
  llvm::raw_ostream* stream_;
};

inline auto ConsoleDiagnosticConsumer() -> DiagnosticConsumer& {
  static auto* consumer = new StreamDiagnosticConsumer(llvm::errs());
  return *consumer;
}

// Diagnostic consumer adaptor that tracks whether any errors have been
// produced.
class ErrorTrackingDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit ErrorTrackingDiagnosticConsumer(DiagnosticConsumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    seen_error_ |= diagnostic.level == DiagnosticLevel::Error;
    next_consumer_->HandleDiagnostic(std::move(diagnostic));
  }

  // Reset whether we've seen an error.
  auto Reset() -> void { seen_error_ = false; }

  // Returns whether we've seen an error since the last reset.
  auto seen_error() const -> bool { return seen_error_; }

 private:
  DiagnosticConsumer* next_consumer_;
  bool seen_error_ = false;
};

// An RAII object that denotes a scope in which any diagnostic produced should
// be annotated in some way.
//
// This object is given a function `annotate` that will be called with a
// `DiagnosticBuilder& builder` for any diagnostic that is emitted through the
// given emitter. That function can annotate the diagnostic by calling
// `builder.Note` to add notes.
template <typename LocationT, typename AnnotateFn>
class DiagnosticAnnotationScope
    : private DiagnosticEmitter<LocationT>::DiagnosticAnnotationScopeBase {
  using Base =
      typename DiagnosticEmitter<LocationT>::DiagnosticAnnotationScopeBase;

 public:
  DiagnosticAnnotationScope(DiagnosticEmitter<LocationT>* emitter,
                            AnnotateFn annotate)
      : Base(emitter), annotate_(annotate) {}

 private:
  auto Annotate(
      typename DiagnosticEmitter<LocationT>::DiagnosticBuilder& builder)
      -> void override {
    annotate_(builder);
  }

  AnnotateFn annotate_;
};

template <typename LocationT, typename AnnotateFn>
DiagnosticAnnotationScope(DiagnosticEmitter<LocationT>* emitter,
                          AnnotateFn annotate)
    -> DiagnosticAnnotationScope<LocationT, AnnotateFn>;

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
