// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
#define TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_

#include <functional>
#include <string>
#include <type_traits>

#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/diagnostics/diagnostic_kind.h"

namespace Carbon {

enum class DiagnosticLevel : int8_t {
  // A warning diagnostic, indicating a likely problem with the program.
  Warning,
  // An error diagnostic, indicating that the program is not valid.
  Error,
};

// Provides a definition of a diagnostic. For example:
//   CARBON_DIAGNOSTIC(MyDiagnostic, Error, "Invalid code!");
//   CARBON_DIAGNOSTIC(MyDiagnostic, Warning, "Found {0}, expected {1}.",
//              llvm::StringRef, llvm::StringRef);
//
// Arguments are passed to llvm::formatv; see:
// https://llvm.org/doxygen/FormatVariadic_8h_source.html
//
// See `DiagnosticEmitter::Emit` for comments about argument lifetimes.
#define CARBON_DIAGNOSTIC(DiagnosticName, Level, Format, ...) \
  static constexpr auto DiagnosticName =                      \
      Internal::DiagnosticBase<__VA_ARGS__>(                  \
          ::Carbon::DiagnosticKind::DiagnosticName,           \
          ::Carbon::DiagnosticLevel::Level, Format)

struct DiagnosticLocation {
  // Name of the file or buffer that this diagnostic refers to.
  // TODO: Move this out of DiagnosticLocation, as part of an expectation that
  // files will be compiled separately, so storing the file's path
  // per-diagnostic is wasteful.
  std::string file_name;
  // 1-based line number.
  int32_t line_number;
  // 1-based column number.
  int32_t column_number;
};

// An instance of a single error or warning.  Information about the diagnostic
// can be recorded into it for more complex consumers.
struct Diagnostic {
  // The diagnostic's kind.
  DiagnosticKind kind;

  // The diagnostic's level.
  DiagnosticLevel level;

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
  llvm::SmallVector<llvm::Any, 0> format_args;

  // Returns the formatted string. By default, this uses llvm::formatv.
  std::function<std::string(const Diagnostic&)> format_fn;
};

// Receives diagnostics as they are emitted.
class DiagnosticConsumer {
 public:
  virtual ~DiagnosticConsumer() = default;

  // Handle a diagnostic.
  virtual auto HandleDiagnostic(const Diagnostic& diagnostic) -> void = 0;

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

  [[nodiscard]] virtual auto GetLocation(LocationT loc)
      -> DiagnosticLocation = 0;
};

namespace Internal {

// Use the DIAGNOSTIC macro to instantiate this.
// This stores static information about a diagnostic category.
template <typename... Args>
struct DiagnosticBase {
  constexpr DiagnosticBase(DiagnosticKind kind, DiagnosticLevel level,
                           llvm::StringLiteral format)
      : Kind(kind), Level(level), Format(format) {}

  // Calls formatv with the diagnostic's arguments.
  auto FormatFn(const Diagnostic& diagnostic) const -> std::string {
    return FormatFnImpl(diagnostic,
                        std::make_index_sequence<sizeof...(Args)>());
  };

  // The diagnostic's kind.
  DiagnosticKind Kind;
  // The diagnostic's level.
  DiagnosticLevel Level;
  // The diagnostic's format for llvm::formatv.
  llvm::StringLiteral Format;

 private:
  // Handles the cast of llvm::Any to Args types for formatv.
  template <std::size_t... N>
  inline auto FormatFnImpl(const Diagnostic& diagnostic,
                           std::index_sequence<N...> /*indices*/) const
      -> std::string {
    assert(diagnostic.format_args.size() == sizeof...(Args));
    return llvm::formatv(diagnostic.format.data(),
                         llvm::any_cast<Args>(diagnostic.format_args[N])...);
  }
};

}  // namespace Internal

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
  void Emit(LocationT location,
            const Internal::DiagnosticBase<Args...>& diagnostic_base,
            // Disable type deduction based on `args`; the type of
            // `diagnostic_base` determines the diagnostic's parameter types.
            typename std::common_type_t<Args>... args) {
    consumer_->HandleDiagnostic({
        .kind = diagnostic_base.Kind,
        .level = diagnostic_base.Level,
        .location = translator_->GetLocation(location),
        .format = diagnostic_base.Format,
        .format_args = {std::move(args)...},
        .format_fn = [&diagnostic_base](const Diagnostic& diagnostic)
            -> std::string { return diagnostic_base.FormatFn(diagnostic); },
    });
  }

 private:
  DiagnosticLocationTranslator<LocationT>* translator_;
  DiagnosticConsumer* consumer_;
};

inline auto ConsoleDiagnosticConsumer() -> DiagnosticConsumer& {
  struct Consumer : DiagnosticConsumer {
    auto HandleDiagnostic(const Diagnostic& diagnostic) -> void override {
      llvm::errs() << diagnostic.location.file_name << ":"
                   << diagnostic.location.line_number << ":"
                   << diagnostic.location.column_number << ": "
                   << diagnostic.format_fn(diagnostic) << "\n";
    }
  };
  static auto* consumer = new Consumer;
  return *consumer;
}

// Diagnostic consumer adaptor that tracks whether any errors have been
// produced.
class ErrorTrackingDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit ErrorTrackingDiagnosticConsumer(DiagnosticConsumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  auto HandleDiagnostic(const Diagnostic& diagnostic) -> void override {
    seen_error_ |= diagnostic.level == DiagnosticLevel::Error;
    next_consumer_->HandleDiagnostic(diagnostic);
  }

  // Reset whether we've seen an error.
  auto Reset() -> void { seen_error_ = false; }

  // Returns whether we've seen an error since the last reset.
  auto seen_error() const -> bool { return seen_error_; }

 private:
  DiagnosticConsumer* next_consumer_;
  bool seen_error_ = false;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
