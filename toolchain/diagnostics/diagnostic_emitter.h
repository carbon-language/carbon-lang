// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
#define TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_

#include <any>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// An enumeration of all diagnostics provided by the toolchain. Diagnostics must
// be added to diagnostic_registry.def, and defined locally to where they're
// used using the `DIAGNOSTIC` macro.
//
// Diagnostic definitions are decentralized because placing all diagnostic
// definitions centrally is expected to create a compilation bottleneck
// long-term, and we also see value to keeping diagnostic format strings close
// to the consuming code.
enum class DiagnosticKind {
#define DIAGNOSTIC_KIND(DiagnosticName) DiagnosticName,
#include "toolchain/diagnostics/diagnostic_registry.def"
};

enum class DiagnosticLevel : int8_t {
  // A warning diagnostic, indicating a likely problem with the program.
  Warning,
  // An error diagnostic, indicating that the program is not valid.
  Error,
};

// Provides a definition of a diagnostic. For example:
//   DIAGNOSTIC(MyDiagnostic, Error, "Invalid code!");
//   DIAGNOSTIC(MyDiagnostic, Warning, "Found {0}, expected {1}.",
//              llvm::StringRef, llvm::StringRef);
//
// See `DiagnosticEmitter::Emit` for comments about argument lifetimes.
#define DIAGNOSTIC(DiagnosticName, Level, Format, ...)                      \
  static constexpr auto DiagnosticName =                                    \
      Internal::DiagnosticBase<__VA_ARGS__>(DiagnosticKind::DiagnosticName, \
                                            DiagnosticLevel::Level, Format);

// Provides a definition of a diagnostic with a custom formatter. The custom
// format function is called with the format string and all type arguments.
//
// For example:
//   DIAGNOSTIC_WITH_FORMAT_FN(
//       MyDiagnostic, Error, "Number is {0}.",
//       [](llvm::StringLiteral format, int radix) {
//         return llvm::formatv(format,
//                              radix == 16 ? "hexadecimal" : "decimal");
//       },
//       int);
#define DIAGNOSTIC_WITH_FORMAT_FN(DiagnosticName, Level, Format, FormatFn,  \
                                  ...)                                      \
  static constexpr auto DiagnosticName =                                    \
      Internal::DiagnosticBase<__VA_ARGS__>(DiagnosticKind::DiagnosticName, \
                                            DiagnosticLevel::Level, Format, \
                                            FormatFn);

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
  const DiagnosticKind kind;

  // The diagnostic's level.
  const DiagnosticLevel level;

  // The calculated location of the diagnostic.
  const DiagnosticLocation location;

  // A std::tuple containing the diagnostic's format plus any format arguments.
  // These will be passed to llvm::formatv.
  const std::any format_args;

  // Returns the formatted string.
  const std::function<std::string(const Diagnostic&)> format_fn;
};

// Receives diagnostics as they are emitted.
class DiagnosticConsumer {
 public:
  virtual ~DiagnosticConsumer() = default;

  // Handle a diagnostic.
  virtual auto HandleDiagnostic(const Diagnostic& diagnostic) -> void = 0;
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
  // This is the underlying format function type. It's wrapped for Diagnostic in
  // order to hide Args type information.
  using RawFormatFnType = std::string (*)(llvm::StringLiteral format,
                                          const Args&... args);

  constexpr DiagnosticBase(DiagnosticKind kind, DiagnosticLevel level,
                           llvm::StringLiteral format,
                           RawFormatFnType raw_format_fn = &DefaultRawFormatFn)
      : Kind(kind), Level(level), Format(format), RawFormatFn(raw_format_fn) {}

  // Calls raw_format_fn with the diagnostic's arguments.
  auto FormatFn(const Diagnostic& diagnostic) const -> std::string {
    return std::apply(RawFormatFn,
                      std::any_cast<std::tuple<llvm::StringLiteral, Args...>>(
                          diagnostic.format_args));
  };

  // The diagnostic's kind.
  const DiagnosticKind Kind;
  // The diagnostic's level.
  const DiagnosticLevel Level;
  // The diagnostic's format for llvm::formatv.
  const llvm::StringLiteral Format;

 private:
  // A generic format function, used when format_fn isn't provided.
  static auto DefaultRawFormatFn(llvm::StringLiteral format,
                                 const Args&... args) -> std::string {
    return llvm::formatv(format.data(), args...);
  }

  // The function to use for formatting.
  const RawFormatFnType RawFormatFn;
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
  void Emit(
      LocationT location,
      const Internal::DiagnosticBase<Args...>& diagnostic_base,
      // Disable type deduction based on `args`; `diagnostic` still applies.
      typename std::common_type_t<Args>... args) {
    consumer_->HandleDiagnostic({
        .kind = diagnostic_base.Kind,
        .level = diagnostic_base.Level,
        .location = translator_->GetLocation(location),
        .format_args = std::make_tuple(diagnostic_base.Format, args...),
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

  // Returns whether we've seen an error since the last reset.
  auto SeenError() const -> bool { return seen_error_; }

  // Reset whether we've seen an error.
  auto Reset() -> void { seen_error_ = false; }

 private:
  DiagnosticConsumer* next_consumer_;
  bool seen_error_ = false;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_EMITTER_H_
