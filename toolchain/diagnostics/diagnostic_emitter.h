// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_DIAGNOSTICS_DIAGNOSTICEMITTER_H_
#define TOOLCHAIN_DIAGNOSTICS_DIAGNOSTICEMITTER_H_

#include <functional>
#include <string>
#include <type_traits>

#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// An instance of a single error or warning.  Information about the diagnostic
// can be recorded into it for more complex consumers.
//
// TODO: turn this into a much more reasonable API when we add some actual
// uses of it.
struct Diagnostic {
  enum Level {
    // A warning diagnostic, indicating a likely problem with the program.
    Warning,
    // An error diagnostic, indicating that the program is not valid.
    Error,
  };

  struct Location {
    // Name of the file or buffer that this diagnostic refers to.
    std::string file_name;
    // 1-based line number.
    int32_t line_number;
    // 1-based column number.
    int32_t column_number;
  };

  Level level;
  Location location;
  llvm::StringRef short_name;
  std::string message;
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
      -> Diagnostic::Location = 0;
};

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

  // Emits an error unconditionally.
  template <typename DiagnosticT>
  auto EmitError(LocationT location, DiagnosticT diag) -> void {
    // TODO: Encode the diagnostic kind in the Diagnostic object rather than
    // hardcoding an "error: " prefix.
    consumer_->HandleDiagnostic({.level = Diagnostic::Error,
                                 .location = translator_->GetLocation(location),
                                 .short_name = DiagnosticT::ShortName,
                                 .message = diag.Format()});
  }

  // Emits a stateless error unconditionally.
  template <typename DiagnosticT>
  auto EmitError(LocationT location)
      -> std::enable_if_t<std::is_empty_v<DiagnosticT>> {
    EmitError<DiagnosticT>(location, {});
  }

  // Emits a warning if `F` returns true.  `F` may or may not be called if the
  // warning is disabled.
  template <typename DiagnosticT>
  auto EmitWarningIf(LocationT location,
                     llvm::function_ref<bool(DiagnosticT&)> f) -> void {
    // TODO(kfm): check if this warning is enabled at `location`.
    DiagnosticT diag;
    if (f(diag)) {
      // TODO: Encode the diagnostic kind in the Diagnostic object rather than
      // hardcoding a "warning: " prefix.
      consumer_->HandleDiagnostic(
          {.level = Diagnostic::Warning,
           .location = translator_->GetLocation(location),
           .short_name = DiagnosticT::ShortName,
           .message = diag.Format()});
    }
  }

 private:
  DiagnosticLocationTranslator<LocationT>* translator_;
  DiagnosticConsumer* consumer_;
};

inline auto ConsoleDiagnosticConsumer() -> DiagnosticConsumer& {
  struct Consumer : DiagnosticConsumer {
    auto HandleDiagnostic(const Diagnostic& d) -> void override {
      if (!d.location.file_name.empty()) {
        llvm::errs() << d.location.file_name << ":" << d.location.line_number
                     << ":" << d.location.column_number << ": ";
      }

      llvm::errs() << d.message << "\n";
    }
  };
  static auto* consumer = new Consumer;
  return *consumer;
}

// CRTP base class for diagnostics with no substitutions.
template <typename Derived>
struct SimpleDiagnostic {
  static auto Format() -> std::string { return Derived::Message.str(); }
};

// Diagnostic consumer adaptor that tracks whether any errors have been
// produced.
class ErrorTrackingDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit ErrorTrackingDiagnosticConsumer(DiagnosticConsumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  auto HandleDiagnostic(const Diagnostic& diagnostic) -> void override {
    seen_error_ |= diagnostic.level == Diagnostic::Error;
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

#endif  // TOOLCHAIN_DIAGNOSTICS_DIAGNOSTICEMITTER_H_
