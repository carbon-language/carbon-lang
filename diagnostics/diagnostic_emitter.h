// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef DIAGNOSTICS_DIAGNOSTICEMITTER_H_
#define DIAGNOSTICS_DIAGNOSTICEMITTER_H_

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
  struct Location {
    // Name of the file or buffer that this diagnostic refers to.
    llvm::StringRef file_name;
    // 1-based line number.
    int32_t line_number;
    // 1-based column number.
    int32_t column_number;
  };

  Location location;
  llvm::StringRef short_name;
  std::string message;
};

// Receives diagnostics as they are emitted.
class DiagnosticConsumer {
 public:
  virtual ~DiagnosticConsumer() {}

  // Handle a diagnostic.
  virtual auto HandleDiagnostic(const Diagnostic& diagnostic) -> void = 0;
};

// An interface that can translate some representation of a location into a
// diagnostic location.
template <typename LocationT>
class DiagnosticLocationTranslator {
 public:
  virtual ~DiagnosticLocationTranslator() {}

  [[nodiscard]] virtual auto GetLocation(LocationT loc)
      -> Diagnostic::Location = 0;
};

// Manages the creation of reports, the testing if diagnostics are enabled, and
// the collection of reports.
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
    consumer_->HandleDiagnostic({.location = translator_->GetLocation(location),
                                 .short_name = DiagnosticT::ShortName,
                                 .message = "error: " + diag.Format()});
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
      consumer_->HandleDiagnostic(
          {.location = translator_->GetLocation(location),
           .short_name = DiagnosticT::ShortName,
           .message = "warning: " + diag.Format()});
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

}  // namespace Carbon

#endif  // DIAGNOSTICS_DIAGNOSTICEMITTER_H_
