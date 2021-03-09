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
  llvm::StringRef short_name;
  std::string message;
};

// Manages the creation of reports, the testing if diagnostics are enabled, and
// the collection of reports.
class DiagnosticEmitter {
 public:
  using Callback = std::function<void(const Diagnostic&)>;

  explicit DiagnosticEmitter(Callback callback)
      : callback_(std::move(callback)) {}
  ~DiagnosticEmitter() = default;

  // Emits an error unconditionally.
  template <typename DiagnosticT>
  auto EmitError(DiagnosticT diag) -> void {
    callback_({.short_name = DiagnosticT::ShortName, .message = diag.Format()});
  }

  // Emits a stateless error unconditionally.
  template <typename DiagnosticT>
  auto EmitError() -> std::enable_if_t<std::is_empty_v<DiagnosticT>> {
    EmitError<DiagnosticT>({});
  }

  // Emits a warning if `F` returns true.  `F` may or may not be called if the
  // warning is disabled.
  template <typename DiagnosticT>
  auto EmitWarningIf(llvm::function_ref<bool(DiagnosticT&)> f) -> void {
    // TODO(kfm): check if this warning is enabled
    DiagnosticT diag;
    if (f(diag)) {
      callback_(
          {.short_name = DiagnosticT::ShortName, .message = diag.Format()});
    }
  }

 private:
  Callback callback_;
};

inline auto ConsoleDiagnosticEmitter() -> DiagnosticEmitter& {
  static auto* emitter = new DiagnosticEmitter(
      [](const Diagnostic& d) { llvm::errs() << d.message << "\n"; });
  return *emitter;
}

inline auto NullDiagnosticEmitter() -> DiagnosticEmitter& {
  static auto* emitter = new DiagnosticEmitter([](const Diagnostic&) {});
  return *emitter;
}

// CRTP base class for diagnostics with no substitutions.
template <typename Derived>
struct SimpleDiagnostic {
  struct Substitutions {};
  static auto Format() -> std::string { return Derived::Message.str(); }
};

}  // namespace Carbon

#endif  // DIAGNOSTICS_DIAGNOSTICEMITTER_H_
