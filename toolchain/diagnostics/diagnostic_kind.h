// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_
#define TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

// An enumeration of all diagnostics provided by the toolchain. Diagnostics must
// be added to diagnostic_registry.def, and defined locally to where they're
// used using the `DIAGNOSTIC` macro in diagnostic_emitter.h.
//
// Diagnostic definitions are decentralized because placing all diagnostic
// definitions centrally is expected to create a compilation bottleneck
// long-term, and we also see value to keeping diagnostic format strings close
// to the consuming code.
class DiagnosticKind {
 private:
  // Note that this must be declared earlier in the class so that its type can
  // be used, for example in the conversion operator.
  enum class KindEnum : int32_t {
#define DIAGNOSTIC_KIND(DiagnosticName) DiagnosticName,
#include "toolchain/diagnostics/diagnostic_registry.def"
  };

 public:
  // The formatting for this macro is weird due to a `clang-format` bug. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define DIAGNOSTIC_KIND(DiagnosticName)                    \
  static constexpr auto DiagnosticName()->DiagnosticKind { \
    return DiagnosticKind(KindEnum::DiagnosticName);       \
  }
#include "toolchain/diagnostics/diagnostic_registry.def"

  // The default constructor is deleted as objects of this type should always be
  // constructed using the above factory functions for each unique kind.
  DiagnosticKind() = delete;

  friend auto operator==(DiagnosticKind lhs, DiagnosticKind rhs) -> bool {
    return lhs.kind_value_ == rhs.kind_value_;
  }
  friend auto operator!=(DiagnosticKind lhs, DiagnosticKind rhs) -> bool {
    return lhs.kind_value_ != rhs.kind_value_;
  }

  // Prints the DiagnosticKind, typically for diagnostics.
  void Print(llvm::raw_ostream& out) const { out << name(); }

  // Get a friendly name for the token for logging or debugging.
  auto name() const -> llvm::StringRef;

 private:
  constexpr explicit DiagnosticKind(KindEnum kind_value)
      : kind_value_(kind_value) {}

  KindEnum kind_value_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_
