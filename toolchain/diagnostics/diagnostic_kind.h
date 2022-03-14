// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_
#define TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_

namespace Carbon {

// An enumeration of all diagnostics provided by the toolchain. Diagnostics must
// be added to diagnostic_registry.def, and defined locally to where they're
// used using the `DIAGNOSTIC` macro in diagnostic_emitter.h.
//
// Diagnostic definitions are decentralized because placing all diagnostic
// definitions centrally is expected to create a compilation bottleneck
// long-term, and we also see value to keeping diagnostic format strings close
// to the consuming code.
enum class DiagnosticKind {
#define DIAGNOSTIC_KIND(DiagnosticName) DiagnosticName,
#include "toolchain/diagnostics/diagnostic_registry.def"
};

}  // namespace Carbon

#endif  // TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_
