// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_

#include <cstdint>

#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_DEFINE_RAW_ENUM_CLASS(DiagnosticKind, uint16_t) {
#define CARBON_DIAGNOSTIC_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/diagnostics/diagnostic_kind.def"
};

// An enumeration of all diagnostics provided by the toolchain. Diagnostics must
// be added to diagnostic_registry.def, and defined locally to where they're
// used using the `DIAGNOSTIC` macro in diagnostic_emitter.h.
//
// Diagnostic definitions are decentralized because placing all diagnostic
// definitions centrally is expected to create a compilation bottleneck
// long-term, and we also see value to keeping diagnostic format strings close
// to the consuming code.
class DiagnosticKind : public CARBON_ENUM_BASE(DiagnosticKind) {
 public:
#define CARBON_DIAGNOSTIC_KIND(Name) CARBON_ENUM_CONSTANT_DECLARATION(Name)
#include "toolchain/diagnostics/diagnostic_kind.def"
};

#define CARBON_DIAGNOSTIC_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(DiagnosticKind, Name)
#include "toolchain/diagnostics/diagnostic_kind.def"

// We expect DiagnosticKind to fit into 2 bits.
static_assert(sizeof(DiagnosticKind) == 2, "DiagnosticKind includes padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_KIND_H_
