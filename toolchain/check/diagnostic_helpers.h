// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_
#define CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_

#include "llvm/ADT/APSInt.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree_node_diagnostic_converter.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Diagnostic locations produced by checking may be either a parse node
// directly, or an inst ID which is later translated to a parse node.
struct SemIRLocation {
  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLocation(SemIR::InstId inst_id)
      : inst_id(inst_id), is_inst_id(true), token_only(false) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLocation(Parse::NodeId node_id) : SemIRLocation(node_id, false) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLocation(SemIR::LocationId loc_id) : SemIRLocation(loc_id, false) {}

  explicit SemIRLocation(SemIR::LocationId loc_id, bool token_only)
      : loc_id(loc_id), is_inst_id(false), token_only(token_only) {}

  union {
    SemIR::InstId inst_id;
    SemIR::LocationId loc_id;
  };

  bool is_inst_id;
  bool token_only;
};

inline auto TokenOnly(SemIR::LocationId loc_id) -> SemIRLocation {
  return SemIRLocation(loc_id, true);
}

// An integer value together with its type. The type is used to determine how to
// format the value in diagnostics.
struct TypedInt {
  using DiagnosticType = DiagnosticTypeInfo<llvm::APSInt>;

  SemIR::TypeId type;
  llvm::APInt value;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_
