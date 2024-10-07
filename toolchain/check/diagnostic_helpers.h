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
struct SemIRLoc {
  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLoc(SemIR::InstId inst_id)
      : inst_id(inst_id), is_inst_id(true), token_only(false) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLoc(Parse::NodeId node_id) : SemIRLoc(node_id, false) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLoc(SemIR::LocId loc_id) : SemIRLoc(loc_id, false) {}

  explicit SemIRLoc(SemIR::LocId loc_id, bool token_only)
      : loc_id(loc_id), is_inst_id(false), token_only(token_only) {}

  union {
    SemIR::InstId inst_id;
    SemIR::LocId loc_id;
  };

  bool is_inst_id;
  bool token_only;
};

inline auto TokenOnly(SemIR::LocId loc_id) -> SemIRLoc {
  return SemIRLoc(loc_id, true);
}

// An integer value together with its type. The type is used to determine how to
// format the value in diagnostics.
struct TypedInt {
  using DiagnosticType = DiagnosticTypeInfo<llvm::APSInt>;

  SemIR::TypeId type;
  llvm::APInt value;
};

// A type value for rendering in a diagnostic without enclosing "`"s. See
// `InstIdAsRawType` for details.
struct TypeIdAsRawType {
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  TypeIdAsRawType(SemIR::TypeId type_id) : type_id(type_id) {}

  SemIR::TypeId type_id;
};

// A type expression, for rendering in a diagnostic. The diagnostic rendering
// will include enclosing "`"s, and may also include extra information about the
// type if it would otherwise be ambiguous.
// TODO: Include such additional information where relevant.
struct InstIdAsType {
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  InstIdAsType(SemIR::InstId inst_id) : inst_id(inst_id) {}

  SemIR::InstId inst_id;
};

// A type expression, for rendering in a diagnostic. Like `InstIdAsType`, but
// the type will not be enclosed in "`"s. Once we start including "aka" types
// for sugared types, such annotations will also not be included for raw types.
// This is intended for cases where the type is part of a larger syntactic
// construct in a diagnostic, such as "redefinition of `impl {0} as {1}`".
struct InstIdAsRawType {
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  InstIdAsRawType(SemIR::InstId inst_id) : inst_id(inst_id) {}

  SemIR::InstId inst_id;
};

// An expression whose type should be rendered in a diagnostic, including
// enclosing "`"s. This should be used instead of `TypeId` as a diagnostic
// argument wherever possible, because we should eventually be able to produce a
// sugared type name in this case, whereas a `TypeId` will render as a canonical
// type.
struct InstIdAsTypeOfExpr {
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  InstIdAsTypeOfExpr(SemIR::InstId inst_id) : inst_id(inst_id) {}

  SemIR::InstId inst_id;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_
