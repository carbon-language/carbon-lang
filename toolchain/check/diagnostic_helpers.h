// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_
#define CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_

#include "llvm/ADT/APSInt.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Tracks a location for diagnostic use, which is either a parse node or an inst
// ID which can be translated to a parse node. Used when code needs to support
// multiple possible ways of reporting a diagnostic location.
class SemIRLoc {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLoc(SemIR::InstId inst_id)
      : inst_id_(inst_id), is_inst_id_(true), token_only_(false) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLoc(Parse::NodeId node_id) : SemIRLoc(node_id, false) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLoc(SemIR::LocId loc_id) : SemIRLoc(loc_id, false) {}

  // If `token_only` is true, refers to the specific node; otherwise, refers to
  // the node and its children.
  explicit SemIRLoc(SemIR::LocId loc_id, bool token_only)
      : loc_id_(loc_id), is_inst_id_(false), token_only_(token_only) {}

 private:
  // Only allow member access for diagnostics.
  friend class SemIRLocDiagnosticEmitter;
  // And also for eval to unwrap a LocId for calling into the rest of Check.
  friend class UnwrapSemIRLoc;

  union {
    SemIR::InstId inst_id_;
    SemIR::LocId loc_id_;
  };

  bool is_inst_id_;
  bool token_only_;
};

using DiagnosticBuilder = Diagnostics::Emitter<SemIRLoc>::DiagnosticBuilder;

// A function that forms a diagnostic for some kind of problem. The
// DiagnosticBuilder is returned rather than emitted so that the caller
// can add contextual notes as appropriate.
using MakeDiagnosticBuilderFn = llvm::function_ref<auto()->DiagnosticBuilder>;

inline auto TokenOnly(SemIR::LocId loc_id) -> SemIRLoc {
  return SemIRLoc(loc_id, true);
}

// An expression whose type should be rendered in a diagnostic. The diagnostic
// rendering will include enclosing "`"s, and may also include extra information
// about the type if it might otherwise be ambiguous or context-dependent, such
// as the targets of aliases used in the type.
//
// TODO: Include such additional information where relevant. For example:
// "`StdString` (aka `Cpp.std.basic_string(Char)`)".
//
// This should be used instead of `TypeId` as a diagnostic argument wherever
// possible, because we should eventually be able to produce a sugared type name
// in this case, whereas a `TypeId` will render as a canonical type.
struct TypeOfInstId {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  TypeOfInstId(SemIR::InstId inst_id) : inst_id(inst_id) {}

  SemIR::InstId inst_id;
};

// A type expression, for rendering in a diagnostic. The diagnostic rendering
// will include enclosing "`"s, and may also include extra information about the
// type if it would otherwise be ambiguous.
//
// TODO: Include such additional information where relevant.
//
// This should be used when the source expression used to construct a type is
// available.
struct InstIdAsType {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  InstIdAsType(SemIR::InstId inst_id) : inst_id(inst_id) {}

  SemIR::InstId inst_id;
};

// A type expression, for rendering in a diagnostic as a raw type. When
// formatting as a raw type in a diagnostic, the type will be formatted as a
// simple Carbon expression, without enclosing "`"s. Once we start including
// extra information about types, such annotations will also not be included for
// raw types.
//
// This is intended for cases where the type is part of a larger syntactic
// construct in a diagnostic, such as "redefinition of `impl {0} as {1}`".
struct InstIdAsRawType {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  InstIdAsRawType(SemIR::InstId inst_id) : inst_id(inst_id) {}

  SemIR::InstId inst_id;
};

// A type value for rendering in a diagnostic without enclosing "`"s. See
// `InstIdAsRawType` for details on raw type formatting.
//
// As with `TypeId`, this should be avoided as a diagnostic argument where
// possible, because it can't be formatted with syntactic sugar such as aliases
// that describe how the type was written.
struct TypeIdAsRawType {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  TypeIdAsRawType(SemIR::TypeId type_id) : type_id(type_id) {}

  SemIR::TypeId type_id;
};

// An integer value together with its type. The type is used to determine how to
// format the value in diagnostics.
struct TypedInt {
  using DiagnosticType = Diagnostics::TypeInfo<llvm::APSInt>;

  SemIR::TypeId type;
  llvm::APInt value;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_
