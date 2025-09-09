// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_
#define CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_

#include <concepts>

#include "llvm/ADT/APSInt.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// The `DiagnosticEmitterBase` is templated on this type so that
// diagnostics can be passed an `InstId` as a location, without having to
// explicitly construct a `LocId` from it first.
class LocIdForDiagnostics {
 public:
  // Constructs a token-only location for a diagnostic.
  //
  // This means the displayed location will include only the location's specific
  // parse node, instead of also including its descendants.
  static auto TokenOnly(Parse::NodeId node_id) -> LocIdForDiagnostics {
    return LocIdForDiagnostics(SemIR::LocId(node_id), true);
  }

  template <typename LocT>
    requires std::constructible_from<SemIR::LocId, LocT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  LocIdForDiagnostics(LocT loc_id)
      : LocIdForDiagnostics(SemIR::LocId(loc_id), false) {}

  auto loc_id() const -> SemIR::LocId { return loc_id_; }

  auto is_token_only() const -> bool { return is_token_only_; }

 private:
  explicit LocIdForDiagnostics(SemIR::LocId loc_id, bool is_token_only)
      : loc_id_(loc_id), is_token_only_(is_token_only) {}

  SemIR::LocId loc_id_;
  bool is_token_only_;
};

// We define the emitter separately for dependencies, so only provide a base
// here.
using DiagnosticEmitterBase = Diagnostics::Emitter<LocIdForDiagnostics>;

using DiagnosticBuilder = DiagnosticEmitterBase::Builder;

// A function that forms a diagnostic for some kind of problem. The
// DiagnosticBuilder is returned rather than emitted so that the caller
// can add contextual notes as appropriate.
using MakeDiagnosticBuilderFn = llvm::function_ref<auto()->DiagnosticBuilder>;

// An expression with a constant value, for rendering in a diagnostic. The
// diagnostic rendering will include enclosing "`"s.
struct InstIdAsConstant {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  InstIdAsConstant(SemIR::InstId inst_id) : inst_id(inst_id) {}

  SemIR::InstId inst_id;
};

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
//
// Note that this is currently an alias for InstIdAsConstant. However, using
// InstIdAsType is clearer when defining CARBON_DIAGNOSTICs, and we may wish to
// distinguish type arguments in diagnostics from more general constants in some
// way in the future.
using InstIdAsType = InstIdAsConstant;

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

struct SpecificInterfaceIdAsRawType {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  SpecificInterfaceIdAsRawType(SemIR::SpecificInterfaceId specific_interface_id)
      : specific_interface_id(specific_interface_id) {}

  SemIR::SpecificInterfaceId specific_interface_id;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DIAGNOSTIC_HELPERS_H_
