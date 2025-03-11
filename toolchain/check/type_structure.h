// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
#define CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_

#include <compare>

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"

namespace Carbon::Check {

// The "type structure" for an impl declaration.
//
// See
// https://docs.carbon-lang.dev/docs/design/generics/overview.html#parameterized-impl-declarations.
//
// Type structures are ordered, and a type structure that is ordered higher is a
// better, more specific, match.
class TypeStructure {
 public:
  // A TypeStructure that has no witness, and is the worst possible match.
  static const TypeStructure None;

  // An instruction with an ImplWitness value for the impl declaration this
  // `TypeStructure` represents.
  auto witness_id() const -> SemIR::InstId { return witness_id_; }

  // Ordering of type structures. A higher value is a better match.
  friend auto operator<=>(TypeStructure lhs, TypeStructure rhs)
      -> std::weak_ordering {
    // A `None` for the witness is the worst possible match. In that case there
    // won't be an interface either.
    if (!lhs.witness_id_.has_value() || !rhs.witness_id_.has_value()) {
      if (lhs.witness_id_.has_value()) {
        return std::weak_ordering::greater;
      } else if (rhs.witness_id_.has_value()) {
        return std::weak_ordering::less;
      } else {
        return std::weak_ordering::equivalent;
      }
    }

    CARBON_CHECK(
        lhs.interface_ == rhs.interface_,
        "Comparing type structures from two different interfaces is not valid");

    // Higher distance is a better match, and `InfiniteDistance` is treated
    // specially as the best possible match.
    if (lhs.distance_to_first_symbolic_type_ !=
        rhs.distance_to_first_symbolic_type_) {
      if (lhs.distance_to_first_symbolic_type_ == InfiniteDistance) {
        return std::weak_ordering::greater;
      } else if (rhs.distance_to_first_symbolic_type_ == InfiniteDistance) {
        return std::weak_ordering::less;
      } else {
        return lhs.distance_to_first_symbolic_type_ <=>
               rhs.distance_to_first_symbolic_type_;
      }
    }
    // Higher priority is a better match.
    return lhs.priority_ordering_ <=> rhs.priority_ordering_;
  }

 private:
  friend class TypeStructureBuilder;

  constexpr explicit TypeStructure(int distance_to_first_symbolic_type,
                                   int priority_ordering,
                                   SemIR::InstId witness_id,
                                   SemIR::InterfaceId interface)
      : distance_to_first_symbolic_type_(distance_to_first_symbolic_type),
        priority_ordering_(priority_ordering),
        witness_id_(witness_id),
        interface_(interface) {}

  static constexpr int InfiniteDistance = -1;

  // Number of concrete types traversed before finding a symbolic type.
  int distance_to_first_symbolic_type_;
  // Priority of the impl. A higher value is a better match.
  int priority_ordering_;
  // The witness ID for the impl.
  SemIR::InstId witness_id_;
  // The interface being implemented by the witness; used to verify type
  // structures are only compared for a single interface.
  SemIR::InterfaceId interface_;
};

constexpr TypeStructure TypeStructure::None =
    TypeStructure(0, 0, SemIR::InstId::None, SemIR::InterfaceId::None);

// Constructs the TypeStructure for an `impl` declaration, which represents the
// location of unknown generic types in the signature and which is ordered by
// them.
//
// The `witness_id` is witness for the impl as a convenience for now since they
// are paired together.
//
// The `priority_ordering` is used to specify a higher priority for impl decls
// in priority blocks.
auto BuildTypeStructure(Context& context, const SemIR::Impl& impl,
                        SemIR::InstId witness_id, int priority_ordering = 0)
    -> TypeStructure;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
