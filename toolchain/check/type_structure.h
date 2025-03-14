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

  // Returns whether the type structure is compatible with `other`. If false,
  // they can not possibly match with one being an `impl` for the other as a
  // lookup query.
  auto IsCompatibleWith(const TypeStructure& other) const -> bool;

  // Ordering of type structures. A higher value is a better match.
  friend auto operator<=>(const TypeStructure& lhs, const TypeStructure& rhs)
      -> std::weak_ordering {
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

  enum class Structural : uint8_t {
    Concrete,
    ConcreteOpenParen,
    ConcreteCloseParen,
    Symbolic,
  };

  static constexpr int InfiniteDistance = -1;

  constexpr explicit TypeStructure(std::vector<Structural> structure,
                                   int distance_to_first_symbolic_type,
                                   int priority_ordering,
                                   SemIR::InterfaceId interface)
      : structure_(std::move(structure)),
        distance_to_first_symbolic_type_(distance_to_first_symbolic_type),
        priority_ordering_(priority_ordering),
        interface_(interface) {}

  // The structural position of concrete and symbolic values in the type.
  std::vector<Structural> structure_;

  // Number of concrete types traversed before finding a symbolic type.
  int distance_to_first_symbolic_type_;
  // Priority of the impl. A higher value is a better match.
  int priority_ordering_;
  // The interface being implemented by the witness; used to verify type
  // structures are only compared for a single interface.
  SemIR::InterfaceId interface_;
};

constexpr TypeStructure TypeStructure::None =
    TypeStructure({}, 0, 0, SemIR::InterfaceId::None);

// Constructs the TypeStructure for an `impl` declaration, which represents the
// location of unknown generic types in the signature and which is ordered by
// them.
//
// The `priority_ordering` is used to specify a higher priority for impl decls
// in priority blocks. See
// https://docs.carbon-lang.dev/docs/design/generics/details.html#prioritization-rule.
auto BuildTypeStructure(Context& context, const SemIR::Impl& impl,
                        int priority_ordering = 0) -> TypeStructure;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
