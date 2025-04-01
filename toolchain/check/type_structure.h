// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
#define CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_

#include <compare>

#include "common/ostream.h"
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
// better, more specified, match.
class TypeStructure : public Printable<TypeStructure> {
 public:
  // Returns whether the type structure is compatible with `other`. If false,
  // they can not possibly match with one being an `impl` for the other as a
  // lookup query.
  auto IsCompatibleWith(const TypeStructure& other) const -> bool;

  // Ordering of type structures. A higher value is a better match.
  friend auto operator<=>(const TypeStructure& lhs, const TypeStructure& rhs)
      -> std::weak_ordering {
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
    // TODO: If they have a symbolic in the same position, we could use further
    // symbolic types to get an ordering.
    return std::weak_ordering::equivalent;
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "TypeStructure = ";
    for (auto s : structure_) {
      switch (s) {
        case Structural::Concrete:
          out << 'c';
          break;
        case Structural::Symbolic:
          out << '?';
          break;
        case Structural::ConcreteOpenParen:
          out << "(";
          break;
        case Structural::ConcreteCloseParen:
          out << ')';
          break;
      }
    }
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

  TypeStructure(llvm::SmallVector<Structural> structure,
                int distance_to_first_symbolic_type)
      : structure_(std::move(structure)),
        distance_to_first_symbolic_type_(distance_to_first_symbolic_type) {}

  // The structural position of concrete and symbolic values in the type.
  llvm::SmallVector<Structural> structure_;

  // Number of concrete types traversed before finding a symbolic type.
  int distance_to_first_symbolic_type_;
};

// Constructs the TypeStructure for a self type or facet value and an interface
// constraint (e.g. `Iface(A, B(C))`), which represents the location of unknown
// symbolic values in the combined signature and which is ordered by them.
//
// Given `impl C as Z {}` the `self_const_id` would be a `C` and the interface
// constraint would be `Z`.
auto BuildTypeStructure(Context& context, SemIR::InstId self_inst_id,
                        SemIR::SpecificInterface interface) -> TypeStructure;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
