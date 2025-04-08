// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
#define CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_

#include <algorithm>

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

  // Ordering of type structures. A lower value is a better match.
  // TODO: switch to operator<=> once we can depend on
  // std::lexicographical_compare_three_way (in particular, once we can
  // require clang-17 or newer, including in places like the GitHub test
  // runners).
  friend auto operator<(const TypeStructure& lhs, const TypeStructure& rhs)
      -> bool {
    return std::lexicographical_compare(
        lhs.symbolic_type_indices_.begin(), lhs.symbolic_type_indices_.end(),
        rhs.symbolic_type_indices_.begin(), rhs.symbolic_type_indices_.end(),
        [](int lhs_index, int rhs_index) {
          // A higher symbolic type index is a better match, so we need to
          // reverse the order.
          return rhs_index < lhs_index;
        });
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
                llvm::SmallVector<int> symbolic_type_indices)
      : structure_(std::move(structure)),
        symbolic_type_indices_(std::move(symbolic_type_indices)) {}

  // The structural position of concrete and symbolic values in the type.
  llvm::SmallVector<Structural> structure_;

  // Indices of the symbolic entries in structure_.
  llvm::SmallVector<int> symbolic_type_indices_;
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
