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

  // Equality of type structures. This compares that the structures are
  // identical, which is a stronger requirement than that they are ordered the
  // same.
  friend auto operator==(const TypeStructure& lhs, const TypeStructure& rhs)
      -> bool {
    return lhs.structure_ == rhs.structure_ &&
           lhs.concrete_types_ == rhs.concrete_types_;
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

  // Elements of the type structure, indicating the presence of a concrete or
  // symbolic element, and for aggregate concrete types (such as generic types),
  // nesting for the types inside.
  enum class Structural : uint8_t {
    // A concrete element in the type structure, such as `bool`.
    Concrete,

    // A concrete element in the type structure that contains nested types
    // within, such as `C(D)` for some classes C and D. It marks the start of
    // the nested and is paired with a ConcreteCloseParen at the end of the
    // nested types.
    ConcreteOpenParen,

    // Closes a ConcreteOpenParen for a concrete type with nested types.
    // Does not have its own concrete type.
    ConcreteCloseParen,

    // A symbolic element in the type structure. When matching type structures,
    // it represents a wildcard that matches against either a single `Concrete`
    // or `Symbolic`, or everything from a `ConcreteOpenParen` to its paired
    // `ConcreteCloseParen`.
    Symbolic,
  };

  // Indicates a concrete element in the type structure which does not add any
  // type information of its own. See `ConcreteType`.
  struct ConcreteNoneType {
    friend auto operator==(ConcreteNoneType /*lhs*/, ConcreteNoneType /*rhs*/)
        -> bool = default;
  };
  // The `concrete_types_` tracks the specific concrete type for each
  // `Structural::Concrete` or `Structural::ConcreteOpenParen` in the type
  // structure. But there are cases where the `ConcreteOpenParen` opens a scope
  // for other concrete types but doesn't add any type data of its own, and
  // `ConcreteNoneType` can appear there.
  using ConcreteType = std::variant<ConcreteNoneType, SemIR::TypeId,
                                    SemIR::ClassId, SemIR::InterfaceId>;

  TypeStructure(llvm::SmallVector<Structural> structure,
                llvm::SmallVector<int> symbolic_type_indices,
                llvm::SmallVector<ConcreteType> concrete_types)
      : structure_(std::move(structure)),
        symbolic_type_indices_(std::move(symbolic_type_indices)),
        concrete_types_(std::move(concrete_types)) {}

  // The structural position of concrete and symbolic values in the type.
  llvm::SmallVector<Structural> structure_;

  // Indices of the symbolic entries in structure_.
  llvm::SmallVector<int> symbolic_type_indices_;

  // The related value for each `Concrete` and `ConcreteOpenParen` entry in
  // the type `structure_`, in the same order. See `ConcreteType`.
  llvm::SmallVector<ConcreteType> concrete_types_;
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
