// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
#define CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_

#include <algorithm>

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

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
  enum class CompareTest {
    // Test whether `this` has the same structure as `other`, or `this` is
    // strictly more specific (has more concrete values) than `other` while
    // maintaining a compatible structure.
    //
    // If false, they can not possibly match with `this` being a lookup query
    // and `other` being an `impl`.
    IsEqualToOrMoreSpecificThan,

    // Tests whether there is a possible query that could match both `this` and
    // `other`, in which case we say `this` has overlap with `other`.
    HasOverlap,
  };

  // Compares the structure of `this` and `other`, and returns whether the
  // structures match according to the specified test.
  auto CompareStructure(CompareTest test, const TypeStructure& other) const
      -> bool;

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

  // Marks an array type. The type and bound will appear as other entries.
  struct ConcreteArrayType {
    friend auto operator==(ConcreteArrayType /*lhs*/, ConcreteArrayType /*rhs*/)
        -> bool = default;
  };
  // Marks a pointer type. The pointee type will appear as another entry.
  struct ConcretePointerType {
    friend auto operator==(ConcretePointerType /*lhs*/,
                           ConcretePointerType /*rhs*/) -> bool = default;
  };
  // Marks a struct type. The field names and types will appear as other
  // entries.
  struct ConcreteStructType {
    friend auto operator==(ConcreteStructType /*lhs*/,
                           ConcreteStructType /*rhs*/) -> bool = default;
  };
  // Marks a tuple type. The type members (if any) will appear as other entries.
  struct ConcreteTupleType {
    friend auto operator==(ConcreteTupleType /*lhs*/, ConcreteTupleType /*rhs*/)
        -> bool = default;
  };
  // The `concrete_types_` tracks the specific concrete type for each
  // `Structural::Concrete` or `Structural::ConcreteOpenParen` in the type
  // structure.
  //
  // `ConstantId` is used strictly for non-type values. For types, `TypeId` is
  // used.
  //
  // `NameId` is used strictly for struct fields, as the field names are part of
  // the struct type.
  using ConcreteType =
      std::variant<ConcreteArrayType, ConcretePointerType, ConcreteStructType,
                   ConcreteTupleType, SemIR::ClassId, SemIR::ConstantId,
                   SemIR::InterfaceId, SemIR::NameId, SemIR::TypeId>;

  TypeStructure(llvm::SmallVector<Structural> structure,
                llvm::SmallVector<int> symbolic_type_indices,
                llvm::SmallVector<ConcreteType> concrete_types)
      : structure_(std::move(structure)),
        symbolic_type_indices_(std::move(symbolic_type_indices)),
        concrete_types_(std::move(concrete_types)) {}

  // A helper for CompareStructure.
  static auto ConsumeRhsSymbolic(
      llvm::SmallVector<Structural>::const_iterator& lhs_cursor,
      llvm::SmallVector<ConcreteType>::const_iterator& lhs_concrete_cursor,
      llvm::SmallVector<Structural>::const_iterator& rhs_cursor) -> bool;

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
//
// Returns nullopt if an ErrorInst is encountered in the self type or facet
// value.
auto BuildTypeStructure(Context& context, SemIR::InstId self_inst_id,
                        SemIR::SpecificInterface interface)
    -> std::optional<TypeStructure>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_TYPE_STRUCTURE_H_
