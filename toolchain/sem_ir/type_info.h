// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_

#include "common/ostream.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// The value representation to use when passing by value.
struct ValueRepr : public Printable<ValueRepr> {
  // Returns information about the value representation to use for a type.
  static auto ForType(const File& file, TypeId type_id) -> ValueRepr;

  auto Print(llvm::raw_ostream& out) const -> void;

  enum Kind : int8_t {
    // The value representation is not yet known. This is used for incomplete
    // types.
    Unknown,
    // The type has no value representation. This is used for empty types, such
    // as `()`, where there is no value.
    None,
    // The value representation is a copy of the value. On call boundaries, the
    // value itself will be passed. `type` is the value type.
    Copy,
    // The value representation is a pointer to the value. When used as a
    // parameter, the argument is a reference expression. `type` is the pointee
    // type.
    Pointer,
    // The value representation has been customized, and has the same behavior
    // as the value representation of some other type.
    // TODO: This is not implemented or used yet.
    Custom,
  };

  enum AggregateKind : int8_t {
    // This type is not an aggregation of other types.
    NotAggregate,
    // This type is an aggregate that holds the value representations of its
    // elements.
    ValueAggregate,
    // This type is an aggregate that holds the object representations of its
    // elements.
    ObjectAggregate,
    // This type is an aggregate for which the value and object representation
    // of all elements are the same, so it effectively holds both.
    ValueAndObjectAggregate,
  };

  // Returns whether this is an aggregate that holds its elements by value.
  auto elements_are_values() const {
    return aggregate_kind == ValueAggregate ||
           aggregate_kind == ValueAndObjectAggregate;
  }

  // The kind of value representation used by this type.
  Kind kind = Unknown;
  // The kind of aggregate representation used by this type.
  AggregateKind aggregate_kind = AggregateKind::NotAggregate;
  // The type used to model the value representation.
  TypeId type_id = TypeId::Invalid;
};

// Information stored about a TypeId corresponding to a complete type.
struct CompleteTypeInfo : public Printable<CompleteTypeInfo> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // The value representation for this type. Will be `Unknown` if the type is
  // not complete.
  ValueRepr value_repr = ValueRepr();
};

// The initializing representation to use when returning by value.
struct InitRepr {
  // Returns information about the initializing representation to use for a
  // type.
  static auto ForType(const File& file, TypeId type_id) -> InitRepr;

  enum Kind : int8_t {
    // The type has no initializing representation. This is used for empty
    // types, where no initialization is necessary.
    None,
    // An initializing expression produces an object representation by value,
    // which is copied into the initialized object.
    ByCopy,
    // An initializing expression takes a location as input, which is
    // initialized as a side effect of evaluating the expression.
    InPlace,
    // No initializing expressions should exist because the type is not
    // complete.
    Incomplete,
    // TODO: Consider adding a kind where the expression takes an advisory
    // location and returns a value plus an indicator of whether the location
    // was actually initialized.
  };
  // The kind of initializing representation used by this type.
  Kind kind;

  // Returns whether the initializing representation information could be fully
  // computed.
  auto is_valid() const -> bool { return kind != Incomplete; }
};

// Information about a function's return type.
struct ReturnTypeInfo {
  // Builds return type information for a given declared return type.
  static auto ForType(const File& file, TypeId type_id) -> ReturnTypeInfo {
    return {.type_id = type_id,
            .init_repr = type_id.is_valid() ? InitRepr::ForType(file, type_id)
                                            : InitRepr{.kind = InitRepr::None}};
  }

  // Builds return type information for a given function.
  static auto ForFunction(const File& file, const Function& function,
                          SpecificId specific_id = SpecificId::Invalid)
      -> ReturnTypeInfo {
    return ForType(file, function.GetDeclaredReturnType(file, specific_id));
  }

  // Returns whether the return information could be fully computed.
  auto is_valid() const -> bool { return init_repr.is_valid(); }

  // Returns whether a function with this return type has a return slot. Can
  // only be called for valid return info.
  auto has_return_slot() const -> bool {
    CARBON_CHECK(is_valid());
    return init_repr.kind == InitRepr::InPlace;
  }

  // The declared return type. Invalid if no return type was specified.
  TypeId type_id;
  // The initializing representation for the return type.
  InitRepr init_repr;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_
