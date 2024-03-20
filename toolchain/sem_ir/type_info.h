// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_

#include "common/ostream.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// The value representation to use when passing by value.
struct ValueRepr : public Printable<ValueRepr> {
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

// Information stored about a TypeId.
struct TypeInfo : public Printable<TypeInfo> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // The constant type value that defines this type.
  ConstantId constant_id;
  // The value representation for this type. Will be `Unknown` if the type is
  // not complete.
  ValueRepr value_repr = ValueRepr();
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_
