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

// Returns information about the value representation to use for a type.
auto GetValueRepr(const File& file, TypeId type_id) -> ValueRepr;

// Information stored about a TypeId corresponding to a complete type.
struct CompleteTypeInfo : public Printable<CompleteTypeInfo> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // The value representation for this type. Will be `Unknown` if the type is
  // not complete.
  ValueRepr value_repr = ValueRepr();
};

// The initializing representation to use when returning by value.
struct InitRepr {
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
    // TODO: Consider adding a kind where the expression takes an advisory
    // location and returns a value plus an indicator of whether the location
    // was actually initialized.
  };
  // The kind of initializing representation used by this type.
  Kind kind;

  // Returns whether a return slot is used when returning this type.
  auto has_return_slot() const -> bool { return kind == InPlace; }
};

// Returns information about the initializing representation to use for a type.
auto GetInitRepr(const File& file, TypeId type_id) -> InitRepr;

// A value that describes whether the function uses a return slot.
enum class ReturnSlot : int8_t {
  // The function is known to not use a return slot.
  Absent,
  // The function has a return slot, and a call to the function is expected to
  // have an additional final argument corresponding to the return slot.
  Present,
  // Computing whether the function should have a return slot failed because
  // the return type was incomplete.
  Incomplete,
};

// Information about how a function returns its return value.
struct ReturnTypeInfo {
  // Builds return information for a given declared return type.
  static auto ForType(const File& file, TypeId type_id) -> ReturnTypeInfo;

  // Returns whether the return information could be fully computed.
  auto is_valid() const -> bool {
    return return_slot != ReturnSlot::Incomplete;
  }

  // Returns whether the function has a return slot. Can only be called for
  // valid return info.
  auto has_return_slot() const -> bool {
    CARBON_CHECK(is_valid());
    return return_slot == ReturnSlot::Present;
  }

  // The return type. Invalid if no return type was specified.
  TypeId type_id;
  // The return slot usage for this function.
  ReturnSlot return_slot;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_
