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
    // The value representation is dependent because the type is symbolic.
    Dependent,
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

  // Returns whether this value representation is a copy of the object
  // representation of the type. `orig_type_id` must be the type for which this
  // is the value representation.
  auto IsCopyOfObjectRepr(const File& file, TypeId orig_type_id) const -> bool;

  // The kind of value representation used by this type.
  Kind kind = Unknown;
  // The kind of aggregate representation used by this type.
  AggregateKind aggregate_kind = AggregateKind::NotAggregate;
  // The type used to model the value representation.
  TypeId type_id = TypeId::None;
};

// A size within an object representation. This stores a bit count, but provides
// convenience methods to access the size in bytes instead.
class ObjectSize : public Printable<ObjectSize> {
 public:
  static constexpr auto Zero() -> ObjectSize { return ObjectSize(0); }

  static constexpr auto Bits(int64_t bits) -> ObjectSize {
    return ObjectSize(bits);
  }

  static constexpr auto Bytes(int64_t bytes) -> ObjectSize {
    return ObjectSize(bytes * 8);
  }

  // Returns the size in bits.
  auto bits() const -> int64_t { return bits_; }

  // Returns the minimum number of bytes that would contain this size. This is
  // the size in bytes, rounded up.
  auto bytes() const -> int64_t { return (bits_ + 7) / 8; }

  // Return this size rounded up to a multiple of `align`, which must be a power
  // of 2.
  [[nodiscard]] auto AlignedTo(ObjectSize align) const -> ObjectSize {
    CARBON_CHECK(llvm::isPowerOf2_64(align.bits_),
                 "Non power-of-2 alignment {0}", align.bits_);
    return Bits((bits_ + align.bits_ - 1) & -align.bits_);
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  friend constexpr auto operator<=>(ObjectSize a, ObjectSize b) {
    return a.bits_ <=> b.bits_;
  }

  friend constexpr auto operator==(ObjectSize a, ObjectSize b) -> bool {
    return a.bits_ == b.bits_;
  }

  // Computes the sum of two object sizes. This is the size of the smallest
  // object that could fit both objects consecutively, without any alignment
  // constraints or other padding bits.
  auto operator+=(ObjectSize other) -> ObjectSize& {
    bits_ += other.bits_;
    return *this;
  }
  friend auto operator+(ObjectSize a, ObjectSize b) -> ObjectSize {
    return a += b;
  }

  // Scales a size by a dimensionless integer. This is the size of `n` objects
  // of the given size, laid out contiguously with no padding between them.
  // Equivalent to adding the size to itself `n` times.
  auto operator*=(int64_t n) -> ObjectSize& {
    CARBON_CHECK(n >= 0, "sizes should not be negative");
    bits_ *= n;
    return *this;
  }
  friend auto operator*(ObjectSize a, int64_t n) -> ObjectSize {
    return a *= n;
  }
  friend auto operator*(int64_t n, ObjectSize a) -> ObjectSize {
    return a *= n;
  }

 private:
  explicit constexpr ObjectSize(int64_t bits) : bits_(bits) {
    CARBON_CHECK(bits >= 0, "sizes should not be negative");
  }

  int64_t bits_;
};

// The layout of an object representation.
struct ObjectLayout {
  // The size of the object representation. This may be zero for empty types.
  // This is not guaranteed to be a multiple of `alignment`, so explicit tail
  // padding may be required if objects using this layout are stored in an
  // array.
  ObjectSize size = ObjectSize::Zero();

  // The alignment of the object representation. This will be a power of 2 for a
  // valid representation, or zero if the representation is unset (such as for
  // the object layout of an incomplete type). This will typically be 1 byte for
  // an empty type.
  ObjectSize alignment = ObjectSize::Zero();

  // Returns the object layout to use for an empty type.
  [[nodiscard]] static auto Empty() -> ObjectLayout {
    return {.size = ObjectSize::Zero(), .alignment = ObjectSize::Bytes(1)};
  }

  // Returns the object layout to use for an array of `count` elements. The
  // result never has tail padding, so this is *not* the same as adding the
  // element to itself `count` times in general.
  static auto ForArray(ObjectLayout element_layout, int64_t count)
      -> ObjectLayout {
    return {.size = element_layout.ArrayStride() * count,
            .alignment = element_layout.alignment};
  }

  // Returns the stride of an array with this element type.
  auto ArrayStride() const -> ObjectSize { return size.AlignedTo(alignment); }

  // Returns the offset that a field with the given layout would have if
  // appended to this struct.
  auto FieldOffset(ObjectLayout field) -> ObjectSize {
    CARBON_CHECK(field.has_value());
    return size.AlignedTo(field.alignment);
  }

  // Updates this aggregate layout by concatenating naturally-aligned space for
  // the given field layout, which is required to be valid.
  auto AppendField(ObjectLayout field) -> void {
    size = FieldOffset(field) + field.size;
    alignment = std::max(alignment, field.alignment);
  }

  // Updates this aggregate layout by concatenating naturally-aligned space for
  // the given field layout, if the field has a valid layout. If any field with
  // an invalid layout is appended, the layout of the aggregate becomes invalid.
  //
  // An invalid layout for a field with a complete type indicates that its
  // layout is dependent on a generic parameter; this causes the layout of the
  // enclosing aggregate to also be dependent on that parameter.
  auto TryAppendField(ObjectLayout field) -> void {
    if (has_value()) {
      if (field.has_value()) {
        AppendField(field);
      } else {
        *this = ObjectLayout();
      }
    }
  }

  // Returns true if the layout has been set. For a complete type, this will be
  // true if the layout is non-dependent.
  auto has_value() const -> bool { return alignment != ObjectSize::Zero(); }
};

// Information stored about a TypeId corresponding to a complete type.
struct CompleteTypeInfo : public Printable<CompleteTypeInfo> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // The value representation for this type. Will be `Unknown` if the type is
  // not complete.
  ValueRepr value_repr = ValueRepr();

  // The layout of the object representation of this type. This will be valid
  // unless the type is incomplete or dependent.
  ObjectLayout object_layout = ObjectLayout();

  // If this type is abstract, this is id of an abstract class it uses.
  ClassId abstract_class_id = ClassId::None;

  // Returns whether the type is abstract.
  //
  // The type must be completed before we can determine if it's abstract.
  auto IsAbstract() const -> bool {
    CARBON_CHECK(value_repr.kind != ValueRepr::Unknown);
    return abstract_class_id.has_value();
  }
};

// The representation to use for an initializing expression of some type.
struct InitRepr : Printable<InitRepr> {
  // Returns information about the initializing representation to use for a
  // type.
  static auto ForType(const File& file, TypeId type_id) -> InitRepr;

  enum Kind : int8_t {
    // The type has no initializing representation. This is used for empty
    // types, where no initialization is necessary.
    None,
    // The initializing representation is dependent because the type is
    // symbolic.
    Dependent,
    // An initializing expression produces an object representation by value,
    // which is copied into the initialized object.
    ByCopy,
    // An initializing expression takes a location as input, which is
    // initialized as a side effect of evaluating the expression.
    InPlace,
    // No initializing expressions should exist because the type is abstract.
    Abstract,
    // No initializing expressions should exist yet, because the type is not
    // complete.
    Incomplete,
    // TODO: Consider adding a kind where the expression takes an advisory
    // location and returns a value plus an indicator of whether the location
    // was actually initialized.
  };
  // The kind of initializing representation used by this type.
  Kind kind;

  // Returns whether the type can be used as the type of an initializing
  // expression in the current context.
  auto is_valid() const -> bool {
    return kind != Incomplete && kind != Abstract;
  }

  // Returns whether the initializing representation is a copy of the object
  // representation of the type. Provided for symmetry with `ValueRepr`.
  auto IsCopyOfObjectRepr() const -> bool { return kind == ByCopy; }

  // Returns whether the initializing representation might be by-copy, and
  // therefore might require a final destination store.
  auto MightBeByCopy() const -> bool {
    return kind == ByCopy || kind == Dependent;
  }

  // Returns whether the initializing representation might be in-place, and
  // therefore might require a destination address to be provided as input.
  auto MightBeInPlace() const -> bool {
    return kind == InPlace || kind == Dependent;
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{kind: ";
    switch (kind) {
      case None:
        out << "None";
        break;
      case Dependent:
        out << "Dependent";
        break;
      case ByCopy:
        out << "ByCopy";
        break;
      case InPlace:
        out << "InPlace";
        break;
      case Abstract:
        out << "Abstract";
        break;
      case Incomplete:
        out << "Incomplete";
        break;
    }
    out << "}";
  }
};

// Information about the numeric type literal that corresponds to a type.
struct NumericTypeLiteralInfo {
  // The kind of a numeric type literal, as determined by the letter that
  // prefixes the bit width.
  enum Kind : char {
    None = 0,
    Int = 'i',
    UInt = 'u',
    Float = 'f',
  };

  static const NumericTypeLiteralInfo Invalid;

  // Returns the numeric type literal that would evaluate to this class type, if
  // any.
  static auto ForType(const File& file, ClassType class_type)
      -> NumericTypeLiteralInfo;

  // Prints the numeric type literal that corresponds to this type.
  auto PrintLiteral(const File& file, llvm::raw_ostream& out) const -> void;

  // Returns whether this is a valid numeric type literal.
  auto is_valid() const -> bool { return kind != None; }

  // The kind of this type literal.
  Kind kind;
  // The bit width of this type literal.
  IntId bit_width_id;
};

// Information about a recognized type, which is either a literal type, or a C++
// builtin.
struct RecognizedTypeInfo {
  enum Kind : char {
    None,
    // A numeric type literal such as `i8`; see `numeric` field for details.
    Numeric,
    // `char` / `Core.Char`.
    Char,
    // `Core.CppCompat.Long32` which is `Cpp.long` when `long` is 32 bits.
    CppLong32,
    // `Core.CppCompat.ULong32` which is `Cpp.unsigned_long` when `unsigned
    // long` is 32 bits.
    CppULong32,
    // `Core.CppCompat.LongLong64` which is `Cpp.long_long` when `long` is 64
    // bits.
    CppLongLong64,
    // `Core.CppCompat.ULongLong64` which is `Cpp.unsigned_long_long` when
    // `unsigned long` is 64 bits.
    CppULongLong64,
    // `Cpp.nullptr_t` / `Core.CppCompat.NullptrT`.
    CppNullptrT,
    // `Cpp.void` / `Core.CppCompat.VoidBase`.
    CppVoidBase,
    // `Core.Optional(...)`.
    Optional,
    // `str` / `Core.String`.
    // TODO: Rename `Core.String` to `Core.Str`.
    Str,
  };

  // Returns the type literal that would evaluate to this class type, if any.
  static auto ForType(const File& file, ClassType class_type)
      -> RecognizedTypeInfo;

  // Prints the type literal or special type name that corresponds to this type,
  // if there is one. Returns true if the type was printed, or false if this
  // type doesn't have special syntax and should be printed directly.
  auto PrintLiteral(const File& file, llvm::raw_ostream& out) const -> bool;

  // Returns whether this is a valid type literal.
  auto is_valid() const -> bool { return kind != None; }

  // The kind of the literal.
  Kind kind;
  // If this is a numeric literal, additional information about the literal.
  NumericTypeLiteralInfo numeric = NumericTypeLiteralInfo::Invalid;
  // If this is a generic type, the arguments.
  InstBlockId args_id = InstBlockId::None;
};

inline constexpr NumericTypeLiteralInfo NumericTypeLiteralInfo::Invalid = {
    .kind = None, .bit_width_id = IntId::None};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPE_INFO_H_
