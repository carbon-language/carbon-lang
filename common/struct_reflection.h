// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_STRUCT_REFLECTION_H_
#define CARBON_COMMON_STRUCT_REFLECTION_H_

// Reflection support for simple struct types.
//
// Example usage:
//
// ```
// struct A { int x; std::string y; };
//
// A a;
// std::tuple<int, std::string> t = StructReflection::AsTuple(a);
// ```
//
// Limitations:
//
// - Only simple aggregate structs are supported. Types with base classes,
//   non-public data members, constructors, or virtual functions are not
//   supported.
// - Structs with more than 6 fields are not supported. This limit is easy to
//   increase if needed, but removing it entirely is hard.
// - Structs containing a reference to the same type are not supported.

#include <tuple>
#include <type_traits>

namespace Carbon::StructReflection {

namespace Internal {

// A type that can be converted to any field type within type T.
template <typename T>
struct AnyField {
  template <typename FieldT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator FieldT&() const;

  template <typename FieldT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator FieldT&&() const;

  // Don't allow conversion to T itself. This ensures we don't match against a
  // copy or move constructor.
  operator T&() const = delete;
  operator T&&() const = delete;
};

// The detection mechanism below intentionally misses field initializers.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"

// Detector for whether we can list-initialize T from the given list of fields.
template <typename T, typename... Fields>
constexpr auto CanListInitialize(decltype(T{Fields()...})* /*unused*/) -> bool {
  return true;
}
template <typename T, typename... Fields>
constexpr auto CanListInitialize(...) -> bool {
  return false;
}

#pragma clang diagnostic pop

// Simple detector to find the number of data fields in a struct. This proceeds
// in two passes:
//
// 1) Add AnyField<T>s until we can initialize T from our list of initializers.
// 2) Add more AnyField<T>s until we can't initialize any more.
template <typename T, bool AnyWorkedSoFar = false, typename... Fields>
constexpr auto CountFields() -> int {
  if constexpr (CanListInitialize<T, Fields...>(0)) {
    return CountFields<T, true, Fields..., AnyField<T>>();
  } else if constexpr (AnyWorkedSoFar) {
    constexpr int NumFields = sizeof...(Fields) - 1;
    static_assert(NumFields <= 6, "Unsupported: too many fields in struct");
    return NumFields;
  } else if constexpr (sizeof...(Fields) > 32) {
    // If we go too far without finding a working initializer, something
    // probably went wrong with our calculation. Bail out before we recurse too
    // deeply.
    static_assert(sizeof...(Fields) <= 32,
                  "Internal error, could not count fields in struct");
  } else {
    return CountFields<T, false, Fields..., AnyField<T>>();
  }
}

// Utility to access fields by index.
template <int NumFields>
struct FieldAccessor;

template <>
struct FieldAccessor<0> {
  template <typename T>
  static auto Get(T& /*value*/) -> auto {
    return std::tuple<>();
  }
};

template <>
struct FieldAccessor<1> {
  template <typename T>
  static auto Get(T& value) -> auto {
    auto& [field0] = value;
    return std::tuple<decltype(field0)>(field0);
  }
};

template <>
struct FieldAccessor<2> {
  template <typename T>
  static auto Get(T& value) -> auto {
    auto& [field0, field1] = value;
    return std::tuple<decltype(field0), decltype(field1)>(field0, field1);
  }
};

template <>
struct FieldAccessor<3> {
  template <typename T>
  static auto Get(T& value) -> auto {
    auto& [field0, field1, field2] = value;
    return std::tuple<decltype(field0), decltype(field1), decltype(field2)>(
        field0, field1, field2);
  }
};

template <>
struct FieldAccessor<4> {
  template <typename T>
  static auto Get(T& value) -> auto {
    auto& [field0, field1, field2, field3] = value;
    return std::tuple<decltype(field0), decltype(field1), decltype(field2),
                      decltype(field3)>(field0, field1, field2, field3);
  }
};

template <>
struct FieldAccessor<5> {
  template <typename T>
  static auto Get(T& value) -> auto {
    auto& [field0, field1, field2, field3, field4] = value;
    return std::tuple<decltype(field0), decltype(field1), decltype(field2),
                      decltype(field3), decltype(field4)>(
        field0, field1, field2, field3, field4);
  }
};

template <>
struct FieldAccessor<6> {
  template <typename T>
  static auto Get(T& value) -> auto {
    auto& [field0, field1, field2, field3, field4, field5] = value;
    return std::tuple<decltype(field0), decltype(field1), decltype(field2),
                      decltype(field3), decltype(field4), decltype(field5)>(
        field0, field1, field2, field3, field4, field5);
  }
};

}  // namespace Internal

// Get the fields of the struct `T` as a tuple.
template <typename T>
auto AsTuple(T value) -> auto {
  // We use aggregate initialization to detect the number of fields.
  static_assert(std::is_aggregate_v<T>, "Only aggregates are supported");
  return Internal::FieldAccessor<Internal::CountFields<T>()>::Get(value);
}

}  // namespace Carbon::StructReflection

#endif  // CARBON_COMMON_STRUCT_REFLECTION_H_
