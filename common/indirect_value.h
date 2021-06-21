// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_INDIRECT_VALUE_H_
#define COMMON_INDIRECT_VALUE_H_

#include <memory>
#include <utility>

namespace Carbon {

template <typename T>
class IndirectValue;

// Creates and returns an IndirectValue<T> with the value T(args...)
template <typename T, typename... Args>
auto MakeIndirectValue(Args&&... args) -> IndirectValue<T>;

// An IndirectValue<T> object stores a T value, using a layer of indirection
// that allows us to name the IndirectValue<T> type, and even access the
// underlying T value, in a context where T is not a complete type. This makes
// it useful for things like defining recursive types.
//
// The underlying value is accessed using the * and -> operators, but
// IndirectValue does not otherwise behave like a pointer: it has no null state,
// and separate IndirectValue objects are never aliases for the same T object.
// Instead, an IndirectValue object behaves as much as possible like a T object:
// the default constructor, copy operations, and move operations all delegate to
// the corresponding operations on T, and a const IndirectValue object provides
// only const access to the underlying T object. The address of the underlying T
// object remains the same throughout the lifetime of the IndirectValue.
//
// IndirectValue is inspired by the indirect_value library proposed in
// http://wg21.link/P1950R1, but makes some different design choices (notably,
// not having an empty state) in order to provide a more value-like API.
template <typename T>
class IndirectValue {
 public:
  // TODO(geoffromer): consider using enable_if to disable constructors and
  // assignment operators when they wouldn't compile, so that traits like
  // std::is_constructible give correct answers.
  IndirectValue() : value_(std::make_unique<T>()) {}

  IndirectValue(const IndirectValue& other)
      : value_(std::make_unique<T>(*other)) {}

  IndirectValue(IndirectValue&& other)
      : value_(std::make_unique<T>(std::move(*other))) {}

  auto operator=(const IndirectValue& other) -> IndirectValue& {
    *value_ = *other.value_;
    return *this;
  }

  auto operator=(IndirectValue&& other) -> IndirectValue& {
    *value_ = std::move(*other.value_);
    return *this;
  }

  auto operator*() -> T& { return *value_; }
  auto operator*() const -> const T& { return *value_; }

  auto operator->() -> T* { return value_.get(); }
  auto operator->() const -> const T* { return value_.get(); }

  // Returns the address of the stored value.
  //
  // TODO(geoffromer): Consider eliminating this method, which is not
  // present in comparable types like indirect_value<T> or optional<T>,
  // once our APIs are less pointer-centric.
  auto GetPointer() -> T* { return value_.get(); }
  auto GetPointer() const -> const T* { return value_.get(); }

 private:
  template <typename TT, typename... Args>
  friend auto MakeIndirectValue(Args&&... args) -> IndirectValue<TT>;

  template <typename... Args>
  IndirectValue(std::in_place_t, Args&&... args)
      : value_(std::make_unique<T>(std::forward<Args...>(args...))) {}

  const std::unique_ptr<T> value_;
};

template <typename T, typename... Args>
auto MakeIndirectValue(Args&&... args) -> IndirectValue<T> {
  return IndirectValue<T>(std::in_place, std::forward<Args...>(args...));
}
}  // namespace Carbon

#endif  // COMMON_INDIRECT_VALUE_H_
