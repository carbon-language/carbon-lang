// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_ERROR_H_
#define COMMON_ERROR_H_

#include <string>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/Twine.h"

namespace Carbon {

// Success values should be represented as the presence of a value in ErrorOr,
// using `ErrorOr<Success>` and `return Success();` if no value needs to be
// returned.
struct Success {};

// Tracks an error message.
class [[nodiscard]] Error {
 public:
  // Represents an error state.
  explicit Error(llvm::Twine message) : message_(message.str()) {
    CHECK(!message_.empty()) << "Errors must have a message.";
  }

  Error(Error&& other) noexcept : message_(std::move(other.message_)) {}

  // Prints the error string. Note this marks as used.
  void Print(llvm::raw_ostream& out) const { out << message(); }

  // Returns the error message.
  auto message() const -> const std::string& { return message_; }

 private:
  // The error message.
  std::string message_;
};

// Holds a value of type `T`, or an Error explaining why the value is
// unavailable.
template <typename T>
class [[nodiscard]] ErrorOr {
 public:
  // Constructs with an error; the error must not be Error::Success().
  // Implicit for easy construction on returns.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ErrorOr(Error err) : val_(std::move(err)) {}

  // Constructs with a value.
  // Implicit for easy construction on returns.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ErrorOr(T val) : val_(std::move(val)) {}

  // Moves held state.
  ErrorOr(ErrorOr&& other) noexcept : val_(std::move(other.val_)) {}

  // Returns true for success.
  auto ok() const -> bool { return std::holds_alternative<T>(val_); }

  // Returns the contained error.
  // REQUIRES: `ok()` is false.
  auto error() const& -> const Error& {
    CHECK(!ok());
    return std::get<Error>(val_);
  }
  auto error() && -> Error {
    CHECK(!ok());
    return std::get<Error>(std::move(val_));
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator*() -> T& {
    CHECK(ok());
    return std::get<T>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator*() const -> const T& {
    CHECK(ok());
    return std::get<T>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator->() -> T* {
    CHECK(ok());
    return &std::get<T>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator->() const -> const T* {
    CHECK(ok());
    return &std::get<T>(val_);
  }

 private:
  // Either an error message or
  std::variant<Error, T> val_;
};

}  // namespace Carbon

#endif  // COMMON_ERROR_H_
