// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_ERROR_H_
#define COMMON_ERROR_H_

#include <optional>
#include <string>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/Twine.h"

namespace Carbon {

// Tracks an error message.
[[nodiscard]]
class Error {
 public:
  // Represents a success state.
  static auto Success() -> Error { return Error(); }

  // Represents an error state.
  explicit Error(llvm::Twine message) : message_(message.str()) {
    CHECK(!message_->empty()) << "Errors must have a message.";
  }

  Error(Error&& other) noexcept
      : used_(other.used_), message_(std::move(other.message_)) {
    // Prevent the other from checking on destruction.
    other.used_ = true;
  }

  // Verifies the error state was examined.
  ~Error() { CHECK(used_); }

  // Prints the error string. Note this marks as used.
  void Print(llvm::raw_ostream& out) const { out << message(); }

  // Returns true for success.
  auto ok() const -> bool {
    used_ = true;
    return !message_.has_value();
  }

  // Returns the error message.
  // REQUIRES: `ok()` is false.
  auto message() const -> const std::string& {
    CHECK(!ok());
    return *message_;
  }

 private:
  // The success state constructor.
  Error() = default;

  // Indicates whether the held state has been examined.
  mutable bool used_ = false;

  // The error message. Set to nullopt for success.
  std::optional<std::string> message_;
};

// Holds a value of type `T`, or an Error explaining why the value is
// unavailable.
template <typename T>
class ErrorOr {
 public:
  // Constructs with an error; the error must not be Error::Success().
  // Implicit for easy construction on returns.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ErrorOr(Error err) : val_(std::move(err)) {
    // Note this marks the Error as used, but the ErrorOr remains unused.
    CHECK(!std::get<Error>(val_).ok());
  }

  // Constructs with a value.
  // Implicit for easy construction on returns.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ErrorOr(T val) : val_(std::move(val)) {}

  // Moves held state.
  ErrorOr(ErrorOr&& other) noexcept
      : used_(other.used_), val_(std::move(other.val_)) {
    // Prevent the other from checking on destruction.
    other.used_ = true;
  }

  ~ErrorOr() { CHECK(used_); }

  // Returns true for success.
  auto ok() -> bool {
    used_ = true;
    return std::holds_alternative<T>(val_);
  }

  // Returns the contained error.
  // REQUIRES: `ok()` is false.
  auto error() const -> const Error& {
    CHECK(!ok());
    return std::get<Error>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator*() -> T& {
    CHECK(ok());
    return std::get<T>(val_);
  }

 private:
  // Used to verify that the held state is examined, preventing dropping
  // values.
  mutable bool used_ = false;

  // Either an error message or
  std::variant<Error, T> val_;
};

}  // namespace Carbon

#endif  // COMMON_ERROR_H_
