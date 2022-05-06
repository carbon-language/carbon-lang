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

  Error(const Error&) = default;
  Error(Error&&) = default;
  auto operator=(const Error&) -> Error& = default;
  auto operator=(Error&&) -> Error& = default;

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

  // ErrorOr<T> is movable and copyable if T is.
  //
  // TODO: Consider SFINAE-disabling these operations when they would be
  // invalid, so that is_constructible and similar traits give correct results.
  ErrorOr(const ErrorOr&) = default;
  ErrorOr(ErrorOr&&) noexcept = default;
  auto operator=(const ErrorOr<T>&) -> ErrorOr<T>& = default;
  auto operator=(ErrorOr<T>&&) noexcept -> ErrorOr<T>& = default;

  // Returns true for success.
  auto ok() const -> bool { return std::holds_alternative<T>(val_); }

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

// A helper class for accumulating error message and converting to
// `Error` and `ErrorOr<T>`.
class ErrorBuilder {
 public:
  ErrorBuilder() : out_(std::make_unique<llvm::raw_string_ostream>(message_)) {}

  // Accumulates string message.
  template <typename T>
  [[nodiscard]] auto operator<<(const T& message) -> ErrorBuilder& {
    *out_ << message;
    return *this;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator Error() { return Error(message_); }

  template <typename T>
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator ErrorOr<T>() {
    return Error(message_);
  }

 private:
  std::string message_;
  // Use a pointer to allow move construction.
  std::unique_ptr<llvm::raw_string_ostream> out_;
};

}  // namespace Carbon

// Macro hackery to get a unique variable name.
#define MAKE_UNIQUE_NAME_IMPL(a, b, c) a##b##c
#define MAKE_UNIQUE_NAME(a, b, c) MAKE_UNIQUE_NAME_IMPL(a, b, c)

#define RETURN_IF_ERROR_IMPL(unique_name, expr)                           \
  if (auto unique_name = (expr); /* NOLINT(bugprone-macro-parentheses) */ \
      !(unique_name).ok()) {                                              \
    return std::move(unique_name).error();                                \
  }

#define RETURN_IF_ERROR(expr) \
  RETURN_IF_ERROR_IMPL(       \
      MAKE_UNIQUE_NAME(_llvm_error_line, __LINE__, __COUNTER__), expr)

#define ASSIGN_OR_RETURN_IMPL(unique_name, var, expr)                 \
  auto unique_name = (expr); /* NOLINT(bugprone-macro-parentheses) */ \
  if (!(unique_name).ok()) {                                          \
    return std::move(unique_name).error();                            \
  }                                                                   \
  var = std::move(*(unique_name)); /* NOLINT(bugprone-macro-parentheses) */

#define ASSIGN_OR_RETURN(var, expr) \
  ASSIGN_OR_RETURN_IMPL(            \
      MAKE_UNIQUE_NAME(_llvm_expected_line, __LINE__, __COUNTER__), var, expr)

#endif  // COMMON_ERROR_H_
