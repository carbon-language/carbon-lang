// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ERROR_H_
#define CARBON_COMMON_ERROR_H_

#include <string>
#include <variant>

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
  explicit Error(llvm::Twine prefix, llvm::Twine location, llvm::Twine message)
      : prefix_(prefix.str()),
        location_(location.str()),
        message_(message.str()) {
    CARBON_CHECK(!message_.empty()) << "Errors must have a message.";
  }

  // Represents an error with no associated prefix or location.
  // TODO: Consider using two different types.
  explicit Error(llvm::Twine message) : Error("", "", message) {}

  Error(Error&& other) noexcept
      : prefix_(std::move(other.prefix_)),
        location_(std::move(other.location_)),
        message_(std::move(other.message_)) {}

  // Prints the error string.
  void Print(llvm::raw_ostream& out) const {
    if (!prefix().empty()) {
      out << prefix() << ": ";
    }
    if (!location().empty()) {
      out << location() << ": ";
    }
    out << message();
  }

  // Returns the prefix to prepend to the error, such as "ERROR".
  auto prefix() const -> const std::string& { return prefix_; }

  // Returns a string describing the location of the error, such as
  // "file.cc:123".
  auto location() const -> const std::string& { return location_; }

  // Returns the error message.
  auto message() const -> const std::string& { return message_; }

 private:
  // A prefix, indicating the kind of error.
  std::string prefix_;
  // The location associated with the error.
  std::string location_;
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
    CARBON_CHECK(!ok());
    return std::get<Error>(val_);
  }
  auto error() && -> Error {
    CARBON_CHECK(!ok());
    return std::get<Error>(std::move(val_));
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator*() -> T& {
    CARBON_CHECK(ok());
    return std::get<T>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator*() const -> const T& {
    CARBON_CHECK(ok());
    return std::get<T>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator->() -> T* {
    CARBON_CHECK(ok());
    return &std::get<T>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator->() const -> const T* {
    CARBON_CHECK(ok());
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
  explicit ErrorBuilder(std::string prefix, std::string location)
      : prefix_(std::move(prefix)),
        location_(std::move(location)),
        out_(std::make_unique<llvm::raw_string_ostream>(message_)) {}

  explicit ErrorBuilder() : ErrorBuilder("", "") {}

  // Accumulates string message.
  template <typename T>
  [[nodiscard]] auto operator<<(const T& message) -> ErrorBuilder& {
    *out_ << message;
    return *this;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator Error() { return Error(prefix_, location_, message_); }

  template <typename T>
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator ErrorOr<T>() {
    return Error(prefix_, location_, message_);
  }

 private:
  std::string prefix_;
  std::string location_;
  std::string message_;
  // Use a pointer to allow move construction.
  std::unique_ptr<llvm::raw_string_ostream> out_;
};

}  // namespace Carbon

// Macro hackery to get a unique variable name.
#define CARBON_MAKE_UNIQUE_NAME_IMPL(a, b, c) a##b##c
#define CARBON_MAKE_UNIQUE_NAME(a, b, c) CARBON_MAKE_UNIQUE_NAME_IMPL(a, b, c)

#define CARBON_RETURN_IF_ERROR_IMPL(unique_name, expr)                    \
  if (auto unique_name = (expr); /* NOLINT(bugprone-macro-parentheses) */ \
      !(unique_name).ok()) {                                              \
    return std::move(unique_name).error();                                \
  }

#define CARBON_RETURN_IF_ERROR(expr) \
  CARBON_RETURN_IF_ERROR_IMPL(       \
      CARBON_MAKE_UNIQUE_NAME(_llvm_error_line, __LINE__, __COUNTER__), expr)

#define CARBON_ASSIGN_OR_RETURN_IMPL(unique_name, var, expr)          \
  auto unique_name = (expr); /* NOLINT(bugprone-macro-parentheses) */ \
  if (!(unique_name).ok()) {                                          \
    return std::move(unique_name).error();                            \
  }                                                                   \
  var = std::move(*(unique_name)); /* NOLINT(bugprone-macro-parentheses) */

#define CARBON_ASSIGN_OR_RETURN(var, expr)                                 \
  CARBON_ASSIGN_OR_RETURN_IMPL(                                            \
      CARBON_MAKE_UNIQUE_NAME(_llvm_expected_line, __LINE__, __COUNTER__), \
      var, expr)

#endif  // CARBON_COMMON_ERROR_H_
