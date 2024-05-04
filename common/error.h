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
//
// This is nodiscard to enforce error handling prior to destruction.
class [[nodiscard]] Error : public Printable<Error> {
 public:
  // Represents an error state.
  explicit Error(llvm::Twine location, llvm::Twine message)
      : location_(location.str()), message_(message.str()) {
    CARBON_CHECK(!message_.empty()) << "Errors must have a message.";
  }

  // Represents an error with no associated location.
  // TODO: Consider using two different types.
  explicit Error(llvm::Twine message) : Error("", message) {}

  Error(Error&& other) noexcept
      : location_(std::move(other.location_)),
        message_(std::move(other.message_)) {}

  auto operator=(Error&& other) noexcept -> Error& {
    location_ = std::move(other.location_);
    message_ = std::move(other.message_);
    return *this;
  }

  // Prints the error string.
  void Print(llvm::raw_ostream& out) const {
    if (!location().empty()) {
      out << location() << ": ";
    }
    out << message();
  }

  // Returns a string describing the location of the error, such as
  // "file.cc:123".
  auto location() const -> const std::string& { return location_; }

  // Returns the error message.
  auto message() const -> const std::string& { return message_; }

 private:
  // The location associated with the error.
  std::string location_;
  // The error message.
  std::string message_;
};

// Holds a value of type `T`, or an Error explaining why the value is
// unavailable.
//
// This is nodiscard to enforce error handling prior to destruction.
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
  // Either an error message or a value.
  std::variant<Error, T> val_;
};

// A helper class for accumulating error message and converting to
// `Error` and `ErrorOr<T>`.
class ErrorBuilder {
 public:
  explicit ErrorBuilder(std::string location = "")
      : location_(std::move(location)),
        out_(std::make_unique<llvm::raw_string_ostream>(message_)) {}

  // Accumulates string message to a temporary `ErrorBuilder`. After streaming,
  // the builder must be converted to an `Error` or `ErrorOr`.
  template <typename T>
  auto operator<<(T&& message) && -> ErrorBuilder&& {
    *out_ << message;
    return std::move(*this);
  }

  // Accumulates string message for an lvalue error builder.
  template <typename T>
  auto operator<<(T&& message) & -> ErrorBuilder& {
    *out_ << message;
    return *this;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator Error() { return Error(location_, message_); }

  template <typename T>
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator ErrorOr<T>() {
    return Error(location_, message_);
  }

 private:
  std::string location_;
  std::string message_;
  // Use a pointer to allow move construction.
  std::unique_ptr<llvm::raw_string_ostream> out_;
};

}  // namespace Carbon

// Macro hackery to get a unique variable name.
#define CARBON_MAKE_UNIQUE_NAME_IMPL(a, b, c) a##b##c
#define CARBON_MAKE_UNIQUE_NAME(a, b, c) CARBON_MAKE_UNIQUE_NAME_IMPL(a, b, c)

// Macro to prevent a top-level comma from being interpreted as a macro
// argument separator.
#define CARBON_PROTECT_COMMAS(...) __VA_ARGS__

#define CARBON_RETURN_IF_ERROR_IMPL(unique_name, expr)  \
  if (auto unique_name = (expr); !(unique_name).ok()) { \
    return std::move(unique_name).error();              \
  }

#define CARBON_RETURN_IF_ERROR(expr)                                    \
  CARBON_RETURN_IF_ERROR_IMPL(                                          \
      CARBON_MAKE_UNIQUE_NAME(_llvm_error_line, __LINE__, __COUNTER__), \
      CARBON_PROTECT_COMMAS(expr))

#define CARBON_ASSIGN_OR_RETURN_IMPL(unique_name, var, expr) \
  auto unique_name = (expr);                                 \
  if (!(unique_name).ok()) {                                 \
    return std::move(unique_name).error();                   \
  }                                                          \
  var = std::move(*(unique_name));

#define CARBON_ASSIGN_OR_RETURN(var, expr)                                 \
  CARBON_ASSIGN_OR_RETURN_IMPL(                                            \
      CARBON_MAKE_UNIQUE_NAME(_llvm_expected_line, __LINE__, __COUNTER__), \
      CARBON_PROTECT_COMMAS(var), CARBON_PROTECT_COMMAS(expr))

#endif  // CARBON_COMMON_ERROR_H_
