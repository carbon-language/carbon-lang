// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ERROR_H_
#define CARBON_COMMON_ERROR_H_

#include <concepts>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "common/check.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/Twine.h"

namespace Carbon {

// Success values should be represented as the presence of a value in ErrorOr,
// using `ErrorOr<Success>` and `return Success();` if no value needs to be
// returned.
struct Success : public Printable<Success> {
  auto Print(llvm::raw_ostream& out) const -> void { out << "Success"; }
};

// Tracks an error message.
//
// This is nodiscard to enforce error handling prior to destruction.
class [[nodiscard]] Error : public Printable<Error> {
 public:
  // Represents an error state.
  explicit Error(llvm::Twine message) : message_(message.str()) {
    CARBON_CHECK(!message_.empty(), "Errors must have a message.");
  }

  // Move-only.
  Error(Error&& other) noexcept = default;
  auto operator=(Error&& other) noexcept -> Error& = default;

  // Prints the error string.
  auto Print(llvm::raw_ostream& out) const -> void { out << message(); }

  // Returns the error message.
  auto message() const -> const std::string& { return message_; }

 private:
  // The error message.
  std::string message_;
};

// A common base class that custom error types should derive from.
//
// This combines the ability to be printed with the ability to convert the error
// to a string and in turn to a non-customized `Error` type by rendering into a
// string.
//
// The goal is that custom error types can be used for errors that are common
// and/or would have cost to fully render the error message to a string. A
// custom type can then be used to allow custom, light-weight handling of errors
// when appropriate. But to avoid these custom types being excessively viral, we
// ensure they can be converted to normal `Error` types when needed by rendering
// fully to a string.
template <typename ErrorT>
class [[nodiscard]] ErrorBase : public Printable<ErrorT> {
 public:
  ErrorBase(const ErrorBase&) = delete;
  auto operator=(const ErrorBase&) -> ErrorBase& = delete;

  auto ToString() const -> std::string {
    RawStringOstream os;
    static_cast<const ErrorT*>(this)->Print(os);
    return os.TakeStr();
  }
  auto ToError() const -> Error { return Error(this->ToString()); }

 protected:
  ErrorBase() = default;
  ErrorBase(ErrorBase&&) noexcept = default;
  auto operator=(ErrorBase&&) noexcept -> ErrorBase& = default;
};

// Holds a value of type `T`, or an `ErrorT` type explaining why the value is
// unavailable.
//
// The `ErrorT` type defaults to `Error` but can be customized where desired
// with a type that derives from `ErrorBase` above. See the documentation for
// `ErrorBase` to understand the expected contract of custom error types.
//
// This is nodiscard to enforce error handling prior to destruction.
template <typename T, typename ErrorT = Error>
class [[nodiscard]] ErrorOr {
 public:
  using ValueT = std::remove_reference_t<T>;

  // Check that the custom error type is structured the way we expect. These
  // need to be `static_assert`s to enable forward declared error types to be
  // used with `ErrorOr` in function signatures.
  static_assert(!std::is_reference_v<ErrorT>);
  static_assert(std::same_as<ErrorT, Error> ||
                std::derived_from<ErrorT, ErrorBase<ErrorT>>);

  // Constructs with an error; the error must not be Error::Success().
  // Implicit for easy construction on returns.
  explicit(false) ErrorOr(ErrorT err) : val_(std::move(err)) {}

  // Constructs from a custom error type derived from `ErrorBase` into an
  // `ErrorOr` for `Error` to facilitate returning errors transparently.
  template <typename OtherErrorT>
    requires(std::same_as<ErrorT, Error> &&
             std::derived_from<OtherErrorT, ErrorBase<OtherErrorT>>)
  // Implicit for easy construction on returns.
  explicit(false) ErrorOr(OtherErrorT other_err) : val_(other_err.ToError()) {}

  // Constructs with any convertible error type, necessary for return statements
  // that are already converting to the `ErrorOr` wrapper.
  //
  // This supports *explicitly* conversions, not just implicit, which is
  // important to make common patterns of returning and adjusting the error
  // type without each error type conversion needing to be implicit.
  template <typename OtherErrorT>
    requires(std::constructible_from<ErrorT, OtherErrorT> &&
             std::derived_from<OtherErrorT, ErrorBase<OtherErrorT>>)
  // Implicit for easy construction on returns.
  explicit(false) ErrorOr(OtherErrorT other_err)
      : val_(std::in_place_type<ErrorT>, std::move(other_err)) {}

  // Constructs with a reference.
  // Implicit for easy construction on returns.
  explicit(false) ErrorOr(T ref)
    requires std::is_reference_v<T>
      : val_(std::ref(ref)) {}

  // Constructs with a value.
  // Implicit for easy construction on returns.
  explicit(false) ErrorOr(T val)
    requires(!std::is_reference_v<T>)
      : val_(std::move(val)) {}

  // Returns true for success.
  auto ok() const -> bool { return std::holds_alternative<StoredT>(val_); }

  // Returns the contained error.
  // REQUIRES: `ok()` is false.
  auto error() const& -> const ErrorT& {
    CARBON_CHECK(!ok());
    return std::get<ErrorT>(val_);
  }
  auto error() && -> ErrorT {
    CARBON_CHECK(!ok());
    return std::get<ErrorT>(std::move(val_));
  }

  // Checks that `ok()` is true.
  // REQUIRES: `ok()` is true.
  auto Check() const -> void { CARBON_CHECK(ok(), "{0}", error()); }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  [[nodiscard]] auto operator*() & -> ValueT& {
    Check();
    return std::get<StoredT>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  [[nodiscard]] auto operator*() const& -> const ValueT& {
    Check();
    return std::get<StoredT>(val_);
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  [[nodiscard]] auto operator*() && -> ValueT&& {
    Check();
    return std::get<StoredT>(std::move(val_));
  }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator->() -> ValueT* { return &**this; }

  // Returns the contained value.
  // REQUIRES: `ok()` is true.
  auto operator->() const -> const ValueT* { return &**this; }

 private:
  using StoredT = std::conditional_t<std::is_reference_v<T>,
                                     std::reference_wrapper<ValueT>, T>;

  // Either an error message or a value.
  std::variant<ErrorT, StoredT> val_;
};

// A helper class for accumulating error message and converting to
// `Error` and `ErrorOr<T>`.
class ErrorBuilder {
 public:
  explicit ErrorBuilder() : out_(std::make_unique<RawStringOstream>()) {}

  ErrorBuilder(ErrorBuilder&&) = default;
  auto operator=(ErrorBuilder&&) -> ErrorBuilder& = default;

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
  explicit(false) operator Error() { return Error(out_->TakeStr()); }

  template <typename T>
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  explicit(false) operator ErrorOr<T>() {
    return Error(out_->TakeStr());
  }

 private:
  std::unique_ptr<RawStringOstream> out_;
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
