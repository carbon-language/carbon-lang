// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_COMMON_ERROR_H_
#define EXPLORER_COMMON_ERROR_H_

#include <optional>

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/source_location.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// A helper class for accumulating error message and converting to
// `Carbon::Error`/`Carbon::ErrorOr<T>`.
class ErrorBuilder {
 public:
  explicit ErrorBuilder(std::optional<SourceLocation> loc = std::nullopt)
      : out_(message_) {
    if (loc.has_value()) {
      out_ << *loc << ": ";
    }
  }

  // Accumulates string message.
  template <typename T>
  [[nodiscard]] auto operator<<(const T& message) -> ErrorBuilder& {
    out_ << message;
    return *this;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator Error() { return Error(message_); }

  template <typename V>
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit cast for returns.
  operator ErrorOr<V>() {
    return Error(message_);
  }

  std::string message_;
  llvm::raw_string_ostream out_;
};

// Builds a Carbon::Error instance with the specified message. This should be
// used for non-recoverable errors with user input.
//
// For example:
//   return FATAL_PROGRAM_ERROR(line_num) << "Line is bad!";
//   return FATAL_PROGRAM_ERROR_NO_LINE() << "Application is bad!";
//
// Where possible, try to identify the error as a compilation or
// runtime error. Use CHECK/FATAL for internal errors. The generic program
// error option is provided as a fallback for cases that don't fit those
// classifications.
//
// TODO: replace below macro invocations with direct `return ErrorBuilder() <<
// xx` calls.

#define FATAL_PROGRAM_ERROR_NO_LINE() \
  Carbon::ErrorBuilder() << "PROGRAM ERROR: "

#define FATAL_PROGRAM_ERROR(line) \
  FATAL_PROGRAM_ERROR_NO_LINE() << (line) << ": "

#define FATAL_COMPILATION_ERROR_NO_LINE() \
  Carbon::ErrorBuilder() << "COMPILATION ERROR: "

#define FATAL_COMPILATION_ERROR(line) \
  FATAL_COMPILATION_ERROR_NO_LINE() << (line) << ": "

#define FATAL_RUNTIME_ERROR_NO_LINE() \
  Carbon::ErrorBuilder() << "RUNTIME ERROR: "

#define FATAL_RUNTIME_ERROR(line) \
  FATAL_RUNTIME_ERROR_NO_LINE() << (line) << ": "

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

}  // namespace Carbon

#endif  // EXPLORER_COMMON_ERROR_H_
