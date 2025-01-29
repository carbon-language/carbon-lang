// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_CHECK_INTERNAL_H_
#define CARBON_COMMON_CHECK_INTERNAL_H_

#include "common/template_string.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Internal {

// Evaluates a condition in a CHECK. This diagnoses if the condition evaluates
// to the constant `true` or `false`.
[[clang::always_inline]] constexpr bool
// Trailing GNU function attributes are incompatible with trailing return types.
// Filed as https://github.com/llvm/llvm-project/issues/118697
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
CheckCondition(bool condition)
    __attribute__((diagnose_if(condition,
                               "CHECK condition is always true; replace with "
                               "static_assert if this is intended",
                               "error")))
    __attribute__((diagnose_if(!condition,
                               "CHECK condition is always false; replace with "
                               "CARBON_FATAL if this is intended",
                               "error"))) {
  return condition;
}

// Implements the check failure message printing.
//
// This is out-of-line and will arrange to stop the program, print any debugging
// information and this string. In `!NDEBUG` mode (`dbg` and `fastbuild`), check
// failures can be made non-fatal by a build flag, so this is not `[[noreturn]]`
// in that case.
//
// This API uses `const char*` C string arguments rather than `llvm::StringRef`
// because we know that these are available as C strings and passing them that
// way lets the code size of calling it be smaller: it only needs to materialize
// a single pointer argument for each. The runtime cost of re-computing the size
// should be minimal. The extra message however might not be compile-time
// guaranteed to be a C string so we use a normal `StringRef` there.
#ifdef NDEBUG
[[noreturn]]
#endif
auto CheckFailImpl(const char* kind, const char* file, int line,
                   const char* condition_str, llvm::StringRef extra_message)
    -> void;

// Allow converting format values; the default behaviour is to just pass them
// through.
template <typename T>
auto ConvertFormatValue(T&& t) -> T&& {
  return std::forward<T>(t);
}

// Convert enums to larger integers so that byte-sized enums are not confused
// with being chars and printed as invalid (or nul-terminating) characters.
// Scoped enums are explicitly converted to integers so they can be printed
// without the user writing a cast.
template <typename T>
  requires(std::is_enum_v<std::remove_reference_t<T>>)
auto ConvertFormatValue(T&& t) {
  if constexpr (std::is_signed_v<
                    std::underlying_type_t<std::remove_reference_t<T>>>) {
    return static_cast<int64_t>(t);
  } else {
    return static_cast<uint64_t>(t);
  }
}

// Prints a check failure, including rendering any user-provided message using
// a format string.
//
// Most of the parameters are passed as compile-time template strings to avoid
// runtime cost of parameter setup in optimized builds. Each of these are passed
// along to the underlying implementation to include in the final printed
// message.
//
// Any user-provided format string and values are directly passed to
// `llvm::formatv` which handles all of the formatting of output.
template <TemplateString Kind, TemplateString File, int Line,
          TemplateString ConditionStr, TemplateString FormatStr, typename... Ts>
#ifdef NDEBUG
[[noreturn]]
#endif
[[gnu::cold, clang::noinline]] auto
CheckFail(Ts&&... values) -> void {
  if constexpr (llvm::StringRef(FormatStr).empty()) {
    // Skip the format string rendering if empty. Note that we don't skip it
    // even if there are no values as we want to have consistent handling of
    // `{}`s in the format string. This case is about when there is no message
    // at all, just the condition.
    CheckFailImpl(Kind.c_str(), File.c_str(), Line, ConditionStr.c_str(), "");
  } else {
    CheckFailImpl(Kind.c_str(), File.c_str(), Line, ConditionStr.c_str(),
                  llvm::formatv(FormatStr.c_str(),
                                ConvertFormatValue(std::forward<Ts>(values))...)
                      .str());
  }
}

}  // namespace Carbon::Internal

// Evaluates the condition of a CHECK as a boolean value.
//
// This performs a contextual conversion to bool, diagnoses if the condition is
// always true or always false, and returns its value.
#define CARBON_INTERNAL_CHECK_CONDITION(cond) \
  (Carbon::Internal::CheckCondition(true && (cond)))

// Implements check messages without any formatted values.
//
// Passes each of the provided components of the message to the template
// parameters of the check failure printing function above, including an empty
// string for the format string. Because there are multiple template arguments,
// the entire call is wrapped in parentheses.
#define CARBON_INTERNAL_CHECK_IMPL(kind, file, line, condition_str) \
  (Carbon::Internal::CheckFail<kind, file, line, condition_str, "">())

// Implements check messages with a format string and potentially formatted
// values.
//
// Each of the main components is passed as a template arguments, and then any
// formatted values are passed as arguments. Because there are multiple template
// arguments, the entire call is wrapped in parentheses.
#define CARBON_INTERNAL_CHECK_IMPL_FORMAT(kind, file, line, condition_str,   \
                                          format_str, ...)                   \
  (Carbon::Internal::CheckFail<kind, file, line, condition_str, format_str>( \
      __VA_ARGS__))

// Implements the failure of a check.
//
// Collects all the metadata about the failure to be printed, such as source
// location and stringified condition, and passes those, any format string and
// formatted arguments to the correct implementation macro above.
#define CARBON_INTERNAL_CHECK(condition, ...)      \
  CARBON_INTERNAL_CHECK_IMPL##__VA_OPT__(_FORMAT)( \
      "CHECK", __FILE__, __LINE__, #condition __VA_OPT__(, ) __VA_ARGS__)

// Implements the fatal macro.
//
// Similar to the check failure macro, but tags the message as a fatal one and
// leaves the stringified condition empty.
#define CARBON_INTERNAL_FATAL(...)                                  \
  (CARBON_INTERNAL_CHECK_IMPL##__VA_OPT__(_FORMAT)(                 \
       "FATAL", __FILE__, __LINE__, "" __VA_OPT__(, ) __VA_ARGS__), \
   CARBON_INTERNAL_FATAL_NORETURN_SUFFIX())

#ifdef NDEBUG
// For `DCHECK` in optimized builds we have a dead check that we want to
// potentially "use" arguments, but otherwise have the minimal overhead. We
// avoid forming interesting format strings here so that we don't have to
// repeatedly instantiate the `Check` function above. This format string would
// be an error if actually used.
#define CARBON_INTERNAL_DEAD_DCHECK(condition, ...) \
  CARBON_INTERNAL_DEAD_DCHECK_IMPL##__VA_OPT__(_FORMAT)(__VA_ARGS__)

#define CARBON_INTERNAL_DEAD_DCHECK_IMPL() \
  Carbon::Internal::CheckFail<"", "", 0, "", "">()

#define CARBON_INTERNAL_DEAD_DCHECK_IMPL_FORMAT(format_str, ...) \
  Carbon::Internal::CheckFail<"", "", 0, "", "">(__VA_ARGS__)

// The CheckFail function itself is noreturn in NDEBUG.
#define CARBON_INTERNAL_FATAL_NORETURN_SUFFIX() void()
#else
#define CARBON_INTERNAL_FATAL_NORETURN_SUFFIX() std::abort()
#endif

#endif  // CARBON_COMMON_CHECK_INTERNAL_H_
