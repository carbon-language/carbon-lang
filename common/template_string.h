// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TEMPLATE_STRING_H_
#define CARBON_COMMON_TEMPLATE_STRING_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Represents a compile-time string in a form suitable for use as a non-type
// template argument.
//
// These arguments are required to be a "structural type", and so we copy the
// string contents into a public array of `char`s. For details, see:
// https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter
//
// Designed to support implicitly deduced construction from a string literal
// template argument. This type will implicitly convert to an `llvm::StringRef`
// for accessing the string contents, and also provides a dedicated `c_str()`
// method to access the string as a C string.
//
// Example usage:
// ```cpp
// template <TemplateString Str> auto F() -> void {
//   llvm::cout() << Str;
// }
//
// auto Example() -> void {
//   F<"string contents here">();
// }
// ```
template <int N>
struct TemplateString {
  // Constructs the template string from a string literal.
  //
  // Intentionally allows implicit conversion from string literals for use as a
  // non-type template parameter.
  //
  // The closest we can get to explicitly accepting a string literal is to
  // accept an array of `const char`s, so we additionally use Clang's constexpr
  // `enable_if` attribute to require the array to be usable as a C string with
  // the expected length. This checks both for null-termination and no embedded
  // `0` bytes.
  //
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr TemplateString(const char (&str)[N + 1]) __attribute__((
      enable_if(__builtin_strlen(str) == N,
                "character array is not null-terminated valid C string"))) {
    // Rely on Clang's constexpr `__builtin_memcpy` to minimize compile time
    // overhead copying the string contents around.
    __builtin_memcpy(storage_, str, N + 1);
  }

  // This type is designed to act as a `StringRef` implicitly while having the
  // storage necessary to be used as a template parameter.
  //
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator llvm::StringRef() const {
    return llvm::StringRef(storage_, N);
  }

  // Accesses the string data directly as a compile-time C string.
  constexpr auto c_str() const -> const char* { return storage_; }

  // Note that this must be public for the type to be structural and available
  // as a template argument, but this is not part of the public API.
  char storage_[N + 1];
};

// Allow deducing `N` when implicitly constructing these so that we can directly
// use a string literal in a template argument. The array needs an extra char
// for the null terminator.
template <int M>
TemplateString(const char (&str)[M]) -> TemplateString<M - 1>;

}  // namespace Carbon

#endif  // CARBON_COMMON_TEMPLATE_STRING_H_
