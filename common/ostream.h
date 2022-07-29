// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_OSTREAM_H_
#define CARBON_COMMON_OSTREAM_H_

#include <ostream>

#include "common/metaprogramming.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// True if T has a method `void Print(llvm::raw_ostream& out) const`.
template <typename T>
static constexpr bool HasPrintMethod = Requires<const T, llvm::raw_ostream>(
    [](auto&& t, auto&& out) -> decltype(t.Print(out)) {});

// Support raw_ostream << for types which implement:
//   void Print(llvm::raw_ostream& out) const;
template <typename T, typename = std::enable_if_t<HasPrintMethod<T>>>
auto operator<<(llvm::raw_ostream& out, const T& obj) -> llvm::raw_ostream& {
  obj.Print(out);
  return out;
}

// Prevents raw_ostream << for pointers to printable types.
template <typename T, typename = std::enable_if_t<HasPrintMethod<T>>>
__attribute__((unavailable(
    "Received a pointer to a printable type, are you missing a `*`? "
    "To print as a pointer, cast to void*."))) auto
operator<<(llvm::raw_ostream& out, const T* /*obj*/) -> llvm::raw_ostream&;

// Support std::ostream << for types which implement:
//   void Print(llvm::raw_ostream& out) const;
template <typename T, typename = std::enable_if_t<HasPrintMethod<T>>>
auto operator<<(std::ostream& out, const T& obj) -> std::ostream& {
  llvm::raw_os_ostream raw_os(out);
  obj.Print(raw_os);
  return out;
}

// Prevents std::ostream << for pointers to printable types.
template <typename T, typename = std::enable_if_t<HasPrintMethod<T>>>
__attribute__((unavailable(
    "Received a pointer to a printable type, are you missing a `*`? "
    "To print as a pointer, cast to void*."))) auto
operator<<(std::ostream& out, const T* /*obj*/) -> std::ostream&;

// Allow GoogleTest and GoogleMock to print even pointers by dereferencing them.
// This is important to allow automatic printing of arguments of mocked APIs.
template <typename T, typename = std::enable_if_t<HasPrintMethod<T>>>
void PrintTo(T* p, std::ostream* out) {
  *out << static_cast<const void*>(p);

  // Also print the object if non-null.
  if (p) {
    *out << " pointing to " << *p;
  }
}

}  // namespace Carbon

namespace llvm {

// Injects an `operator<<` overload into the `llvm` namespace which detects LLVM
// types with `raw_ostream` overloads and uses that to map to a `std::ostream`
// overload. This allows LLVM types to be printed to `std::ostream` via their
// `raw_ostream` operator overloads, which is needed both for logging and
// testing.
//
// To make this overload be unusually low priority, it is designed to take even
// the `std::ostream` parameter as a template, and SFINAE disable itself unless
// that template parameter matches `std::ostream`. This ensures that an
// *explicit* operator will be preferred when provided. Some LLVM types may have
// this, and so we want to prioritize accordingly.
//
// It would be slightly cleaner for LLVM itself to provide this overload in
// `raw_os_ostream.h` so that we wouldn't need to inject into LLVM's namespace,
// but supporting `std::ostream` isn't a priority for LLVM so we handle it
// locally instead.
template <typename S, typename T,
          typename = std::enable_if_t<std::is_base_of_v<
              std::ostream, std::remove_reference_t<std::remove_cv_t<S>>>>,
          typename = std::enable_if_t<!std::is_same_v<
              std::remove_reference_t<std::remove_cv_t<T>>, raw_ostream>>>
auto operator<<(S& standard_out, const T& value) -> S& {
  raw_os_ostream(standard_out) << value;
  return standard_out;
}

}  // namespace llvm

#endif  // CARBON_COMMON_OSTREAM_H_
