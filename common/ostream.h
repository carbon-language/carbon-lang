// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_OSTREAM_H_
#define COMMON_OSTREAM_H_

#include <iosfwd>

#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Support raw_ostream << for types which implement:
//   void Print(llvm::raw_ostream& out) const;
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
auto operator<<(llvm::raw_ostream& out, const T& obj) -> llvm::raw_ostream& {
  obj.Print(out);
  return out;
}

// Prevents raw_ostream << for pointers to printable types.
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
__attribute__((unavailable(
    "Received a pointer to a printable type, are you missing a `*`? "
    "To print as a pointer, cast to void*."))) auto
operator<<(llvm::raw_ostream& out, const T* /*obj*/) -> llvm::raw_ostream&;

// Support std::ostream << for types which implement:
//   void Print(llvm::raw_ostream& out) const;
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
auto operator<<(std::ostream& out, const T& obj) -> std::ostream& {
  llvm::raw_os_ostream raw_os(out);
  obj.Print(raw_os);
  return out;
}

// Prevents std::ostream << for pointers to printable types.
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
__attribute__((unavailable(
    "Received a pointer to a printable type, are you missing a `*`? "
    "To print as a pointer, cast to void*."))) auto
operator<<(std::ostream& out, const T* /*obj*/) -> std::ostream&;

namespace Internal {

void PrintNullPointer(std::ostream& out);

}

// Allow GoogleTest and GoogleMock to print even pointers by dereferencing them.
// This is important to allow automatic printing of arguments of mocked APIs.
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
void PrintTo(const T* p, std::ostream* out) {
  // Handle null pointers directly.
  if (!p) {
    Internal::PrintNullPointer(*out);
    return;
  }

  // For non-null pointers, dereference and delegate.
  *out << *p;
}

}  // namespace Carbon

namespace llvm {

template <typename S, typename T,
          typename = std::enable_if_t<std::is_base_of_v<
              std::ostream, std::remove_reference_t<std::remove_cv_t<S>>>>,
          typename = std::enable_if_t<!std::is_same_v<
              std::remove_reference_t<std::remove_cv_t<T>>, raw_ostream>>>
S& operator<<(S& StandardOS, const T& Value) {
  raw_os_ostream(StandardOS) << Value;
  return StandardOS;
}

}  // namespace llvm

#endif  // COMMON_OSTREAM_H_
