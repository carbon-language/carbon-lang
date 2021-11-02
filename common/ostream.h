// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_OSTREAM_H_
#define COMMON_OSTREAM_H_

#include <type_traits>

#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Support ostream << for types which implement:
//   void Print(llvm::raw_ostream& out) const;
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
auto operator<<(llvm::raw_ostream& out, const T& obj) -> llvm::raw_ostream& {
  obj.Print(out);
  return out;
}

// Prevents ostream << for pointers to printable types.
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
__attribute__((unavailable(
    "Received a pointer to a printable type, are you missing a `*`? "
    "To print as a pointer, cast to void*."))) auto
operator<<(llvm::raw_ostream& out, const T* /*obj*/) -> llvm::raw_ostream&;

template <typename T, typename = void>
struct IsPrintableImpl : public std::false_type {};

template <typename T>
struct IsPrintableImpl<T,
                       std::void_t<decltype(std::declval<llvm::raw_ostream&>()
                                            << std::declval<const T&>())>>
    : public std::true_type {};

// Type trait which is true if T can be streamed to an llvm::raw_ostream.
template <typename T>
constexpr bool IsPrintable = IsPrintableImpl<T>::value;

}  // namespace Carbon

#endif  // COMMON_OSTREAM_H_
