// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_OSTREAM_H_
#define COMMON_OSTREAM_H_

#include "llvm/Support/raw_ostream.h"

namespace Carbon {

template <typename T>
using HasPrintMethod = typename std::enable_if_t<
    std::is_member_function_pointer_v<decltype(&T::Print)>>;

// Support ostream << for types which implement:
//   void Print(llvm::raw_ostream& out) const;
template <typename T, HasPrintMethod<T>* = nullptr>
auto operator<<(llvm::raw_ostream& out, const T& obj) -> llvm::raw_ostream& {
  obj.Print(out);
  return out;
}

// Prevents ostream << for pointers to printable types.
template <typename T, HasPrintMethod<T>* = nullptr>
__attribute__((unavailable(
    "Received a pointer to a printable type, are you missing a `*`? "
    "To print as a pointer, cast to void*."))) auto
operator<<(llvm::raw_ostream& out, const T* /*obj*/) -> llvm::raw_ostream&;

}  // namespace Carbon

#endif  // COMMON_OSTREAM_H_
