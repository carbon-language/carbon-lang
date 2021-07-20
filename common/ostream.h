// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_OSTREAM_H_
#define COMMON_OSTREAM_H_

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

}  // namespace Carbon

#endif  // COMMON_OSTREAM_H_
