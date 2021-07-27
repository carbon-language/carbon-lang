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

// Returns a string containing the output of `value.Print`.
//
// Note that this typically won't be invocable in a debugger, because it's
// a template, so printable types should also provide a `DebugString` method.
// That method should not be inline, because that likewise prevents invoking
// it in a debugger. This function is provided primarily as an implementation
// convenience for those methods.
template <typename T, typename std::enable_if<std::is_member_function_pointer<
                          decltype(&T::Print)>::value>::type* = nullptr>
auto DebugString(const T& value) -> std::string {
  std::string result_string;
  llvm::raw_string_ostream stream(result_string);
  value.Print(stream);
  return result_string;
}

}  // namespace Carbon

#endif  // COMMON_OSTREAM_H_
