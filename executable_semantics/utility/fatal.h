// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_UTILITY_FATAL_H_
#define EXECUTABLE_SEMANTICS_UTILITY_FATAL_H_

namespace Carbon {

// Prints the arguments to std::cerr and exits with code 255.
template <class First, class... Rest>
[[noreturn]] auto fatal(const First& first, const Rest&... rest) -> void {
  std::cerr << first;
  ((std::cerr << rest), ...);
  std::cerr << "\n";
  std::exit(-1);
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_UTILITY_FATAL_H_
