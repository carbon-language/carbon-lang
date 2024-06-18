// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_VERSION_H_
#define CARBON_COMMON_VERSION_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon {

struct Version {
  static const int Major;
  static const int Minor;
  static const int Patch;

  static const llvm::StringLiteral String;

  // A dedicated version information string to use in the toolchain as its
  // command line rendered version. Composed centrally so it can be composed at
  // compile time with potentially build info stamped components.
  static const llvm::StringLiteral ToolchainInfo;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_VERSION_H_
