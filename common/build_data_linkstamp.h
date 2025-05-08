// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_
#define CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_

namespace Carbon {

// See build_data.h; the list of names here should match. When
// build_data_linkstamp.cpp is compiled, this doesn't receive deps, so we can't
// use things like `llvm::StringLiteral`.
struct BuildDataLinkstamp {
  static const char Platform[];
  static const bool BuildCoverageEnabled;
  static const char TargetName[];
  static const char BuildTarget[];
};

}  // namespace Carbon

#endif  // CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_
