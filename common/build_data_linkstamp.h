// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_
#define CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_

namespace Carbon::BuildData::Internal {

// See build_data.h; the list of names here should match.
//
// Bazel will build dependencies on the `:build_data` library (which exposes
// `build_data_linkstamp.h`) throughout the build process, but
// `build_data_linkstamp.cpp` is compiled and linked per-binary -- essentially a
// separate library. In essence, `build_data_linkstamp.h` is exposing values
// that are assigned later (this has consequences like preventing `constexpr`
// use).
//
// Also, when build_data_linkstamp.cpp is compiled, this doesn't receive deps,
// so we can't use things like `llvm::StringRef` here. It should ideally be
// purely hermetic -- not even using STL for `string_view`. As a result, we use
// `build_data.h` as an intermediary to do a `StringRef` wrap.
extern const char platform[];
extern const bool build_coverage_enabled;
extern const char target_name[];
extern const char build_target[];

}  // namespace Carbon::BuildData::Internal

#endif  // CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_
