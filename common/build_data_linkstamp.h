// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_
#define CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_

#include <string_view>

namespace Carbon::BuildData::Internal {

// See build_data.h; the list of names here should match.
//
// These are exposed as non-constexpr so that `build_data_linkstamp.cpp` can be
// compiled at the end, even though dependencies are added earlier.
//
// Also, when build_data_linkstamp.cpp is compiled, this doesn't receive deps,
// so we can't use things like `llvm::StringRef` here. As a result, we use
// `build_data.h` as an intermediary to do a `StringRef` wrap.
extern const std::string_view platform;
extern const bool build_coverage_enabled;
extern const std::string_view target_name;
extern const std::string_view build_target;

}  // namespace Carbon::BuildData::Internal

#endif  // CARBON_COMMON_BUILD_DATA_LINKSTAMP_H_
