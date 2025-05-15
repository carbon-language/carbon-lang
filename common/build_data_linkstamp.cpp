// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/build_data_linkstamp.h"

namespace Carbon::BuildData::Internal {

const char platform[] = GPLATFORM;
const bool build_coverage_enabled = BUILD_COVERAGE_ENABLED;
const char target_name[] = G3_TARGET_NAME;
const char build_target[] = G3_BUILD_TARGET;

}  // namespace Carbon::BuildData::Internal
