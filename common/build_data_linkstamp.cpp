// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/build_data_linkstamp.h"

namespace Carbon {

const char BuildDataLinkstamp::Platform[] = GPLATFORM;
const bool BuildDataLinkstamp::BuildCoverageEnabled = BUILD_COVERAGE_ENABLED;
const char BuildDataLinkstamp::TargetName[] = G3_TARGET_NAME;
const char BuildDataLinkstamp::BuildTarget[] = G3_BUILD_TARGET;

}  // namespace Carbon
