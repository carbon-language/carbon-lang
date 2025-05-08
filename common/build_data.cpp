// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/build_data.h"

#include "common/build_data_linkstamp.h"

namespace Carbon {

const llvm::StringRef BuildData::Platform = BuildDataLinkstamp::Platform;
const bool BuildData::BuildCoverageEnabled =
    BuildDataLinkstamp::BuildCoverageEnabled;
const llvm::StringRef BuildData::TargetName = BuildDataLinkstamp::TargetName;
const llvm::StringRef BuildData::BuildTarget = BuildDataLinkstamp::BuildTarget;

}  // namespace Carbon
