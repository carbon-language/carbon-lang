// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandlePackageApi(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageApi");
}

auto HandlePackageDirective(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageDirective");
}

auto HandlePackageImpl(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageImpl");
}

auto HandlePackageIntroducer(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageIntroducer");
}

auto HandlePackageLibrary(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageLibrary");
}

}  // namespace Carbon::Check
