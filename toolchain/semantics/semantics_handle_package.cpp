// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandlePackageApi(SemanticsContext& context,
                               ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageApi");
}

auto SemanticsHandlePackageDirective(SemanticsContext& context,
                                     ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageDirective");
}

auto SemanticsHandlePackageImpl(SemanticsContext& context,
                                ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageImpl");
}

auto SemanticsHandlePackageIntroducer(SemanticsContext& context,
                                      ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageIntroducer");
}

auto SemanticsHandlePackageLibrary(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePackageLibrary");
}

}  // namespace Carbon
