// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

// `import` and `package` are structured by parsing. As a consequence, no
// checking logic is needed here.

auto HandleImportIntroducer(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandlePackageIntroducer(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandleLibrary(Context& context, Parse::Node /*parse_node*/) -> bool {
  // Pop and discard the library name from the node stack.
  context.node_stack().Pop<Parse::NodeKind::Literal>();
  return true;
}

auto HandlePackageApi(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandlePackageImpl(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandleImportDirective(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  // Pop and discard the identifier from the node stack.
  context.node_stack().Pop<Parse::NodeKind::Name>();
  return true;
}

auto HandlePackageDirective(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  // Pop and discard the identifier from the node stack.
  context.node_stack().Pop<Parse::NodeKind::Name>();
  return true;
}

}  // namespace Carbon::Check
