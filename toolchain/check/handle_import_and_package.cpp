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

auto HandleImportDirective(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandleLibraryIntroducer(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandleLibraryDirective(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandlePackageIntroducer(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandlePackageDirective(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandleLibrarySpecifier(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandlePackageName(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandleLibraryName(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

auto HandleDefaultLibrary(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
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

}  // namespace Carbon::Check
