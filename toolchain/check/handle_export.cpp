// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Check {

auto HandleExportIntroducer(Context& /*context*/,
                            Parse::ExportIntroducerId /*node_id*/) -> bool {
  return true;
}

auto HandleExportDirective(Context& context, Parse::ExportDirectiveId node_id)
    -> bool {
  return context.TODO(node_id, "ExportDirective");
}

}  // namespace Carbon::Check
