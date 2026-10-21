// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/extract.h"

namespace Carbon::Parse {

#define CARBON_PARSE_NODE_KIND(KindName)
#define CARBON_PARSE_NODE_KIND_EXPRESSION(KindName)                    \
  template auto TreeAndSubtrees::TryExtractNodeFromChildren<KindName>( \
      NodeId node_id,                                                  \
      llvm::iterator_range<TreeAndSubtrees::SiblingIterator> children, \
      ErrorBuilder * trace) const -> std::optional<KindName>;
#include "toolchain/parse/node_kind.def"

}  // namespace Carbon::Parse
