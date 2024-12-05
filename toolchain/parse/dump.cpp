// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/dump.h"

#include "llvm/Support/raw_ostream.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse::DumpOverloads {

auto Dump(NodeId /* node_id */, const Tree& /* tree */) -> void {
  llvm::errs() << "A node id\n";
}

}  // namespace Carbon::Parse::DumpOverloads
