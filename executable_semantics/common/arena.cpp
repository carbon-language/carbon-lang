// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/arena.h"

namespace Carbon::ArenaInternal {

llvm::ManagedStatic<std::vector<std::unique_ptr<ArenaPtr>>> arena;

}  // namespace Carbon::ArenaInternal
