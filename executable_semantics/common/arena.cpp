// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/arena.h"

namespace Carbon {

llvm::ManagedStatic<Arena> global_arena;

}  // namespace Carbon
