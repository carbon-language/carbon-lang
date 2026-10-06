// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_hashtable.h"

#include <cstddef>

namespace Carbon::RawHashtable {

#ifndef NDEBUG
// A global variable whose address seeds the iteration entropy pool. This allows
// ASLR to introduce some variation in debug iteration order when enabled via
// the code model for globals.
volatile std::byte global_addr_seed{1};

std::atomic<HashCode> entropy_hash =
    Carbon::HashValue(reinterpret_cast<uint64_t>(&global_addr_seed));
#endif

}  // namespace Carbon::RawHashtable
