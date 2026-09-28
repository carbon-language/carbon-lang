// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_hashtable.h"

#include <cstddef>

namespace Carbon::RawHashtable {

volatile std::byte global_addr_seed{1};

#ifndef NDEBUG
std::atomic<HashCode> entropy_hash =
    Carbon::HashValue(reinterpret_cast<uint64_t>(&global_addr_seed));
#endif

}  // namespace Carbon::RawHashtable
