// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashing.h"

namespace Carbon {

auto Hasher::HashSizedBytesLarge(llvm::ArrayRef<std::byte> bytes) -> void {
  const std::byte* data_ptr = bytes.data();
  const ssize_t size = bytes.size();
  CARBON_DCHECK(size > 32);

  // If we have 64 bytes or more, we're going to handle two 32-byte chunks at a
  // time using a simplified version of the main algorithm. This is based
  // heavily on the 64-byte and larger processing approach used by Abseil. The
  // goal is to mix the input data using as few multiplies (or other operations)
  // as we can and with as much [ILP][1] as we can. The ILP comes largely from
  // creating parallel structures to the operations.
  //
  // [1]: https://en.wikipedia.org/wiki/Instruction-level_parallelism
  auto mix32 = [](const std::byte* data_ptr, uint64_t buffer, uint64_t random0,
                  uint64_t random1) {
    uint64_t a = Read8(data_ptr);
    uint64_t b = Read8(data_ptr + 8);
    uint64_t c = Read8(data_ptr + 16);
    uint64_t d = Read8(data_ptr + 24);
    uint64_t m0 = Mix(a ^ random0, b ^ buffer);
    uint64_t m1 = Mix(c ^ random1, d ^ buffer);
    return (m0 ^ m1);
  };

  // Prefetch the first bytes into cache.
  __builtin_prefetch(data_ptr, 0 /* read */, 0 /* discard after next use */);

  uint64_t buffer0 = buffer ^ StaticRandomData[0];
  uint64_t buffer1 = buffer ^ StaticRandomData[2];
  const std::byte* tail_32b_ptr = data_ptr + (size - 32);
  const std::byte* tail_16b_ptr = data_ptr + (size - 16);
  const std::byte* end_ptr = data_ptr + (size - 64);
  while (data_ptr < end_ptr) {
    // Prefetch the next 64-bytes while we process the current 64-bytes.
    __builtin_prefetch(data_ptr + 64, 0 /* read */,
                       0 /* discard after next use */);

    buffer0 =
        mix32(data_ptr, buffer0, StaticRandomData[4], StaticRandomData[5]);
    buffer1 =
        mix32(data_ptr + 32, buffer1, StaticRandomData[6], StaticRandomData[7]);

    data_ptr += 64;
  }

  // If we haven't reached our 32-byte tail pointer, consume another 32-bytes
  // directly.
  if (data_ptr < tail_32b_ptr) {
    buffer0 =
        mix32(data_ptr, buffer0, StaticRandomData[4], StaticRandomData[5]);
    data_ptr += 32;
  }

  if (data_ptr < tail_16b_ptr) {
    // We have more than 16-bytes in the tail so use a full 32-byte mix from the
    // 32-byte tail pointer.
    buffer1 =
        mix32(tail_32b_ptr, buffer1, StaticRandomData[6], StaticRandomData[7]);
  } else {
    // 16-bytes or less in the tail, do something more minimal instead of a full
    // 32-byte mix. As this only involves a single multiply, we don't decompose
    // further even when the tail is (much) shorter.
    buffer1 = Mix(Read8(tail_16b_ptr) ^ StaticRandomData[6],
                  Read8(tail_16b_ptr + 8) ^ buffer1);
  }

  buffer = buffer0 ^ buffer1;
  HashDense(size);
}

}  // namespace Carbon
