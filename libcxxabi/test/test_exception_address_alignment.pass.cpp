//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Test that the address of the exception object is properly aligned to the
// largest supported alignment for the system.

#include <cstdint>
#include <cassert>

struct __attribute__((aligned)) AlignedType {};
struct MinAligned {  };
static_assert(alignof(MinAligned) == 1 && sizeof(MinAligned) == 1, "");

int main() {
  for (int i=0; i < 10; ++i) {
    try {
      throw MinAligned{};
    } catch (MinAligned const& ref) {
      assert(reinterpret_cast<uintptr_t>(&ref) % alignof(AlignedType) == 0);
    }
  }
}
