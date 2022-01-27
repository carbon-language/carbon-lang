//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: c++03

// The <unwind.h> header provided in the SDK of older Xcodes used to provide
// an incorrectly aligned _Unwind_Exception type on non-ARM. That causes these
// tests to fail when compiling against such a SDK, or when running against a
// system libc++abi that was compiled with an incorrect definition of _Unwind_Exception.
// XFAIL: apple-clang-12.0.0 && !target={{arm.*}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// Test that the address of the exception object is properly aligned as required
// by the relevant ABI

#include <cstdint>
#include <cassert>
#include <__cxxabi_config.h>

#include <unwind.h>

struct __attribute__((aligned)) AlignedType {};

// EHABI  : 8-byte aligned
// Itanium: Largest supported alignment for the system
#if defined(_LIBCXXABI_ARM_EHABI)
#  define EXPECTED_ALIGNMENT 8
#else
#  define EXPECTED_ALIGNMENT alignof(AlignedType)
#endif

static_assert(alignof(_Unwind_Exception) == EXPECTED_ALIGNMENT,
  "_Unwind_Exception is incorrectly aligned. This test is expected to fail");

struct MinAligned {  };
static_assert(alignof(MinAligned) == 1 && sizeof(MinAligned) == 1, "");

int main(int, char**) {
  for (int i=0; i < 10; ++i) {
    try {
      throw MinAligned{};
    } catch (MinAligned const& ref) {
      assert(reinterpret_cast<uintptr_t>(&ref) % EXPECTED_ALIGNMENT == 0);
    }
  }

  return 0;
}
