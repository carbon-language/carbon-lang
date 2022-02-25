//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T N>
//   using make_integer_sequence = integer_sequence<T, 0, 1, ..., N-1>;

// UNSUPPORTED: c++03, c++11

// This test hangs during recursive template instantiation with libstdc++
// UNSUPPORTED: libstdc++

#include <utility>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
  typedef std::make_integer_sequence<int, -3> MakeSeqT;

  // std::make_integer_sequence is implemented using a compiler builtin if available.
  // this builtin has different diagnostic messages than the fallback implementation.
#if TEST_HAS_BUILTIN(__make_integer_seq) && !defined(_LIBCPP_TESTING_FALLBACK_MAKE_INTEGER_SEQUENCE)
  MakeSeqT i; // expected-error@*:* {{integer sequences must have non-negative sequence length}}
#else
  MakeSeqT i; // expected-error@*:* {{static_assert failed "std::make_integer_sequence must have a non-negative sequence length"}}
#endif

  return 0;
}
