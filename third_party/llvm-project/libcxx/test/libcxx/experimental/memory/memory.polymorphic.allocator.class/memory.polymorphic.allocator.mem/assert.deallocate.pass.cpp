//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// T* polymorphic_allocator<T>::deallocate(T*, size_t size)

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "check_assertion.h"
#include "test_memory_resource.h"

namespace ex = std::experimental::pmr;

int main(int, char**) {
    using Alloc = ex::polymorphic_allocator<int>;
    using Traits = std::allocator_traits<Alloc>;
    NullResource R;
    Alloc a(&R);
    const std::size_t maxSize = Traits::max_size(a);

    a.deallocate(nullptr, maxSize); // no assertion
    TEST_LIBCPP_ASSERT_FAILURE(a.deallocate(nullptr, maxSize + 1), "deallocate called for size which exceeds max_size()");

    return 0;
}
