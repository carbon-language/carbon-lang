//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// T* polymorphic_allocator<T>::deallocate(T*, size_t size)

int AssertCount = 0;

// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : (void)::AssertCount++)
#define _LIBCPP_DEBUG 0
#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_memory_resource.h"

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int main(int, char**)
{
    using Alloc = NullAllocator<char>;

    AllocController P;
    ex::resource_adaptor<Alloc> r(Alloc{P});
    ex::memory_resource & m1 = r;

    std::size_t maxSize = std::numeric_limits<std::size_t>::max()
                            - alignof(std::max_align_t);

    m1.deallocate(nullptr, maxSize);
    assert(AssertCount == 0);
    m1.deallocate(nullptr, maxSize + 1);
    assert(AssertCount >= 1);

  return 0;
}
