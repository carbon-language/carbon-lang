//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: sanitizer-new-delete

// Libcxx when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

// test operator new replacement

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <limits>

#include "test_macros.h"

constexpr auto OverAligned = __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2;

bool A_constructed = false;

struct alignas(OverAligned) A {
    A() {A_constructed = true;}
    ~A() {A_constructed = false;}
};


bool B_constructed = false;

struct alignas(std::max_align_t) B
{
    std::max_align_t member;
    B() {B_constructed = true;}
    ~B() {B_constructed = false;}
};

int new_called = 0;

alignas(OverAligned) char DummyData[OverAligned];

void* operator new(std::size_t s, std::align_val_t a) TEST_THROW_SPEC(std::bad_alloc)
{
    assert(new_called == 0); // We already allocated
    assert(s <= sizeof(DummyData));
    assert(static_cast<std::size_t>(a) == OverAligned);
    ++new_called;
    void *Ret = DummyData;
    DoNotOptimize(Ret);
    return Ret;
}

void  operator delete(void* p, std::align_val_t) TEST_NOEXCEPT
{
    assert(new_called == 1);
    --new_called;
    assert(p == DummyData);
    DoNotOptimize(DummyData);
}


int main(int, char**)
{
    {
        A* ap = new A;
        assert(ap);
        assert(A_constructed);
        assert(new_called);
        delete ap;
        assert(!A_constructed);
        assert(!new_called);
    }
    {
        B* bp = new B;
        assert(bp);
        assert(B_constructed);
        assert(!new_called);
        delete bp;
        assert(!new_called);
    }

  return 0;
}
