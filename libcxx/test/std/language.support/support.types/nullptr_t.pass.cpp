//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <type_traits>
#include <cassert>

// typedef decltype(nullptr) nullptr_t;

struct A
{
    A(std::nullptr_t) {}
};

template <class T>
void test_conversions()
{
    {
        T p = 0;
        assert(p == nullptr);
    }
    {
        T p = nullptr;
        assert(p == nullptr);
        assert(nullptr == p);
        assert(!(p != nullptr));
        assert(!(nullptr != p));
    }
}

template <class T>
void test_comparisons()
{
    T p = nullptr;
    assert(p == nullptr);
    assert(p <= nullptr);
    assert(p >= nullptr);
    assert(!(p != nullptr));
    assert(!(p < nullptr));
    assert(!(p > nullptr));
    assert(nullptr == p);
    assert(nullptr <= p);
    assert(nullptr >= p);
    assert(!(nullptr != p));
    assert(!(nullptr < p));
    assert(!(nullptr > p));
}


int main()
{
    static_assert(sizeof(std::nullptr_t) == sizeof(void*),
                  "sizeof(std::nullptr_t) == sizeof(void*)");

    {
        test_conversions<std::nullptr_t>();
        test_conversions<void*>();
        test_conversions<A*>();
        test_conversions<void(*)()>();
        test_conversions<void(A::*)()>();
        test_conversions<int A::*>();
    }
    {
        test_comparisons<std::nullptr_t>();
        test_comparisons<void*>();
        test_comparisons<A*>();
        test_comparisons<void(*)()>();
    }
    {
        bool b = nullptr;
        assert(!b);
    }
}
