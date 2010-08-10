//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

int main()
{
    static_assert(sizeof(std::nullptr_t) == sizeof(void*),
                  "sizeof(std::nullptr_t) == sizeof(void*)");
    A* p = 0;
    assert(p == nullptr);
    assert(nullptr == p);
#if !((__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ <= 5))
    // GCC 4.2 through 4.5 can't handle this
    void (A::*pmf)() = 0;
    assert(pmf == nullptr);
#endif
    int A::*pmd = 0;
    assert(pmd == nullptr);
    A a1(nullptr);
    A a2(0);
    bool b = nullptr;
    assert(!b);
    assert(nullptr == nullptr);
    assert(nullptr <= nullptr);
    assert(nullptr >= nullptr);
    assert(!(nullptr != nullptr));
    assert(!(nullptr < nullptr));
    assert(!(nullptr > nullptr));
    assert(!(&a1 == nullptr));
    assert(!(nullptr == &a1));
    assert(&a1 != nullptr);
    assert(nullptr != &a1);
    assert(nullptr < &a1);
    assert(nullptr <= &a1);
    assert(!(nullptr < p));
    assert(nullptr <= p);
    assert(!(&a1 < nullptr));
    assert(!(&a1 <= nullptr));
    assert(!(p < nullptr));
    assert(p <= nullptr);
    assert(!(nullptr > &a1));
    assert(!(nullptr >= &a1));
    assert(!(nullptr > p));
    assert(nullptr >= p);
}
