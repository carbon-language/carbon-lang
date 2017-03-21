//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// constexpr common_type_t<duration> operator+() const;

#include <chrono>
#include <cassert>

#include <test_macros.h>

int main()
{
    {
    const std::chrono::minutes m(3);
    std::chrono::minutes m2 = +m;
    assert(m.count() == m2.count());
    }
#if TEST_STD_VER >= 11
    {
    constexpr std::chrono::minutes m(3);
    constexpr std::chrono::minutes m2 = +m;
    static_assert(m.count() == m2.count(), "");
    }
#endif
}
