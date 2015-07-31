//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/any>

// Check that the size and alignment of any are what we expect.

#include <experimental/any>
#include "any_helpers.h"

class SmallThrowsDtor
{
public:
    SmallThrowsDtor() {}
    SmallThrowsDtor(SmallThrowsDtor const &) noexcept {}
    SmallThrowsDtor(SmallThrowsDtor &&) noexcept {}
    ~SmallThrowsDtor() noexcept(false) {}
};

int main()
{
    using std::experimental::any;
    using std::experimental::__any_imp::_IsSmallObject;
    static_assert(_IsSmallObject<small>::value, "");
    static_assert(_IsSmallObject<void*>::value, "");
    static_assert(!_IsSmallObject<SmallThrowsDtor>::value, "");
    static_assert(!_IsSmallObject<large>::value, "");
    // long double is over aligned.
    static_assert(sizeof(long double) <= sizeof(void*) * 3, "");
    static_assert(alignof(long double) > alignof(void*), "");
    static_assert(!_IsSmallObject<long double>::value, "");
}
