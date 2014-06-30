//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R()>

// template<class F> function(F);

#define _LIBCPP_HAS_NO_VARIADICS
#include <functional>
#include <cassert>

int main()
{
    std::function<void()> f(static_cast<void(*)()>(0));
    assert(!f);
}
