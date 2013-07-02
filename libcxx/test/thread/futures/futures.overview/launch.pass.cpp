//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// enum class launch
// {
//     async = 1,
//     deferred = 2,
//     any = async | deferred
// };

#include <future>
#include <cassert>

int main()
{
#ifdef _LIBCPP_HAS_NO_STRONG_ENUMS
    static_assert(static_cast<int>(std::launch::any) ==
                 (static_cast<int>(std::launch::async) | static_cast<int>(std::launch::deferred)), "");
#else
    static_assert(std::launch::any == (std::launch::async | std::launch::deferred), "");
    static_assert(std::launch(0) == (std::launch::async & std::launch::deferred), "");
    static_assert(std::launch::any == (std::launch::async ^ std::launch::deferred), "");
    static_assert(std::launch::deferred == ~std::launch::async, "");
    std::launch x = std::launch::async;
    x &= std::launch::deferred;
    assert(x == std::launch(0));
    x = std::launch::async;
    x |= std::launch::deferred;
    assert(x == std::launch::any);
    x ^= std::launch::deferred;
    assert(x == std::launch::async);
#endif
    static_assert(static_cast<int>(std::launch::async) == 1, "");
    static_assert(static_cast<int>(std::launch::deferred) == 2, "");
}
