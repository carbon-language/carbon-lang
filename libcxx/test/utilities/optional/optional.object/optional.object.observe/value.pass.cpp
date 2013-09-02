//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// T& optional<T>::value();

#include <optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

struct X
{
    X() = default;
    X(const X&) = delete;
    constexpr int test() const {return 3;}
    int test() {return 4;}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        std::optional<X> opt;
        opt.emplace();
        assert(opt.value().test() == 4);
    }
    {
        std::optional<X> opt;
        try
        {
            opt.value();
            assert(false);
        }
        catch (const std::bad_optional_access&)
        {
        }
    }
#endif  // _LIBCPP_STD_VER > 11
}
