//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// optional<T>& operator=(nullopt_t) noexcept;

#include <optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

struct X
{
    static bool dtor_called;
    ~X() {dtor_called = true;}
};

bool X::dtor_called = false;

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        std::optional<int> opt;
        static_assert(noexcept(opt = std::nullopt) == true, "");
        opt = std::nullopt;
        assert(static_cast<bool>(opt) == false);
    }
    {
        std::optional<int> opt(3);
        opt = std::nullopt;
        assert(static_cast<bool>(opt) == false);
    }
    {
        std::optional<X> opt;
        static_assert(noexcept(opt = std::nullopt) == true, "");
        assert(X::dtor_called == false);
        opt = std::nullopt;
        assert(X::dtor_called == false);
        assert(static_cast<bool>(opt) == false);
    }
    {
        X x;
        {
            std::optional<X> opt(x);
            assert(X::dtor_called == false);
            opt = std::nullopt;
            assert(X::dtor_called == true);
            assert(static_cast<bool>(opt) == false);
        }
    }
#endif  // _LIBCPP_STD_VER > 11
}
