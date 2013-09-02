//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// template <class U> constexpr T optional<T>::value_or(U&& v) const&;

#include <optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

struct Y
{
    int i_;

    constexpr Y(int i) : i_(i) {}
};

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
    constexpr X(const Y& y) : i_(y.i_) {}
    constexpr X(Y&& y) : i_(y.i_+1) {}
    friend constexpr bool operator==(const X& x, const X& y)
        {return x.i_ == y.i_;}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        constexpr std::optional<X> opt(2);
        constexpr Y y(3);
        static_assert(opt.value_or(y) == 2, "");
    }
    {
        constexpr std::optional<X> opt(2);
        static_assert(opt.value_or(Y(3)) == 2, "");
    }
    {
        constexpr std::optional<X> opt;
        constexpr Y y(3);
        static_assert(opt.value_or(y) == 3, "");
    }
    {
        constexpr std::optional<X> opt;
        static_assert(opt.value_or(Y(3)) == 4, "");
    }
    {
        const std::optional<X> opt(2);
        const Y y(3);
        assert(opt.value_or(y) == 2);
    }
    {
        const std::optional<X> opt(2);
        assert(opt.value_or(Y(3)) == 2);
    }
    {
        const std::optional<X> opt;
        const Y y(3);
        assert(opt.value_or(y) == 3);
    }
    {
        const std::optional<X> opt;
        assert(opt.value_or(Y(3)) == 4);
    }
#endif  // _LIBCPP_STD_VER > 11
}
