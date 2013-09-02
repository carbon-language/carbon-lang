//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// template <class U> T optional<T>::value_or(U&& v) &&;

#include <optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

struct Y
{
    int i_;

    Y(int i) : i_(i) {}
};

struct X
{
    int i_;

    X(int i) : i_(i) {}
    X(X&& x) : i_(x.i_) {x.i_ = 0;}
    X(const Y& y) : i_(y.i_) {}
    X(Y&& y) : i_(y.i_+1) {}
    friend constexpr bool operator==(const X& x, const X& y)
        {return x.i_ == y.i_;}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        std::optional<X> opt(std::in_place, 2);
        Y y(3);
        assert(std::move(opt).value_or(y) == 2);
        assert(*opt == 0);
    }
    {
        std::optional<X> opt(std::in_place, 2);
        assert(std::move(opt).value_or(Y(3)) == 2);
        assert(*opt == 0);
    }
    {
        std::optional<X> opt;
        Y y(3);
        assert(std::move(opt).value_or(y) == 3);
        assert(!opt);
    }
    {
        std::optional<X> opt;
        assert(std::move(opt).value_or(Y(3)) == 4);
        assert(!opt);
    }
#endif  // _LIBCPP_STD_VER > 11
}
