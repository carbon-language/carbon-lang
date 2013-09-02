//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// constexpr optional(T&& v);

#include <optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

class X
{
    int i_;
public:
    X(int i) : i_(i) {}
    X(X&& x) : i_(x.i_) {}

    friend bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

class Y
{
    int i_;
public:
    constexpr Y(int i) : i_(i) {}
    constexpr Y(Y&& x) : i_(x.i_) {}

    friend constexpr bool operator==(const Y& x, const Y& y) {return x.i_ == y.i_;}
};

class Z
{
    int i_;
public:
    Z(int i) : i_(i) {}
    Z(Z&&) {throw 6;}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        typedef int T;
        constexpr std::optional<T> opt(T(5));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 5, "");

        struct test_constexpr_ctor
            : public std::optional<T>
        {
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
    {
        typedef double T;
        constexpr std::optional<T> opt(T(3));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 3, "");

        struct test_constexpr_ctor
            : public std::optional<T>
        {
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
    {
        typedef X T;
        std::optional<T> opt(T(3));
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 3);
    }
    {
        typedef Y T;
        constexpr std::optional<T> opt(T(3));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 3, "");

        struct test_constexpr_ctor
            : public std::optional<T>
        {
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
    {
        typedef Z T;
        try
        {
            std::optional<T> opt(T(3));
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
        }
    }
#endif  // _LIBCPP_STD_VER > 11
}
