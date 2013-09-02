//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// template <class... Args>
//   constexpr explicit optional(in_place_t, Args&&... args);

#include <optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

class X
{
    int i_;
    int j_ = 0;
public:
    X() : i_(0) {}
    X(int i) : i_(i) {}
    X(int i, int j) : i_(i), j_(j) {}

    ~X() {}

    friend bool operator==(const X& x, const X& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

class Y
{
    int i_;
    int j_ = 0;
public:
    constexpr Y() : i_(0) {}
    constexpr Y(int i) : i_(i) {}
    constexpr Y(int i, int j) : i_(i), j_(j) {}

    friend constexpr bool operator==(const Y& x, const Y& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

class Z
{
    int i_;
public:
    Z(int i) : i_(i) {throw 6;}
};


#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        constexpr std::optional<int> opt(std::in_place, 5);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 5, "");

        struct test_constexpr_ctor
            : public std::optional<int>
        {
            constexpr test_constexpr_ctor(std::in_place_t, int i) 
                : std::optional<int>(std::in_place, i) {}
        };

    }
    {
        const std::optional<X> opt(std::in_place);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X());
    }
    {
        const std::optional<X> opt(std::in_place, 5);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(5));
    }
    {
        const std::optional<X> opt(std::in_place, 5, 4);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(5, 4));
    }
    {
        constexpr std::optional<Y> opt(std::in_place);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == Y(), "");

        struct test_constexpr_ctor
            : public std::optional<Y>
        {
            constexpr test_constexpr_ctor(std::in_place_t) 
                : std::optional<Y>(std::in_place) {}
        };

    }
    {
        constexpr std::optional<Y> opt(std::in_place, 5);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == Y(5), "");

        struct test_constexpr_ctor
            : public std::optional<Y>
        {
            constexpr test_constexpr_ctor(std::in_place_t, int i) 
                : std::optional<Y>(std::in_place, i) {}
        };

    }
    {
        constexpr std::optional<Y> opt(std::in_place, 5, 4);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == Y(5, 4), "");

        struct test_constexpr_ctor
            : public std::optional<Y>
        {
            constexpr test_constexpr_ctor(std::in_place_t, int i, int j) 
                : std::optional<Y>(std::in_place, i, j) {}
        };

    }
    {
        try
        {
            const std::optional<Z> opt(std::in_place, 1);
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
        }
    }
#endif  // _LIBCPP_STD_VER > 11
}
