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

#include <experimental/optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;
using std::experimental::in_place_t;
using std::experimental::in_place;

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
        constexpr optional<int> opt(in_place, 5);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 5, "");

        struct test_constexpr_ctor
            : public optional<int>
        {
            constexpr test_constexpr_ctor(in_place_t, int i) 
                : optional<int>(in_place, i) {}
        };

    }
    {
        const optional<X> opt(in_place);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X());
    }
    {
        const optional<X> opt(in_place, 5);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(5));
    }
    {
        const optional<X> opt(in_place, 5, 4);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(5, 4));
    }
    {
        constexpr optional<Y> opt(in_place);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == Y(), "");

        struct test_constexpr_ctor
            : public optional<Y>
        {
            constexpr test_constexpr_ctor(in_place_t) 
                : optional<Y>(in_place) {}
        };

    }
    {
        constexpr optional<Y> opt(in_place, 5);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == Y(5), "");

        struct test_constexpr_ctor
            : public optional<Y>
        {
            constexpr test_constexpr_ctor(in_place_t, int i) 
                : optional<Y>(in_place, i) {}
        };

    }
    {
        constexpr optional<Y> opt(in_place, 5, 4);
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == Y(5, 4), "");

        struct test_constexpr_ctor
            : public optional<Y>
        {
            constexpr test_constexpr_ctor(in_place_t, int i, int j) 
                : optional<Y>(in_place, i, j) {}
        };

    }
    {
        try
        {
            const optional<Z> opt(in_place, 1);
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
        }
    }
#endif  // _LIBCPP_STD_VER > 11
}
