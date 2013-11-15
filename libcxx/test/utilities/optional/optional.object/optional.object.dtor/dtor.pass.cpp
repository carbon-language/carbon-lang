//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// ~optional();

#include <experimental/optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;

class X
{
public:
    static bool dtor_called;
    X() = default;
    ~X() {dtor_called = true;}
};

bool X::dtor_called = false;

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        typedef int T;
        static_assert(std::is_trivially_destructible<T>::value, "");
        static_assert(std::is_trivially_destructible<optional<T>>::value, "");
    }
    {
        typedef double T;
        static_assert(std::is_trivially_destructible<T>::value, "");
        static_assert(std::is_trivially_destructible<optional<T>>::value, "");
    }
    {
        typedef X T;
        static_assert(!std::is_trivially_destructible<T>::value, "");
        static_assert(!std::is_trivially_destructible<optional<T>>::value, "");
        {
            X x;
            optional<X> opt{x};
            assert(X::dtor_called == false);
        }
        assert(X::dtor_called == true);
    }
#endif  // _LIBCPP_STD_VER > 11
}
