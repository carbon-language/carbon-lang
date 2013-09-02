//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// constexpr const T& optional<T>::value() const;

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
        constexpr std::optional<X> opt(std::in_place);
        static_assert(opt.value().test() == 3, "");
    }
    {
        const std::optional<X> opt(std::in_place);
        assert(opt.value().test() == 3);
    }
    {
        const std::optional<X> opt;
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
