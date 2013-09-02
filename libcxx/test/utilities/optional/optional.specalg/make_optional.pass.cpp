//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// template <class T>
//   constexpr
//   optional<typename decay<T>::type>
//   make_optional(T&& v);

#include <optional>
#include <string>
#include <memory>
#include <cassert>

#if _LIBCPP_STD_VER > 11

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        std::optional<int> opt = std::make_optional(2);
        assert(*opt == 2);
    }
    {
        std::string s("123");
        std::optional<std::string> opt = std::make_optional(s);
        assert(*opt == s);
    }
    {
        std::string s("123");
        std::optional<std::string> opt = std::make_optional(std::move(s));
        assert(*opt == "123");
        assert(s.empty());
    }
    {
        std::unique_ptr<int> s(new int(3));
        std::optional<std::unique_ptr<int>> opt = std::make_optional(std::move(s));
        assert(**opt == 3);
        assert(s == nullptr);
    }
#endif  // _LIBCPP_STD_VER > 11
}
