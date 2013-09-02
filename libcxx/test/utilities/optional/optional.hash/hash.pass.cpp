//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// template <class T> struct hash<optional<T>>;

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
        typedef int T;
        std::optional<T> opt;
        assert(std::hash<std::optional<T>>{}(opt) == 0);
        opt = 2;
        assert(std::hash<std::optional<T>>{}(opt) == std::hash<T>{}(*opt));
    }
    {
        typedef std::string T;
        std::optional<T> opt;
        assert(std::hash<std::optional<T>>{}(opt) == 0);
        opt = std::string("123");
        assert(std::hash<std::optional<T>>{}(opt) == std::hash<T>{}(*opt));
    }
    {
        typedef std::unique_ptr<int> T;
        std::optional<T> opt;
        assert(std::hash<std::optional<T>>{}(opt) == 0);
        opt = std::unique_ptr<int>(new int(3));
        assert(std::hash<std::optional<T>>{}(opt) == std::hash<T>{}(*opt));
    }
#endif  // _LIBCPP_STD_VER > 11
}
