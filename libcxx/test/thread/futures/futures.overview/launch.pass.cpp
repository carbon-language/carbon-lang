//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// enum class launch
// {
//     any,
//     async,
//     sync
// };

#include <future>

int main()
{
    static_assert(std::launch::any == 0, "");
    static_assert(std::launch::async == 1, "");
    static_assert(std::launch::sync == 2, "");
}
