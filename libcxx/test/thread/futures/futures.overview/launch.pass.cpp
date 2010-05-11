//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
