//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <shared_mutex>

// class shared_timed_mutex;

// shared_timed_mutex();

#include <shared_mutex>

int main()
{
#if _LIBCPP_STD_VER > 11
    std::shared_timed_mutex m;
#endif
}
