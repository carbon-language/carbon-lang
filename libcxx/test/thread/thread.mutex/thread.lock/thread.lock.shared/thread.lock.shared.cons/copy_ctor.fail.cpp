//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <shared_mutex>

// template <class Mutex> class shared_lock;

// shared_lock(shared_lock const&) = delete;

#include <shared_mutex>

#if _LIBCPP_STD_VER > 11
std::shared_mutex m;
#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    std::shared_lock<std::shared_mutex> lk0(m);
    std::shared_lock<std::shared_mutex> lk = lk0;
#else
#   error
#endif  // _LIBCPP_STD_VER > 11
}
