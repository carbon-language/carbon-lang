//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <shared_mutex>

// class shared_mutex;

// shared_mutex& operator=(const shared_mutex&) = delete;

#include <shared_mutex>

#include "test_macros.h"

int main()
{
#if TEST_STD_VER > 14
    std::shared_mutex m0;
    std::shared_mutex m1;
    m1 = m0;
#else
#   error
#endif
}
