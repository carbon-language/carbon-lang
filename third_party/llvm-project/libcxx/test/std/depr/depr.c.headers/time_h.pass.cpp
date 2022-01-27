//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <time.h>

#include <time.h>
#include <type_traits>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

#ifndef CLOCKS_PER_SEC
#error CLOCKS_PER_SEC not defined
#endif

int main(int, char**)
{
    clock_t c = 0; ((void)c);
    size_t s = 0;
    time_t t = 0;
    tm tmv = {};
    static_assert((std::is_same<decltype(clock()), clock_t>::value), "");
    static_assert((std::is_same<decltype(difftime(t,t)), double>::value), "");
    static_assert((std::is_same<decltype(mktime(&tmv)), time_t>::value), "");
    static_assert((std::is_same<decltype(time(&t)), time_t>::value), "");
    static_assert((std::is_same<decltype(asctime(&tmv)), char*>::value), "");
    static_assert((std::is_same<decltype(ctime(&t)), char*>::value), "");
    static_assert((std::is_same<decltype(gmtime(&t)), tm*>::value), "");
    static_assert((std::is_same<decltype(localtime(&t)), tm*>::value), "");
    char* c1 = 0;
    const char* c2 = 0;
    static_assert((std::is_same<decltype(strftime(c1,s,c2,&tmv)), size_t>::value), "");

  return 0;
}
