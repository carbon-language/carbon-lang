//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cstring>

#include <cstring>
#include <type_traits>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

int main(int, char**)
{
    std::size_t s = 0;
    void* vp = 0;
    const void* vpc = 0;
    char* cp = 0;
    const char* cpc = 0;

    ASSERT_SAME_TYPE(void*,       decltype(std::memcpy(vp, vpc, s)));
    ASSERT_SAME_TYPE(void*,       decltype(std::memmove(vp, vpc, s)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strcpy(cp, cpc)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strncpy(cp, cpc, s)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strcat(cp, cpc)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strncat(cp, cpc, s)));
    ASSERT_SAME_TYPE(int,         decltype(std::memcmp(vpc, vpc, s)));
    ASSERT_SAME_TYPE(int,         decltype(std::strcmp(cpc, cpc)));
    ASSERT_SAME_TYPE(int,         decltype(std::strncmp(cpc, cpc, s)));
    ASSERT_SAME_TYPE(int,         decltype(std::strcoll(cpc, cpc)));
    ASSERT_SAME_TYPE(std::size_t, decltype(std::strxfrm(cp, cpc, s)));
    ASSERT_SAME_TYPE(void*,       decltype(std::memchr(vp, 0, s)));
    ASSERT_SAME_TYPE(const void*, decltype(std::memchr(vpc, 0, s)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strchr(cp, 0)));
    ASSERT_SAME_TYPE(const char*, decltype(std::strchr(cpc, 0)));
    ASSERT_SAME_TYPE(std::size_t, decltype(std::strcspn(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strpbrk(cp, cpc)));
    ASSERT_SAME_TYPE(const char*, decltype(std::strpbrk(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strrchr(cp, 0)));
    ASSERT_SAME_TYPE(const char*, decltype(std::strrchr(cpc, 0)));
    ASSERT_SAME_TYPE(std::size_t, decltype(std::strspn(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strstr(cp, cpc)));
    ASSERT_SAME_TYPE(const char*, decltype(std::strstr(cpc, cpc)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strtok(cp, cpc)));
    ASSERT_SAME_TYPE(void*,       decltype(std::memset(vp, 0, s)));
    ASSERT_SAME_TYPE(char*,       decltype(std::strerror(0)));
    ASSERT_SAME_TYPE(std::size_t, decltype(std::strlen(cpc)));

    return 0;
}
