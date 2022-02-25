//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wstring_convert(const byte_string& byte_err,
//                 const wide_string& wide_err = wide_string());

#include <locale>
#include <codecvt>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::codecvt_utf8<wchar_t> Codecvt;
    typedef std::wstring_convert<Codecvt> Myconv;
#if TEST_STD_VER > 11
    static_assert(!std::is_convertible<std::string, Myconv>::value, "");
    static_assert( std::is_constructible<Myconv, std::string>::value, "");
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        Myconv myconv;
        try
        {
            TEST_IGNORE_NODISCARD myconv.to_bytes(L"\xDA83");
            assert(false);
        }
        catch (const std::range_error&)
        {
        }
        try
        {
            TEST_IGNORE_NODISCARD myconv.from_bytes('\xA5');
            assert(false);
        }
        catch (const std::range_error&)
        {
        }
    }
#endif
    {
        Myconv myconv("byte error");
        std::string bs = myconv.to_bytes(L"\xDA83");
        assert(bs == "byte error");
#ifndef TEST_HAS_NO_EXCEPTIONS
        try
        {
            TEST_IGNORE_NODISCARD myconv.from_bytes('\xA5');
            assert(false);
        }
        catch (const std::range_error&)
        {
        }
#endif
    }
    {
        Myconv myconv("byte error", L"wide error");
        std::string bs = myconv.to_bytes(L"\xDA83");
        assert(bs == "byte error");
        std::wstring ws = myconv.from_bytes('\xA5');
        assert(ws == L"wide error");
    }

  return 0;
}
