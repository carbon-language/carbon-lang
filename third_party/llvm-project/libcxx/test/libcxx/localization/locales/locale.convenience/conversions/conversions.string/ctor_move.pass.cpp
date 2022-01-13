//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 'do_bytes' throws a std::range_error unexpectedly
// XFAIL: LIBCXX-WINDOWS-FIXME, powerpc-ibm-aix

// XFAIL: libcpp-has-no-wide-characters

// UNSUPPORTED: c++03

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wstring_convert(wstring_convert&& other); // EXTENSION

#include <locale>
#include <codecvt>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::codecvt_utf8<wchar_t> Codecvt;
    typedef std::wstring_convert<Codecvt> Myconv;
    // create a converter and perform some conversions to generate some
    // interesting state.
    Myconv myconv;
    myconv.from_bytes("\xF1\x80\x80\x83");
    const auto old_converted = myconv.converted();
    assert(myconv.converted() == 4);
    // move construct a new converter and make sure the state is the same.
    Myconv myconv2(std::move(myconv));
    assert(myconv2.converted() == old_converted);

  return 0;
}
