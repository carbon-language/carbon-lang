//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream

// basic_ifstream();

#include <fstream>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ifstream fs;
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wifstream fs;
    }
#endif

  return 0;
}
