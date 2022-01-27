//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// basic_istream<charT,traits>&
//    ignore(streamsize n = 1, int_type delim = traits::eof());

// https://llvm.org/PR16427

#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int bad=-1;
    std::ostringstream os;
    os << "aaaabbbb" << static_cast<char>(bad)
       << "ccccdddd" << std::endl;
    std::string s=os.str();

    std::istringstream is(s);
    const unsigned int ignoreLen=10;
    std::istringstream::pos_type a=is.tellg();
    is.ignore(ignoreLen);
    std::istringstream::pos_type b=is.tellg();
    assert((b-a)==ignoreLen);

  return 0;
}
