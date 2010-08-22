//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test that <bitset> includes <cstddef>, <string>, <stdexcept> and <iosfwd>

#include <bitset>

#ifndef _LIBCPP_CSTDDEF
#error <cstddef> has not been included
#endif

#ifndef _LIBCPP_STRING
#error <string> has not been included
#endif

#ifndef _LIBCPP_STDEXCEPT
#error <stdexcept> has not been included
#endif

#ifndef _LIBCPP_IOSFWD
#error <iosfwd> has not been included
#endif

int main()
{
}
