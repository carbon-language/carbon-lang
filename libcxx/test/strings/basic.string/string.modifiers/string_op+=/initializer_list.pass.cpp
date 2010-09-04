//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string& operator+=(initializer_list<charT> il);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::string s("123");
        s += {'a', 'b', 'c'};
        assert(s == "123abc");
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
