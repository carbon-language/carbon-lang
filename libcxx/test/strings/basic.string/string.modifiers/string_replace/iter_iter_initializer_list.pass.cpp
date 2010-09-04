//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string& replace(iterator i1, iterator i2, initializer_list<charT> il);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::string s("123def456");
        s.replace(s.begin() + 3, s.begin() + 6, {'a', 'b', 'c'});
        assert(s == "123abc456");
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
