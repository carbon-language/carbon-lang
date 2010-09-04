//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// iterator insert(const_iterator p, initializer_list<charT> il);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::string s("123456");
        std::string::iterator i = s.insert(s.begin() + 3, {'a', 'b', 'c'});
        assert(i - s.begin() == 3);
        assert(s == "123abc456");
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
