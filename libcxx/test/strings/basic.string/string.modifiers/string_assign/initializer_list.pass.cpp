//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string& assign(initializer_list<charT> il);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::string s("123");
        s.assign({'a', 'b', 'c'});
        assert(s == "abc");
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
