//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(initializer_list<charT> il, const Allocator& a = Allocator());

#include <string>
#include <cassert>

#include "../test_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::string s = {'a', 'b', 'c'};
        assert(s == "abc");
    }
    {
        std::wstring s;
        s = {L'a', L'b', L'c'};
        assert(s == L"abc");
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
