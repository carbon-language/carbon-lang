//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// istreambuf_iterator

// istreambuf_iterator() throw();

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::istreambuf_iterator<char> i;
        assert(i == std::istreambuf_iterator<char>());
    }
    {
        std::istreambuf_iterator<wchar_t> i;
        assert(i == std::istreambuf_iterator<wchar_t>());
    }
}
