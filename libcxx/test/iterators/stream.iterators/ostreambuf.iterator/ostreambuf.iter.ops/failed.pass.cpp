//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostreambuf_iterator

// bool failed() const throw();

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::ostreambuf_iterator<char> i(nullptr);
        assert(i.failed());
    }
    {
        std::ostreambuf_iterator<wchar_t> i(nullptr);
        assert(i.failed());
    }
}
