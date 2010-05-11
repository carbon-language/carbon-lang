//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostream_iterator

// ostream_iterator& operator=(const T& value);

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::ostringstream outf;
        std::ostream_iterator<int> i(outf);
        i = 2.4;
        assert(outf.str() == "2");
    }
    {
        std::ostringstream outf;
        std::ostream_iterator<int> i(outf, ", ");
        i = 2.4;
        assert(outf.str() == "2, ");
    }
    {
        std::wostringstream outf;
        std::ostream_iterator<int, wchar_t> i(outf);
        i = 2.4;
        assert(outf.str() == L"2");
    }
    {
        std::wostringstream outf;
        std::ostream_iterator<int, wchar_t> i(outf, L", ");
        i = 2.4;
        assert(outf.str() == L"2, ");
    }
}
