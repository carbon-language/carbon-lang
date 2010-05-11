//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream

// basic_ifstream& operator=(basic_ifstream&& rhs);

#include <fstream>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    {
        std::ifstream fso("test.dat");
        std::ifstream fs;
        fs = move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    {
        std::wifstream fso("test.dat");
        std::wifstream fs;
        fs = move(fso);
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
#endif
}
