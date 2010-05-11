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
// class basic_ofstream

// basic_ofstream(basic_ofstream&& rhs);

#include <fstream>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    {
        std::ofstream fso("test.dat");
        std::ofstream fs = move(fso);
        fs << 3.25;
    }
    {
        std::ifstream fs("test.dat");
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    remove("test.dat");
    {
        std::wofstream fso("test.dat");
        std::wofstream fs = move(fso);
        fs << 3.25;
    }
    {
        std::wifstream fs("test.dat");
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    remove("test.dat");
#endif
}
