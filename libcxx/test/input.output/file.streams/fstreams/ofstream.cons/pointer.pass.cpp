//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// explicit basic_ofstream(const char* s, ios_base::openmode mode = ios_base::out);

#include <fstream>
#include <cassert>

int main()
{
    {
        std::ofstream fs("test.dat");
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
        std::wofstream fs("test.dat");
        fs << 3.25;
    }
    {
        std::wifstream fs("test.dat");
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    remove("test.dat");
}
