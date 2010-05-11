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

// template <class charT, class traits> 
//   void swap(basic_ofstream<charT, traits>& x, basic_ofstream<charT, traits>& y);

#include <fstream>
#include <cassert>

int main()
{
    {
        std::ofstream fs1("test1.dat");
        std::ofstream fs2("test2.dat");
        fs1 << 3.25;
        fs2 << 4.5;
        swap(fs1, fs2);
        fs1 << ' ' << 3.25;
        fs2 << ' ' << 4.5;
    }
    {
        std::ifstream fs("test1.dat");
        double x = 0;
        fs >> x;
        assert(x == 3.25);
        fs >> x;
        assert(x == 4.5);
    }
    remove("test1.dat");
    {
        std::ifstream fs("test2.dat");
        double x = 0;
        fs >> x;
        assert(x == 4.5);
        fs >> x;
        assert(x == 3.25);
    }
    remove("test2.dat");
    {
        std::wofstream fs1("test1.dat");
        std::wofstream fs2("test2.dat");
        fs1 << 3.25;
        fs2 << 4.5;
        swap(fs1, fs2);
        fs1 << ' ' << 3.25;
        fs2 << ' ' << 4.5;
    }
    {
        std::wifstream fs("test1.dat");
        double x = 0;
        fs >> x;
        assert(x == 3.25);
        fs >> x;
        assert(x == 4.5);
    }
    remove("test1.dat");
    {
        std::wifstream fs("test2.dat");
        double x = 0;
        fs >> x;
        assert(x == 4.5);
        fs >> x;
        assert(x == 3.25);
    }
    remove("test2.dat");
}
