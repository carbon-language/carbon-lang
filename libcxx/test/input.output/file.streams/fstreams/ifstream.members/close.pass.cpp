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
// class basic_ifstream

// void close();

#include <fstream>
#include <cassert>

int main()
{
    {
        std::ifstream fs;
        assert(!fs.is_open());
        fs.open("test.dat");
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
    {
        std::wifstream fs;
        assert(!fs.is_open());
        fs.open("test.dat");
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
}
